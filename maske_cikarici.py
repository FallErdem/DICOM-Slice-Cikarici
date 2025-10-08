import os
import re
import json
import cv2
import numpy as np
import shutil
import tempfile
import unicodedata
import pydicom
import SimpleITK as sitk
from pathlib import Path
from rt_utils import RTStructBuilder
from collections import defaultdict
from typing import Dict, Tuple, List, Set

# ============================================
#  GTV ve BRAIN ROI eşleştiriciler (sağlam)
# ============================================
def _strip_punct_space(s: str) -> str:
    """lower + harf/rakam dışını temizle (eşleştirme için güvenli)."""
    return re.sub(r'[^0-9a-z]+', '', (s or '').casefold())

def is_gtv_like(text: str) -> bool:
    """
    GTV eşleştirici:
      - gtv, _gtv, agtv, gtv-1, gtvp, gtvn, gtvboost, igtv
      - gross tumor/tumour volume
    """
    if not text:
        return False
    t = unicodedata.normalize('NFKD', text)
    t = ''.join(ch for ch in t if not unicodedata.combining(ch))
    t = _strip_punct_space(t)
    if 'gtv' in t or 'igtv' in t:
        return True
    soft = re.sub(r'[\-_\.]+', ' ', text, flags=re.IGNORECASE).casefold()
    return bool(re.search(r'gross\s*tumou?r\s*volume', soft, flags=re.IGNORECASE))

def is_brain_roi(name: str) -> bool:
    """
    ROI adı 'beyin/brain' kümesine yakınsa True:
     - Brain, Beyin, Brainstem, Hipofiz/Pituitary, Hippocampus (liste genişletilebilir)
    """
    if not name:
        return False
    n = _strip_punct_space(name)
    brain_tokens = ["brain", "beyin", "brainstem", "hipofiz", "pituitary", "hippocampus"]
    return any(tok in n for tok in brain_tokens)

# =========================
# Genel yardımcılar
# =========================
def find_folder_with_prefix(main_folder, prefix):
    main_path = Path(main_folder)
    for folder in main_path.iterdir():
        if folder.is_dir() and folder.name.upper().startswith(prefix.upper()):
            return folder
    return None

def find_all_rtstructs(main_folder):
    """Klasör ağacındaki tüm RTSTRUCT’leri bul."""
    main_path = Path(main_folder)
    candidates = []
    for dcm_file in main_path.rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(dcm_file, stop_before_pixels=True)
            if getattr(ds, "Modality", "") == "RTSTRUCT":
                candidates.append(dcm_file)
        except Exception:
            continue
    return candidates

def list_rois_in_rtstruct(rtstruct_path):
    """Bir RTSTRUCT içindeki ROIName + Observations bilgisini döndür."""
    ds = pydicom.dcmread(rtstruct_path)
    roi_number_to_name = {}
    if hasattr(ds, 'StructureSetROISequence'):
        for roi in ds.StructureSetROISequence:
            roi_number_to_name[getattr(roi, 'ROINumber', None)] = getattr(roi, 'ROIName', '')

    obs = {}
    if hasattr(ds, 'RTROIObservationsSequence'):
        for ob in ds.RTROIObservationsSequence:
            ref_num = getattr(ob, 'ReferencedROINumber', None)
            obs_label = getattr(ob, 'ROIObservationLabel', '')
            interp_type = getattr(ob, 'RTROIInterpretedType', '')
            obs.setdefault(ref_num, []).append((obs_label, interp_type))

    items = []
    for roinum, roiname in roi_number_to_name.items():
        entries = obs.get(roinum, [])
        texts = [roiname] + [x for pair in entries for x in pair if x]
        items.append({
            "ROINumber": roinum,
            "ROIName": roiname,
            "Observations": entries,
            "IsGTV": any(is_gtv_like(t) for t in texts)
        })
    return items

def choose_best_rtstruct(main_folder):
    """
    GTV içeren ilk RTSTRUCT'u, yoksa ROI sayısı en yüksek olanı seç.
    (Tüm RS'leri listeler; debug amaçlı ROI adlarını loglar.)
    """
    candidates = find_all_rtstructs(main_folder)
    if not candidates:
        return None, []
    print(f"Found {len(candidates)} RTSTRUCT candidates.")
    best_rtstruct, best_items = None, []
    gtv_rtstruct, gtv_items = None, []
    for rs_path in candidates:
        try:
            items = list_rois_in_rtstruct(rs_path)
            roi_names = [it["ROIName"] for it in items]
            has_gtv = any(it["IsGTV"] for it in items)
            print(f"RS: {rs_path} -> ROIs={roi_names}  hasGTV={has_gtv}")
            if has_gtv and gtv_rtstruct is None:
                gtv_rtstruct, gtv_items = rs_path, items
            if len(items) > len(best_items):
                best_rtstruct, best_items = rs_path, items
        except Exception as e:
            print(f"Error scanning RS {rs_path}: {e}")
    return (gtv_rtstruct or best_rtstruct), (gtv_items if gtv_rtstruct else best_items)

def get_referenced_series_uids(rtstruct_path):
    """RS -> ReferencedFrameOfReferenceSequence -> ... -> RTReferencedSeriesSequence -> SeriesInstanceUID seti."""
    uids = set()
    try:
        ds = pydicom.dcmread(rtstruct_path, stop_before_pixels=True)
        if hasattr(ds, "ReferencedFrameOfReferenceSequence"):
            for rfor in ds.ReferencedFrameOfReferenceSequence:
                if hasattr(rfor, "RTReferencedStudySequence"):
                    for rstudy in rfor.RTReferencedStudySequence:
                        if hasattr(rstudy, "RTReferencedSeriesSequence"):
                            for rseries in rstudy.RTReferencedSeriesSequence:
                                uid = getattr(rseries, "SeriesInstanceUID", None)
                                if uid:
                                    uids.add(uid)
    except Exception as e:
        print(f"ERR get_referenced_series_uids: {e}")
    return uids

def get_frame_of_reference_uid_from_first_slice(ct_folder, target_series_uid):
    """Seçilen serinin ilk diliminden FoR UID'i getir (opsiyonel tutarlılık kontrolü)."""
    try:
        for p in Path(ct_folder).rglob("*.dcm"):
            ds = pydicom.dcmread(p, stop_before_pixels=True)
            if getattr(ds, "SeriesInstanceUID", None) == target_series_uid:
                return getattr(ds, "FrameOfReferenceUID", None)
    except Exception:
        pass
    return None

# ============================================
#  CT tarafında SOP -> Seri eşleme haritası
# ============================================
def build_sop_to_series_map(ct_root: Path):
    """
    CT DICOM'larında SOPInstanceUID -> (SeriesInstanceUID, SeriesDescription)
    ve SeriesInstanceUID -> meta (desc, modality, NumSlices) haritalarını üret.
    """
    sop2series = {}
    series_meta = {}
    counts = defaultdict(int)

    for p in ct_root.rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True)
            if getattr(ds, "Modality", "CT") != "CT":
                continue
            suid = getattr(ds, "SeriesInstanceUID", None)
            sop  = getattr(ds, "SOPInstanceUID", None)
            if not suid or not sop:
                continue
            sop2series[sop] = (suid, getattr(ds, "SeriesDescription", ""))
            counts[suid] += 1
            if suid not in series_meta:
                series_meta[suid] = {
                    "SeriesDescription": getattr(ds, "SeriesDescription", ""),
                    "Modality": "CT",
                }
        except Exception:
            continue

    for suid in series_meta:
        series_meta[suid]["NumSlices"] = counts[suid]
    return sop2series, series_meta

# ============================================
#  RS içinde ROI -> CT Serisi eşlemesi
# ============================================
def map_roi_to_series_uids(rtstruct_path: Path, sop2series: dict):
    """
    RTSTRUCT içinden:
      - ROINumber -> dokunduğu SeriesInstanceUID seti
      - ROINumber -> ROIName
    döndür.
    """
    roi_to_series = {}
    roi_number_to_name = {}
    try:
        ds = pydicom.dcmread(rtstruct_path)

        # ROI isimleri
        if hasattr(ds, "StructureSetROISequence"):
            for roi in ds.StructureSetROISequence:
                roi_number_to_name[getattr(roi, 'ROINumber', None)] = getattr(roi, 'ROIName', '')

        # ROIContourSequence -> ContourSequence -> ContourImageSequence -> ReferencedSOPInstanceUID
        if hasattr(ds, "ROIContourSequence"):
            for rc in ds.ROIContourSequence:
                roinum = getattr(rc, "ReferencedROINumber", None)
                touched = set()
                if hasattr(rc, "ContourSequence"):
                    for cs in rc.ContourSequence:
                        if hasattr(cs, "ContourImageSequence"):
                            for cis in cs.ContourImageSequence:
                                sop = getattr(cis, "ReferencedSOPInstanceUID", None)
                                if sop and sop in sop2series:
                                    suid, _ = sop2series[sop]
                                    touched.add(suid)
                if roinum is not None:
                    roi_to_series[roinum] = roi_to_series.get(roinum, set()) | touched

    except Exception as e:
        print(f"ROI→Series map error: {e}")

    return roi_to_series, roi_number_to_name

# ============================================
#  Seçili seri için SOP -> z index haritası
# ============================================
def build_sop_to_z_map(series_dir: Path) -> Dict[str, int]:
    """
    Tek bir CT serisini içeren klasörden SOPInstanceUID -> z_index haritası çıkar.
    Sıralama: ImagePositionPatient[2] (yoksa InstanceNumber).
    """
    entries = []
    for p in series_dir.rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True)
            sop = getattr(ds, "SOPInstanceUID", None)
            if not sop:
                continue
            if hasattr(ds, "ImagePositionPatient") and len(ds.ImagePositionPatient) == 3:
                key = float(ds.ImagePositionPatient[2])
            else:
                key = int(getattr(ds, "InstanceNumber", 0))
            entries.append((key, sop))
        except Exception:
            continue
    entries.sort(key=lambda x: x[0])
    sop_to_z = {}
    for i, (_, sop) in enumerate(entries):
        sop_to_z[sop] = i
    return sop_to_z

def collect_brain_slice_indices_from_rtstruct(rtstruct_path: Path, sop_to_z: Dict[str, int], roinum2name: Dict[int,str]) -> Set[int]:
    """
    RTSTRUCT içindeki brain-like ROINumber'ların ContourImageSequence referanslarından,
    seçili serideki z indekslerini topla.
    """
    zset: Set[int] = set()
    try:
        ds = pydicom.dcmread(rtstruct_path)
        if hasattr(ds, "ROIContourSequence"):
            for rc in ds.ROIContourSequence:
                roinum = getattr(rc, "ReferencedROINumber", None)
                roi_name = roinum2name.get(roinum, "")
                if not is_brain_roi(roi_name):
                    continue
                if hasattr(rc, "ContourSequence"):
                    for cs in rc.ContourSequence:
                        if hasattr(cs, "ContourImageSequence"):
                            for cis in cs.ContourImageSequence:
                                sop = getattr(cis, "ReferencedSOPInstanceUID", None)
                                if sop in sop_to_z:
                                    zset.add(sop_to_z[sop])
    except Exception as e:
        print(f"collect_brain_slice_indices error: {e}")
    return zset

# ============================================
#  CT yükleyiciler
# ============================================
def materialize_series_to_temp(ct_folder: Path, target_series_uid: str) -> Path:
    """
    Hedef CT SeriesInstanceUID'e ait dilimleri tek klasöre hardlink/kopya ile topla.
    rt-utils bu yolu kullanacak (yalnızca doğru seri görsün).
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="ct_series_"))
    created = 0
    for p in ct_folder.rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True)
            if getattr(ds, "Modality", "CT") != "CT":
                continue
            if getattr(ds, "SeriesInstanceUID", None) != target_series_uid:
                continue
            dst = tmp_dir / p.name
            try:
                os.link(p, dst)  # mümkünse hardlink
            except Exception:
                shutil.copy2(p, dst)  # değilse kopya
            created += 1
        except Exception:
            continue
    if created == 0:
        raise RuntimeError("No CT files materialized for target SeriesInstanceUID")
    print(f"Materialized {created} CT slices into temp dir: {tmp_dir}")
    return tmp_dir

def load_ct_images(ct_folder, target_series_uid=None, check_for_uid_only=True):
    """CT hacmini yükle: önce SimpleITK serisi, olmazsa tek tek DICOM."""
    # --- SimpleITK ile ---
    try:
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(ct_folder))
        if series_ids:
            selected = None
            if target_series_uid:
                for sid in series_ids:
                    if sid == target_series_uid:
                        selected = sid
                        break
            if not selected and not check_for_uid_only:
                best_sid, best_len = None, -1
                for sid in series_ids:
                    files = reader.GetGDCMSeriesFileNames(str(ct_folder), sid)
                    if len(files) > best_len:
                        best_sid, best_len = sid, len(files)
                selected = best_sid
            if selected:
                files = reader.GetGDCMSeriesFileNames(str(ct_folder), selected)
                print(f"Selected CT series by UID: {selected} with {len(files)} slices")
                reader.SetFileNames(files)
                ct_image = reader.Execute()
                ct_array = sitk.GetArrayFromImage(ct_image)  # (z,y,x)
                print(f"Loaded CT series with shape: {ct_array.shape}")
                return ct_array, True
            else:
                print("SimpleITK: Target SeriesInstanceUID not found among series IDs.")
    except Exception as e:
        print(f"SimpleITK failed: {e}")

    # --- Tek tek DICOM ile ---
    try:
        print("Trying individual file loading...")
        dcm_paths = list(Path(ct_folder).rglob("*.dcm"))
        if not dcm_paths:
            print("No DICOM files found in CT folder!")
            return None, False

        filtered = []
        if target_series_uid:
            for p in dcm_paths:
                try:
                    ds = pydicom.dcmread(p, stop_before_pixels=True)
                    if getattr(ds, "SeriesInstanceUID", None) == target_series_uid and getattr(ds, "Modality", "CT") == "CT":
                        filtered.append(p)
                except Exception:
                    continue
            if not filtered:
                print("Manual load: No files match the target SeriesInstanceUID.")
                if check_for_uid_only:
                    return None, False
        else:
            for p in dcm_paths:
                try:
                    ds = pydicom.dcmread(p, stop_before_pixels=True)
                    if getattr(ds, "Modality", "CT") == "CT":
                        filtered.append(p)
                except Exception:
                    continue
            if not filtered:
                print("Manual load: No CT files found.")
                return None, False

        def get_sort_key(filepath):
            """z (ImagePositionPatient[2]) yoksa InstanceNumber ile sırala."""
            try:
                ds_local = pydicom.dcmread(filepath, stop_before_pixels=True)
                if hasattr(ds_local, "ImagePositionPatient") and len(ds_local.ImagePositionPatient) == 3:
                    return float(ds_local.ImagePositionPatient[2])  # Z
                return int(getattr(ds_local, "InstanceNumber", 0))
            except Exception:
                return 0

        filtered.sort(key=get_sort_key)

        first_ds = pydicom.dcmread(filtered[0])
        if not hasattr(first_ds, 'pixel_array'):
            print("First DICOM file has no pixel data!")
            return None, False

        height, width = first_ds.pixel_array.shape
        num_slices = len(filtered)
        volume = np.zeros((num_slices, height, width), dtype=np.float32)

        print(f"Loading {num_slices} CT slices...")
        idx = 0
        for dcm_file in filtered:
            try:
                ds_local = pydicom.dcmread(dcm_file)
                if getattr(ds_local, 'Modality', 'CT') != 'CT' or not hasattr(ds_local, 'pixel_array'):
                    continue
                px = ds_local.pixel_array.astype(np.float32)
                slope = getattr(ds_local, 'RescaleSlope', 1.0)
                intercept = getattr(ds_local, 'RescaleIntercept', 0.0)
                volume[idx] = px * slope + intercept
                idx += 1
            except Exception as e:
                print(f"Error loading slice {idx}: {e}")

        if idx == 0:
            print("Manual load: Filtered files did not yield any CT pixels.")
            return None, False
        if idx != num_slices:
            volume = volume[:idx]

        print(f"Loaded CT volume with shape: {volume.shape}")
        return volume, False

    except Exception as e:
        print(f"Individual file loading failed: {e}")
        return None, False

def apply_window_level(image, center=40, width=400):
    """CT penceresi uygula ve 8-bit'e normalle (örn. Window=400/Level=40)."""
    min_val = center - width / 2.0
    max_val = center + width / 2.0
    windowed = np.clip(image, min_val, max_val)
    normalized = ((windowed - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
    return normalized

def make_output_dirs(base_dir: Path) -> Dict[str, Path]:
    """
    Çıktıları alt klasörlere böler:
      base/ct, base/overlay, base/mask, base/healthy
    """
    paths = {
        "ct": base_dir / "ct",
        "overlay": base_dir / "overlay",
        "mask": base_dir / "mask",
        "healthy": base_dir / "healthy",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths

# =========================
# Tek vaka işleyici
# =========================
def convert_dicom_to_png(main_folder, output_folder):
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Processing case: {main_folder} ===")

    # 1) CT klasörü
    ct_folder = find_folder_with_prefix(main_folder, "CT_")
    if not ct_folder:
        print("ERROR: No CT_ folder found!")
        return False
    print(f"CT folder: {ct_folder}")

    # 2) RTSTRUCT seçimi (tüm RS'leri tarayıp en iyi adayı alır)
    rtstruct_file, rs_items = choose_best_rtstruct(main_folder)
    if not rtstruct_file:
        print("ERROR: No RT-STRUCT file found anywhere in the case folder!")
        return False
    print(f"Selected RTSTRUCT: {rtstruct_file}")

    # 3) RS -> Referenced SeriesInstanceUID(ler)
    ref_uids = get_referenced_series_uids(rtstruct_file)
    print(f"Referenced SeriesInstanceUIDs in RS: {list(ref_uids)}")

    # 4) CT tarafında SOP→Seri eşlemesini kur
    sop2series, series_meta = build_sop_to_series_map(Path(ct_folder))

    # 5) RS içinde ROI'lerin hangi serilere dokunduğunu çıkar
    roi2series, roinum2name = map_roi_to_series_uids(Path(rtstruct_file), sop2series)

    # 6) Brain-like ROI'leri bul ve dokundukları serileri topla
    brain_series: Set[str] = set()
    for roinum, name in roinum2name.items():
        if is_brain_roi(name):
            brain_series |= set(roi2series.get(roinum, set()))
    print(f"Series touched by 'Brain-like' ROIs: {list(brain_series)}")

    # 7) Seçim: RS'in referans verdikleri ∩ Brain ROI serileri
    candidates = sorted([uid for uid in brain_series if (not ref_uids or uid in ref_uids)])
    if not candidates:
        print("No CT series touched by a 'Brain'-like ROI within the referenced series. Skipping.")
        return False

    # Adayları açıklayıcı biçimde listele
    print("Brain candidates (SeriesUID | Desc | NumSlices):")
    for uid in candidates:
        meta = series_meta.get(uid, {})
        print(f"  - {uid} | '{meta.get('SeriesDescription','')}' | {meta.get('NumSlices',0)}")

    # 8) Birden fazla varsa en çok slice'lı olanı seç
    candidates.sort(key=lambda u: (-series_meta.get(u, {}).get("NumSlices", 0), u))
    target_uid = candidates[0]
    meta = series_meta.get(target_uid, {})
    print(f"Selected CT Series by ROI(Brain): UID={target_uid} | "
          f"Desc='{meta.get('SeriesDescription','')}' | Slices={meta.get('NumSlices',0)}")

    # (Opsiyonel) FoR kontrol
    try:
        ds_rs = pydicom.dcmread(rtstruct_file, stop_before_pixels=True)
        rs_for_uid = getattr(ds_rs, "FrameOfReferenceUID", None)
        ct_for_uid = get_frame_of_reference_uid_from_first_slice(ct_folder, target_uid)
        print(f"FrameOfReferenceUID RS={rs_for_uid}  CT={ct_for_uid}")
        if rs_for_uid and ct_for_uid and rs_for_uid != ct_for_uid:
            print("WARN: FrameOfReferenceUID mismatch between RS and CT. Check registration!")
    except Exception as e:
        print(f"FoR check error: {e}")

    # 9) CT hacmini yükle (sadece hedef seri)
    ct_volume, used_sitk = load_ct_images(ct_folder, target_series_uid=target_uid, check_for_uid_only=True)
    if ct_volume is None:
        print("ERROR: Failed to load CT images for the selected series!")
        return False

    # 10) rt-utils, yalnızca hedef seriyi görsün diye temp klasör hazırla
    ct_series_tmp = None
    try:
        ct_series_tmp = materialize_series_to_temp(Path(ct_folder), target_uid)
    except Exception as e:
        print(f"ERROR creating temp CT series dir: {e}")
        return False

    # 10.1) Seçili seri için SOP->z haritası (brain-only healthy slice çıkarmak için)
    sop_to_z = build_sop_to_z_map(ct_series_tmp)

    # 11) GTV isimleri (seçili RS)
    print("=== ROI Diagnostic (selected RS) ===")
    for it in rs_items:
        print(f"ROINumber={it['ROINumber']}  ROIName='{it['ROIName']}'  Obs={it['Observations']}  IsGTV={it['IsGTV']}")

    gtv_structures = [it["ROIName"] for it in rs_items if it["IsGTV"] and it["ROIName"]]
    if not gtv_structures:
        print("ERROR: No GTV-like structures in selected RTSTRUCT. Nothing to export.")
        if ct_series_tmp:
            shutil.rmtree(ct_series_tmp, ignore_errors=True)
        return False

    # 12) RT-STRUCT'tan GTV maskeleri (yalnız hedef seri ile)
    try:
        rtstruct = RTStructBuilder.create_from(
            dicom_series_path=str(ct_series_tmp),
            rt_struct_path=str(rtstruct_file)
        )
        print("RT-STRUCT loaded successfully")
        all_rois = rtstruct.get_roi_names()
        print(f"All ROIs (rt-utils): {all_rois}")

        # Önce rt-utils ROI isimlerinde GTV yakala; yoksa RS'ten gelen GTV isimleriyle dener
        gtv_roi_names = [name for name in all_rois if is_gtv_like(name)]
        if not gtv_roi_names:
            inter = [n for n in gtv_structures if n in all_rois]
            gtv_roi_names = inter if inter else gtv_structures

        if not gtv_roi_names:
            print("ERROR: No GTV-like ROIs found in either rt-utils or DICOM sequences.")
            if ct_series_tmp:
                shutil.rmtree(ct_series_tmp, ignore_errors=True)
            return False

        print(f"GTV-like ROIs to use: {gtv_roi_names}")

        combined_mask = None
        for roi_name in gtv_roi_names:
            try:
                print(f"Processing GTV ROI: {roi_name}")
                roi_mask = rtstruct.get_roi_mask_by_name(roi_name)  # beklenen (z,y,x) aynı seri
                if roi_mask.shape != ct_volume.shape:
                    print(f"Mask shape {roi_mask.shape} != CT {ct_volume.shape}. Trying axis fixes...")
                    candidates_axes = [(2,0,1),(2,1,0),(1,2,0),(0,2,1)]
                    fixed = False
                    for axes in candidates_axes:
                        if len(roi_mask.shape) == 3 and roi_mask.transpose(axes).shape == ct_volume.shape:
                            roi_mask = roi_mask.transpose(axes)
                            print(f"Fixed by transpose{axes}")
                            fixed = True
                            break
                    if not fixed:
                        raise ValueError(f"Cannot align ROI mask {roi_mask.shape} to CT {ct_volume.shape}")
                roi_mask_bool = roi_mask.astype(bool)
                combined_mask = roi_mask_bool if combined_mask is None else np.logical_or(combined_mask, roi_mask_bool)
            except Exception as e:
                print(f"Error processing ROI '{roi_name}': {e}")

        if combined_mask is None:
            print("ERROR: No GTV masks could be loaded!")
            if ct_series_tmp:
                shutil.rmtree(ct_series_tmp, ignore_errors=True)
            return False

    except Exception as e:
        print(f"ERROR loading/aligning RT-STRUCT: {e}")
        if ct_series_tmp:
            shutil.rmtree(ct_series_tmp, ignore_errors=True)
        return False

    # 13) GTV içeren slice'ları bul
    slices_with_gtv = sorted([z for z in range(combined_mask.shape[0]) if np.any(combined_mask[z])])
    if not slices_with_gtv:
        print("No slices contain GTV after mask alignment. Skipping.")
        if ct_series_tmp:
            shutil.rmtree(ct_series_tmp, ignore_errors=True)
        return False
    gtv_slice_set = set(slices_with_gtv)

    # 13.1) Brain ROI içeren slice'ları çıkar (Contour referanslarıyla)
    brain_slice_indices = collect_brain_slice_indices_from_rtstruct(Path(rtstruct_file), sop_to_z, roinum2name)

    # 13.2) Healthy = Brain var ∧ GTV yok
    healthy_slices = sorted(list(brain_slice_indices - gtv_slice_set))

    print(f"Found GTV in {len(slices_with_gtv)} slices: {min(slices_with_gtv)} to {max(slices_with_gtv)}")
    print(f"Found HEALTHY brain-only slices (no GTV): {len(healthy_slices)}")

    # 14) PNG çıktıları: ayrı klasörler (ct / overlay / mask / healthy)
    out_dirs = make_output_dirs(output_path)
    saved_count = 0

    # küçük metadata bırak
    meta_json = {
        "selected_series_uid": target_uid,
        "selected_series_desc": meta.get('SeriesDescription', ''),
        "num_slices_in_series": meta.get('NumSlices', 0),
        "exported_slices": slices_with_gtv,
        "healthy_slices": healthy_slices,
        "gtv_rois": gtv_roi_names,
    }
    (output_path / "export_meta.json").write_text(json.dumps(meta_json, indent=2), encoding="utf-8")

    # 14.a) GTV'li slice'lar (ct / overlay / mask)
    for z in slices_with_gtv:
        ct_slice = ct_volume[z]
        ct_windowed = apply_window_level(ct_slice, center=40, width=400)

        ct_rgb = cv2.cvtColor(ct_windowed, cv2.COLOR_GRAY2RGB)
        overlay = ct_rgb.copy()
        overlay[combined_mask[z]] = [255, 0, 0]  # kırmızı GTV
        blended = cv2.addWeighted(ct_rgb, 0.7, overlay, 0.3, 0.0)

        fname = f"z{z:04d}.png"
        cv2.imwrite(str(out_dirs["ct"]      / fname), ct_windowed)
        cv2.imwrite(str(out_dirs["overlay"] / fname), blended)
        mask_img = (combined_mask[z] * 255).astype(np.uint8)
        cv2.imwrite(str(out_dirs["mask"]    / fname), mask_img)
        saved_count += 1

    # 14.b) HEALTHY slice'lar (yalnızca CT)
    healthy_saved = 0
    for z in healthy_slices:
        ct_slice = ct_volume[z]
        ct_windowed = apply_window_level(ct_slice, center=40, width=400)
        fname = f"z{z:04d}.png"
        cv2.imwrite(str(out_dirs["healthy"] / fname), ct_windowed)
        healthy_saved += 1

    print(f"Saved {saved_count} GTV slices to '{output_folder}\\ct/overlay/mask'")
    print(f"Saved {healthy_saved} healthy brain-only slices to '{output_folder}\\healthy'")

    # 15) Geçici klasörü temizle
    if ct_series_tmp:
        shutil.rmtree(ct_series_tmp, ignore_errors=True)

    return True

# =========================
# Batch bulucu & çalıştırıcı
# =========================
def is_case_dir(path: Path) -> bool:
    """Bir vaka klasörü mü? CT_* klasörü ve (altında) en az bir RTSTRUCT olmalı."""
    if not path.is_dir():
        return False
    ct = find_folder_with_prefix(path, "CT_")
    if not ct:
        return False
    rs = find_all_rtstructs(path)
    return len(rs) > 0

def iter_case_dirs_under(root_dir: Path):
    """Kök altında dolaşıp vaka klasörleri üretir."""
    # Önce kökün kendisi vaka ise onu ver
    if is_case_dir(root_dir):
        yield root_dir
        return
    # Değilse altları tara
    for p in root_dir.rglob("*"):
        try:
            if p.is_dir() and is_case_dir(p):
                yield p
        except Exception:
            continue

def batch_process(root_dirs: List[str], output_root: str):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "total_cases": 0,
        "processed": 0,
        "succeeded": 0,
        "failed": 0,
        "fail_list": []
    }

    for root in root_dirs:
        root_path = Path(root)
        if not root_path.exists():
            print(f"\n!! Skipping missing root: {root}")
            continue

        print(f"\n##### Scanning root: {root} #####")
        for case_dir in iter_case_dirs_under(root_path):
            summary["total_cases"] += 1
            # Çıkış yolunu, köke göre ayna yapıda kur
            try:
                rel = case_dir.relative_to(root_path)
                out_dir = output_root / root_path.name / rel
            except Exception:
                # Güvenli fallback: sadece son iki segment
                parts = case_dir.parts[-2:]
                out_dir = output_root / root_path.name / Path(*parts)

            print(f"\n--- Case: {case_dir} -> Output: {out_dir} ---")
            ok = convert_dicom_to_png(str(case_dir), str(out_dir))
            summary["processed"] += 1
            if ok:
                summary["succeeded"] += 1
            else:
                summary["failed"] += 1
                summary["fail_list"].append(str(case_dir))

    # Özet yaz
    print("\n====== BATCH SUMMARY ======")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    (output_root / "batch_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"Summary saved to: {output_root / 'batch_summary.json'}")

# =========================
# Örnek kullanım
# =========================
if __name__ == "__main__":
    root_dirs = [
        r"C:\Users\YZE\Desktop\Dcom_Verılerı\Fatih Bey\BEYIN4\beyin4.1",
        r"C:\Users\YZE\Desktop\Dcom_Verılerı\Fatih Bey\BEYIN4\beyin4.2",
        r"C:\Users\YZE\Desktop\Dcom_Verılerı\Fatih Bey\BEYIN1\beyin1.1",
        r"C:\Users\YZE\Desktop\Dcom_Verılerı\Fatih Bey\BEYIN1\beyin1.2",
        r"C:\Users\YZE\Desktop\Dcom_Verılerı\Fatih Bey\BEYIN2",
        r"C:\Users\YZE\Desktop\Dcom_Verılerı\Fatih Bey\BEYIN3\beyin3.1",
        r"C:\Users\YZE\Desktop\Dcom_Verılerı\Fatih Bey\BEYIN3\beyin3.2",
        r"C:\Users\YZE\Desktop\Dcom_Verılerı\Fatih Bey\BEYIN3\beyin3.3",
        r"C:\Users\YZE\Desktop\Dcom_Verılerı\Fatih Bey\BEYIN3\beyin3.4",
        r"C:\Users\YZE\Desktop\Dcom_Verılerı\Fatih Bey\BEYİN5\beyin5.1",
        r"C:\Users\YZE\Desktop\Dcom_Verılerı\Fatih Bey\BEYİN5\beyin5.2",
    ]

    output_root = r"C:\Users\YZE\Desktop\output_png_batch"

    batch_process(root_dirs, output_root)
