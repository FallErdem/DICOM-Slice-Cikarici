#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Optional, Callable, Tuple, List, Any

import numpy as np
from PIL import Image
import pydicom
import SimpleITK as sitk
from rt_utils import RTStructBuilder
import csv

# ------------------ KULLANICI AYARLARI ------------------
class Config:
    MODE = 'BATCH'  # 'SINGLE' veya 'BATCH'
    SINGLE_PATIENT_ID = '1548735'
    INPUT_DIRECTORY = Path(r"D:/beyin_3_4_5/beyin3.2")
    OUTPUT_DIRECTORY = Path(r"D:/beyin_maske_ciktilari_FINAL")

    ROI_PREFIX = 'gtv'

    SAVE_OVERLAYS = True
    SAVE_MASK_PNG = True
    SAVE_FULL_MASK_AS = 'none'  # 'none' | 'npy' | 'nii'

    LOG_FILENAME = "islem_gunlugu_v10.log"
# -------------------------------------------------------


def setup_logging(log_path: Path):
    log_path.parent.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    repl = {'ı':'i', 'İ':'i', 'ğ':'g', 'Ğ':'g', 'ü':'u', 'Ü':'u', 'ş':'s', 'Ş':'s', 'ö':'o', 'Ö':'o', 'ç':'c', 'Ç':'c'}
    t = text.lower()
    for a,b in repl.items():
        t = t.replace(a,b)
    return t


def find_rtstruct_file(rs_dir: Path) -> Optional[Path]:
    for file_path in rs_dir.rglob('*'):
        if file_path.is_file():
            try:
                ds = pydicom.dcmread(str(file_path), stop_before_pixels=True, force=True)
                if hasattr(ds, "SOPClassUID") and ds.SOPClassUID == pydicom.uid.RTStructureSetStorage:
                    return file_path
            except Exception:
                continue
    return None


def select_gtv_roi_names(rtstruct: RTStructBuilder, prefix: str) -> List[str]:
    all_rois = rtstruct.get_roi_names() or []
    pref = normalize_text(prefix)
    matches = [r for r in all_rois if normalize_text(r).startswith(pref)]
    return matches


def compute_zpos_from_ds(ds: pydicom.dataset.FileDataset) -> float:
    try:
        return float(ds.ImagePositionPatient[2])
    except Exception:
        try:
            return float(ds.SliceLocation)
        except Exception:
            try:
                return float(ds.InstanceNumber)
            except Exception:
                return 0.0


def prepare_ct_reader(series_data: Any) -> Tuple[Callable[[int], np.ndarray], int, int, int, dict]:
    if isinstance(series_data, sitk.Image):
        img = series_data
        size = list(img.GetSize())
        cols, rows = int(size[0]), int(size[1])
        num_slices = int(size[2]) if len(size) > 2 else 1

        def get_slice(z: int) -> np.ndarray:
            extractor = sitk.ExtractImageFilter()
            size3 = [cols, rows, 0]
            index3 = [0, 0, int(z)]
            slice_img = extractor.Execute(img, size3, index3)
            arr = sitk.GetArrayFromImage(slice_img)
            return arr.astype(np.float32)

        meta = {
            'spacing': img.GetSpacing(),
            'origin': img.GetOrigin(),
            'direction': img.GetDirection()
        }
        return get_slice, num_slices, rows, cols, meta

    if isinstance(series_data, (list, tuple)) and len(series_data) > 0:
        first = series_data[0]
        if isinstance(first, pydicom.dataset.FileDataset):
            slices = list(series_data)
            slices_sorted = sorted(slices, key=compute_zpos_from_ds)
            rows = int(slices_sorted[0].Rows)
            cols = int(slices_sorted[0].Columns)
            num_slices = len(slices_sorted)
            zpos_list = [compute_zpos_from_ds(s) for s in slices_sorted]

            def get_slice(z: int) -> np.ndarray:
                ds = slices_sorted[int(z)]
                arr = ds.pixel_array.astype(np.float32)
                slope = float(getattr(ds, 'RescaleSlope', 1.0))
                intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
                arr = arr * slope + intercept
                return arr

            meta = {
                'spacing': (float(slices_sorted[0].PixelSpacing[0]),
                            float(slices_sorted[0].PixelSpacing[1]),
                            abs(zpos_list[1] - zpos_list[0]) if num_slices > 1 else 1.0),
                'origin': None,
                'direction': None
            }
            return get_slice, num_slices, rows, cols, meta

        if isinstance(first, sitk.Image):
            slices = list(series_data)
            rows = int(slices[0].GetSize()[1])
            cols = int(slices[0].GetSize()[0])
            num_slices = len(slices)

            def get_slice(z: int) -> np.ndarray:
                arr = sitk.GetArrayFromImage(slices[int(z)])
                return arr.astype(np.float32)

            meta = {'spacing': None, 'origin': None, 'direction': None}
            return get_slice, num_slices, rows, cols, meta

    raise TypeError(f"Hazırlanamayan series_data tipi: {type(series_data)}")


def align_mask_to_target(mask: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    if mask.shape == target_shape:
        return mask.astype(bool)
    from itertools import permutations
    for perm in permutations(range(3)):
        permuted = np.transpose(mask, perm)
        if permuted.shape == target_shape:
            logging.info(f"Mask için uygun permütasyon bulundu: {perm}")
            return permuted.astype(bool)
    raise ValueError(f"Mask shape {mask.shape} target shape {target_shape} ile eşleşemedi.")


def slice_to_u8(slice_arr: np.ndarray) -> np.ndarray:
    if slice_arr.size == 0:
        return slice_arr.astype(np.uint8)
    lo, hi = np.percentile(slice_arr, (0.5, 99.5))
    if hi <= lo:
        lo, hi = float(np.min(slice_arr)), float(np.max(slice_arr))
    if hi <= lo:
        return np.zeros_like(slice_arr, dtype=np.uint8)
    arr = np.clip(slice_arr, lo, hi)
    arr = (arr - lo) / (hi - lo)
    return (arr * 255.0).astype(np.uint8)


def save_mask_png(mask_slice: np.ndarray, out_path: Path):
    Image.fromarray((mask_slice.astype(np.uint8) * 255)).save(out_path)


def save_overlay_png(ct_u8_slice: np.ndarray, mask_slice_bool: np.ndarray, out_path: Path):
    rgb = np.stack([ct_u8_slice] * 3, axis=-1)
    overlay_color = np.array([255, 0, 0], dtype=np.uint8)
    mask_idx = mask_slice_bool.astype(bool)
    if mask_idx.any():
        rgb[mask_idx] = (rgb[mask_idx] * 0.6 + overlay_color * 0.4).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(out_path)


def process_patient(patient_path: Path, config: Config) -> dict:
    patient_id = patient_path.name
    logging.info(f"--- HASTA İŞLEMİ BAŞLATILDI: {patient_id} ---")

    try:
        ct_dir = next(patient_path.glob("CT_*"))
        try:
            rs_dir = next(patient_path.glob("RS_*"))
        except StopIteration:
            rs_dir = next(patient_path.glob("RT_*"))

        rtstruct_file = find_rtstruct_file(rs_dir)
        if not rtstruct_file:
            raise FileNotFoundError("RTSTRUCT bulunamadı")
    except (StopIteration, FileNotFoundError):
        logging.critical(f"KRİTİK HATA: Hasta {patient_id} için gerekli CT/RS/RTSTRUCT dosyaları bulunamadı.")
        return {"patient_id": patient_id, "status": "HATA_DOSYA_YOK"}

    # Çalışma bilgilerini logla
    first_ct_file = next(ct_dir.rglob("*.dcm"))
    ds_ct = pydicom.dcmread(str(first_ct_file), stop_before_pixels=True, force=True)
    study_uid = getattr(ds_ct, "StudyInstanceUID", "Bilinmiyor")
    study_desc = getattr(ds_ct, "StudyDescription", "Açıklama yok")
    logging.info(f"Kullanılan çalışma: StudyDescription='{study_desc}', StudyInstanceUID='{study_uid}'")

    try:
        rtstruct = RTStructBuilder.create_from(dicom_series_path=str(ct_dir), rt_struct_path=str(rtstruct_file))
    except Exception as e:
        logging.critical("KRİTİK HATA: rt-utils verileri yükleyemedi.", exc_info=True)
        return {"patient_id": patient_id, "status": "HATA_RTUTILS_YUKLEME", "error_message": str(e)}

    gtv_names = select_gtv_roi_names(rtstruct, config.ROI_PREFIX)
    if not gtv_names:
        logging.warning(f"Hasta {patient_id}: 'GTV' ile başlayan ROI bulunamadı.")
        return {"patient_id": patient_id, "status": "HATA_GTV_BULUNAMADI"}

    selected_gtv = gtv_names[0]
    logging.info(f"Seçilen GTV ROI: {selected_gtv}")

    try:
        series_data = rtstruct.series_data
        get_slice, num_slices, rows, cols, meta = prepare_ct_reader(series_data)
    except Exception as e:
        logging.critical("CT serisi okunurken hata oluştu.", exc_info=True)
        return {"patient_id": patient_id, "status": "HATA_CT_OKUMA", "error_message": str(e)}

    try:
        mask_union = rtstruct.get_roi_mask_by_name(selected_gtv).astype(bool)
    except Exception as e:
        logging.critical("ROI maskesi alınırken hata oluştu.", exc_info=True)
        return {"patient_id": patient_id, "status": "HATA_ROI_MASK_ALMA", "error_message": str(e)}

    target_shape = (num_slices, rows, cols)
    try:
        mask_aligned = align_mask_to_target(mask_union, target_shape)
    except Exception as e:
        logging.critical("Maske CT ile hizalanamadı.", exc_info=True)
        return {"patient_id": patient_id, "status": "HATA_BOYUT_ESLESMESI", "error_message": str(e)}

    nonzero_z = np.where(mask_aligned.any(axis=(1, 2)))[0]
    if nonzero_z.size == 0:
        logging.warning(f"UYARI: ROI boş. Hasta: {patient_id}")
        return {"patient_id": patient_id, "status": "UYARI_ROI_BOS", "roi_name": selected_gtv}

    patient_out = config.OUTPUT_DIRECTORY / patient_id
    masks_out = patient_out / 'masks'
    overlays_out = patient_out / 'overlays'
    masks_out.mkdir(parents=True, exist_ok=True)
    if config.SAVE_OVERLAYS:
        overlays_out.mkdir(parents=True, exist_ok=True)

    count = 0
    for z in nonzero_z:
        try:
            ct_slice = get_slice(int(z))
            ct_u8 = slice_to_u8(ct_slice)
            mask_slice = mask_aligned[int(z)]

            if config.SAVE_MASK_PNG:
                save_mask_png(mask_slice, masks_out / f"mask_{z:04d}.png")
            if config.SAVE_OVERLAYS:
                save_overlay_png(ct_u8, mask_slice, overlays_out / f"overlay_{z:04d}.png")
            count += 1
        except Exception as e:
            logging.error(f"Slice {z} sırasında hata: {e}")

    logging.info(f"{count} adet maske/overlay kaydedildi.")
    return {
        'patient_id': patient_id,
        'status': 'BASARILI',
        'roi_name': selected_gtv,
        'study_description': study_desc,
        'study_uid': study_uid,
        'num_mask_slices': int(len(nonzero_z)),
        'total_slices': int(num_slices)
    }


def main():
    config = Config()
    setup_logging(config.OUTPUT_DIRECTORY / config.LOG_FILENAME)

    logging.info("="*60)
    logging.info("Maske Çıkarıcı v10 - Tek GTV, çalışma bilgili")
    logging.info("="*60)

    results = []
    if config.MODE == 'SINGLE':
        patient_path = config.INPUT_DIRECTORY / config.SINGLE_PATIENT_ID
        if not patient_path.is_dir():
            logging.critical(f"HATA: Belirtilen hasta klasörü bulunamadı: {patient_path}")
            sys.exit(1)
        results.append(process_patient(patient_path, config))
    elif config.MODE == 'BATCH':
        patient_paths = sorted([p for p in config.INPUT_DIRECTORY.iterdir() if p.is_dir()])
        for p in patient_paths:
            results.append(process_patient(p, config))
    else:
        logging.critical(f"Geçersiz mod: {config.MODE}")
        sys.exit(1)

    results = [r for r in results if r]
    csv_path = config.OUTPUT_DIRECTORY / 'ozet_rapor_v10.csv'
    if results:
        try:
            csv_path.parent.mkdir(exist_ok=True, parents=True)
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                headers = list(results[0].keys())
                writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(results)
            logging.info(f"Özet rapor oluşturuldu: {csv_path}")
        except Exception as e:
            logging.error(f"CSV raporu yazılamadı: {e}")

    logging.info("="*60)
    logging.info("Tüm işlemler tamamlandı.")
    logging.info(f"Çıktılar: {config.OUTPUT_DIRECTORY}")
    logging.info("="*60)

#TODO: full otomatik hale getirilecek.
if __name__ == '__main__':
    main()
