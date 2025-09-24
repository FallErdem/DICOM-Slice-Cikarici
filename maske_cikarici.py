#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gtv_extractor_full.py
- Ana dizin ver (hasta klasörü). Kod CT_*/RS_*/RT_* klasörlerini bulur.
- RTSTRUCT adaylarını tarar, her RT için referenced SOPInstanceUID'leri toplar.
- Her CT serisi için SOPInstanceUID setleri oluşturur.
- En iyi RT-CT eşini seçer (overlap >= min_overlap & overlap_count >= min_count).
- Seçilen RT'yi contour-level rasterize eder (ContourImageSequence -> ReferencedSOPInstanceUID kullanır;
  yoksa contour z-ort. ile en yakın slice'a yerleştirir).
- GTV'leri union alır. 'BRAIN' ROI'si tercih edilir (brainstem bir fallback'tir, uyarı verir).
- Son olarak sadece GTV ∩ BRAIN dilimlerini mask+overlay olarak kaydeder.
- Çıktılar / raporlar: masks/, overlays/, selection_report.json, debug_report.json
"""

import argparse
import logging
import json
import os
from pathlib import Path
import sys
import glob
from collections import defaultdict
import itertools

import numpy as np
import pydicom
import SimpleITK as sitk
import cv2

# ---------------- CONFIG (CLI overrideable) ----------------
DEFAULT_MIN_OVERLAP = 0.95
DEFAULT_MIN_COUNT = 3
ALPHA = 0.6
WINDOW_CENTER_FALLBACK = 40.0
WINDOW_WIDTH_FALLBACK = 80.0
# ----------------------------------------------------------

# ---------- Helpers ----------
def setup_logging(verbose=True):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")

def normalize_text(t: str) -> str:
    if not t:
        return ""
    repl = {'ı':'i','İ':'i','ğ':'g','Ğ':'g','ü':'u','Ü':'u','ş':'s','Ş':'s','ö':'o','Ö':'o','ç':'c','Ç':'c'}
    s = t.lower()
    for a,b in repl.items():
        s = s.replace(a,b)
    return s

def find_candidate_ct_roots(base_dir: Path):
    """
    1) Öncelikle base_dir içinde CT_* isimli alt klasörleri al.
    2) Yoksa, base_dir altındaki her bir alt klasörü tara; Modality==CT içeren dizinleri CT root olarak al.
    """
    ct_roots = []
    # direct CT_* first
    for p in base_dir.iterdir():
        if p.is_dir() and p.name.upper().startswith("CT_"):
            ct_roots.append(p)
    if ct_roots:
        return ct_roots

    # fallback: scan immediate subdirs for presence of CT DICOMs
    for p in base_dir.iterdir():
        if not p.is_dir():
            continue
        try:
            # look for some .dcm files and check Modality
            found = False
            for f in p.glob("**/*.dcm"):
                try:
                    ds = pydicom.dcmread(str(f), stop_before_pixels=True, specific_tags=["Modality"])
                    if getattr(ds, "Modality", "").upper() == "CT":
                        ct_roots.append(p)
                        found = True
                        break
                except Exception:
                    continue
            if found:
                continue
        except Exception:
            continue
    return ct_roots

def find_rtstruct_candidates(base_dir: Path):
    """
    RS_* and RT_* folders, and base_dir itself, search for DICOM files with Modality == RTSTRUCT
    """
    candidates = []
    # prefer RS_/RT_ folders
    roots = []
    for p in base_dir.iterdir():
        if p.is_dir() and (p.name.upper().startswith("RS_") or p.name.upper().startswith("RT_")):
            roots.append(p)
    # always include base_dir too
    roots.append(base_dir)
    seen = set()
    for r in roots:
        if r in seen or not r.exists():
            continue
        seen.add(r)
        for f in r.rglob("*.dcm"):
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True, specific_tags=["Modality"])
                if getattr(ds, "Modality", "").upper() == "RTSTRUCT":
                    candidates.append(str(f))
            except Exception:
                continue
    return sorted(set(candidates))

def get_series_map_from_ct_root(ct_root: Path):
    """
    Return dict SeriesInstanceUID -> list-of-files (full paths)
    Uses SimpleITK ImageSeriesReader.
    """
    reader = sitk.ImageSeriesReader()
    try:
        series_ids = reader.GetGDCMSeriesIDs(str(ct_root)) or []
    except Exception as e:
        logging.debug(f"SimpleITK GetGDCMSeriesIDs hata: {e}")
        series_ids = []
    series_map = {}
    for sid in series_ids:
        try:
            files = reader.GetGDCMSeriesFileNames(str(ct_root), sid)
            series_map[sid] = list(files)
        except Exception:
            continue
    # fallback: if none found, group by SeriesInstanceUID by scanning files
    if not series_map:
        logging.debug(f"No series via SimpleITK in {ct_root}, grouping by SeriesInstanceUID via pydicom scan.")
        files = list(ct_root.rglob("*.dcm"))
        temp = defaultdict(list)
        for f in files:
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True, specific_tags=["SeriesInstanceUID"])
                sid = getattr(ds, "SeriesInstanceUID", "NO_SERIES")
                temp[sid].append(str(f))
            except Exception:
                continue
        series_map = {k: v for k, v in temp.items() if v}
    return series_map

def build_sop_uid_set(file_list):
    s = set()
    for f in file_list:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, specific_tags=["SOPInstanceUID"])
            uid = getattr(ds, "SOPInstanceUID", None)
            if uid:
                s.add(uid)
        except Exception:
            continue
    return s

def extract_referenced_sop_uids_from_rt(rt_path):
    """
    Extract ReferencedSOPInstanceUID values from ROIContourSequence -> ContourSequence -> ContourImageSequence.
    Returns a set of UIDs.
    """
    uids = set()
    try:
        ds = pydicom.dcmread(rt_path, stop_before_pixels=True)
        if hasattr(ds, "ROIContourSequence"):
            for rc in ds.ROIContourSequence:
                if hasattr(rc, "ContourSequence"):
                    for cs in rc.ContourSequence:
                        if hasattr(cs, "ContourImageSequence"):
                            for im in cs.ContourImageSequence:
                                uid = getattr(im, "ReferencedSOPInstanceUID", None)
                                if uid:
                                    uids.add(uid)
        # also try other places if needed in future (not implemented now)
    except Exception as e:
        logging.debug(f"RTSTRUCT parse hata {rt_path}: {e}")
    return uids

def select_best_rt_ct_pair(rt_candidates, ct_series_map, min_overlap, min_count):
    """
    For each RT candidate, compute overlap with each CT series (ct_series_map: sid->files)
    Returns dict with best match info or None if below thresholds.
    """
    # build sop uid sets for ct series
    series_uid_sets = {}
    for sid, flist in ct_series_map.items():
        series_uid_sets[sid] = build_sop_uid_set(flist)

    best = {"rt": None, "series": None, "ratio": 0.0, "overlap": 0, "ref_count": 0}
    for rt in rt_candidates:
        ref_uids = extract_referenced_sop_uids_from_rt(rt)
        if not ref_uids:
            logging.info(f"RT candidate {Path(rt).name} has no referenced SOPInstanceUIDs -> skipping")
            continue
        for sid, uidset in series_uid_sets.items():
            overlap = len(ref_uids & uidset)
            ratio = overlap / float(len(ref_uids)) if len(ref_uids) > 0 else 0.0
            logging.info(f"RT {Path(rt).name} vs CT series {sid}: overlap {overlap}/{len(ref_uids)} -> {ratio:.3f}")
            if ratio > best["ratio"]:
                best.update(rt=rt, series=sid, ratio=ratio, overlap=overlap, ref_count=len(ref_uids))
    if best["rt"] is None:
        return None
    if best["ratio"] < min_overlap or best["overlap"] < min_count:
        return None
    return best

# coordinate transforms & rasterization
def world_to_pixel(pt_world, slice_ds):
    """
    Convert (x,y,z) world coordinates to (col,row) pixel coordinates on given slice dataset.
    Requires ImagePositionPatient, ImageOrientationPatient, PixelSpacing.
    """
    ipp = np.array(slice_ds.ImagePositionPatient, dtype=float)
    iop = np.array(slice_ds.ImageOrientationPatient, dtype=float)
    row_dir = iop[0:3]
    col_dir = iop[3:6]
    ps = np.array(slice_ds.PixelSpacing, dtype=float)  # [row_spacing, col_spacing]
    rel = np.array(pt_world, dtype=float) - ipp
    col = np.dot(rel, col_dir) / ps[1]
    row = np.dot(rel, row_dir) / ps[0]
    return float(col), float(row)

def rasterize_contour_to_slice(contour_pts, slice_idx, index_to_ds, rows, cols):
    """
    contour_pts: flat list of floats [x1,y1,z1, x2,y2,z2, ...]
    returns 2D uint8 mask (rows,cols) filled with 1 where polygon is.
    """
    ds = index_to_ds.get(slice_idx)
    if ds is None:
        return np.zeros((rows, cols), dtype=np.uint8)
    pts = np.array(contour_pts).reshape(-1,3)
    poly = []
    for p in pts:
        try:
            colf, rowf = world_to_pixel(p, ds)
            cx = int(round(colf)); ry = int(round(rowf))
            cx = max(0, min(cols-1, cx)); ry = max(0, min(rows-1, ry))
            poly.append([cx, ry])
        except Exception:
            continue
    if len(poly) < 3:
        return np.zeros((rows, cols), dtype=np.uint8)
    arr = np.array(poly, dtype=np.int32).reshape((-1,1,2))
    mask = np.zeros((rows, cols), dtype=np.uint8)
    cv2.fillPoly(mask, [arr], color=1)
    return mask

def rasterize_rt_rois_to_3d(rt_ds, uid_to_index, z_positions, index_to_ds, ct_shape):
    """
    Return dict roi_name -> 3D boolean mask (Z,Y,X)
    """
    rows = int(index_to_ds[0].Rows)
    cols = int(index_to_ds[0].Columns)
    zcount = ct_shape[0]
    roi_masks = {}
    # map ROINumber->ROIName
    num2name = {}
    if hasattr(rt_ds, "StructureSetROISequence"):
        for roi in rt_ds.StructureSetROISequence:
            num2name[roi.ROINumber] = getattr(roi, "ROIName", f"ROI_{roi.ROINumber}")
    if not hasattr(rt_ds, "ROIContourSequence"):
        return roi_masks
    for rc in rt_ds.ROIContourSequence:
        refnum = getattr(rc, "ReferencedROINumber", None)
        roi_name = num2name.get(refnum, f"ROI_{refnum}")
        mask3d = np.zeros((zcount, rows, cols), dtype=np.uint8)
        if not hasattr(rc, "ContourSequence"):
            continue
        for cs in rc.ContourSequence:
            if not hasattr(cs, "ContourData"):
                continue
            contour_pts = cs.ContourData
            # try ContourImageSequence -> ReferencedSOPInstanceUID
            slice_idx = None
            if hasattr(cs, "ContourImageSequence"):
                try:
                    for im in cs.ContourImageSequence:
                        uid = getattr(im, "ReferencedSOPInstanceUID", None)
                        if uid and uid in uid_to_index:
                            slice_idx = uid_to_index[uid]
                            break
                except Exception:
                    slice_idx = None
            if slice_idx is None:
                # fallback: compute mean z and find nearest slice by z_positions
                pts = np.array(contour_pts).reshape(-1,3)
                zmean = float(np.mean(pts[:,2]))
                slice_idx = int(np.argmin(np.abs(z_positions - zmean)))
            if 0 <= slice_idx < zcount:
                m2d = rasterize_contour_to_slice(contour_pts, slice_idx, index_to_ds, rows, cols)
                if m2d.any():
                    mask3d[slice_idx] = np.logical_or(mask3d[slice_idx], m2d).astype(np.uint8)
            else:
                logging.warning(f"Contour mapped outside CT z-range (slice_idx={slice_idx}) for ROI {roi_name}")
        if mask3d.any():
            roi_masks[roi_name] = mask3d.astype(bool)
            logging.info(f"ROI '{roi_name}' rasterized, nonzero slices={int(np.count_nonzero(mask3d.max(axis=(1,2))>0))}")
        else:
            logging.info(f"ROI '{roi_name}' rasterized empty.")
    return roi_masks

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="GTV extractor — CT-RT safe contour rasterization")
    parser.add_argument("--base", "-b", required=True, help="BASE_DIR (hasta klasörü), örn D:\\...\\1559224")
    parser.add_argument("--out", "-o", default=None, help="OUTPUT_DIR (varsayılan: BASE_DIR/out_gtv)")
    parser.add_argument("--min-overlap", type=float, default=DEFAULT_MIN_OVERLAP, help="Min overlap ratio (0..1), default 0.95")
    parser.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT, help="Min overlap count")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    base = Path(args.base)
    if not base.exists():
        logging.critical(f"BASE_DIR bulunamadı: {base}")
        sys.exit(1)
    out_dir = Path(args.out) if args.out else (base / "gtv_extractor_out")
    masks_dir = out_dir / "masks"
    overlays_dir = out_dir / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"BASE_DIR: {base}")
    logging.info("CT kökleri aranıyor...")
    ct_roots = find_candidate_ct_roots(base)
    if not ct_roots:
        logging.critical("CT kökleri bulunamadı. Dizin içinde CT_* klasörlerini veya CT DICOM içeren alt dizinleri kontrol et.")
        sys.exit(1)
    logging.info(f"CT kökleri: {ct_roots}")

    # aggregate series across all CT roots
    ct_series_map = {}
    for root in ct_roots:
        smap = get_series_map_from_ct_root(root)
        if smap:
            for sid, files in smap.items():
                # avoid duplicate series ids (two CT roots might share same sid) - keep first
                if sid not in ct_series_map:
                    ct_series_map[sid] = files

    if not ct_series_map:
        logging.critical("Hiç CT serisi bulunamadı (SimpleITK veya fallback scan başarısız).")
        sys.exit(1)
    logging.info(f"Toplam CT serisi: {len(ct_series_map)}")

    logging.info("RTSTRUCT adayları aranıyor (RS_ / RT_ alt klasörleri ve base)...")
    rt_candidates = find_rtstruct_candidates(base)
    if not rt_candidates:
        logging.critical("RTSTRUCT dosyası bulunamadı (RS_/RT_ veya base içinde).")
        sys.exit(1)
    logging.info(f"RTSTRUCT aday sayısı: {len(rt_candidates)}")

    logging.info("En iyi RT-CT eşini seçmek için overlap hesaplanıyor...")
    best = select_best_rt_ct_pair(rt_candidates, ct_series_map, args.min_overlap, args.min_count)
    if not best:
        logging.critical(f"Hiçbir RT-CT çifti minimum overlap şartlarını sağlamadı (min_overlap={args.min_overlap}, min_count={args.min_count}).")
        # Save list of candidates and summary for debugging
        dbg = {
            "ct_series_ids": list(ct_series_map.keys()),
            "rt_candidates": [str(p) for p in rt_candidates],
            "min_overlap": args.min_overlap,
            "min_count": args.min_count
        }
        with open(out_dir / "selection_failed_debug.json", "w", encoding="utf-8") as f:
            json.dump(dbg, f, indent=2)
        sys.exit(1)

    logging.info(f"Seçilen RT: {best['rt']}")
    logging.info(f"Eşlenen CT series: {best['series']} (overlap {best['overlap']}/{best['ref_count']} => {best['ratio']:.3f})")

    selected_rt_path = Path(best["rt"])
    selected_series_id = best["series"]
    selected_ct_files = sorted(ct_series_map[selected_series_id])

    # Load CT series via SimpleITK for pixel data
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(selected_ct_files)
    try:
        ct_img = reader.Execute()
    except Exception as e:
        logging.critical(f"SimpleITK CT serisini yüklerken hata: {e}")
        sys.exit(1)
    ct_np = sitk.GetArrayFromImage(ct_img).astype(np.float32)  # (Z,Y,X)
    ct_shape = ct_np.shape
    logging.info(f"Loaded CT (Z,Y,X) = {ct_shape}")

    # build uid->index and index->pydicom ds maps & z positions
    uid_to_index = {}
    index_to_ds = {}
    z_positions = []
    for idx, f in enumerate(selected_ct_files):
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            uid = getattr(ds, "SOPInstanceUID", f"NOUID_{idx}")
            uid_to_index[uid] = idx
            index_to_ds[idx] = ds
            try:
                z = float(ds.ImagePositionPatient[2])
            except Exception:
                try:
                    z = float(ds.SliceLocation)
                except Exception:
                    z = float(idx)
            z_positions.append(z)
        except Exception:
            uid_to_index[f"NOUID_{idx}"] = idx
            index_to_ds[idx] = None
            z_positions.append(float(idx))
    z_positions = np.array(z_positions)

    # parse RT ds
    rt_ds = pydicom.dcmread(str(selected_rt_path), stop_before_pixels=True)
    # rasterize ROI contours
    roi_masks = rasterize_rt_rois_to_3d(rt_ds, uid_to_index, z_positions, index_to_ds, ct_shape)

    all_roi_names = list(roi_masks.keys())
    logging.info(f"Rasterize sonucu ROI sayısı: {len(all_roi_names)}")

    # find GTV names
    gtv_names = [n for n in all_roi_names if normalize_text(n).startswith(normalize_text("gtv"))]
    if not gtv_names:
        logging.critical("RT içinde rasterize ile GTV bulunamadı. debug_report.json kaydediliyor.")
        with open(out_dir / "debug_report.json", "w", encoding="utf-8") as f:
            json.dump({"roi_names": all_roi_names}, f, indent=2, ensure_ascii=False)
        sys.exit(1)
    logging.info(f"GTV isimleri: {gtv_names}")

    # choose brain ROI: prefer exact "brain", else any name that contains 'brain' but not 'brainstem',
    # else 'brainstem' fallback, else 'body' fallback, else threshold fallback.
    brain_name = None
    normalized = {n: normalize_text(n) for n in all_roi_names}
    # exact match
    for n, norm in normalized.items():
        if norm == "brain":
            brain_name = n; break
    if not brain_name:
        # contains 'brain' but exclude 'brainstem'
        for n, norm in normalized.items():
            if "brain" in norm and "brainstem" not in norm:
                brain_name = n; break
    if not brain_name:
        # brainstem fallback
        for n, norm in normalized.items():
            if "brainstem" in norm:
                brain_name = n; 
                logging.warning(f"BRAIN ROI tam olarak bulunamadı; 'brainstem' bulundu ve fallback olarak kullanılıyor: {brain_name}")
                break
    if not brain_name:
        # body fallback
        for n, norm in normalized.items():
            if "body" in norm:
                brain_name = n
                logging.warning(f"BRAIN/brainstem yok; 'BODY' ROI fallback olarak kullanılıyor: {brain_name}")
                break

    # build union of GTV masks
    gtv_union = np.zeros(ct_shape, dtype=bool)
    for nm in gtv_names:
        gtv_union = np.logical_or(gtv_union, roi_masks.get(nm, np.zeros(ct_shape, dtype=bool)))

    # brain mask
    if brain_name:
        brain_mask = roi_masks.get(brain_name, np.zeros(ct_shape, dtype=bool))
    else:
        # threshold fallback using HU > -300
        logging.warning("No brain/body ROI found; using HU threshold fallback for body/head region.")
        bm = (ct_np > -300.0).astype(np.uint8)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        for z in range(bm.shape[0]):
            m = (bm[z]*255).astype(np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kern, iterations=1)
            bm[z] = (m>127).astype(np.uint8)
        brain_mask = bm.astype(bool)

    final_mask = np.logical_and(gtv_union, brain_mask)
    z_has = np.where(final_mask.any(axis=(1,2)))[0]
    if z_has.size == 0:
        logging.warning("GTV ∩ BRAIN boş — debug raporu kaydediliyor.")
        debug = {
            "selected_rt": str(selected_rt_path),
            "selected_series": selected_series_id,
            "gtv_nonzero_slices": int(np.count_nonzero(gtv_union.max(axis=(1,2))>0)),
            "brain_nonzero_slices": int(np.count_nonzero(brain_mask.max(axis=(1,2))>0)),
            "gtv_names": gtv_names,
            "all_roi_names": all_roi_names
        }
        with open(out_dir / "debug_report.json", "w", encoding="utf-8") as f:
            json.dump(debug, f, indent=2, ensure_ascii=False)
        logging.info("debug_report.json kaydedildi; lütfen içeriğini gönderin, birlikte inceleyelim.")
        sys.exit(0)

    # window params try read
    wc = WINDOW_CENTER_FALLBACK; ww = WINDOW_WIDTH_FALLBACK
    for f in selected_ct_files[:20]:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            wc0 = getattr(ds, "WindowCenter", None); ww0 = getattr(ds, "WindowWidth", None)
            if wc0 is not None and ww0 is not None:
                if isinstance(wc0, pydicom.multival.MultiValue): wc0 = float(sum(map(float,wc0))/len(wc0))
                else: wc0 = float(wc0)
                if isinstance(ww0, pydicom.multival.MultiValue): ww0 = float(sum(map(float,ww0))/len(ww0))
                else: ww0 = float(ww0)
                wc, ww = wc0, ww0
                break
        except Exception:
            continue

    def apply_window(img2d, center, width):
        low = center - width/2.0; high = center + width/2.0
        arr = np.clip(img2d, low, high)
        arr = (arr - low) / (high - low + 1e-6) * 255.0
        return arr.astype(np.uint8)

    # Save slices that contain final mask
    masks_out = masks_dir
    overlays_out = overlays_dir
    frame_idx = 0
    for z in z_has:
        try:
            m0 = final_mask[z].astype(np.uint8)
            ct_slice = ct_np[z]
            g = apply_window(ct_slice, wc, ww)
            rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            overlay = rgb.copy()
            overlay[m0 > 0] = (0,0,255)
            blended = cv2.addWeighted(rgb, 1.0, overlay, ALPHA, 0.0)
            overlay_path = overlays_out / f"overlay_{frame_idx:03d}.png"
            mask_path = masks_out / f"mask_{frame_idx:03d}.png"
            cv2.imwrite(str(overlay_path), blended)
            cv2.imwrite(str(mask_path), (m0*255).astype(np.uint8))
            frame_idx += 1
        except Exception as e:
            logging.error(f"Slice {z} kaydedilemedi: {e}")

    logging.info(f"Kaydedilen dilim sayısı: {frame_idx}")

    # final report
    report = {
        "base_dir": str(base),
        "selected_rt": str(selected_rt_path),
        "selected_series": selected_series_id,
        "selected_ct_file_count": len(selected_ct_files),
        "gtv_names": gtv_names,
        "brain_roi": brain_name,
        "saved_slices": frame_idx,
        "overlap_info": {"overlap": best["overlap"], "ref_count": best["ref_count"], "ratio": best["ratio"]},
    }
    with open(out_dir / "selection_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logging.info("İşlem tamamlandı. Çıktıları kontrol et:")
    logging.info(f" - overlays: {overlays_out}")
    logging.info(f" - masks: {masks_out}")
    logging.info(f" - report: {out_dir / 'selection_report.json'}")
    return

if __name__ == "__main__":
    main()
