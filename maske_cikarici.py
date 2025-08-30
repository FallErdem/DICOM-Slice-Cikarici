#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contour-level RTSTRUCT -> CT mask extractor
- Doğru CT serisini seçer (UID overlap ile).
- RTSTRUCT içindeki contour'ları doğrudan rasterize eder (world -> pixel).
- GTV(ler) alınır, BRAIN (veya BODY fallback) ile kesişim hesaplanır.
- Yalnızca GTV'nin BRAIN içinde olduğu dilimler mask/overlay olarak kaydedilir.
"""
import os
import glob
import logging
from pathlib import Path
from collections import defaultdict
import numpy as np
import pydicom
import SimpleITK as sitk
import cv2
import itertools
import json
import sys

# ------------- AYARLAR -------------
BASE_DIR = r"D:\beyin_3_4_5\beyin3.2\1559224"   # Hasta klasörü
OUTPUT_DIR = r"D:\beyincikti\beyinv5"
ROI_PREFIX = "gtv"        # GTV önek
MIN_RTSTRUCT_OVERLAP = 0.20
MIN_OVERLAP_COUNT = 3
ALPHA = 0.6
# -----------------------------------

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def normalize_text(t: str) -> str:
    if not t:
        return ""
    repl = {'ı':'i','İ':'i','ğ':'g','Ğ':'g','ü':'u','Ü':'u','ş':'s','Ş':'s','ö':'o','Ö':'o','ç':'c','Ç':'c'}
    s = t.lower()
    for a,b in repl.items():
        s = s.replace(a,b)
    return s

# CT serilerini bul
reader = sitk.ImageSeriesReader()
series_ids = reader.GetGDCMSeriesIDs(BASE_DIR) or []
if not series_ids:
    logging.critical("CT serisi bulunamadı. BASE_DIR içerik yapısını kontrol et.")
    sys.exit(1)

series_files_map = {}
for sid in series_ids:
    files = reader.GetGDCMSeriesFileNames(BASE_DIR, sid)
    series_files_map[sid] = files
    logging.info(f"CT series: id={sid}, files={len(files)}")

# Her seri için SOPInstanceUID seti
def build_uid_set(file_list):
    s = set()
    for f in file_list:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, specific_tags=["SOPInstanceUID"])
            uid = getattr(ds, "SOPInstanceUID", None)
            if uid:
                s.add(uid)
        except Exception:
            continue
    return s

series_uid_sets = {sid: build_uid_set(files) for sid, files in series_files_map.items()}

# RTSTRUCT adaylarını tara (RS_, RT_, BASE_DIR)
def find_rtstruct_candidates():
    roots = []
    # hypthetical common subdirs
    for pref in ("RS_", "RT_"):
        for d in os.listdir(BASE_DIR):
            p = os.path.join(BASE_DIR, d)
            if os.path.isdir(p) and d.upper().startswith(pref):
                roots.append(p)
    roots.append(BASE_DIR)
    found = []
    seen = set()
    for r in roots:
        if not r or r in seen: continue
        seen.add(r)
        for p in glob.glob(os.path.join(r, "**", "*.dcm"), recursive=True):
            try:
                ds = pydicom.dcmread(p, stop_before_pixels=True, specific_tags=["Modality"])
                if getattr(ds, "Modality", "") == "RTSTRUCT":
                    found.append(p)
            except Exception:
                continue
    return found

candidates = find_rtstruct_candidates()
logging.info(f"RTSTRUCT aday sayısı: {len(candidates)}")
if not candidates:
    logging.critical("RTSTRUCT bulunamadı.")
    sys.exit(1)

# RTSTRUCT içindeki Referenced SOP UIDs çek
def extract_reference_uids(rtpath):
    uids = set()
    try:
        ds = pydicom.dcmread(rtpath, stop_before_pixels=True)
        if hasattr(ds, "ROIContourSequence"):
            for rc in ds.ROIContourSequence:
                if hasattr(rc, "ContourSequence"):
                    for cs in rc.ContourSequence:
                        if hasattr(cs, "ContourImageSequence"):
                            for im in cs.ContourImageSequence:
                                uid = getattr(im, "ReferencedSOPInstanceUID", None)
                                if uid:
                                    uids.add(uid)
    except Exception as e:
        logging.debug(f"RT read error {rtpath}: {e}")
    return uids

# Her aday için her CT serisi ile overlap hesapla
best = {"rt": None, "series": None, "ratio": 0.0, "count":0, "ref_count":0}
for rt in candidates:
    ref_uids = extract_reference_uids(rt)
    if not ref_uids:
        logging.info(f"Aday RT {os.path.basename(rt)} ReferencedSOPInstanceUID yok -> atla")
        continue
    for sid, uidset in series_uid_sets.items():
        overlap = len(ref_uids & uidset)
        ratio = overlap / float(len(ref_uids)) if len(ref_uids)>0 else 0.0
        logging.info(f"RT {os.path.basename(rt)} vs CT series {sid}: overlap {overlap}/{len(ref_uids)} -> {ratio:.3f}")
        if ratio > best["ratio"]:
            best.update(rt=rt, series=sid, ratio=ratio, count=overlap, ref_count=len(ref_uids))

if best["rt"] is None:
    logging.critical("Hiçbir RTSTRUCT referans UID içermiyor. Çıkılıyor.")
    sys.exit(1)

logging.info(f"Seçilen RT: {best['rt']}  CT series: {best['series']}  overlap={best['count']}/{best['ref_count']} ({best['ratio']:.3f})")

if best["ratio"] < MIN_RTSTRUCT_OVERLAP or best["count"] < MIN_OVERLAP_COUNT:
    logging.critical(f"Eşleşme yetersiz (ratio {best['ratio']:.3f} veya count {best['count']}). Çıkılıyor.")
    sys.exit(1)

selected_rt = best["rt"]
selected_series_id = best["series"]
selected_ct_files = series_files_map[selected_series_id]
logging.info(f"Seçilen CT dosya sayısı: {len(selected_ct_files)}")

# --- CT slice metadata: uid->index, z positions, ds map ---
uid_to_index = {}
index_to_ds = {}
z_positions = []
for idx, f in enumerate(selected_ct_files):
    try:
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        uid = getattr(ds, "SOPInstanceUID", None) or f"NOUID_{idx}"
        uid_to_index[uid] = idx
        index_to_ds[idx] = ds
        # z position: prefer ImagePositionPatient[2], fallback SliceLocation
        try:
            z = float(ds.ImagePositionPatient[2])
        except Exception:
            try:
                z = float(ds.SliceLocation)
            except Exception:
                z = float(getattr(ds, "InstanceNumber", idx))
        z_positions.append(z)
    except Exception:
        uid_to_index[f"NOUID_{idx}"] = idx
        index_to_ds[idx] = None
        z_positions.append(float(idx))

z_positions = np.array(z_positions)
# Ensure ordering: SimpleITK reader probably returned ordered list, but let's build index sort mapping if needed.
# But uid_to_index is built from the file order returned by GetGDCMSeriesFileNames, which is usually correct.

# --- RTSTRUCT DICOM dataset ---
rt_ds = pydicom.dcmread(selected_rt, stop_before_pixels=True)
# map ROINumber -> ROIName
num2name = {}
if hasattr(rt_ds, "StructureSetROISequence"):
    for roi in rt_ds.StructureSetROISequence:
        num2name[roi.ROINumber] = getattr(roi, "ROIName", f"ROI_{roi.ROINumber}")

# Utility: world (x,y,z) -> pixel (col,row) for a given slice ds
def world_to_pixel(pt_world, slice_ds):
    # slice_ds must have ImagePositionPatient, ImageOrientationPatient, PixelSpacing
    ipp = np.array(slice_ds.ImagePositionPatient, dtype=float)
    iop = np.array(slice_ds.ImageOrientationPatient, dtype=float)
    row_dir = iop[0:3]
    col_dir = iop[3:6]
    ps = np.array(slice_ds.PixelSpacing, dtype=float)  # [row_spacing, col_spacing]
    rel = np.array(pt_world, dtype=float) - ipp
    # col = dot(rel, col_dir)/col_spacing ; row = dot(rel, row_dir)/row_spacing
    col = np.dot(rel, col_dir) / ps[1]
    row = np.dot(rel, row_dir) / ps[0]
    return float(col), float(row)

# Rasterize a single contour (ContourData points) into 2D mask for given slice index
def rasterize_contour_to_slice(contour_points, slice_idx, rows, cols):
    ds = index_to_ds.get(slice_idx)
    if ds is None:
        return np.zeros((rows, cols), dtype=np.uint8)
    pts = np.array(contour_points).reshape(-1,3)
    poly = []
    for p in pts:
        colf, rowf = world_to_pixel(p, ds)
        # round
        cx = int(round(colf))
        ry = int(round(rowf))
        # clip
        cx = max(0, min(cols-1, cx))
        ry = max(0, min(rows-1, ry))
        poly.append([cx, ry])
    if len(poly) < 3:
        return np.zeros((rows, cols), dtype=np.uint8)
    # prepare for cv2.fillPoly: array of shape (n,1,2) with dtype int32
    arr = np.array(poly, dtype=np.int32)
    arr = arr.reshape((-1,1,2))
    mask = np.zeros((rows, cols), dtype=np.uint8)
    cv2.fillPoly(mask, [arr], color=1)
    return mask

# Parse ROIContourSequence and rasterize into 3D masks (Z,Y,X)
def rasterize_rt_rois(rt_ds, uid_to_index_map, z_positions_array, shape_ct):
    """
    Returns dict: roi_name -> 3D boolean mask (Z,Y,X)
    """
    rows = int(index_to_ds[0].Rows)
    cols = int(index_to_ds[0].Columns)
    zcount = shape_ct[0]
    roi_masks = {}
    if not hasattr(rt_ds, "ROIContourSequence"):
        return roi_masks
    for rc in rt_ds.ROIContourSequence:
        refnum = getattr(rc, "ReferencedROINumber", None)
        roi_name = num2name.get(refnum, f"ROI_{refnum}")
        mask3d = np.zeros((zcount, rows, cols), dtype=np.uint8)
        if not hasattr(rc, "ContourSequence"):
            continue
        for cs in rc.ContourSequence:
            # get contour points
            if not hasattr(cs, "ContourData"):
                continue
            contour_pts = cs.ContourData
            # try to find referenced SOPInstanceUID (preferred)
            slice_idx = None
            if hasattr(cs, "ContourImageSequence"):
                # usually only one item
                try:
                    for im in cs.ContourImageSequence:
                        uid = getattr(im, "ReferencedSOPInstanceUID", None)
                        if uid and uid in uid_to_index_map:
                            slice_idx = uid_to_index_map[uid]
                            break
                except Exception:
                    slice_idx = None
            if slice_idx is None:
                # fallback: compute mean z of contour and find nearest slice by z_positions
                pts = np.array(contour_pts).reshape(-1,3)
                zmean = float(np.mean(pts[:,2]))
                # find closest index (abs difference)
                slice_idx = int(np.argmin(np.abs(z_positions_array - zmean)))
            # rasterize to that slice (if in range)
            if 0 <= slice_idx < zcount:
                m2d = rasterize_contour_to_slice(contour_pts, slice_idx, rows, cols)
                mask3d[slice_idx] = np.logical_or(mask3d[slice_idx], m2d).astype(np.uint8)
            else:
                logging.warning(f"Contour mapped outside CT z range: slice_idx={slice_idx} zmean maybe out of range.")
        # store mask if any
        if mask3d.any():
            roi_masks[roi_name] = mask3d.astype(bool)
            logging.info(f"ROI {roi_name}: rasterized, nonzero slices = {np.count_nonzero(mask3d.max(axis=(1,2))>0)}")
        else:
            logging.info(f"ROI {roi_name}: rasterize sonucu boş.")
    return roi_masks

# Rasterize all ROIs
ct_shape = (len(selected_ct_files), int(index_to_ds[0].Rows), int(index_to_ds[0].Columns))
logging.info(f"CT shape used for rasterization: {ct_shape}")
roi_masks = rasterize_rt_rois(rt_ds, uid_to_index, z_positions, ct_shape)

# Hedef: GTV listesi ve BRAIN (veya BODY)
all_roi_names = list(roi_masks.keys())
logging.info(f"Rasterize ile elde edilen ROI sayısı: {len(all_roi_names)}")
gtv_names = [n for n in all_roi_names if normalize_text(n).startswith(normalize_text(ROI_PREFIX))]
if not gtv_names:
    logging.critical("Rasterize sonucu GTV bulunamadı.")
    sys.exit(1)
logging.info(f"GTV isimleri (rasterized): {gtv_names}")

# BRAIN seçimi
brain_candidates = [n for n in all_roi_names if "brain" in normalize_text(n)]
body_candidates  = [n for n in all_roi_names if "body" in normalize_text(n)]
brain_name = None
if brain_candidates:
    brain_name = brain_candidates[0]
    logging.info(f"BRAIN ROI kullanılacak: {brain_name}")
elif body_candidates:
    brain_name = body_candidates[0]
    logging.info(f"BRAIN yok, BODY ROI kullanılacak fallback: {brain_name}")
else:
    logging.info("No BRAIN/BODY ROI found -- fallback to simple CT thresholding for body region.")

# build final gtv union mask
gtv_union = np.zeros(ct_shape, dtype=bool)
for nm in gtv_names:
    gtv_union = np.logical_or(gtv_union, roi_masks[nm])

# brain mask
if brain_name:
    brain_mask = roi_masks.get(brain_name, np.zeros(ct_shape, dtype=bool))
else:
    # fallback threshold-based body estimation
    # load CT as array via SimpleITK
    reader.SetFileNames(selected_ct_files)
    ct_img = reader.Execute()
    ct_np = sitk.GetArrayFromImage(ct_img).astype(np.float32)  # (Z,Y,X)
    # simple threshold: HU > -300 as body (slice-level postprocess)
    bm = (ct_np > -300.0).astype(np.uint8)
    # morphological close per slice to fill holes
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    for z in range(bm.shape[0]):
        m = (bm[z]*255).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kern, iterations=1)
        bm[z] = (m>127).astype(np.uint8)
    brain_mask = bm.astype(bool)

# final intersection
final_mask = np.logical_and(gtv_union, brain_mask)
z_has = np.where(final_mask.any(axis=(1,2)))[0]
if z_has.size == 0:
    logging.warning("GTV ∩ BRAIN boş — kayıt yapılacak dilim yok. (kontur / hizalama kontrol et)")
    # save debug info
    debug_report = {
        "selected_rt": selected_rt,
        "selected_series": selected_series_id,
        "gtv_names": gtv_names,
        "brain_name": brain_name,
        "gtv_nonzero_slices": int(np.count_nonzero(gtv_union.max(axis=(1,2))>0)),
        "brain_nonzero_slices": int(np.count_nonzero(brain_mask.max(axis=(1,2))>0))
    }
    with open(Path(OUTPUT_DIR)/"debug_report.json", "w", encoding="utf-8") as f:
        json.dump(debug_report, f, indent=2, ensure_ascii=False)
    logging.info("Debug raporu kaydedildi -> debug_report.json")
    sys.exit(0)

logging.info(f"Final mask dilimleri: {len(z_has)} (z aralığı {z_has.min()}..{z_has.max()})")

# load CT image array for saving overlays (if not loaded)
if 'ct_np' not in globals():
    reader.SetFileNames(selected_ct_files)
    ct_img = reader.Execute()
    ct_np = sitk.GetArrayFromImage(ct_img).astype(np.float32)

# try read window center/width from first slices
WINDOW_CENTER = 40.0
WINDOW_WIDTH = 80.0
for f in selected_ct_files[:20]:
    try:
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        wc = getattr(ds, "WindowCenter", None)
        ww = getattr(ds, "WindowWidth", None)
        if wc is not None and ww is not None:
            if isinstance(wc, pydicom.multival.MultiValue): wc = float(sum(map(float,wc))/len(wc))
            else: wc = float(wc)
            if isinstance(ww, pydicom.multival.MultiValue): ww = float(sum(map(float,ww))/len(ww))
            else: ww = float(ww)
            WINDOW_CENTER, WINDOW_WIDTH = wc, ww
            break
    except Exception:
        continue

def apply_window(img2d, center, width):
    low = center - width/2.0
    high = center + width/2.0
    arr = np.clip(img2d, low, high)
    arr = (arr - low) / (high - low + 1e-6) * 255.0
    return arr.astype(np.uint8)

mask_out_dir = Path(OUTPUT_DIR)/"masks"
overlay_out_dir = Path(OUTPUT_DIR)/"overlays"
mask_out_dir.mkdir(parents=True, exist_ok=True)
overlay_out_dir.mkdir(parents=True, exist_ok=True)

frame_idx = 0
for z in z_has:
    m0 = final_mask[z].astype(np.uint8)
    ct_slice = ct_np[z]
    g = apply_window(ct_slice, WINDOW_CENTER, WINDOW_WIDTH)
    rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    overlay = rgb.copy()
    overlay[m0 > 0] = (0,0,255)
    blended = cv2.addWeighted(rgb, 1.0, overlay, ALPHA, 0.0)
    cv2.imwrite(str(overlay_out_dir / f"overlay_{frame_idx:03d}.png"), blended)
    cv2.imwrite(str(mask_out_dir / f"mask_{frame_idx:03d}.png"), (m0*255).astype(np.uint8))
    frame_idx += 1

logging.info(f"Kaydedilen dilim sayısı: {frame_idx}")
# rapor
report = {
    "selected_rt": selected_rt,
    "selected_series": selected_series_id,
    "saved_slices": frame_idx,
    "gtv_names": gtv_names,
    "brain_name": brain_name
}
with open(Path(OUTPUT_DIR)/"selection_result.json","w",encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

logging.info("İşlem tamamlandı. Çıktılar kaydedildi.")
