#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GT V Extractor - contour-level, CT-RT güvenli eşleme, sadece GTV∩BRAIN dilimlerini kaydeder.
- Ayarlar CONFIG bölümünde.
- Çıktılar: OUTPUT_DIR/masks, OUTPUT_DIR/overlays, ayrıca selection_report.json & debug_report.json
"""

import os
import glob
import logging
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
import pydicom
import SimpleITK as sitk
import cv2
import itertools
import sys

# ---------------- CONFIG ----------------
CONFIG = {
    "BASE_DIR": r"D:\beyin_3_4_5\beyin3.2\1559224\CT_1559224",  
    "OUTPUT_DIR": r"D:\beyincikti\beyinv6",   
    "ROI_PREFIX": "gtv",
    "MIN_RTSTRUCT_OVERLAP": 0.20,    # en iyi RT-CT overlap oranı eşiği
    "MIN_OVERLAP_COUNT": 3,          # en az örtüşen UID sayısı
    "ALPHA": 0.6,                    # overlay alpha
    "WINDOW_CENTER_FALLBACK": 40.0,
    "WINDOW_WIDTH_FALLBACK": 80.0,
    "VERBOSE": True,
}
# ----------------------------------------

BASE_DIR = Path(CONFIG["BASE_DIR"])
OUT_DIR = Path(CONFIG["OUTPUT_DIR"])
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "masks").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "overlays").mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO if CONFIG["VERBOSE"] else logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# ---------- Helpers ----------
def normalize_text(t: str) -> str:
    if not t:
        return ""
    repl = {'ı':'i','İ':'i','ğ':'g','Ğ':'g','ü':'u','Ü':'u','ş':'s','Ş':'s','ö':'o','Ö':'o','ç':'c','Ç':'c'}
    s = t.lower()
    for a,b in repl.items():
        s = s.replace(a,b)
    return s

def find_rtstruct_candidates(base_dir: Path):
    """
    RS_/RT_ alt dizinleri ve base_dir içinde RTSTRUCT (.dcm, Modality==RTSTRUCT) dosyalarını ara.
    """
    roots = []
    for name in os.listdir(base_dir):
        p = base_dir / name
        if p.is_dir() and name.upper().startswith(("RS_", "RT_")):
            roots.append(p)
    roots.append(base_dir)
    found = []
    seen = set()
    for r in roots:
        if not r.exists() or r in seen: continue
        seen.add(r)
        for f in r.rglob("*.dcm"):
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True, specific_tags=["Modality"])
                if getattr(ds, "Modality", "") == "RTSTRUCT":
                    found.append(str(f))
            except Exception:
                continue
    return sorted(found)

def find_ct_series(base_dir: Path):
    """
    SimpleITK kullanarak base_dir içerisindeki tüm CT serilerini bul (SeriesInstanceUID -> file list)
    """
    reader = sitk.ImageSeriesReader()
    try:
        series_ids = reader.GetGDCMSeriesIDs(str(base_dir)) or []
    except Exception as e:
        logging.critical(f"SimpleITK.GetGDCMSeriesIDs hata: {e}")
        return {}
    series_files = {}
    for sid in series_ids:
        try:
            files = reader.GetGDCMSeriesFileNames(str(base_dir), sid)
            series_files[sid] = list(files)
        except Exception:
            continue
    return series_files

def build_uid_set_for_files(file_list):
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

def extract_referenced_uids_from_rt(rt_path):
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
    except Exception as e:
        logging.debug(f"RT parse hata {rt_path}: {e}")
    return uids

# world -> pixel coordinate conversion for a slice ds
def world_to_pixel(pt_world, slice_ds):
    # pt_world: (x,y,z)
    ipp = np.array(slice_ds.ImagePositionPatient, dtype=float)   # origin of slice
    iop = np.array(slice_ds.ImageOrientationPatient, dtype=float) # 6 values
    # iop: [row_x,row_y,row_z,col_x,col_y,col_z]
    row_dir = iop[0:3]
    col_dir = iop[3:6]
    ps = np.array(slice_ds.PixelSpacing, dtype=float)  # [row_spacing, col_spacing]
    rel = np.array(pt_world, dtype=float) - ipp
    # column = dot(rel, col_dir)/col_spacing ; row = dot(rel, row_dir)/row_spacing
    col = np.dot(rel, col_dir) / ps[1]
    row = np.dot(rel, row_dir) / ps[0]
    return float(col), float(row)

def rasterize_contour_to_slice(contour_points, slice_idx, index_to_ds, rows, cols):
    ds = index_to_ds.get(slice_idx)
    if ds is None:
        return np.zeros((rows, cols), dtype=np.uint8)
    pts = np.array(contour_points).reshape(-1,3)
    poly = []
    for p in pts:
        colf, rowf = world_to_pixel(p, ds)
        cx = int(round(colf)); ry = int(round(rowf))
        cx = max(0, min(cols-1, cx)); ry = max(0, min(rows-1, ry))
        poly.append([cx, ry])
    if len(poly) < 3:
        return np.zeros((rows, cols), dtype=np.uint8)
    arr = np.array(poly, dtype=np.int32).reshape((-1,1,2))
    mask = np.zeros((rows, cols), dtype=np.uint8)
    cv2.fillPoly(mask, [arr], color=1)
    return mask

def rasterize_rt_rois_to_3d(rt_ds, uid_to_index, z_positions, index_to_ds, ct_shape):
    """
    rt_ds: pydicom dataset for RTSTRUCT
    uid_to_index: dict SOPInstanceUID -> z index in CT series
    z_positions: numpy array of z positions for CT slices
    index_to_ds: map index->pydicom.Dataset for CT slices
    ct_shape: (Z,Y,X)
    returns roi_masks: dict roi_name -> 3D bool mask (Z,Y,X)
    """
    rows = int(index_to_ds[0].Rows)
    cols = int(index_to_ds[0].Columns)
    zcount = ct_shape[0]
    roi_masks = {}
    # map ROINumber->ROIName from StructureSetROISequence
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
                # fallback: place by mean Z
                pts = np.array(contour_pts).reshape(-1,3)
                zmean = float(np.mean(pts[:,2]))
                slice_idx = int(np.argmin(np.abs(z_positions - zmean)))
            if 0 <= slice_idx < zcount:
                m2d = rasterize_contour_to_slice(contour_pts, slice_idx, index_to_ds, rows, cols)
                mask3d[slice_idx] = np.logical_or(mask3d[slice_idx], m2d).astype(np.uint8)
            else:
                logging.warning(f"Contour outside CT z range (mapped idx {slice_idx})")
        if mask3d.any():
            roi_masks[roi_name] = mask3d.astype(bool)
            logging.info(f"ROI '{roi_name}' rasterized, nonzero slices={int(np.count_nonzero(mask3d.max(axis=(1,2))>0))}")
        else:
            logging.info(f"ROI '{roi_name}' rasterized empty.")
    return roi_masks

# ---------- Main flow ----------
def main():
    base = BASE_DIR
    if not base.exists():
        logging.critical(f"BASE_DIR yok: {base}")
        return

    # 1) CT serilerini bul
    logging.info("CT serileri aranıyor...")
    series_files = find_ct_series(base)
    if not series_files:
        logging.critical("CT serisi bulunamadı (SimpleITK). Dizin yapısını kontrol et.")
        return
    logging.info(f"Bulunan CT serisi sayısı: {len(series_files)}")

    # build UID sets for each CT series
    series_uid_sets = {}
    for sid, files in series_files.items():
        series_uid_sets[sid] = build_uid_set_for_files(files)
        logging.info(f"Series {sid}: {len(files)} dosya, {len(series_uid_sets[sid])} SOPInstanceUID")

    # 2) RTSTRUCT adayları
    candidates = find_rtstruct_candidates(base)
    logging.info(f"Found RTSTRUCT candidates: {len(candidates)}")
    if not candidates:
        logging.critical("RTSTRUCT bulunamadı.")
        return

    # 3) compute overlap RT vs CT series
    best = {"rt": None, "series": None, "ratio": 0.0, "overlap": 0, "ref_count":0}
    for rt in candidates:
        ref_uids = extract_referenced_uids_from_rt(rt)
        if not ref_uids:
            logging.info(f"{os.path.basename(rt)}: no referenced SOPInstanceUIDs, skipping.")
            continue
        for sid, uidset in series_uid_sets.items():
            overlap = len(ref_uids & uidset)
            ratio = overlap / float(len(ref_uids)) if len(ref_uids)>0 else 0.0
            logging.info(f"RT {os.path.basename(rt)} vs CT series {sid}: overlap {overlap}/{len(ref_uids)} -> {ratio:.3f}")
            if ratio > best["ratio"]:
                best.update(rt=rt, series=sid, ratio=ratio, overlap=overlap, ref_count=len(ref_uids))

    if best["rt"] is None:
        logging.critical("Hiçbir RTSTRUCT içinde Referenced SOPInstanceUID bulunamadı.")
        return

    logging.info(f"Seçilen RT: {best['rt']}  CT series: {best['series']}  overlap={best['overlap']}/{best['ref_count']} ({best['ratio']:.3f})")

    # security threshold
    if best["ratio"] < CONFIG["MIN_RTSTRUCT_OVERLAP"] or best["overlap"] < CONFIG["MIN_OVERLAP_COUNT"]:
        logging.critical("En iyi eşleşme yetersiz. İşlem iptal edildi. (Yanlış RT olabilir.)")
        # save debug info
        with open(OUT_DIR/"selection_debug.json", "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2)
        return

    selected_rt_path = best["rt"]
    selected_series_id = best["series"]
    selected_ct_files = series_files[selected_series_id]
    selected_ct_files = sorted(selected_ct_files)  # ensure consistent order

    logging.info(f"Loading CT series (files: {len(selected_ct_files)}) with SimpleITK...")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(selected_ct_files)
    ct_img = reader.Execute()
    ct_np = sitk.GetArrayFromImage(ct_img).astype(np.float32)  # (Z,Y,X)
    ct_shape = ct_np.shape
    logging.info(f"CT shape: {ct_shape}")

    # build uid->index map and index->ds map and z positions
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
                try: z = float(ds.SliceLocation)
                except Exception: z = float(idx)
            z_positions.append(z)
        except Exception:
            uid_to_index[f"NOUID_{idx}"] = idx
            index_to_ds[idx] = None
            z_positions.append(float(idx))
    z_positions = np.array(z_positions)

    # load RT ds
    rt_ds = pydicom.dcmread(selected_rt_path, stop_before_pixels=True)
    # rasterize ROIs
    roi_masks = rasterize_rt_rois_to_3d(rt_ds, uid_to_index, z_positions, index_to_ds, ct_shape)

    # analyze ROI names
    all_roi_names = list(roi_masks.keys())
    logging.info(f"Rasterized ROI count: {len(all_roi_names)}")
    gtv_names = [n for n in all_roi_names if normalize_text(n).startswith(normalize_text(CONFIG["ROI_PREFIX"]))]
    if not gtv_names:
        logging.critical("GTV (rasterized) bulunamadı.")
        # save debug info
        with open(OUT_DIR/"debug_empty_gtv.json","w",encoding="utf-8") as f:
            json.dump({"roi_names": all_roi_names}, f, indent=2)
        return
    logging.info(f"GTV names (rasterized): {gtv_names}")

    # choose brain/body ROI
    brain_candidates = [n for n in all_roi_names if "brain" in normalize_text(n)]
    body_candidates = [n for n in all_roi_names if "body" in normalize_text(n)]
    brain_name = None
    if brain_candidates:
        brain_name = brain_candidates[0]
        logging.info(f"Using brain ROI: {brain_name}")
    elif body_candidates:
        brain_name = body_candidates[0]
        logging.info(f"No BRAIN ROI, using BODY ROI fallback: {brain_name}")
    else:
        logging.info("No BRAIN/BODY ROI found. Using threshold fallback for body region.")

    # build GTV union mask
    gtv_union = np.zeros(ct_shape, dtype=bool)
    for nm in gtv_names:
        gtv_union = np.logical_or(gtv_union, roi_masks.get(nm, np.zeros(ct_shape, dtype=bool)))

    # build brain mask
    if brain_name:
        brain_mask = roi_masks.get(brain_name, np.zeros(ct_shape, dtype=bool))
    else:
        # fallback: simple threshold on HU to approximate body/head region
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
        logging.warning("GTV ∩ BRAIN boş. Debug raporu oluşturuluyor.")
        debug = {
            "selected_rt": selected_rt_path,
            "selected_series": selected_series_id,
            "gtv_nonzero_slices": int(np.count_nonzero(gtv_union.max(axis=(1,2))>0)),
            "brain_nonzero_slices": int(np.count_nonzero(brain_mask.max(axis=(1,2))>0)),
            "gtv_names": gtv_names,
            "all_roi_names": all_roi_names
        }
        with open(OUT_DIR/"debug_report.json","w",encoding="utf-8") as f:
            json.dump(debug, f, indent=2, ensure_ascii=False)
        logging.info("debug_report.json kaydedildi.")
        return

    # window params
    wc = CONFIG["WINDOW_CENTER_FALLBACK"]
    ww = CONFIG["WINDOW_WIDTH_FALLBACK"]
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
        low = center - width/2.0
        high = center + width/2.0
        arr = np.clip(img2d, low, high)
        arr = (arr - low) / (high - low + 1e-6) * 255.0
        return arr.astype(np.uint8)

    # save overlays + masks for z_has slices
    mask_out_dir = OUT_DIR/"masks"
    overlay_out_dir = OUT_DIR/"overlays"
    mask_out_dir.mkdir(parents=True, exist_ok=True)
    overlay_out_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    for z in z_has:
        m0 = final_mask[z].astype(np.uint8)
        ct_slice = ct_np[z]
        g = apply_window(ct_slice, wc, ww)
        rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        overlay = rgb.copy()
        overlay[m0 > 0] = (0,0,255)
        blended = cv2.addWeighted(rgb, 1.0, overlay, CONFIG["ALPHA"], 0.0)
        cv2.imwrite(str(overlay_out_dir / f"overlay_{frame_idx:03d}.png"), blended)
        cv2.imwrite(str(mask_out_dir / f"mask_{frame_idx:03d}.png"), (m0*255).astype(np.uint8))
        frame_idx += 1

    logging.info(f"Kaydedilen dilim sayısı: {frame_idx}")

    # save report
    report = {
        "selected_rt": selected_rt_path,
        "selected_series": selected_series_id,
        "saved_slices": frame_idx,
        "gtv_names": gtv_names,
        "brain_name": brain_name
    }
    with open(OUT_DIR / "selection_result.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logging.info("İşlem tamamlandı. Çıktılar dizininde kontrol et.")
    return

if __name__ == "__main__":
    main()
