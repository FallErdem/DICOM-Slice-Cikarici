#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gelişmiş RT->CT eşleştirme ve sadece GTV∩BRAIN dilimlerini kaydetme scripti.
- Tüm CT serilerini tarar, RTSTRUCT adaylarını tarar.
- RTSTRUCT içindeki ReferencedSOPInstanceUID ile her CT serisinin
  SOPInstanceUID setini karşılaştırarak en iyi RT-CT çiftini seçer.
- Seçilen CT serisi yüklenir, RTStruct yüklenir.
- GTV(ler) alınır, BRAIN veya BODY ROI'si varsa onunla kesişim alınır.
- Sadece GTV'nin beyin içinde olduğu dilimler overlay ve mask olarak kaydedilir.
"""
import os
import glob
import logging
from pathlib import Path
import itertools
from collections import defaultdict

import numpy as np
import pydicom
import SimpleITK as sitk
from rt_utils import RTStructBuilder
import cv2
import re
import sys

# ------------------ KULLANICI AYARLARI ------------------
BASE_DIR = r"D:\beyin_3_4_5\beyin3.2\1559224"   # hasta klasörü (içinde CT_*, RS_*, RT_* vb.)
OUTPUT_DIR = r"D:\beyincikti\beyinv4"
ROI_PREFIX = "gtv"           # hangi ROI öneki ile GTV'leri seçeceğiz
MIN_RTSTRUCT_OVERLAP = 0.25  # en iyi RTSTRUCT'un overlap oranı bu eşiğin altında ise dur (güvenlik)
MIN_OVERLAP_COUNT = 5        # en az bu sayıda örtüşme olmalı (opsiyonel ek güvenlik)
SAVE_OVERLAY = True
ALPHA = 0.6
# -------------------------------------------------------

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

def find_subdir_with_prefix(base_dir: str, prefix: str):
    # CT_ / RS_ / RT_ klasörlerini bulmak için
    cands = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.upper().startswith(prefix)]
    if not cands:
        return None
    cands.sort(key=len, reverse=True)
    return os.path.join(base_dir, cands[0])

CT_ROOT = find_subdir_with_prefix(BASE_DIR, "CT_") or BASE_DIR  # bazı yapılar farklı olabilir; fallback BASE_DIR
RS_ROOT = find_subdir_with_prefix(BASE_DIR, "RS_")
RT_ROOT = find_subdir_with_prefix(BASE_DIR, "RT_")

logging.info(f"Base: {BASE_DIR}")
logging.info(f"CT root (tahmini): {CT_ROOT}")
logging.info(f"RS root: {RS_ROOT}, RT root: {RT_ROOT}")

# --- 1) CT serilerini bul ve her seri için file listesi --- 
reader = sitk.ImageSeriesReader()
series_ids = reader.GetGDCMSeriesIDs(CT_ROOT) or []
if not series_ids:
    logging.critical("CT serisi bulunamadı. Dizin düzenini kontrol et.")
    sys.exit(1)

series_files_map = {}
for sid in series_ids:
    files = reader.GetGDCMSeriesFileNames(CT_ROOT, sid)
    series_files_map[sid] = files
    logging.info(f"Found CT series id={sid} with {len(files)} files")

# --- 2) her seri için SOPInstanceUID setlerini çıkar (hızlı, stop_before_pixels) ---
def build_uid_set_for_files(file_list):
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

series_uid_sets = {sid: build_uid_set_for_files(files) for sid, files in series_files_map.items()}

# --- 3) RTSTRUCT adaylarını tara (RS_ROOT, RT_ROOT, BASE_DIR) ---
def find_candidate_rtstructs():
    roots = []
    if RS_ROOT: roots.append(RS_ROOT)
    if RT_ROOT: roots.append(RT_ROOT)
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

candidates = find_candidate_rtstructs()
logging.info(f"RTSTRUCT aday sayısı: {len(candidates)}")
if not candidates:
    logging.critical("Herhangi bir RTSTRUCT bulunamadı (RS_/RT_/BASE/alt).")
    sys.exit(1)

# --- 4) RTSTRUCT içindeki ReferencedSOPInstanceUID setlerini çıkar ve her CT serisi ile karşılaştır ---
def extract_rt_ref_uids(rt_path):
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
        logging.debug(f"RT okunurken hata ({rt_path}): {e}")
    return uids

best = {
    "rt_path": None,
    "series_id": None,
    "overlap_ratio": 0.0,
    "overlap_count": 0,
    "ref_count": 0
}

for rt_path in candidates:
    ref_uids = extract_rt_ref_uids(rt_path)
    if not ref_uids:
        logging.info(f"Aday RT {os.path.basename(rt_path)}: ReferencedSOPInstanceUID yok/boş. Atlanıyor.")
        continue
    # her CT serisi ile overlap hesapla, en iyi seriyi seç
    for sid, uidset in series_uid_sets.items():
        overlap_count = len(ref_uids & uidset)
        # overlap oranını ref_uids üzerinden normalize edelim
        overlap_ratio = overlap_count / float(len(ref_uids)) if len(ref_uids) > 0 else 0.0
        logging.info(f"RT {os.path.basename(rt_path)} vs CT series {sid}: overlap {overlap_count}/{len(ref_uids)} -> {overlap_ratio:.3f}")
        if overlap_ratio > best["overlap_ratio"]:
            best.update(rt_path=rt_path, series_id=sid, overlap_ratio=overlap_ratio,
                        overlap_count=overlap_count, ref_count=len(ref_uids))

if best["rt_path"] is None:
    logging.critical("Hiçbir RTSTRUCT içinde ReferencedSOPInstanceUID bulunamadı. Çıkılıyor.")
    sys.exit(1)

logging.info(f"Seçilen RTSTRUCT: {best['rt_path']}, CT series: {best['series_id']}, overlap={best['overlap_count']}/{best['ref_count']} ({best['overlap_ratio']:.3f})")

# Güvenlik eşiği
if best["overlap_ratio"] < MIN_RTSTRUCT_OVERLAP or best["overlap_count"] < MIN_OVERLAP_COUNT:
    logging.critical(f"En iyi eşleşme yetersiz: overlap_ratio={best['overlap_ratio']:.3f}, count={best['overlap_count']}. (eşik: ratio>={MIN_RTSTRUCT_OVERLAP} ve count>={MIN_OVERLAP_COUNT})")
    logging.critical("Muhtemelen RTSTRUCT farklı seriyi referanslıyor. İşlem sonlandırıldı.")
    sys.exit(1)

selected_rt_path = best["rt_path"]
selected_series_id = best["series_id"]
selected_ct_files = series_files_map[selected_series_id]
logging.info(f"Seçilen CT dosya sayısı: {len(selected_ct_files)} (series_id={selected_series_id})")

# --- 5) seçilen CT serisini SimpleITK ile yükle ---
reader = sitk.ImageSeriesReader()
reader.SetFileNames(selected_ct_files)
ct_img = reader.Execute()
ct_np = sitk.GetArrayFromImage(ct_img).astype(np.float32)  # (Z,Y,X)
ct_shape = ct_np.shape
logging.info(f"Yüklenen CT shape (Z,Y,X): {ct_shape}")

# --- 6) rt-utils ile seçilen RTSTRUCT'ı yükle (bu RT artık seçilmiş) ---
try:
    rtstruct = RTStructBuilder.create_from(dicom_series_path=CT_ROOT, rt_struct_path=selected_rt_path)
except Exception as e:
    logging.critical(f"rt-utils yüklemede hata: {e}")
    sys.exit(1)

all_rois = rtstruct.get_roi_names() or []
logging.info(f"RT içindeki ROI'lar: {all_rois}")

# --- 7) GTV'leri seç (prefix ile) ---
gtv_list = [r for r in all_rois if normalize_text(r).startswith(normalize_text(ROI_PREFIX))]
if not gtv_list:
    logging.critical("RT içinde GTV bulunamadı.")
    sys.exit(1)
logging.info(f"GTV listesi: {gtv_list}")

# Helper: ROI mask'ı al ve CT ile hizala
def find_permutation_to_match(mask_shape, target_shape):
    # mask_shape: tuple mask array
    # target_shape: (Z,Y,X)
    for perm in itertools.permutations(range(3)):
        if tuple(np.transpose(np.zeros(mask_shape), perm).shape) == tuple(target_shape):
            return perm
    return None

def align_mask(mask_raw: np.ndarray, target_shape):
    """
    Deneyerek permute et, eğer Z boyutu farklısa nearest resample et.
    Hataları açıklayıcı fırlat.
    """
    mask_shape = mask_raw.shape
    # hızlı yol: eğer zaten aynı
    if mask_shape == target_shape:
        return mask_raw.astype(bool)

    # tüm permütasyonları dene
    for perm in itertools.permutations((0,1,2)):
        permuted = np.transpose(mask_raw, perm)
        if permuted.shape == target_shape:
            logging.info(f"Mask permütasyonu bulundu: {perm}")
            return permuted.astype(bool)

    # eğer permute ettikten sonra rows/cols eşleşen varsa, z'yi resample et
    # bul permüsyon hangi perm ile rows/cols == target Y,X
    for perm in itertools.permutations((0,1,2)):
        permuted = np.transpose(mask_raw, perm)
        if permuted.shape[1] == target_shape[1] and permuted.shape[2] == target_shape[2]:
            # permuted shape = (Z_src, Y, X)
            z_src = permuted.shape[0]
            z_tgt = target_shape[0]
            if z_src != z_tgt:
                # nearest neighbour sampling along Z
                idx = np.round(np.linspace(0, z_src-1, z_tgt)).astype(int)
                resampled = permuted[idx,:,:]
                logging.info(f"Mask Z yeniden örneklendi: {z_src} -> {z_tgt} (perm {perm})")
                return resampled.astype(bool)
            else:
                return permuted.astype(bool)

    raise ValueError(f"Mask shape {mask_shape} target {target_shape} ile eşleşemedi (tüm perm denendi).")

# GTV union mask oluştur
gtv_union_raw = None
for gtv in gtv_list:
    try:
        m = rtstruct.get_roi_mask_by_name(gtv).astype(np.uint8)
    except Exception as e:
        logging.warning(f"ROI mask alınamadı ({gtv}): {e}")
        continue
    if gtv_union_raw is None:
        gtv_union_raw = m.astype(np.uint8)
    else:
        gtv_union_raw = np.logical_or(gtv_union_raw, m).astype(np.uint8)

if gtv_union_raw is None:
    logging.critical("Hiçbir GTV maskesi alınamadı.")
    sys.exit(1)

# align
try:
    gtv_aligned = align_mask(gtv_union_raw, ct_shape)
except Exception as e:
    logging.critical(f"GTV maskesi CT ile hizalanamadı: {e}")
    sys.exit(1)

# --- 8) BRAIN veya BODY ROI'sini bul (önce brain, sonra body) ---
brain_candidates = [r for r in all_rois if "brain" in normalize_text(r) and len(normalize_text(r))>0]
body_candidates = [r for r in all_rois if "body" in normalize_text(r)]

brain_mask_aligned = None
if brain_candidates:
    brain_name = brain_candidates[0]
    logging.info(f"BRAIN ROI bulundu: {brain_name} kullanılıyor.")
    raw = rtstruct.get_roi_mask_by_name(brain_name).astype(np.uint8)
    try:
        brain_mask_aligned = align_mask(raw, ct_shape)
    except Exception as e:
        logging.warning(f"BRAIN maskesi hizalanamadı: {e}")
        brain_mask_aligned = None

if brain_mask_aligned is None and body_candidates:
    body_name = body_candidates[0]
    logging.info(f"BODY ROI bulundu: {body_name} kullanılıyor.")
    raw = rtstruct.get_roi_mask_by_name(body_name).astype(np.uint8)
    try:
        brain_mask_aligned = align_mask(raw, ct_shape)  # treat as "brain area" for clipping
    except Exception as e:
        logging.warning(f"BODY maskesi hizalanamadı: {e}")
        brain_mask_aligned = None

# Fallback: basit eşikleme ile body maskesi oluştur
def estimate_body_mask_from_ct(ct_np_arr):
    # Basit: HU > -300 (vücut), slice-per-slice morphological close fill holes
    th = -300.0
    mask = (ct_np_arr > th).astype(np.uint8)
    # morphological closing/keep largest component per slice
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    for z in range(mask.shape[0]):
        m = (mask[z]*255).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kern, iterations=1)
        # threshold back
        mask[z] = (m>127).astype(np.uint8)
    # Optionally keep largest connected component in 3D; skip heavy ops for speed
    return mask.astype(bool)

if brain_mask_aligned is None:
    logging.info("BRAIN/BODY ROI yok veya hizalanamadı. CT eşikleme ile beden alanı tahmini yapılıyor (fallback).")
    brain_mask_aligned = estimate_body_mask_from_ct(ct_np)

# --- 9) GTV ∩ BRAIN mask'ı ve sadece bu dilimleri kaydet ---
final_mask = np.logical_and(gtv_aligned.astype(bool), brain_mask_aligned.astype(bool))
z_has = np.where(final_mask.any(axis=(1,2)))[0]
if z_has.size == 0:
    logging.warning("GTV ve BRAIN kesişimi boş. Kayıt yapılacak dilim yok.")
    sys.exit(0)

z_start, z_end = int(z_has.min()), int(z_has.max())
logging.info(f"Final (GTV∩BRAIN) Z aralığı: {z_start}..{z_end} (toplam dilim: {len(z_has)})")

# window param (try read from CT files)
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
logging.info(f"Using window C={WINDOW_CENTER}, W={WINDOW_WIDTH}")

def apply_window(img2d, center, width):
    low = center - width / 2.0
    high = center + width / 2.0
    arr = np.clip(img2d, low, high)
    arr = (arr - low) / (high - low + 1e-6) * 255.0
    return arr.astype(np.uint8)

out_mask_dir = Path(OUTPUT_DIR) / "masks"
out_overlay_dir = Path(OUTPUT_DIR) / "overlays"
out_mask_dir.mkdir(parents=True, exist_ok=True)
out_overlay_dir.mkdir(parents=True, exist_ok=True)

frame_idx = 0
for z in z_has:
    try:
        m0 = final_mask[z].astype(np.uint8)
        ct_slice = ct_np[z]
        g0 = apply_window(ct_slice, WINDOW_CENTER, WINDOW_WIDTH)
        rgb0 = cv2.cvtColor(g0, cv2.COLOR_GRAY2BGR)
        overlay0 = rgb0.copy()
        overlay0[m0 > 0] = (0,0,255)  # BGR red
        blended0 = cv2.addWeighted(rgb0, 1.0, overlay0, ALPHA, 0.0)
        # save
        cv2.imwrite(str(out_overlay_dir / f"overlay_{frame_idx:03d}.png"), blended0)
        cv2.imwrite(str(out_mask_dir / f"mask_{frame_idx:03d}.png"), (m0*255).astype(np.uint8))
        frame_idx += 1
    except Exception as e:
        logging.error(f"Slice {z} kaydedilemedi: {e}")

logging.info(f"Kaydedilen dilim sayısı: {frame_idx}")
logging.info(f"Maskler -> {out_mask_dir}")
logging.info(f"Overlay -> {out_overlay_dir}")

# Ayrıca rapor dosyası yaz (hangi RT/CT seçildi)
report = {
    "patient_folder": BASE_DIR,
    "selected_rtstruct": selected_rt_path,
    "selected_ct_series_id": selected_series_id,
    "selected_ct_file_count": len(selected_ct_files),
    "gtv_list": gtv_list,
    "z_slices_saved": int(frame_idx)
}
import json
with open(Path(OUTPUT_DIR) / "selection_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

logging.info("İşlem tamamlandı.")
