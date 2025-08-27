#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTSTRUCT/RS otomatik seçici ve maske çıkarıcı
- CT klasöründeki SOPInstanceUID'lerle RTSTRUCT içindeki ReferencedSOPInstanceUID'leri karşılaştırıp
  en yüksek örtüşme oranına sahip RTSTRUCT'ı seçer.
- Sonrasında seçilen RTSTRUCT ile mask oluşturup CT'ye hizalar.
- Eğer en iyi örtüşme oranı düşükse program uyarı verip sonlanır (güvenlik için).
"""

import os
import glob
import re
import numpy as np
import SimpleITK as sitk
import pydicom
from rt_utils import RTStructBuilder
import cv2
import itertools
from collections import defaultdict
import logging
from pathlib import Path
import sys

# ========== AYARLAR ==========
BASE_DIR   = r"D:\beyin_3_4_5\beyin3.2\1559224"       # Hasta ana klasörü (içinde CT_..., RS_... vb.)
OUTPUT_DIR = r"D:\beyincikti\beyinv3"
ROI_PREFIX = "gtv"
MIN_RTSTRUCT_OVERLAP = 0.30   # en iyi RTSTRUCT'ın CT ile örtüşme oranı >= bu eşik olmalı
SAVE_OVERLAY = True
ALPHA = 0.6
# ============================

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Helper: CT dizinini bul
def find_dir(prefix):
    cands = [d for d in os.listdir(BASE_DIR)
             if os.path.isdir(os.path.join(BASE_DIR, d)) and d.upper().startswith(prefix)]
    if not cands:
        return None
    # tercihen en uzun isme sahip olanı al (orijinal kodun yaklaşımı)
    cands.sort(key=len, reverse=True)
    return os.path.join(BASE_DIR, cands[0])

CT_DIR = find_dir("CT_")
RS_DIR = find_dir("RS_")
RT_DIR = find_dir("RT_")

if not CT_DIR:
    logging.critical("CT_* klasörü bulunamadı. Çıkılıyor.")
    sys.exit(1)

logging.info(f"CT klasörü: {CT_DIR}")
logging.info(f"RS klasörü: {RS_DIR}")
logging.info(f"RT klasörü: {RT_DIR}")

# --- CT seri dosyalarını oku ve SOPInstanceUID seti oluştur ---
reader = sitk.ImageSeriesReader()
series_ids = reader.GetGDCMSeriesIDs(CT_DIR)
if not series_ids:
    logging.critical("CT serisi bulunamadı. Çıkılıyor.")
    sys.exit(1)

# choose first series for now (you can refine)
series_files_map = {}
for sid in series_ids:
    series_files_map[sid] = reader.GetGDCMSeriesFileNames(CT_DIR, sid)

first_sid = series_ids[0]
ct_files = series_files_map[first_sid]
reader.SetFileNames(ct_files)
ct_img = reader.Execute()
ct_np = sitk.GetArrayFromImage(ct_img).astype(np.float32)  # (Z,Y,X)

# Build SOPInstanceUID set for CT series
ct_sop_uids = set()
for f in ct_files:
    try:
        ds = pydicom.dcmread(f, stop_before_pixels=True, specific_tags=["SOPInstanceUID"])
        uid = getattr(ds, "SOPInstanceUID", None)
        if uid:
            ct_sop_uids.add(uid)
    except Exception:
        continue

logging.info(f"Found CT series with {len(ct_files)} files, SOP UID count = {len(ct_sop_uids)}")
logging.info(f"CT shape (Z,Y,X): {ct_np.shape}")

# --- RTSTRUCT candidate'larını tara: RS_DIR, RT_DIR, BASE_DIR ---
def find_candidate_rtstruct_paths():
    roots = []
    if RS_DIR: roots.append(RS_DIR)
    if RT_DIR: roots.append(RT_DIR)
    roots.append(BASE_DIR)
    found = []
    seen = set()
    for root in roots:
        if not root or root in seen: 
            continue
        seen.add(root)
        for p in glob.glob(os.path.join(root, "**", "*.dcm"), recursive=True):
            try:
                ds = pydicom.dcmread(p, stop_before_pixels=True, specific_tags=["Modality"])
                if getattr(ds, "Modality", "") == "RTSTRUCT":
                    found.append(p)
            except Exception:
                continue
    return found

candidates = find_candidate_rtstruct_paths()
if not candidates:
    logging.critical("Herhangi bir RTSTRUCT bulunamadı (RS_/RT_/BASE taramalarında). Çıkılıyor.")
    sys.exit(1)

logging.info(f"RTSTRUCT aday sayısı: {len(candidates)}")

# --- Bir RTSTRUCT dosyasından Referenced SOPInstanceUID setini çıkar ---
def extract_referenced_sop_uids_from_rtstruct(path):
    uids = set()
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        # ROIContourSequence -> ContourSequence -> ContourImageSequence -> ReferencedSOPInstanceUID
        if hasattr(ds, "ROIContourSequence"):
            for rc in ds.ROIContourSequence:
                if hasattr(rc, "ContourSequence"):
                    for cs in rc.ContourSequence:
                        if hasattr(cs, "ContourImageSequence"):
                            for img in cs.ContourImageSequence:
                                uid = getattr(img, "ReferencedSOPInstanceUID", None)
                                if uid:
                                    uids.add(uid)
        # Fallback: bazı RT'lerde farklı isimlendirme/yerleşim olabilir; try StructureSetROISequence references is not enough.
    except Exception as e:
        logging.debug(f"RTSTRUCT okunurken hata: {e}")
    return uids

# --- Her aday için örtüşme oranını hesapla ---
best_path = None
best_overlap = 0.0
best_overlap_count = 0
best_ref_count = 0

for p in candidates:
    ref_uids = extract_referenced_sop_uids_from_rtstruct(p)
    if not ref_uids:
        # Eğer RTSTRUCT'ta ReferencedSOPInstanceUID yoksa, atla — bu genelde contour/image mapping yok demektir.
        logging.info(f"Aday RTSTRUCT {os.path.basename(p)} : ReferencedSOPInstanceUID yok/boş, atlanıyor.")
        continue
    overlap_count = len(ref_uids & ct_sop_uids)
    ref_count = len(ref_uids)
    overlap_ratio = overlap_count / float(ref_count) if ref_count > 0 else 0.0
    logging.info(f"Aday RT: {os.path.basename(p)} -> ref_count={ref_count}, overlap={overlap_count}, ratio={overlap_ratio:.3f}")
    if overlap_ratio > best_overlap:
        best_overlap = overlap_ratio
        best_path = p
        best_overlap_count = overlap_count
        best_ref_count = ref_count

if best_path is None:
    logging.critical("Hiçbir RTSTRUCT'ta ReferencedSOPInstanceUID bulunmadı. Çıkılıyor (güvenlik).")
    sys.exit(1)

logging.info(f"En iyi RTSTRUCT: {best_path} (overlap_ratio={best_overlap:.3f}, {best_overlap_count}/{best_ref_count})")

if best_overlap < MIN_RTSTRUCT_OVERLAP:
    logging.critical(f"En iyi RTSTRUCT bile örtüşme eşiği ({MIN_RTSTRUCT_OVERLAP}) altında: {best_overlap:.3f}. Muhtemelen farklı hastaya ait RT seçildi. Çıkılıyor.")
    sys.exit(1)

rtstruct_path = best_path

# --- rt-utils ile yükle (seçilen RTSTRUCT) ---
logging.info("rt-utils ile RTSTRUCT yükleniyor...")
try:
    rtstruct = RTStructBuilder.create_from(dicom_series_path=CT_DIR, rt_struct_path=rtstruct_path)
except Exception as e:
    logging.critical(f"rt-utils yüklemede hata: {e}")
    sys.exit(1)

all_rois = rtstruct.get_roi_names() or []
logging.info(f"Tüm ROI'lar: {all_rois}")

# Basit GTV seçimi (prefix ile)
def normalize_text(t):
    if not t: return ""
    repl = {'ı':'i','İ':'i','ğ':'g','Ğ':'g','ü':'u','Ü':'u','ş':'s','Ş':'s','ö':'o','Ö':'o','ç':'c','Ç':'c'}
    s = t.lower()
    for a,b in repl.items(): s = s.replace(a,b)
    return s

gtv_names = [r for r in all_rois if normalize_text(r).startswith(normalize_text(ROI_PREFIX))]
if not gtv_names:
    logging.warning("GTV ile başlayan ROI bulunamadı. Çıkılıyor.")
    sys.exit(1)
selected_gtv = gtv_names[0]
logging.info(f"Seçilen GTV ROI: {selected_gtv}")

# --- yardımlar: CT SOP uid -> z index mapping ---
def build_ct_uid_to_z_map(ct_files_list):
    uid_to_z = {}
    for z, f in enumerate(ct_files_list):
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, specific_tags=["SOPInstanceUID"])
            uid = getattr(ds, "SOPInstanceUID", None)
            if uid:
                uid_to_z[uid] = z
            else:
                uid_to_z[f"NOUID_{z}"] = z
        except Exception:
            uid_to_z[f"NOUID_{z}"] = z
    return uid_to_z

uid_to_z = build_ct_uid_to_z_map(ct_files)

# --- Mask hizalama: permütasyon + Z resample ---
def find_permutation_for_rowcol(mask_shape, target_rows, target_cols):
    for perm in itertools.permutations((0,1,2)):
        perm_shape = tuple(np.array(mask_shape)[list(perm)])
        if perm_shape[1] == target_rows and perm_shape[2] == target_cols:
            return perm
    return None

def resample_z_nearest(mask_zyx, target_z):
    src_z = mask_zyx.shape[0]
    if src_z == target_z:
        return mask_zyx
    src_indices = np.round(np.linspace(0, src_z - 1, target_z)).astype(int)
    return mask_zyx[src_indices, :, :]

def align_mask_to_ct(mask_raw, ct_shape):
    target_z, target_y, target_x = ct_shape
    mask_shape = mask_raw.shape
    perm = find_permutation_for_rowcol(mask_shape, target_y, target_x)
    if perm is None:
        raise ValueError(f"Mask permütasyonu bulunamadı. mask.shape={mask_shape}, ct_rows={target_y}, ct_cols={target_x}")
    permuted = np.transpose(mask_raw, axes=perm)  # (Z_src, Y, X)
    z_src = permuted.shape[0]
    if z_src != target_z:
        permuted = resample_z_nearest(permuted, target_z)
    return permuted.astype(bool)

# --- Mask oluştur / birleştir ---
mask_3d = np.zeros(ct_np.shape, dtype=np.uint8)
# İstersen birden fazla GTV'yi birleştirebilirsin; biz sadece seçileni alıyoruz
try:
    m_raw = rtstruct.get_roi_mask_by_name(selected_gtv).astype(np.uint8)
except Exception as e:
    logging.critical(f"ROI maskesi alınamadı: {e}")
    sys.exit(1)

try:
    m_aligned = align_mask_to_ct(m_raw, ct_np.shape)
except Exception as e:
    logging.critical(f"Maske CT ile hizalanamadı: {e}")
    sys.exit(1)

mask_3d = np.logical_or(mask_3d, m_aligned).astype(np.uint8)
logging.info(f"Mask hazır. Toplam mask dilimi sayısı: {np.count_nonzero(mask_3d.max(axis=(1,2))>0)}")

# --- Görselleştirme ve kaydetme ---
def apply_window(img2d, center, width):
    low = center - width / 2.0
    high = center + width / 2.0
    arr = np.clip(img2d, low, high)
    arr = (arr - low) / (high - low + 1e-6) * 255.0
    return arr.astype(np.uint8)

# Ortalama window fallback: basit sabitler
WINDOW_CENTER = 40.0
WINDOW_WIDTH = 80.0

# attempt to get better window from CT files (first few)
for i, f in enumerate(ct_files[:20]):
    try:
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        wc = getattr(ds, "WindowCenter", None)
        ww = getattr(ds, "WindowWidth", None)
        if wc is not None and ww is not None:
            if isinstance(wc, pydicom.multival.MultiValue): wc = float(sum(map(float,wc))/len(wc))
            else: wc = float(wc)
            if isinstance(ww, pydicom.multival.MultiValue): ww = float(sum(map(float,ww))/len(ww))
            else: ww = float(ww)
            WINDOW_CENTER = wc; WINDOW_WIDTH = ww
            break
    except Exception:
        continue

logging.info(f"Using window C={WINDOW_CENTER}, W={WINDOW_WIDTH}")

z_has = np.where(mask_3d.max(axis=(1,2)) > 0)[0]
if len(z_has) == 0:
    logging.warning("Maske bulunan dilim yok. Çıkılıyor.")
    sys.exit(0)
z_start, z_end = int(z_has.min()), int(z_has.max())
logging.info(f"Kombine maske Z aralığı: {z_start}..{z_end}")

frame_idx = 0
for z in range(z_start, z_end + 1):
    m0 = mask_3d[z]
    g0 = apply_window(ct_np[z], WINDOW_CENTER, WINDOW_WIDTH)
    rgb0 = cv2.cvtColor(g0, cv2.COLOR_GRAY2BGR)
    overlay0 = rgb0.copy()
    overlay0[m0 > 0] = (0, 0, 255)
    blended0 = cv2.addWeighted(rgb0, 1.0, overlay0, ALPHA, 0.0)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"overlay_{frame_idx:03d}.png"), blended0)
    if m0.max() > 0:
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"mask_{frame_idx:03d}.png"), (m0*255).astype(np.uint8))
    frame_idx += 1

    # interpolation
    if z < z_end:
        ct_mid = 0.5*ct_np[z] + 0.5*ct_np[z+1]
        m1 = mask_3d[z+1].astype(np.float32)
        m_mid = ((m0.astype(np.float32) + m1) * 0.5) >= 0.5
        m_mid = m_mid.astype(np.uint8)
    else:
        ct_mid = ct_np[z]
        m_mid = m0

    g_mid = apply_window(ct_mid, WINDOW_CENTER, WINDOW_WIDTH)
    rgb_mid = cv2.cvtColor(g_mid, cv2.COLOR_GRAY2BGR)
    overlay_mid = rgb_mid.copy(); overlay_mid[m_mid > 0] = (0, 0, 255)
    blended_mid = cv2.addWeighted(rgb_mid, 1.0, overlay_mid, ALPHA, 0.0)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"overlay_{frame_idx:03d}.png"), blended_mid)
    if m_mid.max() > 0:
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"mask_{frame_idx:03d}.png"), (m_mid*255).astype(np.uint8))
    frame_idx += 1

logging.info(f"Toplam overlay sayısı: {frame_idx}")
logging.info(f"Çıktılar → {OUTPUT_DIR}")
