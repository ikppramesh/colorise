#!/usr/bin/env python3
"""
Pathala Bhairavi — Watermark Removal (v12)

Detection fix — larger vertical Gaussian sigma (120 vs previous 60):
  With sigma=60 the text zone (y=480-582, height=102px) contributes ~60% of the
  blur weight at y=520.  This self-contamination shrinks the apparent excess to
  only 40% of alpha*(255-bg), making faint or bright-background text undetectable.

  With sigma=120 (MARGIN=360, WIDE zone y=120-942), the text zone weight drops
  to ~33%, so the apparent excess rises to ~67% of the true signal:
    alpha=0.30, bg=180 → apparent_excess ≈ 15.1  (caught at THRESH_LO=10)
    alpha=0.30, bg=220 → apparent_excess ≈  7.1  (caught at THRESH_LO=7)

Dual threshold [THRESH_LO, THRESH_HI):
  Upper cap of 110 rejects very bright static scene elements (pipe, beam)
  whose vertical excess can reach 100-200 but would otherwise pass detection.

Reconstruction: vertical column interpolation (same as v10/v11, per-frame,
  uses N_REF_REC=20 clean rows above/below text zone).

Logo: inpaint TELEA (unchanged).
"""

import cv2
import numpy as np
import subprocess
import os
from tqdm import tqdm

SRC = "/Users/rameshinampudi/Documents/Projects/IMAX_MOVIES/Colorise/Source/Pathala Bhairavi Telugu Full Movie 2K ｜ NTR ｜ Savitri ｜ SVR ｜ Ghantasala ｜ Old Telugu Classic Movies [1400p].mp4"

CLIP_START = "00:05:00"
CLIP_DUR   = 10

OUT        = "/Users/rameshinampudi/Documents/Projects/IMAX_MOVIES/Colorise/Output/TestClips/PB_WatermarkRemoved_10sec.mp4"
TMP_CLIP   = "/tmp/pb_raw_clip.mp4"
TMP_SILENT = "/tmp/pb_nowm_silent.mp4"

# ── Watermark zones ───────────────────────────────────────────────────────────
LOGO_Y1, LOGO_Y2 = 0,   265
LOGO_X1, LOGO_X2 = 0,   210

TEXT_Y1, TEXT_Y2 = 480, 582    # actual letter rows
TEXT_X1          = 1150         # skip fence (ends ~x=1176); TEXT_X2 = W

# ── Detection parameters ──────────────────────────────────────────────────────
# Larger sigma → less self-contamination from text zone rows in the blur
# sigma=120: text zone weight at y=520 drops to 33% (vs 60% with sigma=60)
# apparent_excess = 0.67 * alpha*(255-bg)
SIGMA_Y   = 120     # vertical Gaussian sigma
MARGIN    = 360     # px above/below TEXT zone for vertical blur (= 3*SIGMA_Y)

THRESH_LO = 8       # minimum excess  (catches text with alpha≈0.15 on bg<180)
THRESH_HI = 110     # maximum excess  (rejects very bright static scene elements)

# ── Reconstruction parameters ─────────────────────────────────────────────────
N_REF_REC = 20      # rows above/below text zone to average for fill

# ── Step 1: extract raw clip ──────────────────────────────────────────────────
print("[1/4] Extracting raw clip ...")
subprocess.run([
    'ffmpeg', '-y', '-ss', CLIP_START, '-t', str(CLIP_DUR),
    '-i', SRC, '-c:v', 'copy', '-c:a', 'copy', TMP_CLIP
], check=True, stderr=subprocess.DEVNULL)

# ── Step 2: build static text mask (large-sigma vertical blur) ────────────────
print(f"[2/4] Building temporal text mask (sigma_y={SIGMA_Y}, thresh={THRESH_LO}-{THRESH_HI}) ...")
cap   = cv2.VideoCapture(TMP_CLIP)
fps   = cap.get(cv2.CAP_PROP_FPS)
W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
TEXT_X2 = W
H_zone  = TEXT_Y2 - TEXT_Y1    # 102
W_zone  = TEXT_X2 - TEXT_X1    # 1410

WIDE_Y1 = max(0, TEXT_Y1 - MARGIN)   # 120
WIDE_Y2 = min(H, TEXT_Y2 + MARGIN)   # 942

ksize_y = int(6 * SIGMA_Y) | 1       # force odd  (= 721)
KSIZE   = (1, ksize_y)               # pure vertical kernel

N_SAMPLE = min(40, total)
per_frame_excess = []

for _ in range(N_SAMPLE):
    ret, f = cap.read()
    if not ret:
        break

    wide_gray = cv2.cvtColor(
        f[WIDE_Y1:WIDE_Y2, TEXT_X1:TEXT_X2], cv2.COLOR_BGR2GRAY
    ).astype(np.float32)

    blurred = cv2.GaussianBlur(wide_gray, KSIZE, sigmaX=0, sigmaY=SIGMA_Y)
    excess  = wide_gray - blurred

    ts = TEXT_Y1 - WIDE_Y1   # index into WIDE zone where text zone starts
    te = TEXT_Y2 - WIDE_Y1
    per_frame_excess.append(excess[ts:te, :])   # (H_zone, W_zone)

# Temporal minimum: pixels persistently above vertical background = watermark
min_excess = np.stack(per_frame_excess).min(axis=0)   # (H_zone, W_zone)

# Dual threshold: catch watermark band, reject very bright static scene elements
raw_mask = np.where(
    (min_excess > THRESH_LO) & (min_excess < THRESH_HI),
    np.uint8(255), np.uint8(0)
)

# Close gaps within letter shapes
closed = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))

# Keep text-shaped blobs: area > 100, aspect (W/H) > 0.8, not spanning full zone height
num_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(
    closed, connectivity=8
)
filtered = np.zeros_like(raw_mask)
for i in range(1, num_labels):
    x0, y0, bw, bh, area = stats[i]
    aspect = bw / max(bh, 1)
    if area > 100 and aspect > 0.8 and bh < (H_zone - 5):
        filtered[label_map == i] = 255

static_text_mask = cv2.dilate(filtered, np.ones((5, 5), np.uint8), iterations=1)

n_px    = int(static_text_mask.sum()) // 255
zone_px = H_zone * W_zone
print(f"  Text mask: {n_px} px ({n_px / zone_px * 100:.1f}% of "
      f"{H_zone}×{W_zone} zone)")
cv2.imwrite('/tmp/pb_mask_v12.png', static_text_mask)

# ── Build logo mask from first frame ──────────────────────────────────────────
def build_logo_mask(logo_zone):
    hsv  = cv2.cvtColor(logo_zone, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(logo_zone, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(logo_zone.shape[:2], dtype=np.uint8)
    mask[hsv[:, :, 1] > 30] = 255
    mask[gray > 200]         = 255
    return cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=1)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, first_frame = cap.read()
static_logo_mask = build_logo_mask(
    first_frame[LOGO_Y1:LOGO_Y2, LOGO_X1:LOGO_X2]
)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ── Precompute reconstruction weights ────────────────────────────────────────
ts_rec    = np.linspace(0, 1, H_zone, dtype=np.float32)
bool_mask = static_text_mask > 0   # (H_zone, W_zone)

# ── Step 3: process all frames ────────────────────────────────────────────────
print(f"[3/4] Processing {total} frames @ {fps:.1f} fps ({W}×{H}) ...")
writer = cv2.VideoWriter(
    TMP_SILENT, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H)
)

with tqdm(total=total, unit='fr') as bar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = frame.copy()

        # ── Logo removal (inpaint) ───────────────────────────────────────────
        logo_zone = result[LOGO_Y1:LOGO_Y2, LOGO_X1:LOGO_X2].copy()
        result[LOGO_Y1:LOGO_Y2, LOGO_X1:LOGO_X2] = cv2.inpaint(
            logo_zone, static_logo_mask, 11, cv2.INPAINT_TELEA
        )

        # ── Text removal: vertical column interpolation ──────────────────────
        above = result[TEXT_Y1 - N_REF_REC : TEXT_Y1,
                       TEXT_X1 : TEXT_X2].astype(np.float32)     # (N_REF_REC, W_zone, 3)
        below = result[TEXT_Y2 : TEXT_Y2 + N_REF_REC,
                       TEXT_X1 : TEXT_X2].astype(np.float32)     # (N_REF_REC, W_zone, 3)

        above_mean = above.mean(axis=0)   # (W_zone, 3)
        below_mean = below.mean(axis=0)   # (W_zone, 3)

        bg_est = (above_mean[np.newaxis] * (1.0 - ts_rec[:, np.newaxis, np.newaxis]) +
                  below_mean[np.newaxis] *          ts_rec[:, np.newaxis, np.newaxis])

        zone = result[TEXT_Y1:TEXT_Y2, TEXT_X1:TEXT_X2].astype(np.float32)
        zone[bool_mask] = bg_est[bool_mask]
        result[TEXT_Y1:TEXT_Y2, TEXT_X1:TEXT_X2] = np.clip(zone, 0, 255).astype(np.uint8)

        writer.write(result)
        bar.update(1)

cap.release()
writer.release()

# ── Step 4: mux audio ─────────────────────────────────────────────────────────
print("[4/4] Muxing audio ...")
subprocess.run([
    'ffmpeg', '-y',
    '-i', TMP_SILENT, '-i', TMP_CLIP,
    '-map', '0:v:0', '-map', '1:a:0',
    '-c:v', 'libx264', '-crf', '16', '-preset', 'fast',
    '-c:a', 'aac', '-b:a', '192k',
    OUT
], check=True, stderr=subprocess.DEVNULL)

os.remove(TMP_CLIP)
os.remove(TMP_SILENT)
print(f"\n[done] {OUT}")
