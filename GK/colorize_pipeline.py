#!/usr/bin/env python3
"""
Gundamma Katha – Full Colorization Pipeline
  1. Remove watermarks via inpainting (CLASSIC CINEMA logo + center text)
  2. Colorize with DDColor artistic model (PyTorch / Apple MPS)
  3. Unsharp-mask sharpening (no blur in output)
  4. Reassemble with original audio via ffmpeg
"""

import cv2
import numpy as np
import torch
import sys
import os
import subprocess
from tqdm import tqdm
from huggingface_hub import hf_hub_download

# ── DDColor from cloned repo ─────────────────────────────────────────────────
sys.path.insert(0, '/tmp/DDColor')
from ddcolor.model import DDColor
from ddcolor.pipeline import ColorizationPipeline, build_ddcolor_model

# ── Paths ────────────────────────────────────────────────────────────────────
INPUT_VIDEO  = "/Users/rameshinampudi/Documents/Projects/IMAX_MOVIES/Colorise/Output/Gundamma_Katha_Test_2min.mp4"
TEMP_VIDEO   = "/Users/rameshinampudi/Documents/Projects/IMAX_MOVIES/Colorise/Output/_temp_no_audio.mp4"
OUTPUT_VIDEO = "/Users/rameshinampudi/Documents/Projects/IMAX_MOVIES/Colorise/Output/Gundamma_Katha_Test_2min_Colorised.mp4"
MODEL_CACHE  = "/tmp/ddcolor_cache"

# ── Device ───────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"[device] {device}")

# ── Download model ───────────────────────────────────────────────────────────
print("[model] Downloading DDColor artistic weights ...")
model_path = hf_hub_download(
    repo_id="piddnad/ddcolor_artistic",
    filename="pytorch_model.bin",
    cache_dir=MODEL_CACHE,
)
print(f"[model] Weights at: {model_path}")

# ── Build model ───────────────────────────────────────────────────────────────
print("[model] Building DDColor-large ...")
try:
    model = build_ddcolor_model(
        DDColor,
        model_path=model_path,
        input_size=512,
        model_size="large",
        decoder_type="MultiScaleColorDecoder",
        device=device,
    )
    # Quick MPS sanity check
    dummy = torch.zeros(1, 3, 512, 512, device=device)
    with torch.no_grad():
        _ = model(dummy)
    print(f"[model] Ready on {device}")
except Exception as e:
    print(f"[model] {device} failed ({e}), falling back to CPU")
    device = torch.device("cpu")
    model = build_ddcolor_model(
        DDColor,
        model_path=model_path,
        input_size=512,
        model_size="large",
        decoder_type="MultiScaleColorDecoder",
        device=device,
    )
    print("[model] Ready on CPU")

pipeline = ColorizationPipeline(model, input_size=512, device=device)

# ── Watermark regions (1920x1080) ─────────────────────────────────────────────
# (y_start, y_end, x_start, x_end)
WM_REGIONS = [
    (90,  220, 1715, 1920),  # top-right: CLASSIC CINEMA (TELUGU) coloured logo
    (320, 450,   40,  530),  # centre-left: semi-transparent CLASSIC CINEMA text
]

def make_mask(h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    for y1, y2, x1, x2 in WM_REGIONS:
        mask[y1:y2, x1:x2] = 255
    return mask

def remove_watermarks(frame, mask):
    """Reconstruct watermark area via fast-marching inpainting (sharp, no blur)."""
    return cv2.inpaint(frame, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

def sharpen(frame):
    """Unsharp mask – increases perceived sharpness without adding blur."""
    blurred = cv2.GaussianBlur(frame, (0, 0), 1.2)
    out = cv2.addWeighted(frame, 1.45, blurred, -0.45, 0)
    return np.clip(out, 0, 255).astype(np.uint8)

# ── Open input ────────────────────────────────────────────────────────────────
cap   = cv2.VideoCapture(INPUT_VIDEO)
fps   = cap.get(cv2.CAP_PROP_FPS)
W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"[video] {W}x{H} @ {fps:.2f} fps | {total} frames")

wm_mask = make_mask(H, W)

# ── Output writer (intermediate raw frames) ───────────────────────────────────
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(TEMP_VIDEO, fourcc, fps, (W, H))

# ── Process ───────────────────────────────────────────────────────────────────
print("[process] Starting colorization ...")
n = 0
try:
    with tqdm(total=total, unit="frame") as bar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            clean     = remove_watermarks(frame, wm_mask)
            colorized = pipeline.process(clean)
            sharp     = sharpen(colorized)
            writer.write(sharp)
            n += 1
            bar.update(1)
finally:
    cap.release()
    writer.release()

print(f"[process] {n} frames written -> {TEMP_VIDEO}")

# ── Merge audio ────────────────────────────────────────────────────────────────
print("[audio] Merging original audio ...")
cmd = [
    'ffmpeg', '-y',
    '-i', TEMP_VIDEO,
    '-i', INPUT_VIDEO,
    '-c:v', 'libx264',
    '-crf', '17',
    '-preset', 'medium',
    '-c:a', 'aac',
    '-b:a', '192k',
    '-map', '0:v:0',
    '-map', '1:a:0',
    OUTPUT_VIDEO
]
subprocess.run(cmd, check=True)

# ── Cleanup ────────────────────────────────────────────────────────────────────
if os.path.exists(TEMP_VIDEO):
    os.remove(TEMP_VIDEO)

print(f"\nDone -> {OUTPUT_VIDEO}")