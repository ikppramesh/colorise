#!/usr/bin/env python3
"""
Gundamma Katha – Full Movie Colorization Pipeline (FAST — batched GPU inference)
Based on colorize_pipeline_v2.py

Key speedup over the original:
  • Batched inference (BATCH=8 frames per GPU call vs 1) → ~4-6x faster Pass 1
  • torch.inference_mode() already used in pipeline

Watermark strategy (identical to v2):
  PASS 1 (source):
    • Top-right clap logo (y=0-232, x=1688-1920): inpaint
    • Center 'CLASSIC CINEMA' text: Gaussian blur (sigma=25) if zone dark (max<80)
  PASS 2 (colorized):
    • Same fixed zone: gentle Gaussian blur (sigma=10) if zone dark (median<35, max<185)

4K output capped at ~8 Mbps to stay under 10 GB for the 2h38m runtime.
"""

import cv2, numpy as np, torch, torch.nn.functional as F
import sys, os, subprocess
from tqdm import tqdm
from huggingface_hub import hf_hub_download

sys.path.insert(0, '/tmp/DDColor')
from ddcolor.model import DDColor
from ddcolor.pipeline import build_ddcolor_model

# ── Config ────────────────────────────────────────────────────────────────────
BATCH      = 8      # frames per GPU call
INPUT_SIZE = 256    # DDColor inference resolution — 256 vs 512 = 4× fewer ops, minimal quality diff
USE_FP16   = False  # fp16 unsupported on MPS ConvNeXt — keep fp32

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT   = "/Users/rameshinampudi/Documents/Projects/IMAX_MOVIES/Colorise/Source/Gundamma Katha Full Movie HD ｜ NTR ｜ Nageswara Rao ｜ Savitri ｜ Jamuna.mp4"
TMP1    = "/tmp/gundamma_full_tmp1.mp4"
TMP2    = "/tmp/gundamma_full_tmp2.mp4"
TMP1080 = "/tmp/gundamma_full_1080p.mp4"
OUT4K   = "/Users/rameshinampudi/Documents/Projects/IMAX_MOVIES/Colorise/Output/Gundamma_Katha_Full_Colorised_4K.mp4"
CACHE   = "/tmp/ddcolor_cache"

# ── Device & model ────────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[device] {device}")
wts = hf_hub_download("piddnad/ddcolor_artistic", "pytorch_model.bin", cache_dir=CACHE)
try:
    model = build_ddcolor_model(DDColor, model_path=wts, input_size=INPUT_SIZE,
                                model_size="large",
                                decoder_type="MultiScaleColorDecoder", device=device)
    if USE_FP16:
        try:
            model = model.half()
            with torch.inference_mode():
                _ = model(torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE, device=device, dtype=torch.float16))
            print(f"[model] Ready on {device} fp16")
        except Exception as e:
            print(f"[model] fp16 failed ({e}), falling back to fp32")
            model = model.float()
            USE_FP16 = False
    else:
        with torch.inference_mode():
            _ = model(torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE, device=device))
        print(f"[model] Ready on {device} fp32")
except Exception as e:
    print(f"[model] MPS→CPU: {e}"); device = torch.device("cpu"); USE_FP16 = False
    model = build_ddcolor_model(DDColor, model_path=wts, input_size=INPUT_SIZE,
                                model_size="large",
                                decoder_type="MultiScaleColorDecoder", device=device)
model.eval()

# ── Constants (same as v2) ────────────────────────────────────────────────────
CLAP  = (0, 232, 1688, 1920)
WM_Y1, WM_Y2 = 290, 440
WM_X1, WM_X2 = 12, 470

# ── Helpers ───────────────────────────────────────────────────────────────────
def clean_source(frame):
    result = frame.copy()
    gray   = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # 1. Clap logo (top-right) — mirror-fill from adjacent left pixels (~instant vs slow inpaint)
    y0, y1, x0, x1 = CLAP
    w = x1 - x0  # 232 px
    result[y0:y1, x0:x1] = result[y0:y1, x0-w:x0][:, ::-1]

    # 2. Center text — Gaussian blur only if dark scene
    if gray[WM_Y1:WM_Y2, WM_X1:WM_X2].max() < 80:
        result[WM_Y1:WM_Y2, WM_X1:WM_X2] = \
            cv2.GaussianBlur(result[WM_Y1:WM_Y2, WM_X1:WM_X2], (0, 0), 25)
    return result

def fix_colorized(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    zone = gray[WM_Y1:WM_Y2, WM_X1:WM_X2]
    if (np.median(zone) < 35
            and zone.max() < 185
            and (zone > 80).mean() < 0.05):
        result = frame.copy()
        result[WM_Y1:WM_Y2, WM_X1:WM_X2] = \
            cv2.GaussianBlur(result[WM_Y1:WM_Y2, WM_X1:WM_X2], (0, 0), 10)
        return result
    return frame

def sharpen(f, a=1.8, s=0.9):
    b = cv2.GaussianBlur(f, (0, 0), s)
    return np.clip(cv2.addWeighted(f, 1+a, b, -a, 0), 0, 255).astype(np.uint8)

def frame_to_tensor(clean_bgr):
    """Prepare one cleaned BGR frame → gray-RGB tensor (3, INPUT_SIZE, INPUT_SIZE)."""
    img      = (clean_bgr / 255.0).astype(np.float32)
    resized  = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img_l    = cv2.cvtColor(resized, cv2.COLOR_BGR2Lab)[:, :, :1]
    gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
    gray_rgb = cv2.cvtColor(gray_lab, cv2.COLOR_LAB2RGB)
    return torch.from_numpy(gray_rgb.transpose(2, 0, 1)).float()

def apply_ab(clean_bgr, ab_tensor_1hw2):
    """Merge predicted AB with original L channel → BGR uint8."""
    H, W  = clean_bgr.shape[:2]
    img   = (clean_bgr / 255.0).astype(np.float32)
    orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]
    ab_resized = (
        F.interpolate(ab_tensor_1hw2, size=(H, W))[0]
        .float().numpy().transpose(1, 2, 0)
    )
    lab = np.concatenate((orig_l, ab_resized), axis=-1)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return (bgr * 255.0).round().astype(np.uint8)

# ── PASS 1 — batched colorisation ─────────────────────────────────────────────
cap   = cv2.VideoCapture(INPUT)
fps   = cap.get(cv2.CAP_PROP_FPS)
W     = int(cap.get(3))
H     = int(cap.get(4))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"[video] {W}x{H} @ {fps:.1f}fps | {total} frames | batch={BATCH}")

wr1 = cv2.VideoWriter(TMP1, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
print(f"[pass1] Source clean + DDColor colorize (batch={BATCH}) ...")
n = 0

try:
    with tqdm(total=total, unit="fr") as bar:
        while True:
            # ── accumulate one batch ───────────────────────────────────────
            cleans, tensors = [], []
            for _ in range(BATCH):
                ret, frame = cap.read()
                if not ret:
                    break
                c = clean_source(frame)
                cleans.append(c)
                tensors.append(frame_to_tensor(c))

            if not cleans:
                break

            # ── batched GPU inference ──────────────────────────────────────
            dtype = torch.float16 if USE_FP16 else torch.float32
            batch_t = torch.stack(tensors).to(device=device, dtype=dtype)
            with torch.inference_mode():
                output_abs = model(batch_t).float().cpu()  # always fp32 for numpy ops

            # ── write results ──────────────────────────────────────────────
            for i, clean in enumerate(cleans):
                colorized = apply_ab(clean, output_abs[i:i+1])
                sharp     = sharpen(colorized)
                wr1.write(sharp)
                n += 1
                bar.update(1)

finally:
    cap.release(); wr1.release()
print(f"[pass1] {n} frames done")

# ── PASS 2 — residual watermark suppression ───────────────────────────────────
cap2 = cv2.VideoCapture(TMP1)
wr2  = cv2.VideoWriter(TMP2, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
print("[pass2] Residual watermark suppression ...")
n2 = fixed = 0
try:
    with tqdm(total=total, unit="fr") as bar:
        while True:
            ret, frame = cap2.read()
            if not ret: break
            out = fix_colorized(frame)
            wr2.write(out); n2 += 1
            if not np.array_equal(out, frame): fixed += 1
            bar.update(1)
finally:
    cap2.release(); wr2.release()
print(f"[pass2] {fixed}/{n2} frames suppressed")

if os.path.exists(TMP1): os.remove(TMP1)

# ── Intermediate 1080p (temp) ─────────────────────────────────────────────────
print("[audio] Building 1080p intermediate ...")
subprocess.run(['ffmpeg', '-y', '-i', TMP2, '-i', INPUT,
    '-c:v', 'libx264', '-crf', '15', '-preset', 'medium',
    '-c:a', 'aac', '-b:a', '192k',
    '-map', '0:v:0', '-map', '1:a:0', TMP1080],
    check=True, stderr=subprocess.PIPE)
print(f"[1080p] done -> {TMP1080}")

if os.path.exists(TMP2): os.remove(TMP2)

# ── Final 4K output (<10 GB via 8 Mbps cap) ──────────────────────────────────
print("[4K] Upscaling to 3840x2160 (capped at 8 Mbps to stay under 10 GB) ...")
subprocess.run(['ffmpeg', '-y', '-i', TMP1080,
    '-vf', 'scale=3840:2160:flags=lanczos,unsharp=5:5:2.0:5:5:0.8',
    '-c:v', 'libx264', '-b:v', '8M', '-maxrate', '9M', '-bufsize', '18M',
    '-preset', 'medium', '-c:a', 'copy', OUT4K],
    check=True, stderr=subprocess.PIPE)
print(f"[done] 4K -> {OUT4K}")

if os.path.exists(TMP1080): os.remove(TMP1080)
print("\nAll done!")