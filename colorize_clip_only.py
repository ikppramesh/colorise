#!/usr/bin/env python3
"""
Colorize-only pipeline — no watermark removal, no upscaling.
Applies DDColor (artistic) to any input clip and muxes original audio.
"""

import cv2, numpy as np, torch, torch.nn.functional as F
import sys, os, subprocess
from tqdm import tqdm
from huggingface_hub import hf_hub_download

sys.path.insert(0, '/tmp/DDColor')
from ddcolor.model import DDColor
from ddcolor.pipeline import build_ddcolor_model

# ── Config ─────────────────────────────────────────────────────────────────────
BATCH      = 8
INPUT_SIZE = 512    # higher quality for short clips
USE_FP16   = False  # fp16 unsupported on MPS ConvNeXt

# ── Paths ──────────────────────────────────────────────────────────────────────
INPUT  = "/Users/rameshinampudi/Documents/Projects/IMAX_MOVIES/Colorise/Output/TestClips/PB_WatermarkRemoved_10sec.mp4"
TMP    = "/tmp/pb_colorised_tmp.mp4"
OUTPUT = "/Users/rameshinampudi/Documents/Projects/IMAX_MOVIES/Colorise/Output/TestClips/PB_WatermarkRemoved_10sec_Colorised.mp4"
CACHE  = "/tmp/ddcolor_cache"

# ── Device & model ─────────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[device] {device}")

wts = hf_hub_download("piddnad/ddcolor_artistic", "pytorch_model.bin", cache_dir=CACHE)
try:
    model = build_ddcolor_model(DDColor, model_path=wts, input_size=INPUT_SIZE,
                                model_size="large",
                                decoder_type="MultiScaleColorDecoder", device=device)
    with torch.inference_mode():
        _ = model(torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE, device=device))
    print(f"[model] Ready on {device} fp32")
except Exception as e:
    print(f"[model] MPS→CPU fallback: {e}")
    device = torch.device("cpu")
    model = build_ddcolor_model(DDColor, model_path=wts, input_size=INPUT_SIZE,
                                model_size="large",
                                decoder_type="MultiScaleColorDecoder", device=device)
model.eval()

# ── Helpers ────────────────────────────────────────────────────────────────────
def frame_to_tensor(bgr):
    img     = (bgr / 255.0).astype(np.float32)
    resized = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img_l   = cv2.cvtColor(resized, cv2.COLOR_BGR2Lab)[:, :, :1]
    gray_lab = np.concatenate((img_l,
                                np.zeros_like(img_l),
                                np.zeros_like(img_l)), axis=-1)
    gray_rgb = cv2.cvtColor(gray_lab, cv2.COLOR_LAB2RGB)
    return torch.from_numpy(gray_rgb.transpose(2, 0, 1)).float()

def apply_ab(bgr, ab_tensor_1hw2):
    H, W   = bgr.shape[:2]
    img    = (bgr / 255.0).astype(np.float32)
    orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]
    ab     = (F.interpolate(ab_tensor_1hw2, size=(H, W))[0]
              .float().numpy().transpose(1, 2, 0))
    lab    = np.concatenate((orig_l, ab), axis=-1)
    bgr_out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return (bgr_out * 255.0).round().astype(np.uint8)

def sharpen(f, a=1.0, s=0.9):
    # a=1.0 (was 1.8) — less aggressive to avoid over-brightening
    b = cv2.GaussianBlur(f, (0, 0), s)
    return np.clip(cv2.addWeighted(f, 1+a, b, -a, 0), 0, 255).astype(np.uint8)

# ── Colorization pass ──────────────────────────────────────────────────────────
cap   = cv2.VideoCapture(INPUT)
fps   = cap.get(cv2.CAP_PROP_FPS)
W     = int(cap.get(3))
H     = int(cap.get(4))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"[video] {W}x{H} @ {fps:.3f}fps | {total} frames | batch={BATCH}")

wr = cv2.VideoWriter(TMP, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
print("[colorize] DDColor batched inference ...")
n = 0

try:
    with tqdm(total=total, unit="fr") as bar:
        while True:
            frames, tensors = [], []
            for _ in range(BATCH):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                tensors.append(frame_to_tensor(frame))

            if not frames:
                break

            batch_t = torch.stack(tensors).to(device=device, dtype=torch.float32)
            with torch.inference_mode():
                output_abs = model(batch_t).float().cpu()

            for i, frame in enumerate(frames):
                colorized = apply_ab(frame, output_abs[i:i+1])
                sharp     = sharpen(colorized)
                wr.write(sharp)
                n += 1
                bar.update(1)
finally:
    cap.release()
    wr.release()
print(f"[colorize] {n} frames done")

# ── Mux audio from original ────────────────────────────────────────────────────
print("[ffmpeg] Re-encoding + muxing audio ...")
subprocess.run([
    'ffmpeg', '-y',
    '-i', TMP,
    '-i', INPUT,
    # Color grade: slight brightness pull-down, softer contrast, vibrant saturation
    '-vf', 'eq=brightness=-0.08:contrast=0.82:saturation=1.55:gamma=1.05,hue=s=1.1',
    '-c:v', 'libx264', '-crf', '16', '-preset', 'medium',
    '-c:a', 'aac', '-b:a', '192k',
    '-map', '0:v:0', '-map', '1:a?',   # audio optional (silent clips ok)
    OUTPUT
], check=True, stderr=subprocess.PIPE)

if os.path.exists(TMP):
    os.remove(TMP)

print(f"\n[done] Output -> {OUTPUT}")
