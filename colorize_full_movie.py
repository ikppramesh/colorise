#!/usr/bin/env python3
"""
Colorize-only pipeline — no watermark removal, no upscaling.
Applies DDColor (artistic) to any input clip and muxes original audio.

Parameterized version of colorize_clip_only.py: accepts --input/--output/
--start/--duration so the same logic can run a short test clip or the
entire movie.
"""

import argparse
import cv2, numpy as np, torch, torch.nn.functional as F
import sys, os, subprocess, time
from tqdm import tqdm
from huggingface_hub import hf_hub_download

sys.path.insert(0, '/tmp/DDColor')
from ddcolor.model import DDColor
from ddcolor.pipeline import build_ddcolor_model

# ── CLI args ───────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument('--input', required=True)
ap.add_argument('--output', required=True)
ap.add_argument('--start', type=float, default=0.0, help='start offset in seconds')
ap.add_argument('--duration', type=float, default=None, help='duration in seconds (default: to end)')
ap.add_argument('--tmp', default='/tmp/colorize_full_movie_tmp.mp4')
ap.add_argument('--grade', default='eq=brightness=0.03:contrast=0.85:saturation=1.1:gamma=1.12,'
                                    'colortemperature=temperature=7500:mix=0.4:pl=1',
                 help='ffmpeg -vf filter chain for color grading')
ap.add_argument('--keep-raw', default=None,
                 help='also save the ungraded colorized clip to this path (for fast re-grading later)')
ap.add_argument('--input-size', type=int, default=512, help='DDColor inference resolution (256 = ~4x faster, lower detail)')
args = ap.parse_args()

# ── Config ─────────────────────────────────────────────────────────────────────
BATCH      = 8
INPUT_SIZE = args.input_size
USE_FP16   = False  # fp16 unsupported on MPS ConvNeXt

INPUT  = args.input
TMP    = args.tmp
OUTPUT = args.output
CACHE  = "/tmp/ddcolor_cache"

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

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
src_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

start_frame = int(round(args.start * fps))
if args.duration is not None:
    n_frames = int(round(args.duration * fps))
    total = min(n_frames, max(src_total - start_frame, 0))
else:
    total = max(src_total - start_frame, 0)

if start_frame > 0:
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

print(f"[video] {W}x{H} @ {fps:.3f}fps | processing {total} frames "
      f"(start_frame={start_frame}) | batch={BATCH}")

wr = cv2.VideoWriter(TMP, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
print("[colorize] DDColor batched inference ...")
n = 0
t0 = time.time()

try:
    with tqdm(total=total, unit="fr") as bar:
        while n < total:
            frames, tensors = [], []
            for _ in range(min(BATCH, total - n)):
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

elapsed = time.time() - t0
fr_per_sec = n / elapsed if elapsed > 0 else 0
print(f"[colorize] {n} frames done in {elapsed:.1f}s ({fr_per_sec:.2f} fr/s)")

if args.keep_raw:
    os.makedirs(os.path.dirname(args.keep_raw), exist_ok=True)
    subprocess.run(['cp', TMP, args.keep_raw], check=True)
    print(f"[keep-raw] ungraded colorized clip saved -> {args.keep_raw}")

# ── Mux audio from original ────────────────────────────────────────────────────
# TMP already covers exactly [start, start+duration), so only seek into the
# original INPUT (for audio) — not into TMP.
print("[ffmpeg] Re-encoding + muxing audio ...")
audio_in_cmd = []
if args.start > 0:
    audio_in_cmd += ['-ss', str(args.start)]
audio_in_cmd += ['-i', INPUT]
if args.duration is not None:
    audio_in_cmd += ['-t', str(args.duration)]

ffmpeg_cmd = ['ffmpeg', '-y', '-i', TMP] + audio_in_cmd + [
    '-vf', args.grade,
    '-c:v', 'libx264', '-crf', '16', '-preset', 'medium',
    '-c:a', 'aac', '-b:a', '192k',
    '-map', '0:v:0', '-map', '1:a?',   # audio optional (silent clips ok)
    OUTPUT
]
subprocess.run(ffmpeg_cmd, check=True, stderr=subprocess.PIPE)

if os.path.exists(TMP):
    os.remove(TMP)

print(f"\n[done] Output -> {OUTPUT}")
print(f"[stats] {n} frames | {elapsed:.1f}s colorize time | {fr_per_sec:.2f} fr/s")
