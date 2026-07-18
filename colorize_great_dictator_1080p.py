#!/usr/bin/env python3
"""
Colorize-only pipeline for The Great Dictator (1940), upscaled to 1080p
with a hard output-size cap.

Duplicate of colorize_full_movie.py with two additions:
  1. Upscale to 1080p at the final encode step (native 4:3 aspect
     preserved -> 1440x1080, since the source is 1024x768, not 16:9).
  2. Two-pass libx264 encoding with a video bitrate computed from
     --target-size-gb so the final file lands close to (but under) the cap,
     instead of the reference script's fixed crf=16 (which has no size
     guarantee).

Colorization pipeline itself (DDColor batched inference + color grade)
is unchanged from colorize_full_movie.py.
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
ap.add_argument('--input', default='/Users/rameshinampudi/Documents/Projects/IMAX_MOVIES/Colorise/Source/The.Great.Dictator.1940.720p.BrRip.x264.YIFY.mp4')
ap.add_argument('--output', default='/Users/rameshinampudi/Documents/Projects/IMAX_MOVIES/Colorise/Output/Final/The.Great.Dictator.1940.Colorized.1080p.mp4')
ap.add_argument('--start', type=float, default=0.0, help='start offset in seconds')
ap.add_argument('--duration', type=float, default=None, help='duration in seconds (default: to end)')
ap.add_argument('--tmp', default='/tmp/colorize_great_dictator_tmp.mp4')
ap.add_argument('--grade', default='eq=brightness=0.03:contrast=0.85:saturation=1.1:gamma=1.12,'
                                    'colortemperature=temperature=7500:mix=0.4:pl=1',
                 help='ffmpeg -vf color-grade filter chain (upscale is appended automatically)')
ap.add_argument('--scale', default='-2:1080',
                 help='ffmpeg scale filter args (w:h). Default preserves source 4:3 aspect -> 1440x1080')
ap.add_argument('--keep-raw', default=None,
                 help='also save the ungraded colorized clip to this path (for fast re-grading later)')
ap.add_argument('--input-size', type=int, default=512, help='DDColor inference resolution (256 = ~4x faster, lower detail)')
ap.add_argument('--target-size-gb', type=float, default=3.2, help='approximate output file size cap in GB')
ap.add_argument('--audio-bitrate-kbps', type=int, default=192)
ap.add_argument('--video-bitrate-kbps', type=int, default=None,
                 help='override the auto-computed video bitrate instead of deriving it from --target-size-gb')
args = ap.parse_args()

# ── Config ─────────────────────────────────────────────────────────────────────
BATCH      = 8
INPUT_SIZE = args.input_size
USE_FP16   = False  # fp16 unsupported on MPS ConvNeXt

INPUT  = args.input
TMP    = args.tmp
OUTPUT = args.output
CACHE  = "/tmp/ddcolor_cache"
PASSLOG = '/tmp/ffmpeg2pass_great_dictator'

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

# ── Bitrate budget for the size cap ────────────────────────────────────────────
duration_sec = n / fps if fps > 0 else 0
if args.video_bitrate_kbps is not None:
    video_kbps = args.video_bitrate_kbps
else:
    target_bits = args.target_size_gb * 8 * (1024 ** 3)
    audio_bits  = args.audio_bitrate_kbps * 1000 * duration_sec
    video_kbps  = max(500, int((target_bits - audio_bits) / duration_sec / 1000))

print(f"[bitrate] duration={duration_sec:.1f}s target={args.target_size_gb}GB "
      f"-> video={video_kbps}kbps audio={args.audio_bitrate_kbps}kbps")

# ── Mux audio from original + upscale + 2-pass encode to hit the size cap ──────
# TMP already covers exactly [start, start+duration), so only seek into the
# original INPUT (for audio) — not into TMP.
print("[ffmpeg] 2-pass encode: grading + 1080p upscale + muxing audio ...")
audio_in_cmd = []
if args.start > 0:
    audio_in_cmd += ['-ss', str(args.start)]
audio_in_cmd += ['-i', INPUT]
if args.duration is not None:
    audio_in_cmd += ['-t', str(args.duration)]

vf_chain = f'{args.grade},scale={args.scale}'
maxrate  = int(video_kbps * 1.5)
bufsize  = int(video_kbps * 2)

pass1_cmd = [
    'ffmpeg', '-y', '-i', TMP,
    '-vf', vf_chain,
    '-c:v', 'libx264', '-b:v', f'{video_kbps}k',
    '-maxrate', f'{maxrate}k', '-bufsize', f'{bufsize}k',
    '-preset', 'medium', '-passlogfile', PASSLOG,
    '-pass', '1', '-an', '-f', 'mp4', '/dev/null'
]
pass2_cmd = [
    'ffmpeg', '-y', '-i', TMP] + audio_in_cmd + [
    '-vf', vf_chain,
    '-c:v', 'libx264', '-b:v', f'{video_kbps}k',
    '-maxrate', f'{maxrate}k', '-bufsize', f'{bufsize}k',
    '-preset', 'medium', '-passlogfile', PASSLOG,
    '-pass', '2',
    '-c:a', 'aac', '-b:a', f'{args.audio_bitrate_kbps}k',
    '-map', '0:v:0', '-map', '1:a?',   # audio optional (silent clips ok)
    OUTPUT
]
subprocess.run(pass1_cmd, check=True, stderr=subprocess.PIPE)
subprocess.run(pass2_cmd, check=True, stderr=subprocess.PIPE)

for ext in ['-0.log', '-0.log.mbtree']:
    p = PASSLOG + ext
    if os.path.exists(p):
        os.remove(p)

if os.path.exists(TMP):
    os.remove(TMP)

out_size_gb = os.path.getsize(OUTPUT) / (1024 ** 3)
print(f"\n[done] Output -> {OUTPUT}")
print(f"[stats] {n} frames | {elapsed:.1f}s colorize time | {fr_per_sec:.2f} fr/s | "
      f"final size {out_size_gb:.2f}GB")
