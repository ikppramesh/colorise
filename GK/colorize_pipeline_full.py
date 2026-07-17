#!/usr/bin/env python3
"""
Gundamma Katha – Full Movie Colorization Pipeline
Based on colorize_pipeline_v2.py (same logic, full movie input)

Watermark strategy (same as v2):
  PASS 1 (source):
    • Top-right clap logo (y=0-232, x=1688-1920): inpaint
    • Center 'CLASSIC CINEMA' text: Gaussian blur (sigma=25) if zone is dark (max<80)
  PASS 2 (colorized, catches DDColor amplification residuals):
    • Same fixed zone: gentle Gaussian blur (sigma=10) if zone is dark (median<35, max<185)

4K output capped at ~8 Mbps to stay under 10 GB for the 2h38m runtime.
"""

import cv2, numpy as np, torch, sys, os, subprocess
from tqdm import tqdm
from huggingface_hub import hf_hub_download

sys.path.insert(0, '/tmp/DDColor')
from ddcolor.model import DDColor
from ddcolor.pipeline import ColorizationPipeline, build_ddcolor_model

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT    = "/Users/rameshinampudi/Documents/Projects/IMAX_MOVIES/Colorise/Source/Gundamma Katha Full Movie HD ｜ NTR ｜ Nageswara Rao ｜ Savitri ｜ Jamuna.mp4"
TMP1     = "/tmp/gundamma_full_tmp1.mp4"
TMP2     = "/tmp/gundamma_full_tmp2.mp4"
TMP1080  = "/tmp/gundamma_full_1080p.mp4"   # intermediate – deleted after 4K encode
OUT4K    = "/Users/rameshinampudi/Documents/Projects/IMAX_MOVIES/Colorise/Output/Gundamma_Katha_Full_Colorised_4K.mp4"
CACHE    = "/tmp/ddcolor_cache"

# ── Device & model ────────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[device] {device}")
wts = hf_hub_download("piddnad/ddcolor_artistic", "pytorch_model.bin", cache_dir=CACHE)
try:
    model = build_ddcolor_model(DDColor, model_path=wts, input_size=512, model_size="large",
                                decoder_type="MultiScaleColorDecoder", device=device)
    with torch.no_grad(): _ = model(torch.zeros(1, 3, 512, 512, device=device))
    print(f"[model] Ready on {device}")
except Exception as e:
    print(f"[model] MPS→CPU: {e}"); device = torch.device("cpu")
    model = build_ddcolor_model(DDColor, model_path=wts, input_size=512, model_size="large",
                                decoder_type="MultiScaleColorDecoder", device=device)
pipe = ColorizationPipeline(model, input_size=512, device=device)

# ── Constants (same zones as v2) ──────────────────────────────────────────────
CLAP  = (0, 232, 1688, 1920)   # top-right clap logo: y0,y1,x0,x1
WM_Y1, WM_Y2 = 290, 440        # center 'CLASSIC CINEMA' text vertical range
WM_X1, WM_X2 = 12, 470         # center text horizontal range

# ── Helpers (identical to v2) ─────────────────────────────────────────────────
def clean_source(frame):
    """Remove watermarks from source frame before colorization."""
    H, W  = frame.shape[:2]
    result = frame.copy()
    gray   = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # 1. Top-right clap logo – inpaint
    m = np.zeros((H, W), np.uint8)
    m[CLAP[0]:CLAP[1], CLAP[2]:CLAP[3]] = 255
    result = cv2.inpaint(result, m, 7, cv2.INPAINT_TELEA)

    # 2. Center text – blur fixed zone only if scene is dark (skips title/censor/bright scenes)
    if gray[WM_Y1:WM_Y2, WM_X1:WM_X2].max() < 80:
        result[WM_Y1:WM_Y2, WM_X1:WM_X2] = \
            cv2.GaussianBlur(result[WM_Y1:WM_Y2, WM_X1:WM_X2], (0, 0), 25)

    return result

def fix_colorized(frame):
    """Gentle blur on the same fixed zone to suppress any DDColor-amplified residuals."""
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

# ── PASS 1 ────────────────────────────────────────────────────────────────────
cap   = cv2.VideoCapture(INPUT)
fps   = cap.get(cv2.CAP_PROP_FPS)
W     = int(cap.get(3))
H     = int(cap.get(4))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"[video] {W}x{H} @ {fps:.1f}fps | {total} frames")

wr1 = cv2.VideoWriter(TMP1, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
print("[pass1] Source clean + DDColor colorize ...")
n = 0
try:
    with tqdm(total=total, unit="fr") as bar:
        while True:
            ret, frame = cap.read()
            if not ret: break
            clean = clean_source(frame)
            col   = pipe.process(clean)
            sharp = sharpen(col)
            wr1.write(sharp); n += 1; bar.update(1)
finally:
    cap.release(); wr1.release()
print(f"[pass1] {n} frames done")

# ── PASS 2 ────────────────────────────────────────────────────────────────────
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

# ── Intermediate 1080p (temp – for 4K input) ──────────────────────────────────
print("[audio] Building 1080p intermediate ...")
subprocess.run(['ffmpeg', '-y', '-i', TMP2, '-i', INPUT,
    '-c:v', 'libx264', '-crf', '15', '-preset', 'medium',
    '-c:a', 'aac', '-b:a', '192k',
    '-map', '0:v:0', '-map', '1:a:0', TMP1080],
    check=True, stderr=subprocess.PIPE)
print(f"[1080p] done -> {TMP1080}")

if os.path.exists(TMP2): os.remove(TMP2)

# ── Final 4K output (<10 GB cap via 8 Mbps bitrate) ──────────────────────────
# Full movie is 2h38m (9503s). At 8 Mbps video + 192k audio the output is ~9.5 GB.
print("[4K] Upscaling to 3840x2160 (capped at 8 Mbps to stay under 10 GB) ...")
subprocess.run(['ffmpeg', '-y', '-i', TMP1080,
    '-vf', 'scale=3840:2160:flags=lanczos,unsharp=5:5:2.0:5:5:0.8',
    '-c:v', 'libx264', '-b:v', '8M', '-maxrate', '9M', '-bufsize', '18M',
    '-preset', 'medium', '-c:a', 'copy', OUT4K],
    check=True, stderr=subprocess.PIPE)
print(f"[done] 4K -> {OUT4K}")

if os.path.exists(TMP1080): os.remove(TMP1080)

print("\nAll done!")