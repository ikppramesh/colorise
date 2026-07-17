#!/usr/bin/env python3
"""
IMAX Colourisation Engine v3
=============================================================
  • Zhang et al. ECCV16 deep colorization model
  • CLAHE contrast enhancement on L channel before inference
  • 1.35x saturation boost on AB channels
  • Output scaled + letterboxed to 1998x1080 (IMAX 1.90:1)

Usage:
  python3 colorise_imax_v3.py <preprocessed_input.mp4> <output_silent.mp4>

Input is expected to already have black bars removed and
watermark cleaned (handled upstream by FFmpeg in the shell script).
"""

import sys
import os
import time

try:
    import cv2
    import numpy as np
    import torch
    from colorizers import eccv16, preprocess_img, postprocess_tens
except ImportError as e:
    print(f"MISSING: {e}", flush=True)
    print("Run from inside the project venv: source .venv/bin/activate", flush=True)
    sys.exit(2)

# ── IMAX target — 1.43:1 (traditional 70mm IMAX film ratio) ───────────────────
# 1544x1080 = 1.4296:1 ≈ 1.43:1  (no black bars, fill by height, crop width)
IMAX_W = 1544
IMAX_H = 1080

# ── Processing parameters ──────────────────────────────────────────────────────
SAT_BOOST  = 1.35
CLAHE_CLIP = 2.5
CLAHE_GRID = (8, 8)
INFER_HW   = (256, 256)

# ── Console colours ────────────────────────────────────────────────────────────
G = "\033[0;32m"; C = "\033[0;36m"; Y = "\033[1;33m"
R = "\033[0;31m"; RS = "\033[0m"

def log(m):  print(f"{C}[INFO]{RS}  {m}", flush=True)
def ok(m):   print(f"{G}[DONE]{RS}  {m}", flush=True)
def warn(m): print(f"{Y}[WARN]{RS}  {m}", flush=True)
def err(m):  print(f"{R}[ERR ]{RS}  {m}", file=sys.stderr, flush=True)

_clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)


def load_model():
    log("Loading ECCV16 colorization model (downloads ~130 MB on first run)...")
    colorizer = eccv16(pretrained=True).eval()
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        log("Device  : Apple MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        log("Device  : CUDA GPU")
    else:
        device = torch.device("cpu")
        warn("Device  : CPU — colorisation will be slow")
    colorizer = colorizer.to(device)
    ok("Model ready")
    return colorizer, device


def enhance_contrast(frame_bgr):
    """CLAHE on the L channel for better local contrast before inference."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def colorize_frame(colorizer, device, frame_bgr):
    frame_bgr  = enhance_contrast(frame_bgr)
    frame_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tens_l_orig, tens_l_rs = preprocess_img(frame_rgb, HW=INFER_HW)
    tens_l_rs  = tens_l_rs.to(device)
    with torch.no_grad():
        out_ab = colorizer(tens_l_rs).cpu() * SAT_BOOST
    out_rgb = postprocess_tens(tens_l_orig, out_ab)
    return cv2.cvtColor(
        (np.clip(out_rgb, 0, 1) * 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR
    )


def scale_to_imax(frame_bgr):
    """
    Scale to IMAX 1.43:1 (1544x1080) — NO black bars.
    Strategy: scale by HEIGHT to fill 1080px fully, then center-crop
    width to 1544px. Content always fills the entire frame.
    """
    h, w    = frame_bgr.shape[:2]
    scale   = IMAX_H / h                          # fill full height
    new_w   = int(round(w * scale))
    resized = cv2.resize(frame_bgr, (new_w, IMAX_H), interpolation=cv2.INTER_LANCZOS4)
    # Center-crop width — no padding, no bars
    x_off   = (new_w - IMAX_W) // 2
    x_off   = max(x_off, 0)
    cropped = resized[:, x_off : x_off + IMAX_W]
    # Safety: if source is narrower than IMAX_W, pad (rare with 2.28:1 source)
    if cropped.shape[1] < IMAX_W:
        canvas = np.zeros((IMAX_H, IMAX_W, 3), dtype=np.uint8)
        cx = (IMAX_W - cropped.shape[1]) // 2
        canvas[:, cx : cx + cropped.shape[1]] = cropped
        return canvas
    return cropped


def colorize_video(input_path, output_path, colorizer, device):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        err(f"Cannot open video: {input_path}")
        return False

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    log(f"Input   : {w}x{h} @ {fps:.2f} fps  |  {total} frames")
    log(f"Output  : {IMAX_W}x{IMAX_H} IMAX 1.90:1")
    log(f"Engine  : ECCV16  |  CLAHE clip={CLAHE_CLIP}  |  Sat x{SAT_BOOST}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (IMAX_W, IMAX_H))
    if not writer.isOpened():
        err(f"Cannot open writer: {output_path}")
        cap.release()
        return False

    n = 0; t0 = time.time(); BAR = 44
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        n += 1
        colored    = colorize_frame(colorizer, device, frame)
        imax_frame = scale_to_imax(colored)
        writer.write(imax_frame)

        pct    = n / max(total, 1)
        filled = int(BAR * pct)
        bar    = '█' * filled + '░' * (BAR - filled)
        elapsed = time.time() - t0
        fps_r  = n / elapsed if elapsed > 0 else 0
        eta    = (total - n) / fps_r if fps_r > 0 else 0
        print(
            f"\r  [{bar}] {int(pct*100):3d}%  "
            f"frame {n}/{total}  {fps_r:.2f} f/s  "
            f"ETA {int(eta//60)}m{int(eta%60):02d}s   ",
            end='', flush=True
        )

    print()
    cap.release()
    writer.release()
    ok(f"Colorised {n} frames in {int(time.time()-t0)}s")
    return True


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 colorise_imax_v3.py <input.mp4> <output_silent.mp4>")
        sys.exit(1)

    input_path  = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.isfile(input_path):
        err(f"Input not found: {input_path}")
        sys.exit(1)

    colorizer, device = load_model()
    success = colorize_video(input_path, output_path, colorizer, device)
    sys.exit(0 if success else 1)
