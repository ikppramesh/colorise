#!/usr/bin/env python3
"""
B&W to Colour Video Engine — v2 (ECCV16 + CLAHE + Saturation Boost)
Uses Zhang et al. ECCV16 model with enhanced colour post-processing.

Improvements over v1:
  • CLAHE on L channel before inference → better local contrast input
  • LAB saturation boost (×1.25 on AB channels) → more vibrant colours
  • Additional FFmpeg saturation lift applied in merge step

https://github.com/richzhang/colorization
"""

import sys
import os
import time

# ── Dependency check ───────────────────────────────────────────────────────────
try:
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from skimage import color
    import skimage.transform
    from colorizers import eccv16, preprocess_img, postprocess_tens
except ImportError as e:
    print(f"MISSING:{e}", flush=True)
    sys.exit(2)

# ── Colours ────────────────────────────────────────────────────────────────────
G = "\033[0;32m"; C = "\033[0;36m"; Y = "\033[1;33m"
R = "\033[0;31m"; B = "\033[1m";    RS = "\033[0m"

def log(m):  print(f"{C}[INFO]{RS}  {m}", flush=True)
def ok(m):   print(f"{G}[DONE]{RS}  {m}", flush=True)
def err(m):  print(f"{R}[ERROR]{RS} {m}", file=sys.stderr, flush=True)


# ── CLAHE for L-channel contrast enhancement ──────────────────────────────────
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def enhance_L(frame_bgr):
    """Apply CLAHE to the L channel of a BGR frame for better local contrast."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ── Model loader ───────────────────────────────────────────────────────────────
def load_model():
    log("Loading ECCV16 colorization model...")
    colorizer = eccv16(pretrained=True).eval()
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        log("Device  : Apple MPS (GPU)")
    else:
        device = torch.device("cpu")
        log("Device  : CPU")
    colorizer = colorizer.to(device)
    ok("Model ready")
    return colorizer, device


# ── Single frame colouriser ───────────────────────────────────────────────────
SAT_BOOST = 1.25   # multiply AB channels by this factor after colourisation

def colorize_frame(colorizer, device, frame_bgr):
    """
    Input : BGR uint8 numpy frame from OpenCV
    Output: BGR uint8 colourised frame (SIGGRAPH17 + saturation boost)
    """
    # CLAHE contrast enhancement on input
    frame_bgr = enhance_L(frame_bgr)

    # BGR → RGB (what colorizers expects)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Preprocess: returns (full-res L tensor, resized L tensor)
    tens_l_orig, tens_l_rs = preprocess_img(frame_rgb, HW=(256, 256))
    tens_l_rs = tens_l_rs.to(device)

    with torch.no_grad():
        out_ab = colorizer(tens_l_rs).cpu()

    # Saturation boost: scale AB channels in-place
    out_ab = out_ab * SAT_BOOST

    # Merge L + AB → RGB [0,1] numpy
    out_rgb = postprocess_tens(tens_l_orig, out_ab)

    # RGB [0,1] → BGR uint8
    out_bgr = cv2.cvtColor(
        (np.clip(out_rgb, 0, 1) * 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR
    )
    return out_bgr


# ── Video processor ────────────────────────────────────────────────────────────
def colorize_video(input_path, output_path, colorizer, device):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        err(f"Cannot open: {input_path}")
        return False

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    log(f"Input   : {width}×{height} @ {fps:.2f} fps · {total} frames")
    log(f"Model   : ECCV16  |  Sat boost: ×{SAT_BOOST}  |  CLAHE: on")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_n = 0
    t0      = time.time()
    BAR     = 42

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_n += 1

        coloured = colorize_frame(colorizer, device, frame)
        writer.write(coloured)

        pct     = frame_n / max(total, 1)
        filled  = int(BAR * pct)
        bar     = '█' * filled + '░' * (BAR - filled)
        elapsed = time.time() - t0
        fps_r   = frame_n / elapsed if elapsed > 0 else 0
        eta     = (total - frame_n) / fps_r if fps_r > 0 else 0
        print(f"\r  [{bar}] {int(pct*100):3d}%  "
              f"frame {frame_n}/{total}  "
              f"{fps_r:.2f} f/s  "
              f"ETA {int(eta//60)}m{int(eta%60):02d}s",
              end='', flush=True)

    print()
    cap.release()
    writer.release()
    ok(f"Done — {frame_n} frames in {int(time.time()-t0)}s")
    return True


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 colorise_engine_v2.py <input_video> <output_silent_video>")
        sys.exit(1)

    colorizer, device = load_model()
    success = colorize_video(sys.argv[1], sys.argv[2], colorizer, device)
    sys.exit(0 if success else 1)
