#!/usr/bin/env python3
"""
IMAX Colourisation Engine v4
=============================================================
  Improvements over v3:
  • Watermark removal via cv2.inpaint() BEFORE the AI sees the frame
    → AI never "learns" the watermark pattern; no artefacts
  • Model upgraded: ECCV16 → SIGGRAPH17 (far more vibrant, scene-aware)
  • Color post-processing:
      - 2.0× boost on LAB A/B channels (kills the brown/sepia cast)
      - HSV Vibrance pass (lifts muted areas without blowing saturation)
  • IMAX 1.43:1 — scale by height, center-crop width → 1544×1080, no bars

Usage:
  python3 colorise_imax_v4.py <preprocessed_input.mp4> <output_silent.mp4>

The input should have black bars already removed by FFmpeg (crop=1280:560:0:80).
Watermark removal is done here in Python on every frame.
"""

import sys
import os
import time

try:
    import cv2
    import numpy as np
    import torch
    from colorizers import siggraph17, preprocess_img, postprocess_tens
except ImportError as e:
    print(f"MISSING: {e}", flush=True)
    sys.exit(2)

# ── IMAX target — 1.43:1 ──────────────────────────────────────────────────────
IMAX_W = 1998
IMAX_H = 1080

# ── Model inference resolution ────────────────────────────────────────────────
INFER_HW = (256, 256)

# ── Watermark mask  (coordinates in the pre-processed 1280×560 frame) ─────────
# "Shalimar" semi-transparent text — generous region to catch full text
WM_X,  WM_Y  = 82,  148
WM_W,  WM_H  = 240, 135    # covers all possible vertical positions of the mark
_wm_mask = None   # built lazily from actual frame size

# ── Color post-processing parameters ─────────────────────────────────────────
LAB_AB_BOOST   = 2.0    # multiply A/B channels after colorisation
VIBRANCE_GAIN  = 0.65   # HSV vibrance: how much to lift low-saturation areas
SAT_UNIFORM    = 1.10   # uniform HSV saturation multiplier on top of vibrance

# ── Console colours ────────────────────────────────────────────────────────────
G = "\033[0;32m"; C = "\033[0;36m"; Y = "\033[1;33m"
R = "\033[0;31m"; RS = "\033[0m"

def log(m):  print(f"{C}[INFO]{RS}  {m}", flush=True)
def ok(m):   print(f"{G}[DONE]{RS}  {m}", flush=True)
def warn(m): print(f"{Y}[WARN]{RS}  {m}", flush=True)
def err(m):  print(f"{R}[ERR ]{RS}  {m}", file=sys.stderr, flush=True)

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# ── Watermark removal ─────────────────────────────────────────────────────────
def _build_wm_mask(h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    x1 = max(WM_X, 0);          x2 = min(WM_X + WM_W, w)
    y1 = max(WM_Y, 0);          y2 = min(WM_Y + WM_H, h)
    mask[y1:y2, x1:x2] = 255
    return mask

def remove_watermark(frame_bgr):
    """Inpaint the 'Shalimar' watermark region before colorisation."""
    global _wm_mask
    h, w = frame_bgr.shape[:2]
    if _wm_mask is None or _wm_mask.shape != (h, w):
        _wm_mask = _build_wm_mask(h, w)
        log(f"Watermark mask built  ({w}×{h}  region {WM_X},{WM_Y}+{WM_W}×{WM_H})")
    return cv2.inpaint(frame_bgr, _wm_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)


# ── Model loader ──────────────────────────────────────────────────────────────
def load_model():
    log("Loading SIGGRAPH17 colorization model (downloads ~150 MB on first run)...")
    colorizer = siggraph17(pretrained=True).eval()
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        log("Device  : Apple MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        log("Device  : CUDA GPU")
    else:
        device = torch.device("cpu")
        warn("Device  : CPU — this will be slow")
    colorizer = colorizer.to(device)
    ok("SIGGRAPH17 model ready")
    return colorizer, device


# ── Per-frame pipeline ────────────────────────────────────────────────────────
def enhance_contrast(frame_bgr):
    """CLAHE on L channel — improves detail before inference."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def colorize_frame(colorizer, device, frame_bgr: np.ndarray) -> np.ndarray:
    frame_bgr  = enhance_contrast(frame_bgr)
    frame_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tens_l_orig, tens_l_rs = preprocess_img(frame_rgb, HW=INFER_HW)
    tens_l_rs = tens_l_rs.to(device)

    with torch.no_grad():
        # SIGGRAPH17: pass only L (no user hints) — still far better than ECCV16
        out_ab = colorizer(tens_l_rs).cpu()

    out_rgb = postprocess_tens(tens_l_orig, out_ab)
    return cv2.cvtColor(
        (np.clip(out_rgb, 0, 1) * 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR
    )


def boost_colors(frame_bgr):
    """
    Two-stage color boost:
      1. LAB AB  ×2.0  — destroys the brown/sepia cast, forces vivid hues
      2. HSV Vibrance   — lifts muted, undercolored regions without clipping
    """
    # ── Stage 1: LAB saturation boost ────────────────────────────────────────
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)

    # OpenCV LAB: A & B are stored in [0, 255] where 128 = neutral gray
    a_c = (a - 128.0) * LAB_AB_BOOST
    b_c = (b - 128.0) * LAB_AB_BOOST

    lab_out = cv2.merge([
        np.clip(l,         0, 255).astype(np.uint8),
        np.clip(a_c + 128, 0, 255).astype(np.uint8),
        np.clip(b_c + 128, 0, 255).astype(np.uint8),
    ])
    frame_bgr = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

    # ── Stage 2: HSV Vibrance ─────────────────────────────────────────────────
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    s_n = s / 255.0
    # Vibrance: muted colours get the biggest lift; already-saturated stay put
    s_n = s_n + (1.0 - s_n) * s_n * VIBRANCE_GAIN
    s_n = np.clip(s_n * SAT_UNIFORM, 0, 1)

    hsv_out = cv2.merge([
        h.astype(np.uint8),
        (s_n * 255).astype(np.uint8),
        v.astype(np.uint8),
    ])
    return cv2.cvtColor(hsv_out, cv2.COLOR_HSV2BGR)


# ── IMAX scale  (fill height, center-crop width — no bars) ───────────────────
def scale_to_imax(frame_bgr):
    h, w   = frame_bgr.shape[:2]
    scale  = IMAX_H / h
    new_w  = int(round(w * scale))
    resized = cv2.resize(frame_bgr, (new_w, IMAX_H), interpolation=cv2.INTER_LANCZOS4)
    x_off  = max((new_w - IMAX_W) // 2, 0)
    cropped = resized[:, x_off : x_off + IMAX_W]
    if cropped.shape[1] < IMAX_W:          # source narrower than target (rare)
        canvas = np.zeros((IMAX_H, IMAX_W, 3), dtype=np.uint8)
        cx = (IMAX_W - cropped.shape[1]) // 2
        canvas[:, cx : cx + cropped.shape[1]] = cropped
        return canvas
    return cropped


# ── Main video processor ──────────────────────────────────────────────────────
def colorize_video(input_path, output_path, colorizer, device):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        err(f"Cannot open: {input_path}")
        return False

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    log(f"Input   : {w}×{h} @ {fps:.2f} fps  |  {total} frames")
    log(f"Output  : {IMAX_W}×{IMAX_H}  IMAX 1.90:1  (no black bars)")
    log(f"Model   : SIGGRAPH17  |  LAB boost ×{LAB_AB_BOOST}  |  Vibrance {VIBRANCE_GAIN}")
    log(f"WM mask : x={WM_X} y={WM_Y} w={WM_W} h={WM_H}  (Shalimar inpaint)")

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

        clean   = remove_watermark(frame)     # 1. inpaint watermark
        colored = colorize_frame(colorizer, device, clean)  # 2. SIGGRAPH17
        vivid   = boost_colors(colored)       # 3. LAB + Vibrance
        out     = scale_to_imax(vivid)        # 4. IMAX 1544×1080

        writer.write(out)

        pct    = n / max(total, 1)
        filled = int(BAR * pct)
        bar    = '█' * filled + '░' * (BAR - filled)
        el     = time.time() - t0
        fps_r  = n / el if el > 0 else 0
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
    ok(f"Done — {n} frames in {int(time.time()-t0)}s")
    return True


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 colorise_imax_v4.py <input.mp4> <output_silent.mp4>")
        sys.exit(1)

    inp, out = sys.argv[1], sys.argv[2]
    if not os.path.isfile(inp):
        err(f"Input not found: {inp}")
        sys.exit(1)

    colorizer, device = load_model()
    sys.exit(0 if colorize_video(inp, out, colorizer, device) else 1)
