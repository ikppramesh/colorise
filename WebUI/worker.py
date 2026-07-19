#!/usr/bin/env python3
"""
Generic colorize worker for the web UI.

Same DDColor colorize + color-grade pipeline as colorize_full_movie.py, but:
  - Generalized to any input video (not a fixed movie).
  - Optional resolution rescale (original / 720p / 1080p), aspect preserved.
  - Bitrate-targeted 2-pass encode (no size-guessing math — bitrate is set
    directly, same value approved for The Great Dictator run: 3500kbps).
  - Emits machine-readable progress to --status-file as JSON so a web
    frontend can poll it (colorize %, encode pass 1/2 % with ffmpeg -progress).
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback

import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colorize_lib import load_model, frame_to_tensor, apply_ab, sharpen, SCALE_MAP

# ── CLI args ────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument('--input', required=True)
ap.add_argument('--output', required=True)
ap.add_argument('--status-file', required=True)
ap.add_argument('--tmp', required=True)
ap.add_argument('--resolution', choices=['original', '720p', '1080p'], default='1080p')
ap.add_argument('--video-bitrate-kbps', type=int, default=3500)
ap.add_argument('--target-size-gb', type=float, default=None,
                 help='if set, compute video bitrate from this output-size cap instead of --video-bitrate-kbps')
ap.add_argument('--audio-bitrate-kbps', type=int, default=192)
ap.add_argument('--audio-mode', choices=['off', 'stereo', 'surround51'], default='off',
                 help='off: plain AAC re-encode; stereo: loudness-normalize + denoise; '
                      'surround51: upmix to 5.1 + Dolby Digital Plus (E-AC3) — not real Dolby Atmos, '
                      'that requires a licensed Dolby encoder unavailable to ffmpeg')
ap.add_argument('--surround-bitrate-kbps', type=int, default=448)
ap.add_argument('--grade', default='eq=brightness=0.03:contrast=0.85:saturation=1.1:gamma=1.12,'
                                    'colortemperature=temperature=7500:mix=0.4:pl=1')
ap.add_argument('--input-size', type=int, default=512)
args = ap.parse_args()

BATCH = 8  # measured optimum on Apple Silicon MPS — bigger batches OOM / slow down, see profiling
INPUT_SIZE = args.input_size
INPUT = args.input
TMP = args.tmp
OUTPUT = args.output
CACHE = "/tmp/ddcolor_cache"
PASSLOG = args.tmp + '.passlog'

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
os.makedirs(os.path.dirname(args.status_file), exist_ok=True)

# ── Status helpers ────────────────────────────────────────────────────────────
STAGE_WEIGHTS = {'loading_model': 0.0, 'colorizing': 0.80, 'encode_pass1': 0.10, 'encode_pass2': 0.10}
STAGE_ORDER = ['loading_model', 'colorizing', 'encode_pass1', 'encode_pass2']

def stage_base(stage):
    idx = STAGE_ORDER.index(stage)
    return sum(STAGE_WEIGHTS[s] for s in STAGE_ORDER[:idx]) * 100

def write_status(**kwargs):
    stage = kwargs.get('stage')
    local_pct = kwargs.pop('local_percent', 0.0)
    overall = stage_base(stage) + local_pct * STAGE_WEIGHTS.get(stage, 0) if stage in STAGE_WEIGHTS else kwargs.get('percent', 0)
    payload = {'percent': round(min(overall, 100), 1), 'timestamp': time.time(), **kwargs}
    tmp_path = args.status_file + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(payload, f)
    os.replace(tmp_path, args.status_file)

def fail(msg):
    write_status(stage='error', message=msg, percent=0)
    sys.exit(1)

try:
    write_status(stage='loading_model', message='Loading DDColor model...')

    device, model = load_model(INPUT_SIZE)

    cap = cv2.VideoCapture(INPUT)
    if not cap.isOpened():
        fail(f"Could not open input video: {INPUT}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(3))
    H = int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    write_status(stage='colorizing', local_percent=0, frame=0, total_frames=total, fps_proc=0)

    wr = cv2.VideoWriter(TMP, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    n = 0
    t0 = time.time()
    try:
        while n < total:
            frames, tensors = [], []
            for _ in range(min(BATCH, total - n)):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                tensors.append(frame_to_tensor(frame, INPUT_SIZE))
            if not frames:
                break

            batch_t = torch.stack(tensors).to(device=device, dtype=torch.float32)
            with torch.inference_mode():
                output_abs = model(batch_t).float().cpu()

            for i, frame in enumerate(frames):
                colorized = apply_ab(frame, output_abs[i:i + 1])
                sharp = sharpen(colorized)
                wr.write(sharp)
                n += 1

            elapsed = time.time() - t0
            fr_per_sec = n / elapsed if elapsed > 0 else 0
            eta = (total - n) / fr_per_sec if fr_per_sec > 0 else None
            write_status(stage='colorizing', local_percent=100 * n / total if total else 0,
                         frame=n, total_frames=total, fps_proc=round(fr_per_sec, 2),
                         eta_seconds=round(eta) if eta else None)
    finally:
        cap.release()
        wr.release()

    colorize_elapsed = time.time() - t0
    duration_sec = n / fps if fps > 0 else 0

    # ── Resolution scale ──────────────────────────────────────────────────────
    scale = SCALE_MAP[args.resolution]
    vf_chain = args.grade if scale is None else f'{args.grade},scale={scale}'

    # ── Bitrate: either fixed, or derived from a target output-size cap ───────
    audio_kbps_for_budget = args.surround_bitrate_kbps if args.audio_mode == 'surround51' else args.audio_bitrate_kbps
    if args.target_size_gb is not None and duration_sec > 0:
        target_bits = args.target_size_gb * 8 * (1024 ** 3)
        audio_bits = audio_kbps_for_budget * 1000 * duration_sec
        video_kbps = max(300, int((target_bits - audio_bits) / duration_sec / 1000))
    else:
        video_kbps = args.video_bitrate_kbps
    maxrate = int(video_kbps * 1.5)
    bufsize = int(video_kbps * 2)

    def run_ffmpeg_with_progress(cmd, stage_name, **extra_status):
        progress_file = TMP + f'.progress_{stage_name}.txt'
        full_cmd = cmd + ['-progress', progress_file, '-nostats']
        proc = subprocess.Popen(full_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        while proc.poll() is None:
            time.sleep(1)
            try:
                with open(progress_file) as f:
                    lines = f.read().strip().split('\n')
                out_time_ms = None
                for line in reversed(lines):
                    if line.startswith('out_time_ms='):
                        val = line.split('=')[1]
                        if val.strip().lstrip('-').isdigit():
                            out_time_ms = int(val)
                        break
                if out_time_ms is not None and duration_sec > 0:
                    pct = max(0, min(100, out_time_ms / 1_000_000 / duration_sec * 100))
                    write_status(stage=stage_name, local_percent=pct, frame=n, total_frames=total, **extra_status)
            except FileNotFoundError:
                pass
        proc.wait()
        if os.path.exists(progress_file):
            os.remove(progress_file)
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, full_cmd)

    pass1_cmd = [
        'ffmpeg', '-y', '-i', TMP,
        '-vf', vf_chain,
        '-c:v', 'libx264', '-b:v', f'{video_kbps}k',
        '-maxrate', f'{maxrate}k', '-bufsize', f'{bufsize}k',
        '-preset', 'medium', '-passlogfile', PASSLOG,
        '-pass', '1', '-an', '-f', 'mp4',
    ] + (['/dev/null'])

    # ── Audio enhancement ──────────────────────────────────────────────────────
    # "surround51" upmixes to 5.1 and encodes as E-AC3 (Dolby Digital Plus) — a
    # real Dolby-branded codec ffmpeg can produce. It is NOT Dolby Atmos: Atmos
    # is object-based audio requiring Dolby's own licensed encoder, which isn't
    # available here. loudnorm/afftdn clean up old, often noisy/quiet mono tracks.
    AUDIO_FILTERS = {
        'off': None,
        'stereo': 'loudnorm=I=-16:TP=-1.5:LRA=11,afftdn=nf=-25',
        'surround51': 'loudnorm=I=-16:TP=-1.5:LRA=11,afftdn=nf=-25,surround=chl_out=5.1',
    }
    audio_filter = AUDIO_FILTERS[args.audio_mode]

    if args.audio_mode == 'surround51':
        audio_codec_args = ['-c:a', 'eac3', '-b:a', f'{args.surround_bitrate_kbps}k', '-ac', '6']
    else:
        audio_codec_args = ['-c:a', 'aac', '-b:a', f'{args.audio_bitrate_kbps}k']

    audio_in_cmd = ['-i', INPUT]
    pass2_cmd = [
        'ffmpeg', '-y', '-i', TMP] + audio_in_cmd + [
        '-vf', vf_chain,
    ] + (['-af', audio_filter] if audio_filter else []) + [
        '-c:v', 'libx264', '-b:v', f'{video_kbps}k',
        '-maxrate', f'{maxrate}k', '-bufsize', f'{bufsize}k',
        '-preset', 'medium', '-passlogfile', PASSLOG,
        '-pass', '2',
    ] + audio_codec_args + [
        '-map', '0:v:0', '-map', '1:a?',
        OUTPUT
    ]

    encode_extra = {'video_kbps': video_kbps, 'target_size_gb': args.target_size_gb}
    write_status(stage='encode_pass1', local_percent=0, frame=n, total_frames=total, **encode_extra)
    run_ffmpeg_with_progress(pass1_cmd, 'encode_pass1', **encode_extra)

    write_status(stage='encode_pass2', local_percent=0, frame=n, total_frames=total, **encode_extra)
    run_ffmpeg_with_progress(pass2_cmd, 'encode_pass2', **encode_extra)

    for ext in ['-0.log', '-0.log.mbtree']:
        p = PASSLOG + ext
        if os.path.exists(p):
            os.remove(p)
    if os.path.exists(TMP):
        os.remove(TMP)

    out_size_mb = os.path.getsize(OUTPUT) / (1024 ** 2)
    total_elapsed = time.time() - t0
    write_status(stage='done', percent=100, frame=n, total_frames=total,
                 output_size_mb=round(out_size_mb, 1),
                 colorize_seconds=round(colorize_elapsed),
                 total_seconds=round(total_elapsed),
                 **encode_extra)

except Exception:
    fail(traceback.format_exc())
