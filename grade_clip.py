#!/usr/bin/env python3
"""
Re-apply color grading to an already-colorized (raw, ungraded) clip without
re-running DDColor inference. Pairs with --keep-raw in colorize_full_movie.py.
"""

import argparse
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument('--raw', required=True, help='ungraded colorized clip (silent)')
ap.add_argument('--audio-source', required=True, help='original movie file, for audio')
ap.add_argument('--start', type=float, default=0.0)
ap.add_argument('--duration', type=float, default=None)
ap.add_argument('--output', required=True)
ap.add_argument('--grade', required=True, help='ffmpeg -vf filter chain')
args = ap.parse_args()

audio_in_cmd = []
if args.start > 0:
    audio_in_cmd += ['-ss', str(args.start)]
audio_in_cmd += ['-i', args.audio_source]
if args.duration is not None:
    audio_in_cmd += ['-t', str(args.duration)]

cmd = ['ffmpeg', '-y', '-i', args.raw] + audio_in_cmd + [
    '-vf', args.grade,
    '-c:v', 'libx264', '-crf', '16', '-preset', 'medium',
    '-c:a', 'aac', '-b:a', '192k',
    '-map', '0:v:0', '-map', '1:a?',
    args.output
]
subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
print(f"[done] Output -> {args.output}")
