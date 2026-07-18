#!/usr/bin/env python3
"""
Local web UI for the DDColor movie-colorize pipeline.

Upload a video, pick resolution/bitrate, watch live progress (colorize %,
then 2-pass encode %), preview + download the finished file.

Run:  python3 app.py
Then open http://127.0.0.1:5151
"""

import json
import os
import signal
import subprocess
import sys
import threading
import uuid

import cv2
import torch
from flask import Flask, jsonify, request, send_file, send_from_directory, abort

from colorize_lib import load_model, frame_to_tensor, apply_ab, sharpen, build_grade_filter, SCALE_MAP

ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(ROOT, 'uploads')
JOBS_DIR = os.path.join(ROOT, 'jobs')
OUTPUT_DIR = os.path.join(ROOT, 'outputs')
STATIC_DIR = os.path.join(ROOT, 'static')
PREVIEW_DIR = os.path.join(ROOT, 'previews')
WORKER = os.path.join(ROOT, 'worker.py')

for d in (UPLOAD_DIR, JOBS_DIR, OUTPUT_DIR, STATIC_DIR, PREVIEW_DIR):
    os.makedirs(d, exist_ok=True)

ALLOWED_EXT = {'.mp4', '.mkv', '.mov', '.avi', '.m4v', '.webm'}
PREVIEW_INPUT_SIZE = 512

app = Flask(__name__, static_folder=None)

# job_id -> subprocess.Popen, kept only for the life of this server process
RUNNING = {}

# Sample-frame preview reuses one in-process model (loaded lazily, kept warm)
# instead of spawning a subprocess per request — full renders still use a
# fresh subprocess (worker.py) since that measurably avoids MPS slowdown
# over long sustained runs; a handful of preview frames doesn't hit that.
_model_lock = threading.Lock()
_model_state = {'device': None, 'model': None}


def get_model():
    with _model_lock:
        if _model_state['model'] is None:
            device, model = load_model(PREVIEW_INPUT_SIZE)
            _model_state['device'] = device
            _model_state['model'] = model
        return _model_state['device'], _model_state['model']


def grade_params_from_form(form):
    def grade_float(name, default, lo, hi):
        try:
            v = float(form.get(name, default))
        except ValueError:
            v = default
        return max(lo, min(v, hi))

    return {
        'brightness': grade_float('brightness', 0.03, -0.5, 0.5),
        'contrast': grade_float('contrast', 0.85, 0.1, 2.0),
        'saturation': grade_float('saturation', 1.1, 0.0, 3.0),
        'gamma': grade_float('gamma', 1.12, 0.1, 3.0),
        'temperature': grade_float('temperature', 7500, 2000, 15000),
        'mix': grade_float('mix', 0.4, 0.0, 1.0),
    }


def job_paths(job_id):
    return {
        'status': os.path.join(JOBS_DIR, job_id, 'status.json'),
        'log': os.path.join(JOBS_DIR, job_id, 'log.txt'),
        'tmp': os.path.join(JOBS_DIR, job_id, 'raw.mp4'),
        'output': os.path.join(OUTPUT_DIR, f'{job_id}.mp4'),
    }


@app.route('/')
def index():
    return send_from_directory(STATIC_DIR, 'index.html')


@app.route('/api/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'no file provided'}), 400
    f = request.files['file']
    if not f.filename:
        return jsonify({'error': 'empty filename'}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({'error': f'unsupported file type {ext}'}), 400

    resolution = request.form.get('resolution', '1080p')
    if resolution not in ('original', '720p', '1080p'):
        return jsonify({'error': 'invalid resolution'}), 400
    try:
        bitrate = int(request.form.get('bitrate', 3500))
    except ValueError:
        return jsonify({'error': 'invalid bitrate'}), 400
    bitrate = max(500, min(bitrate, 20000))
    grade = build_grade_filter(**grade_params_from_form(request.form))

    job_id = uuid.uuid4().hex[:12]
    os.makedirs(os.path.join(JOBS_DIR, job_id), exist_ok=True)
    input_path = os.path.join(UPLOAD_DIR, f'{job_id}{ext}')
    f.save(input_path)

    paths = job_paths(job_id)
    log_f = open(paths['log'], 'w')
    cmd = [
        sys.executable, WORKER,
        '--input', input_path,
        '--output', paths['output'],
        '--status-file', paths['status'],
        '--tmp', paths['tmp'],
        '--resolution', resolution,
        '--video-bitrate-kbps', str(bitrate),
        '--grade', grade,
    ]
    # start_new_session=True puts worker.py in its own process group, so
    # cancelling can kill it *and* any ffmpeg child it spawned in one shot.
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=ROOT, start_new_session=True)
    RUNNING[job_id] = proc

    return jsonify({'job_id': job_id})


@app.route('/api/status/<job_id>')
def status(job_id):
    paths = job_paths(job_id)
    if not os.path.exists(paths['status']):
        return jsonify({'stage': 'starting', 'percent': 0})
    with open(paths['status']) as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError:
            return jsonify({'stage': 'starting', 'percent': 0})
    return jsonify(data)


@app.route('/api/cancel/<job_id>', methods=['POST'])
def cancel(job_id):
    proc = RUNNING.get(job_id)
    if proc is None or proc.poll() is not None:
        return jsonify({'error': 'job not running or already finished'}), 404

    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            proc.wait()
    except ProcessLookupError:
        pass
    RUNNING.pop(job_id, None)

    paths = job_paths(job_id)
    for p in (paths['tmp'], paths['output']):
        if os.path.exists(p):
            os.remove(p)
    passlog = paths['tmp'] + '.passlog'
    for ext in ('-0.log', '-0.log.mbtree'):
        p = passlog + ext
        if os.path.exists(p):
            os.remove(p)

    tmp_status = paths['status'] + '.tmp'
    with open(tmp_status, 'w') as fh:
        json.dump({'stage': 'cancelled', 'percent': 0, 'message': 'Cancelled by user'}, fh)
    os.replace(tmp_status, paths['status'])

    return jsonify({'status': 'cancelled'})


@app.route('/api/output/<job_id>')
def output(job_id):
    paths = job_paths(job_id)
    if not os.path.exists(paths['output']):
        abort(404)
    as_attachment = request.args.get('download') == '1'
    return send_file(paths['output'], as_attachment=as_attachment,
                      download_name=f'colorized_{job_id}.mp4')


def _apply_grade_to_previews(pdir, frames_meta, grade, scale):
    vf = grade if scale is None else f'{grade},scale={scale}'
    images = []
    for m in frames_meta:
        raw_path = os.path.join(pdir, m['raw'])
        graded_name = f"graded_{m['idx']}.png"
        graded_path = os.path.join(pdir, graded_name)
        subprocess.run(['ffmpeg', '-y', '-i', raw_path, '-vf', vf, graded_path],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        images.append({'idx': m['idx'], 'timestamp': m['timestamp'], 'orig': m['orig'], 'graded': graded_name})
    return images


@app.route('/api/preview/colorize', methods=['POST'])
def preview_colorize():
    if 'file' not in request.files:
        return jsonify({'error': 'no file provided'}), 400
    f = request.files['file']
    if not f.filename:
        return jsonify({'error': 'empty filename'}), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({'error': f'unsupported file type {ext}'}), 400

    try:
        num_frames = int(request.form.get('num_frames', 6))
    except ValueError:
        num_frames = 6
    num_frames = max(2, min(num_frames, 10))

    resolution = request.form.get('resolution', '1080p')
    if resolution not in SCALE_MAP:
        return jsonify({'error': 'invalid resolution'}), 400
    grade = build_grade_filter(**grade_params_from_form(request.form))

    preview_id = uuid.uuid4().hex[:12]
    pdir = os.path.join(PREVIEW_DIR, preview_id)
    os.makedirs(pdir, exist_ok=True)
    src_path = os.path.join(pdir, f'source{ext}')
    f.save(src_path)

    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        return jsonify({'error': 'could not open video'}), 400
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return jsonify({'error': 'could not read frame count'}), 400

    positions = [(i + 1) / (num_frames + 1) for i in range(num_frames)]
    indices = sorted(set(min(total - 1, max(0, round(p * total))) for p in positions))

    raw_frames, timestamps = [], []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        raw_frames.append(frame)
        timestamps.append(idx / fps if fps > 0 else 0)
    cap.release()

    if not raw_frames:
        return jsonify({'error': 'no frames could be extracted'}), 400

    device, model = get_model()
    tensors = [frame_to_tensor(fr, PREVIEW_INPUT_SIZE) for fr in raw_frames]
    with _model_lock:
        batch_t = torch.stack(tensors).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            output_abs = model(batch_t).float().cpu()

    frames_meta = []
    for i, fr in enumerate(raw_frames):
        colorized = apply_ab(fr, output_abs[i:i + 1])
        sharp = sharpen(colorized)
        orig_name, raw_name = f'orig_{i}.png', f'raw_{i}.png'
        cv2.imwrite(os.path.join(pdir, orig_name), fr)
        cv2.imwrite(os.path.join(pdir, raw_name), sharp)
        frames_meta.append({'idx': i, 'timestamp': timestamps[i], 'orig': orig_name, 'raw': raw_name})

    with open(os.path.join(pdir, 'meta.json'), 'w') as fh:
        json.dump({'frames': frames_meta, 'resolution': resolution}, fh)

    images = _apply_grade_to_previews(pdir, frames_meta, grade, SCALE_MAP[resolution])
    return jsonify({'preview_id': preview_id, 'frames': images})


@app.route('/api/preview/grade', methods=['POST'])
def preview_grade():
    preview_id = request.form.get('preview_id')
    if not preview_id:
        return jsonify({'error': 'missing preview_id'}), 400
    pdir = os.path.join(PREVIEW_DIR, preview_id)
    meta_path = os.path.join(pdir, 'meta.json')
    if not os.path.exists(meta_path):
        return jsonify({'error': 'unknown preview_id'}), 404

    with open(meta_path) as fh:
        stored = json.load(fh)

    resolution = request.form.get('resolution', stored.get('resolution', '1080p'))
    if resolution not in SCALE_MAP:
        return jsonify({'error': 'invalid resolution'}), 400
    grade = build_grade_filter(**grade_params_from_form(request.form))

    images = _apply_grade_to_previews(pdir, stored['frames'], grade, SCALE_MAP[resolution])
    return jsonify({'preview_id': preview_id, 'frames': images})


@app.route('/api/preview/image/<preview_id>/<name>')
def preview_image(preview_id, name):
    pdir = os.path.abspath(os.path.join(PREVIEW_DIR, preview_id))
    path = os.path.abspath(os.path.join(pdir, name))
    if not path.startswith(pdir + os.sep) or not os.path.exists(path):
        abort(404)
    return send_from_directory(pdir, name)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5151))
    print(f"\n  Movie Colorizer running -> http://127.0.0.1:{port}\n")
    app.run(host='127.0.0.1', port=port, debug=False, threaded=True)
