#!/usr/bin/env python3
"""Shared DDColor inference + grading helpers used by worker.py and app.py."""

import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

sys.path.insert(0, '/tmp/DDColor')
from ddcolor.model import DDColor
from ddcolor.pipeline import build_ddcolor_model

CACHE = "/tmp/ddcolor_cache"

SCALE_MAP = {'original': None, '720p': '-2:720', '1080p': '-2:1080'}


def load_model(input_size=512):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    wts = hf_hub_download("piddnad/ddcolor_artistic", "pytorch_model.bin", cache_dir=CACHE)
    try:
        model = build_ddcolor_model(DDColor, model_path=wts, input_size=input_size,
                                    model_size="large", decoder_type="MultiScaleColorDecoder", device=device)
        with torch.inference_mode():
            _ = model(torch.zeros(1, 3, input_size, input_size, device=device))
    except Exception:
        device = torch.device("cpu")
        model = build_ddcolor_model(DDColor, model_path=wts, input_size=input_size,
                                    model_size="large", decoder_type="MultiScaleColorDecoder", device=device)
    model.eval()
    return device, model


def frame_to_tensor(bgr, input_size):
    img = (bgr / 255.0).astype(np.float32)
    resized = cv2.resize(img, (input_size, input_size))
    img_l = cv2.cvtColor(resized, cv2.COLOR_BGR2Lab)[:, :, :1]
    gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
    gray_rgb = cv2.cvtColor(gray_lab, cv2.COLOR_LAB2RGB)
    return torch.from_numpy(gray_rgb.transpose(2, 0, 1)).float()


def apply_ab(bgr, ab_tensor_1hw2):
    H, W = bgr.shape[:2]
    img = (bgr / 255.0).astype(np.float32)
    orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]
    ab = (F.interpolate(ab_tensor_1hw2, size=(H, W))[0].float().numpy().transpose(1, 2, 0))
    lab = np.concatenate((orig_l, ab), axis=-1)
    bgr_out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return (bgr_out * 255.0).round().astype(np.uint8)


def sharpen(f, a=1.0, s=0.9):
    b = cv2.GaussianBlur(f, (0, 0), s)
    return np.clip(cv2.addWeighted(f, 1 + a, b, -a, 0), 0, 255).astype(np.uint8)


def build_grade_filter(brightness, contrast, saturation, gamma, temperature, mix):
    return (f'eq=brightness={brightness}:contrast={contrast}:saturation={saturation}:gamma={gamma},'
            f'colortemperature=temperature={temperature}:mix={mix}:pl=1')
