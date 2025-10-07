#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a single LDR (PNG/JPG) + HDR (EXR) pair:
- Runs your HDRNet to predict NEW HDR EXR (saved to disk)
- Computes metrics (MAE, RMSE, PSNR, SSIM) in linear & log1p space
- Saves tone-mapped previews, over-1 masks, and error heatmap
"""

import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import json
import math
import argparse
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm

# -----------------------------
# Utilities (matching your repo)
# -----------------------------

def srgb_to_linear(img):
    """Convert sRGB [0..1] to linear RGB."""
    img = np.float32(img)
    thr = 0.04045
    below = img <= thr
    above = ~below
    out = np.empty_like(img, dtype=np.float32)
    out[below] = img[below] / 12.92
    out[above] = ((img[above] + 0.055) / 1.055) ** 2.4
    return out

def load_exr(path: str) -> np.ndarray:
    exr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if exr is None:
        raise RuntimeError(f"Failed to read EXR: {path}")
    if exr.ndim == 2:
        exr = np.stack([exr, exr, exr], axis=-1)
    if exr.shape[-1] >= 3:
        exr = cv2.cvtColor(exr, cv2.COLOR_BGR2RGB)
    return np.nan_to_num(exr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

def save_exr(path: str, rgb: np.ndarray):
    bgr = rgb[..., ::-1].astype(np.float32)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"cv2.imwrite failed for {path}")

def load_ldr_linear(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read LDR: {path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return srgb_to_linear(rgb)

def tone_map_log(hdr: np.ndarray, exposure: float=1.0, gamma: float=2.2) -> np.ndarray:
    """Log tone map + gamma for preview PNG (0..1)."""
    x = np.clip(hdr, 0, None) * float(exposure)
    x = np.log1p(x)
    x /= (np.max(x) + 1e-8)
    x = np.clip(x, 0, 1)
    if gamma and gamma != 1.0:
        x = np.clip(x, 1e-8, 1.0) ** (1.0/gamma)
    return x

def to_png(path: str, rgb01: np.ndarray):
    """Save [0..1] RGB image to PNG."""
    arr = np.clip(rgb01, 0, 1)
    u8 = (arr * 255.0 + 0.5).astype(np.uint8)
    bgr = u8[..., ::-1]
    cv2.imwrite(path, bgr)

def luma709(rgb_t: torch.Tensor) -> torch.Tensor:
    """BT.709 luma on torch tensor [B,3,H,W]."""
    w = torch.tensor([0.2126, 0.7152, 0.0722], device=rgb_t.device, dtype=rgb_t.dtype).view(1,3,1,1)
    return (rgb_t * w).sum(1, keepdim=True)

# -----------------------------
# Simple metrics (linear & log)
# -----------------------------

def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2, dtype=np.float64))

def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b), dtype=np.float64))

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return math.sqrt(mse(a, b))

def psnr(a: np.ndarray, b: np.ndarray, data_range: float) -> float:
    m = mse(a, b)
    if m <= 0.0:
        return float("inf")
    return 10.0 * math.log10((data_range ** 2) / m)

def ssim_gray(a: np.ndarray, b: np.ndarray, data_range: float) -> float:
    """
    Lightweight SSIM on single-channel arrays (H,W) using 11x11 Gaussian approx via box filters.
    Not a full reference implementation; good enough for quick QA.
    """
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    k1, k2 = 0.01, 0.03
    L = float(data_range)
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2

    # 11x11 box filter as cheap local mean (you can replace with Gaussian if needed)
    def blur(x):
        x = cv2.blur(x, (11, 11))
        return x

    mu_a = blur(a)
    mu_b = blur(b)
    sigma_a2 = blur(a * a) - mu_a * mu_a
    sigma_b2 = blur(b * b) - mu_b * mu_b
    sigma_ab = blur(a * b) - mu_a * mu_b

    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / ((mu_a**2 + mu_b**2 + c1) * (sigma_a2 + sigma_b2 + c2) + 1e-12)
    return float(np.clip(np.mean(ssim_map), -1.0, 1.0))

def rgb_to_luma(a: np.ndarray) -> np.ndarray:
    w = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)[None, None, :]
    y = (a * w).sum(axis=-1)
    return y

# -----------------------------
# Model (mirror your HDRNet)
# -----------------------------

import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.bn1   = nn.BatchNorm2d(channels)
        self.act   = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act(x + y)

class HDRNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, num_resblocks=8):
        super().__init__()
        self.enc1 = nn.Conv2d(in_channels, base_channels, 3, stride=1, padding=1)
        self.enc2 = nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1)

        blocks = []
        for i in range(num_resblocks):
            dilation = 2 if i % 2 == 0 else 1
            blocks.append(ResidualBlock(base_channels*4, dilation))
        self.bottleneck = nn.Sequential(*blocks)

        self.dec3 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(base_channels*2, base_channels, 4, stride=2, padding=1)
        self.dec1 = nn.Conv2d(base_channels, out_channels, 3, stride=1, padding=1)

        self.out_act = nn.Softplus(beta=1.0)

    def forward(self, x):
        e1 = F.relu(self.enc1(x), inplace=True)
        e2 = F.relu(self.enc2(e1), inplace=True)
        e3 = F.relu(self.enc3(e2), inplace=True)
        b  = self.bottleneck(e3)
        d3 = F.relu(self.dec3(b), inplace=True)
        d2 = F.relu(self.dec2(d3), inplace=True)
        out = self.dec1(d2)
        return self.out_act(out)

# -----------------------------
# Sliding-window inference
# -----------------------------

def hann2d(h, w):
    wy = torch.hann_window(h, periodic=False).unsqueeze(1)
    wx = torch.hann_window(w, periodic=False).unsqueeze(0)
    w2d = (wy @ wx).float()
    return w2d.clamp_min(1e-6)

@torch.no_grad()
def infer_sliding(model, ldr_chw: torch.Tensor, device, patch=1024, overlap=256, channels_last=False):
    model.eval().to(device)
    C, H, W = ldr_chw.shape
    padH = (-H) % 4
    padW = (-W) % 4
    if padH or padW:
        ldr_chw = F.pad(ldr_chw, (0, padW, 0, padH), mode="reflect")
    _, Hp, Wp = ldr_chw.shape

    stride = max(1, patch - overlap)
    weight = hann2d(patch, patch).to(device)  # [P,P]
    out = torch.zeros(3, Hp, Wp, device=device)
    acc = torch.zeros(1, Hp, Wp, device=device)

    for y in tqdm(range(0, Hp - patch + 1, stride), desc="Tiling Y"):
        for x in range(0, Wp - patch + 1, stride):
            tile = ldr_chw[:, y:y+patch, x:x+patch].unsqueeze(0).to(device)
            if channels_last:
                tile = tile.to(memory_format=torch.channels_last)
            with autocast("cuda", enabled=torch.cuda.is_available()):
                pred = model(tile)[0]
            w2d = weight
            out[:, y:y+patch, x:x+patch] += pred * w2d
            acc[:, y:y+patch, x:x+patch] += w2d

    pred_full = (out / acc).clamp_min(0.0)
    pred_full = pred_full[:, :H, :W] if (padH or padW) else pred_full
    return pred_full  # [3,H,W], scaled model space (same as training output)

# -----------------------------
# Main
# -----------------------------

def build_args():
    ap = argparse.ArgumentParser("Evaluate single LDR+HDR pair")
    ap.add_argument("--ldr", required=True, help="Path to LDR PNG/JPG")
    ap.add_argument("--gt", required=True, help="Path to GT HDR EXR")
    ap.add_argument("--ckpt", required=True, help="Path to model checkpoint (.pt)")
    ap.add_argument("--outdir", required=True, help="Output folder for results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tile", type=int, default=1024, help="Sliding window tile")
    ap.add_argument("--overlap", type=int, default=256, help="Sliding window overlap")
    ap.add_argument("--scale-hdr", type=float, default=4.0, help="Training scaling factor (targets were divided by this)")
    ap.add_argument("--exposure", type=float, default=1.0, help="Preview exposure")
    ap.add_argument("--gamma", type=float, default=2.2, help="Preview gamma")
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--vizdir", default="./viz", help="folder for training visualizations")
    ap.add_argument("--viz-interval", type=int, default=5, help="save previews every N epochs")
    return ap.parse_args()

def main():
    args = build_args()
    outp = Path(args.outdir)
    outp.mkdir(parents=True, exist_ok=True)

    # Load data
    ldr_lin = load_ldr_linear(args.ldr)              # [H,W,3] linear 0..1
    gt_hdr  = load_exr(args.gt)                      # [H,W,3] linear HDR (scene)
    H, W, _ = ldr_lin.shape

    # To tensors
    ldr_t = torch.from_numpy(ldr_lin.transpose(2,0,1)).unsqueeze(0).float()  # [1,3,H,W]
    device = args.device

    # Build model & load checkpoint
    model = HDRNet()
    d = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(d["model_state"])
    model.to(device)

    # Inference (sliding)
    pred_scaled = infer_sliding(
        model,
        ldr_t[0],  # [3,H,W]
        device=device,
        patch=args.tile,
        overlap=args.overlap,
        channels_last=args.channels_last
    )  # [3,H,W], in model's scaled space

    # Undo HDR scaling used in training targets
    pred_hdr = (pred_scaled * args.scale_hdr).clamp_min(0.0).cpu().numpy().transpose(1,2,0)

    # Save predicted EXR
    save_exr(str(outp / "pred.exr"), pred_hdr)

    # Save previews
    to_png(str(outp / "ldr_input.png"), np.clip(ldr_lin, 0, 1))
    to_png(str(outp / "pred_tm.png"),   tone_map_log(pred_hdr, exposure=args.exposure, gamma=args.gamma))
    to_png(str(outp / "gt_tm.png"),     tone_map_log(gt_hdr,   exposure=args.exposure, gamma=args.gamma))

    # Over-1 masks
    over1_pred = (pred_hdr > 1.0).any(axis=-1).astype(np.uint8) * 255
    over1_gt   = (gt_hdr   > 1.0).any(axis=-1).astype(np.uint8) * 255
    cv2.imwrite(str(outp / "over1_pred.png"), over1_pred)
    cv2.imwrite(str(outp / "over1_gt.png"),   over1_gt)

    # Log-domain error heatmap
    log_pred = np.log1p(np.clip(pred_hdr, 0, None))
    log_gt   = np.log1p(np.clip(gt_hdr,  0, None))
    log_diff = np.abs(log_pred - log_gt)
    # normalize to 0..255 for colormap
    norm = log_diff / (np.percentile(log_diff, 99.5) + 1e-8)
    heat = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(outp / "log_error_heatmap.png"), heat_color)

    # Metrics (linear & log1p), computed on luma and RGB
    report = {}

    def add_metrics(tag: str, A: np.ndarray, B: np.ndarray, data_range: float):
        rep = {}
        rep["mae_rgb"]  = mae(A, B)
        rep["rmse_rgb"] = rmse(A, B)
        rep["psnr_rgb"] = psnr(A, B, data_range=data_range)

        Ya = rgb_to_luma(A); Yb = rgb_to_luma(B)
        rep["mae_luma"]  = mae(Ya, Yb)
        rep["rmse_luma"] = rmse(Ya, Yb)
        rep["psnr_luma"] = psnr(Ya, Yb, data_range=data_range)

        # SSIM on luma only (faster & more stable for HDR eval)
        rep["ssim_luma"] = ssim_gray(Ya, Yb, data_range=data_range)
        report[tag] = rep

    # Linear metrics: choose data_range from combined 99.9 percentile to be robust
    rng_lin = float(np.percentile(np.concatenate([gt_hdr.reshape(-1,3), pred_hdr.reshape(-1,3)], axis=0), 99.9))
    rng_lin = max(rng_lin, 1.0)
    add_metrics("linear", pred_hdr, gt_hdr, data_range=rng_lin)

    # Log1p metrics (range ~ log1p(max))
    max_log = float(np.max(log_gt))
    max_log = max(max_log, 1.0)
    add_metrics("log1p", log_pred, log_gt, data_range=max_log)

    # Additional stats
    report["pred_stats"] = {
        "max": float(np.max(pred_hdr)),
        "p99": float(np.percentile(pred_hdr, 99)),
        "over1_ratio": float(np.mean((pred_hdr > 1.0)))
    }
    report["gt_stats"] = {
        "max": float(np.max(gt_hdr)),
        "p99": float(np.percentile(gt_hdr, 99)),
        "over1_ratio": float(np.mean((gt_hdr > 1.0)))
    }

    # Save reports
    (outp / "metrics.json").write_text(json.dumps(report, indent=2))
    with open(outp / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(report, indent=2))
    print("\n== Metrics ==")
    print(json.dumps(report, indent=2))
    print(f"\nSaved results to: {outp.resolve()}")

if __name__ == "__main__":
    main()
# EOF