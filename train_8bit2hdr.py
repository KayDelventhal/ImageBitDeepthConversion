#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: train_8bit2hdr.py
"""
8-bit → HDR (EXR) training script (residual encoder-decoder, AMP, robust logging)

- Safe log-loss (no NaNs)
- Mixed precision + grad clipping + grad accumulation
- Checkpointing (best/latest), resume, start-epoch
- CSV/JSONL logs, per-epoch timing, ETA
- Early stop via file or patience
- Lock file to avoid concurrent runs
- Full env/spec printout
- Optional TensorBoard logging
"""

from __future__ import annotations
import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import sys
import time
import json
import math
import gc
import csv
import signal
import random
import argparse
from datetime import datetime
from pathlib import Path
from glob import glob

import numpy as np

try:
    import psutil
except Exception:
    psutil = None
    
import shutil
from textwrap import shorten
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from torch.amp import autocast, GradScaler

# install tensorflow/tensorboard # for tensorboard support
# pip install tensorboard
# pip install matplotlib scikit-image imageio

import matplotlib.pyplot as plt

# add near other imports
try:
    from skimage.metrics import structural_similarity as ssim_sk, peak_signal_noise_ratio as psnr_sk
except Exception:
    ssim_sk = None
    psnr_sk = None

# Optional matplotlib is not required for training; import lazily if needed
try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

# --------------------------------------------------------------------------------------
# Utilities expected to exist in your repo; fallback stubs if missing
# --------------------------------------------------------------------------------------
def _safe_import_exr_utils():
    try:
        from exr_to_ldr import srgb_to_linear, load_exr, tone_map, batch_convert
        return srgb_to_linear, load_exr, tone_map, batch_convert
    except Exception:
        # Minimal fallbacks (no EXR read/write); training can still run with PNG pairs
        def srgb_to_linear(x):
            x = np.clip(x, 0.0, 1.0)
            a = 0.055
            return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)
        def load_exr(path):
            raise RuntimeError("load_exr not available; ensure exr_to_ldr.py is importable.")
        def tone_map(x, exposure=1.0):
            return (1.0 - np.exp(-exposure * x))
        def batch_convert(*args, **kwargs):
            raise RuntimeError("batch_convert not available; ensure exr_to_ldr.py is importable.")
        return srgb_to_linear, load_exr, tone_map, batch_convert

srgb_to_linear, load_exr, tone_map, batch_convert = _safe_import_exr_utils()

# --------------------------------------------------------------------------------------
# Reproducibility & device helpers
# --------------------------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def device_info():
    info = {
        "python": sys.version.replace("\n", " "),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.version.cuda else None,
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_version": torch.backends.cudnn.version(),
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False,
        "channels_last_supported": True,
        "env": {k: v for k, v in os.environ.items() if k.startswith(("CUDA", "CUDNN", "OPENCV", "TORCH", "PYTORCH"))},
    }
    if torch.cuda.is_available():
        g = torch.cuda.get_device_properties(0)
        info.update({
            "gpu_name": g.name,
            "gpu_total_vram_gb": round(g.total_memory / (1024**3), 2),
            "gpu_sm_count": getattr(g, "multi_processor_count", None),
            "gpu_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        })
    if psutil:
        mem = psutil.virtual_memory()
        info.update({
            "cpu_count": psutil.cpu_count(logical=True),
            "ram_total_gb": round(mem.total / (1024**3), 2),
        })
    return info

def print_specs():
    print("=" * 80)
    print("ENV / SPECIFICATIONS")
    print("=" * 80)
    info = device_info()
    print(json.dumps(info, indent=2))
    print("=" * 80, flush=True)
    
def pad_to_multiple(chw: torch.Tensor, m: int = 4) -> tuple[torch.Tensor, tuple[int,int,int,int]]:
    _, H, W = chw.shape
    padH = (-H) % m
    padW = (-W) % m
    if padH==0 and padW==0:
        return chw, (0,0,0,0)
    pad = (0, padW, 0, padH)  # L,R,T,B
    return F.pad(chw, pad, mode="reflect"), pad

class ValPadToMult4:
    def __call__(self, ldr, hdr):
        ldr, _ = pad_to_multiple(ldr, 4)
        hdr, _ = pad_to_multiple(hdr, 4)
        return ldr, hdr
# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
class HDRDataset(Dataset):
    """
    Paired dataset:
      - LDR: 8-bit (JPG/PNG) → linear float
      - HDR: 32-bit EXR (float) or HDR PNG (if EXR not available)
    Assumes matched filenames by base name in both folders.
    """
    def __init__(self, ldr_folder: str, hdr_folder: str, scale_hdr: float = 4.0,
                 allow_non_exr: bool = False, transform=None, min_hw: int = 1024):
        self.scale_hdr = scale_hdr
        self.allow_non_exr = allow_non_exr
        self.transform = transform
        self.min_hw = min_hw

        def base(p): return Path(p).stem
        ldr_files = {base(p): p for p in sorted(glob(os.path.join(ldr_folder, "*"))) if os.path.isfile(p)}
        hdr_files = {base(p): p for p in sorted(glob(os.path.join(hdr_folder, "*"))) if os.path.isfile(p)}
        common = sorted(set(ldr_files) & set(hdr_files))
        pairs = []

        for k in common:
            ldr_p, hdr_p = ldr_files[k], hdr_files[k]
            # read sizes cheaply; EXR via cv2.imread (float) or your load_exr
            try:
                if Path(hdr_p).suffix.lower() == ".exr":
                    hdr_arr = load_exr(hdr_p)  # linear float
                elif allow_non_exr:
                    hdr_arr = cv2.imread(hdr_p, cv2.IMREAD_UNCHANGED)
                    if hdr_arr is None:
                        continue
                    if hdr_arr.ndim == 2:
                        hdr_arr = np.stack([hdr_arr, hdr_arr, hdr_arr], axis=-1)
                    hdr_arr = cv2.cvtColor(hdr_arr, cv2.COLOR_BGR2RGB).astype(np.float32)
                else:
                    continue
            except Exception:
                continue

            H, W = hdr_arr.shape[:2]
            if H < self.min_hw or W < self.min_hw:
                continue

            ldr_img = cv2.imread(ldr_p, cv2.IMREAD_COLOR)
            if ldr_img is None:
                continue
            if ldr_img.shape[0] != H or ldr_img.shape[1] != W:
                # sizes must match; strict
                continue

            pairs.append((ldr_p, hdr_p))

        if not pairs:
            raise RuntimeError(f"No valid pairs ≥{self.min_hw} found in {ldr_folder} / {hdr_folder}")
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def _read_ldr_linear(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read LDR image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return srgb_to_linear(img)

    def _read_hdr(self, path: str) -> np.ndarray:
        ext = Path(path).suffix.lower()
        if ext == ".exr":
            arr = load_exr(path)
        elif self.allow_non_exr:
            # fallback to reading as float PNG/TIF if present
            arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise RuntimeError(f"Failed to read HDR fallback: {path}")
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB).astype(np.float32)
        else:
            raise RuntimeError(f"HDR must be EXR (got {path}). Set allow_non_exr=True to override.")
        arr = np.nan_to_num(arr, nan=0.0, posinf=10.0, neginf=0.0).astype(np.float32)
        return arr

    def __getitem__(self, idx: int):
        ldr_path, hdr_path = self.pairs[idx]
        ldr = self._read_ldr_linear(ldr_path)            # [H,W,3] float32 0..1 (linear)
        hdr = self._read_hdr(hdr_path)                   # [H,W,3] float32 linear HDR
        hdr = hdr / self.scale_hdr                       # normalized target

        # To tensors [C,H,W]
        ldr_t = torch.from_numpy(ldr.transpose(2, 0, 1)).contiguous()
        hdr_t = torch.from_numpy(hdr.transpose(2, 0, 1)).contiguous()
        # apply paired transform (e.g., 1024x1024 tiles)
        if self.transform is not None:
            ldr_t, hdr_t = self.transform(ldr_t, hdr_t)
        return ldr_t, hdr_t

# --- Add to your script (near dataset) ---
"""
Details
Loading: _read_hdr() reads the EXR as linear RGB float and returns the full frame (e.g., 3840×2160). No scale, no pad, no crop.
Pairing: LDR/HDR are matched by basename; both are loaded at their own native sizes. The code assumes the pair has the same H×W.
Network down/upsampling: The model downsamples twice (stride-2) and upsamples twice, so for perfect size symmetry the input H and W should be divisible by 4.
Most camera plates (like your 3840×2160) already satisfy this.
Non-multiple-of-4 sizes usually still work, but can risk a one-pixel mismatch in some architectures; your transposed conv setup is the standard invertible pair for multiples of 2.
Memory: Big EXRs (e.g., 3840×2160 float32 RGB ≈ 95–100 MB each) will drive VRAM. Batch size and AMP determine feasibility; there’s no automatic tiling in the current script.
Training targets: Only a numeric normalization (divide by scale_hdr=4.0) is applied. Geometry is unchanged.
Inference: Outputs keep the original size. Before writing EXR, predictions are multiplied back by scale_hdr (no resize).
If you need tiling or resizing
Tiling: add a patch-crop in the dataset (e.g., random 512/1024 tiles with consistent crops on LDR/HDR). This is the usual way to fit 8 GB VRAM.
Resizing: avoid for VFX truth unless you really must; cropping is safer than resampling.
So: any size can be used as long as LDR and HDR match, and ideally H,W % 4 == 0 for perfect down/up symmetry.

Training: tile (and optionally scale-jitter) into 1024×1024
Always feed paired crops (same window on LDR & HDR).
Prefer random tiles during training for coverage; optional multi-scale jitter.
Keep H,W multiples of 4 inside the net (your encoder/decoder strides).
"""
class PairedRandomTiler:
    def __init__(self, patch=1024, scale_jitter=None):
        self.patch = patch
        self.scale_jitter = scale_jitter

    def __call__(self, ldr_chw: torch.Tensor, hdr_chw: torch.Tensor):
        _, H, W = ldr_chw.shape
        assert hdr_chw.shape[-2:] == (H, W), "LDR/HDR must match size"

        if self.scale_jitter is not None:
            s = float(torch.empty(1).uniform_(*self.scale_jitter))
            newH, newW = max(8, int(round(H * s))), max(8, int(round(W * s)))
            ldr_chw = F.interpolate(ldr_chw.unsqueeze(0), size=(newH, newW), mode="bilinear", align_corners=False).squeeze(0)
            hdr_chw = F.interpolate(hdr_chw.unsqueeze(0), size=(newH, newW), mode="bilinear", align_corners=False).squeeze(0)
            H, W = newH, newW

        needH = max(self.patch, (H + 3)//4 * 4)
        needW = max(self.patch, (W + 3)//4 * 4)
        padH, padW = max(0, needH - H), max(0, needW - W)
        if padH or padW:
            pad = (0, padW, 0, padH)
            ldr_chw = F.pad(ldr_chw, pad, mode="reflect")
            hdr_chw = F.pad(hdr_chw, pad, mode="reflect")
            H += padH; W += padW

        if H == self.patch and W == self.patch:
            y0, x0 = 0, 0
        else:
            y0 = torch.randint(0, H - self.patch + 1, (1,)).item()
            x0 = torch.randint(0, W - self.patch + 1, (1,)).item()

        return (ldr_chw[:, y0:y0+self.patch, x0:x0+self.patch].contiguous(),
                hdr_chw[:, y0:y0+self.patch, x0:x0+self.patch].contiguous())

# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
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
    """
    Residual encoder-decoder with dilated residual blocks.
    Output activation uses Softplus to keep values >=0 but unbounded above (HDR-safe).
    """
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

        self.out_act = nn.Softplus(beta=1.0)  # >=0, unbounded

    def forward(self, x):
        e1 = F.relu(self.enc1(x), inplace=True)
        e2 = F.relu(self.enc2(e1), inplace=True)
        e3 = F.relu(self.enc3(e2), inplace=True)
        b  = self.bottleneck(e3)
        d3 = F.relu(self.dec3(b), inplace=True)
        d2 = F.relu(self.dec2(d3), inplace=True)
        out = self.dec1(d2)
        return self.out_act(out)  # linear HDR ≥ 0

# --------------------------------------------------------------------------------------
# Loss (safe)
# --------------------------------------------------------------------------------------
def hdr_loss(pred, target, alpha=0.5, eps=1e-6):
    # both ≥0 due to Softplus; extra clamp for numerical safety
    pred_c = torch.clamp(pred, min=0.0)
    targ_c = torch.clamp(target, min=0.0)
    l1 = F.l1_loss(pred_c, targ_c)
    log_l1 = F.l1_loss(torch.log1p(pred_c + eps), torch.log1p(targ_c + eps))
    return alpha * l1 + (1 - alpha) * log_l1

def _to_img_np(t: torch.Tensor, clamp_min=0.0, clamp_max=None):
    # expects [B,C,H,W] or [C,H,W] in linear HDR "normalized" space (÷scale_hdr)
    if t.ndim == 4: t = t[0]
    t = t.detach().float().cpu()
    if clamp_max is not None:
        t = t.clamp(min=clamp_min, max=clamp_max)
    else:
        t = t.clamp(min=clamp_min)
    x = t.permute(1, 2, 0).numpy()  # HWC
    return x

def compute_psnr(pred_np: np.ndarray, tgt_np: np.ndarray, data_range: float = 1.0) -> float:
    # prefer skimage if available (handles float inputs well)
    if psnr_sk is not None:
        return float(psnr_sk(tgt_np, pred_np, data_range=data_range))
    # fallback: manual
    mse = np.mean((pred_np - tgt_np) ** 2, dtype=np.float64)
    if mse <= 1e-12: 
        return 99.0
    return float(10.0 * np.log10((data_range ** 2) / mse))

def compute_ssim(pred_np: np.ndarray, tgt_np: np.ndarray, data_range: float = 1.0) -> float:
    if ssim_sk is not None:
        # channel_axis=2 for HWC; fallback to multichannel=True if older skimage
        try:
            return float(ssim_sk(tgt_np, pred_np, data_range=data_range, channel_axis=2))
        except TypeError:
            return float(ssim_sk(tgt_np, pred_np, data_range=data_range, multichannel=True))
    # super-lightweight grayscale SSIM fallback
    def _to_gray(x):
        if x.shape[-1] == 1: 
            return x[...,0]
        # Rec. 709 luma
        return 0.2126*x[...,0] + 0.7152*x[...,1] + 0.0722*x[...,2]
    X = _to_gray(pred_np); Y = _to_gray(tgt_np)
    K1, K2 = 0.01, 0.03
    L = data_range
    C1, C2 = (K1*L)**2, (K2*L)**2
    muX, muY = X.mean(), Y.mean()
    sigX2 = ((X - muX)**2).mean()
    sigY2 = ((Y - muY)**2).mean()
    sigXY = ((X - muX)*(Y - muY)).mean()
    ssim = ((2*muX*muY + C1)*(2*sigXY + C2)) / ((muX**2 + muY**2 + C1)*(sigX2 + sigY2 + C2) + 1e-12)
    return float(ssim)

@torch.no_grad()
def eval_metrics(model: nn.Module, loader: DataLoader, device: str, *,
                 max_batches: int = 2, channels_last=False,
                 mode: str = "linear_pct99", hi_pct: float = 99.5) -> tuple[float,float]:
    """
    Compute mean PSNR/SSIM on a few val batches with configurable normalization:
      - linear_clip1: clamp to [0,1] (old behavior)
      - linear_pct99: divide by hi percentile across pred+gt, then clip to [0,1]
      - log:         take log1p, percentile-normalize, then [0,1]
      - tm:          use preview tone-mapper, then [0,1]
    Metrics are still computed on [0,1] inputs with data_range=1.0 (stable & comparable).
    """
    def _to_np(x):  # x: [B,C,H,W] or [C,H,W]
        if x.ndim == 4: x = x[0]
        return x.detach().float().cpu().clamp_min(0).permute(1,2,0).numpy()  # HWC, ≥0

    def _tm(img):  # same style as previews (log1p normalized by max)
        m = np.max(img)
        if not np.isfinite(m) or m <= 0: return np.zeros_like(img, dtype=np.float32)
        y = np.log1p(img) / np.log1p(m)
        return np.clip(y, 0, 1).astype(np.float32)

    model.eval().to(device)
    psnrs, ssims, n = [], [], 0

    for i, (ldr, hdr) in enumerate(loader):
        if i >= max_batches: break
        ldr = ldr.to(device)
        if channels_last: ldr = ldr.to(memory_format=torch.channels_last)
        pred = model(ldr).clamp_min(0.0)  # normalized HDR (÷ scale_hdr)

        p = _to_np(pred)   # ≥0, HWC
        t = _to_np(hdr)    # ≥0, HWC

        if mode == "linear_clip1":
            p, t = np.clip(p, 0, 1), np.clip(t, 0, 1)

        elif mode == "linear_pct99":
            hi = np.percentile(np.concatenate([p.reshape(-1,3), t.reshape(-1,3)], axis=0), hi_pct)
            hi = max(float(hi), 1e-6)
            p, t = np.clip(p/hi, 0, 1), np.clip(t/hi, 0, 1)

        elif mode == "log":
            p, t = np.log1p(p), np.log1p(t)
            hi = np.percentile(np.concatenate([p.reshape(-1,3), t.reshape(-1,3)], axis=0), hi_pct)
            hi = max(float(hi), 1e-6)
            p, t = np.clip(p/hi, 0, 1), np.clip(t/hi, 0, 1)

        elif mode == "tm":
            p, t = _tm(p), _tm(t)

        else:
            # fallback to old behavior
            p, t = np.clip(p, 0, 1), np.clip(t, 0, 1)

        # Now compute metrics on [0,1]
        psnrs.append(compute_psnr(p, t, data_range=1.0))
        ssims.append(compute_ssim(p, t, data_range=1.0))
        n += 1

    if n == 0: return float("nan"), float("nan")
    return float(np.mean(psnrs)), float(np.mean(ssims))

# --------------------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------------------
def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_checkpoint(model, optimizer, scaler, epoch, train_loss, val_loss,
                    checkpoint_dir="./checkpoints", tag=None, meta:dict=None):
    checkpoint_dir = ensure_dir(checkpoint_dir)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = tag or f"e{epoch:03d}"
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "timestamp": ts,
        "meta": meta or {}
    }
    ckpt_path = Path(checkpoint_dir) / f"ckpt_{tag}_{ts}.pt"
    torch.save(payload, ckpt_path)
    latest_path = Path(checkpoint_dir) / "latest.pt"
    torch.save(payload, latest_path)
    return str(ckpt_path)

def load_checkpoint(model, optimizer=None, scaler=None, checkpoint_path: str | Path = "./checkpoints/latest.pt", map_location=None):
    d = torch.load(str(checkpoint_path), map_location=map_location)
    model.load_state_dict(d["model_state"])
    if optimizer is not None and "optimizer_state" in d and d["optimizer_state"] is not None:
        optimizer.load_state_dict(d["optimizer_state"])
    if scaler is not None and d.get("scaler_state") is not None:
        scaler.load_state_dict(d["scaler_state"])
    return d

def list_checkpoints(checkpoint_dir="./checkpoints"):
    p = Path(checkpoint_dir)
    return sorted(map(str, p.glob("ckpt_*.pt")))

def get_latest_checkpoint(checkpoint_dir="./checkpoints"):
    p = Path(checkpoint_dir) / "latest.pt"
    return str(p) if p.exists() else None

def get_best_checkpoint(checkpoint_dir="./checkpoints"):
    p = Path(checkpoint_dir)
    best_path, best_val = None, float("inf")
    for f in p.glob("ckpt_*.pt"):
        try:
            d = torch.load(f, map_location="cpu")
            v = float(d.get("val_loss", float("inf")))
            if v < best_val:
                best_val, best_path = v, f
        except Exception:
            pass
    return str(best_path) if best_path else None

# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------
class Logger:
    def __init__(self, logdir: str | Path):
        self.logdir = Path(logdir)
        ensure_dir(self.logdir)
        self.csv_path = self.logdir / "train_log.csv"
        self.jsonl_path = self.logdir / "train_log.jsonl"
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                w = csv.writer(f); w.writerow(["epoch","phase","loss","time_sec","lr","timestamp"])
        self._csv = open(self.csv_path, "a", newline=""); self._csv_writer = csv.writer(self._csv)
        self._jsonl = open(self.jsonl_path, "a")

    def log(self, epoch: int, phase: str, loss: float, tsec: float, lr: float):
        ts = datetime.now().isoformat()
        self._csv_writer.writerow([epoch, phase, f"{loss:.6f}", f"{tsec:.3f}", f"{lr:.6e}", ts])
        self._csv.flush()
        self._jsonl.write(json.dumps({
            "epoch": epoch, "phase": phase, "loss": float(loss),
            "time_sec": float(tsec), "lr": float(lr), "timestamp": ts
        }) + "\n")
        self._jsonl.flush()

    def close(self):
        try:
            self._csv.close()
            self._jsonl.close()
        except Exception:
            pass

class MetricsLogger:
    def __init__(self, logdir: str | Path):
        self.path = Path(logdir) / "metrics.csv"
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(["epoch","psnr","ssim","timestamp"])
        self._f = open(self.path, "a", newline="")
        self._w = csv.writer(self._f)
    def log(self, epoch: int, psnr: float, ssim: float):
        self._w.writerow([epoch, f"{psnr:.4f}", f"{ssim:.4f}", datetime.now().isoformat()])
        self._f.flush()
    def close(self):
        try: self._f.close()
        except Exception: pass

def _timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

class RunContext:
    def __init__(self, args):
        ts = _timestamp() if (args.exp is None) else args.exp
        if args.tag:
            ts = f"{ts}_{args.tag}"
        self.run_dir = Path(args.run_root) / ts
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.log_dir  = self.run_dir / "logs"
        self.viz_dir  = Path(args.vizdir) if args.vizdir else (self.run_dir / "viz")
        self.samples_dir = self.run_dir / "samples"
        for d in [self.run_dir, self.ckpt_dir, self.log_dir, self.viz_dir, self.samples_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # optional: snapshot key source files for reproducibility
        for fname in ["train_8bit2hdr.py", "exr_to_ldr.py"]:
            p = Path(fname)
            if p.exists():
                shutil.copy2(p, self.run_dir / f"snapshot_{p.name}")

        # tensorboard
        self.tb = None
        if args.tb and SummaryWriter is not None:
            self.tb = SummaryWriter(log_dir=str(self.run_dir / "tb"))
        elif args.tb and SummaryWriter is None:
            print("[TB] torch.utils.tensorboard not available; install tensorflow/tensorboard to enable.")

    def write_config(self, args, extra=None):
        cfg = {
            "timestamp": _timestamp(),
            "args": vars(args),
            "device_info": device_info(),
        }
        if extra: cfg.update(extra)
        (self.run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    def close(self):
        if self.tb: self.tb.close()

def plot_losses(csv_path: Path, out_png: Path):
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if "phase" not in df.columns:
        return
    fig = plt.figure(figsize=(7,4))
    for phase in ["train","val"]:
        sub = df[df["phase"]==phase]
        if len(sub):
            plt.plot(sub["epoch"], sub["loss"], label=phase)
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.grid(True, alpha=.3)
    plt.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    
def plot_metrics(metrics_csv: Path, out_png: Path):
    try:
        import pandas as pd, matplotlib.pyplot as plt
    except Exception:
        return
    if not metrics_csv.exists(): return
    df = pd.read_csv(metrics_csv)
    if not len(df): return
    fig = plt.figure(figsize=(7,4))
    if "psnr" in df.columns: plt.plot(df["epoch"], df["psnr"], label="PSNR")
    if "ssim" in df.columns: plt.plot(df["epoch"], df["ssim"], label="SSIM")
    plt.xlabel("epoch"); plt.grid(True, alpha=.3); plt.legend(); plt.tight_layout()
    fig.savefig(out_png, dpi=120); plt.close(fig)

# --------------------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------------------
class GracefulKiller:
    stop_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)   # Ctrl-C
        signal.signal(signal.SIGTERM, self.exit_gracefully)  # kill
    def exit_gracefully(self, *args):
        print("\n[Signal] Received stop signal. Finishing current step and exiting...", flush=True)
        self.stop_now = True

def should_stop_via_file(stop_file: str = "STOP_TRAINING") -> bool:
    return Path(stop_file).exists()

def acquire_lock(lock_path: str):
    p = Path(lock_path)
    if p.exists():
        raise RuntimeError(f"Lock file exists ({lock_path}). Another run may be active.")
    p.write_text(datetime.now().isoformat())
    return p

def release_lock(p: Path):
    try:
        p.unlink(missing_ok=True)
    except Exception:
        pass

def train(model: nn.Module,
          train_loader: DataLoader,
          val_loader: DataLoader,
          *,
          rc: RunContext | None = None,
          epochs: int = 10,
          lr: float = 1e-4,
          device: str = "cuda",
          logdir: str = "./logs",
          checkpoint_dir: str = "./checkpoints",
          resume: str | None = None,  # "latest" | "best" | "/path/to.ckpt"
          start_epoch: int = 0,
          grad_clip: float = 1.0,
          use_amp: bool = True,
          grad_accum_steps: int = 1,
          patience: int | None = None,
          channels_last: bool = False,
          stop_file: str = "STOP_TRAINING",
          lock_file: str = "TRAINING.LOCK",
          alpha_loss: float = 0.5,
          args=None):

    torch.backends.cudnn.benchmark = not getattr(args, "no_cudnn_benchmark", False)
    if getattr(args, "deterministic", False):
        torch.use_deterministic_algorithms(True, warn_only=True)
        
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -------- AMP / device / pinning decided ONCE (no re-check every epoch) --------
    is_cuda = (
        (isinstance(device, str) and device.startswith("cuda")) or
        (isinstance(device, torch.device) and device.type == "cuda")
    ) and torch.cuda.is_available()
    amp_enabled = (use_amp and is_cuda)
    pin_mem = is_cuda
    scaler = GradScaler("cuda", enabled=amp_enabled)

    best_val = float("inf")
    best_epoch = -1

    # Resume
    if resume:
        if resume in ("latest", "best"):
            ckpt = get_latest_checkpoint(checkpoint_dir) if resume == "latest" else get_best_checkpoint(checkpoint_dir)
        else:
            ckpt = resume
        if ckpt and Path(ckpt).exists():
            data = load_checkpoint(model, optimizer, scaler, ckpt, map_location=device)
            start_epoch = max(start_epoch, int(data.get("epoch", -1)) + 1)
            best_val = float(data.get("val_loss", best_val))
            print(f"[Resume] {ckpt} -> start_epoch={start_epoch}, best_val={best_val:.6f}")
        else:
            print(f"[Resume] No checkpoint found for '{resume}', starting fresh.")

    model.to(device)
    killer = GracefulKiller()
    logger = Logger(rc.log_dir if rc else logdir)
    metrics_logger = MetricsLogger(rc.log_dir if rc else logdir)
    lockp = acquire_lock(lock_file)

    def _epoch_pass(dl, train_phase: bool, epoch_idx: int):
        model.train(train_phase)
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0

        prog = getattr(args, "progress", "bar")
        show_bar = (prog == "bar")
        show_dots = (prog == "dots")

        iterator = tqdm(dl, leave=False, ncols=100, desc=("Train" if train_phase else "Val")) if show_bar else dl
        torch.set_grad_enabled(train_phase)

        for bi, (ldr, hdr) in enumerate(iterator):
            if killer.stop_now or should_stop_via_file(stop_file):
                break

            ldr = ldr.to(device, non_blocking=pin_mem)
            hdr = hdr.to(device, non_blocking=pin_mem)
            if channels_last:
                ldr = ldr.to(memory_format=torch.channels_last)
                hdr = hdr.to(memory_format=torch.channels_last)

            with autocast(device_type=("cuda" if is_cuda else "cpu"), enabled=amp_enabled):
                pred = model(ldr)                 # [B,C,H,W]
                loss = hdr_loss(pred, hdr, alpha=alpha_loss)

            if train_phase:
                optimizer.zero_grad(set_to_none=True)
                if amp_enabled:
                    scaler.scale(loss / grad_accum_steps).backward()
                else:
                    (loss / grad_accum_steps).backward()

                if (bi + 1) % grad_accum_steps == 0:
                    if grad_clip and grad_clip > 0:
                        if amp_enabled:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    if amp_enabled:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

            if not torch.isfinite(loss):
                raise RuntimeError("NaN/Inf in loss; check normalization and loss terms.")

            total_loss += float(loss.detach())
            n_batches += 1

            if show_bar:
                avg_so_far = total_loss / max(1, n_batches)
                post = {"loss": f"{avg_so_far:.4f}", "b": bi+1, "lr": f"{optimizer.param_groups[0]['lr']:.2e}"}
                if is_cuda:
                    post["memGB"] = f"{torch.cuda.memory_allocated()/1e9:.2f}"
                iterator.set_postfix(post)
            elif show_dots and (bi % 50 == 0):
                print(f"  [{bi+1}/{len(dl)}] loss:{total_loss/max(1,n_batches):.4f}", flush=True)

        dt = time.time() - t0
        avg = total_loss / max(1, n_batches)
        return avg, dt

    try:
        for epoch in range(start_epoch, epochs):
            print(f"\nEpoch {epoch+1}/{epochs}  (best={best_val:.6f} @ {best_epoch})")
            t_epoch_start = time.time()

            # Train
            train_loss, train_dt = _epoch_pass(train_loader, True, epoch)
            logger.log(epoch, "train", train_loss, train_dt, optimizer.param_groups[0]["lr"])

            # Val# Val
            val_loss, val_dt = _epoch_pass(val_loader, False, epoch)
            logger.log(epoch, "val", val_loss, val_dt, optimizer.param_groups[0]["lr"])

            # Quality metrics (PSNR/SSIM) on a few val batches
            try:
                psnr_val, ssim_val = eval_metrics(
                    model, val_loader, device,
                    max_batches=getattr(args, "metrics_batches", 2),
                    channels_last=channels_last,
                    mode=getattr(args, "metrics_mode", "linear_pct99"),
                    hi_pct=getattr(args, "metrics_hi_pct", 99.5),
                )
            except RuntimeError as e:
                if "CUDNN_STATUS_INTERNAL_ERROR" in str(e):
                    print("[Warn] cuDNN internal error during metrics. Retrying with benchmark=False + FP32…")
                    torch.backends.cudnn.benchmark = False
                    psnr_val, ssim_val = eval_metrics(model, val_loader, device,
                                                    max_batches=getattr(args,"metrics_batches",2),
                                                    channels_last=channels_last,
                                                    amp_enabled=False)
                else:
                    raise
            metrics_logger.log(epoch, psnr_val, ssim_val)

            # Optional TB
            if rc and rc.tb:
                rc.tb.add_scalar("metrics/psnr_val", psnr_val, epoch)
                rc.tb.add_scalar("metrics/ssim_val", ssim_val, epoch)

            # Save checkpoints
            ck_saved = save_checkpoint(model, optimizer, scaler, epoch, train_loss, val_loss,
                                    checkpoint_dir=rc.ckpt_dir if rc else checkpoint_dir,
                                    tag=f"e{epoch+1:03d}",
                                    meta={"tile": args.tile, "overlap": args.overlap,
                                            "scale_hdr": args.scale_hdr, "alpha_loss": alpha_loss,
                                            "notes": args.notes})
            print(f"[CKPT] Saved: {ck_saved}")
            
            if val_loss < best_val:
                best_val = val_loss; best_epoch = epoch
                save_checkpoint(model, optimizer, scaler, epoch, train_loss, val_loss,
                                checkpoint_dir=rc.ckpt_dir if rc else checkpoint_dir,
                                tag="best",
                                meta={"tile": args.tile, "overlap": args.overlap,
                                    "scale_hdr": args.scale_hdr, "alpha_loss": alpha_loss,
                                    "notes": args.notes})
                
            # TensorBoard scalars
            if rc and rc.tb:
                rc.tb.add_scalar("loss/train", train_loss, epoch)
                rc.tb.add_scalar("loss/val",   val_loss,   epoch)
                rc.tb.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

            # periodic visuals (just once, via rc.viz_dir)
            if rc and ((epoch+1) % getattr(args, "viz_interval", 5) == 0):
                save_visuals(model, val_loader, device, rc.viz_dir, epoch+1, scale_hdr=args.scale_hdr)

            # update loss plot
            if rc and ((epoch+1) % getattr(args,"plot_interval",1) == 0):
                plot_losses(Path(logger.csv_path), rc.run_dir / "loss_curves.png")
                plot_metrics(Path(rc.log_dir) / "metrics.csv", rc.run_dir / "metrics_curves.png")


            # Epoch timing
            t_epoch = time.time() - t_epoch_start
            remaining = (epochs - (epoch + 1)) * t_epoch
            print(f"Train {train_loss:.6f} ({train_dt:.1f}s) | Val {val_loss:.6f} ({val_dt:.1f}s) | "
                f"PSNR {psnr_val:.2f} | SSIM {ssim_val:.4f} | Epoch {t_epoch:.1f}s | ETA {remaining/60:.1f} min")


            # Patience
            if patience is not None:
                if (epoch - best_epoch) >= patience:
                    print(f"[EarlyStop] No improvement for {patience} epochs. Stopping.")
                    break

            # Housekeeping
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # External stop
            if killer.stop_now or should_stop_via_file(stop_file):
                print("[Stop] Requested. Exiting after epoch boundary.")
                break
            
        print(f"[Best] val={best_val:.6f} at epoch {best_epoch+1}")
        print(f"[Dir] Run: {rc.run_dir if rc else Path(logdir).parent}")
    
    except KeyboardInterrupt:
        print("\n[KeyboardInterrupt] Graceful shutdown.")
    finally:
        logger.close()
        release_lock(lockp)
        if rc: rc.close()
        metrics_logger.close()
        torch.set_grad_enabled(True)

# --------------------------------------------------------------------------------------
# Inference / EXR writing (optional)
# --------------------------------------------------------------------------------------
def _write_exr_cv2(path: Path, rgb: np.ndarray):
    # expects float32 RGB in [0, +inf)
    bgr = rgb[..., ::-1].astype(np.float32)
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for {path}")

def save_as_exr_batch(pred_bchw: np.ndarray, out_dir: str, prefix="pred"):
    outp = ensure_dir(out_dir)
    b = pred_bchw.shape[0]
    for j in range(b):
        out_path = outp / f"{prefix}_{j:05d}.exr"
        try:
            if imageio is not None:
                imageio.imwrite(str(out_path), pred_bchw[j].astype(np.float32), format="EXR")
            else:
                _write_exr_cv2(out_path, pred_bchw[j])
        except Exception:
            _write_exr_cv2(out_path, pred_bchw[j])

@torch.no_grad()
def run_inference(model: nn.Module, loader: DataLoader, device="cuda",
                  out_exr_dir: str | None = None, scale_hdr: float = 4.0, channels_last=False):
    model.eval().to(device)
    for i, (ldr, _) in enumerate(loader):
        ldr = ldr.to(device)
        if channels_last:
            ldr = ldr.to(memory_format=torch.channels_last)
        pred = model(ldr).float().cpu().numpy()  # [B,C,H,W]
        pred = np.clip(pred * scale_hdr, 0, None)  # undo HDR scaling
        pred = pred.transpose(0, 2, 3, 1)         # [B,H,W,C]
        if out_exr_dir:
            save_as_exr_batch(pred, out_exr_dir, prefix=f"pred_{i:04d}")

def hann2d(h, w):
    # separable 2D Hann weights in [0,1], reduce seams
    wy = torch.hann_window(h, periodic=False).unsqueeze(1)
    wx = torch.hann_window(w, periodic=False).unsqueeze(0)
    w2d = (wy @ wx).float()
    return w2d.clamp_min(1e-6)  # avoid divide-by-0 on borders

@torch.no_grad()
def enrich_ldr_sliding(model, ldr_chw: torch.Tensor, device, patch=1024, overlap=256, scale_hdr=4.0, channels_last=False):
    model.eval().to(device)
    C, H, W = ldr_chw.shape
    # pad to multiple of 4
    padH = (-H) % 4; padW = (-W) % 4
    if padH or padW:
        ldr_chw = F.pad(ldr_chw, (0, padW, 0, padH), mode="reflect")
    _, Hp, Wp = ldr_chw.shape

    stride = patch - overlap
    weight = hann2d(patch, patch).to(device)  # [P,P]
    out = torch.zeros(3, Hp, Wp, device=device)
    acc = torch.zeros(1, Hp, Wp, device=device)

    for y in range(0, Hp - patch + 1, stride):
        for x in range(0, Wp - patch + 1, stride):
            tile = ldr_chw[:, y:y+patch, x:x+patch].unsqueeze(0).to(device)
            if channels_last:
                tile = tile.to(memory_format=torch.channels_last)
            with autocast("cuda", enabled=torch.cuda.is_available()):
                pred = model(tile)[0]  # [C,P,P], scaled (÷scale_hdr)
            w2d = weight
            out[:, y:y+patch, x:x+patch] += pred * w2d
            acc[:, y:y+patch, x:x+patch] += w2d

    pred_full = (out / acc).clamp_min(0.0) * scale_hdr  # rescale to scene range
    # remove pad
    pred_full = pred_full[:, :H, :W] if (padH or padW) else pred_full
    return pred_full  # [3,H,W] float32 linear HDR

@torch.no_grad()
def save_visuals(model, val_loader, device, out_dir, epoch, scale_hdr=4.0, exposure=1.0, gamma=2.2, max_batches=1):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)

    for i, (ldr, hdr) in enumerate(val_loader):
        if i >= max_batches:
            break
        ldr = ldr.to(device)
        hdr = hdr.to(device)

        pred = model(ldr).clamp_min(0.0) * scale_hdr
        pred_np = pred[0].detach().cpu().numpy().transpose(1,2,0)
        hdr_np  = (hdr[0].detach().cpu().numpy().transpose(1,2,0)) * scale_hdr
        ldr_np  = (ldr[0].detach().cpu().numpy().transpose(1,2,0))

        def tone_map(x):
            x = np.clip(x,0,None) * exposure
            x = np.log1p(x) / np.log1p(np.max(x)+1e-8)
            x = np.clip(x,0,1)
            if gamma!=1.0:
                x = np.clip(x,1e-8,1)**(1/gamma)
            return x

        ldr_tm  = np.clip(ldr_np,0,1)
        pred_tm = tone_map(pred_np)
        hdr_tm  = tone_map(hdr_np)

        fig, axs = plt.subplots(1,3, figsize=(12,4))
        axs[0].imshow(ldr_tm);  axs[0].set_title("Input LDR")
        axs[1].imshow(pred_tm); axs[1].set_title("Pred HDR→LDR")
        axs[2].imshow(hdr_tm);  axs[2].set_title("GT HDR→LDR")
        for ax in axs: ax.axis("off")

        out_path = os.path.join(out_dir, f"epoch{epoch:04d}_sample{i}.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        print(f"[Viz] Saved {out_path}")

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_argparser():
    ap = argparse.ArgumentParser(description="Train 8-bit→HDR model")
    ap.add_argument("--ldr", required=True, help="LDR folder (8-bit)")
    ap.add_argument("--hdr", required=True, help="HDR folder (EXR)")
    ap.add_argument("--tile", type=int, default=1024, help="training crop size (patch resolution, default=1024)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--workers", type=int, default=0, help="DataLoader workers")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--val-batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--scale-hdr", type=float, default=4.0)
    ap.add_argument("--accum", type=int, default=1, help="grad accumulation steps")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true", help="enable mixed precision")
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--logdir", default="./logs")
    ap.add_argument("--ckptdir", default="./checkpoints")
    ap.add_argument("--resume", default=None, help="'latest'|'best'|/path/to.ckpt")
    ap.add_argument("--start-epoch", type=int, default=0)
    ap.add_argument("--patience", type=int, default=None, help="early stop patience (epochs)")
    ap.add_argument("--stop-file", default="STOP_TRAINING")
    ap.add_argument("--lock-file", default="TRAINING.LOCK")
    ap.add_argument("--alpha", type=float, default=0.5, help="alpha for loss mix")
    ap.add_argument("--allow-non-exr", action="store_true", help="accept non-EXR HDR targets")
    ap.add_argument("--min-hw", type=int, default=1024, help="skip pairs smaller than this height or width")
    
    ap.add_argument("--run-root", default="./runs", help="root folder for experiments")
    ap.add_argument("--exp", default=None, help="experiment name; if omitted, auto from timestamp")
    ap.add_argument("--tag", default="", help="short tag to append to exp name (e.g., b2_tile512)")
    ap.add_argument("--notes", default="", help="free-form notes stored with the run")

    # training / inference tiling metadata
    ap.add_argument("--overlap", type=int, default=256, help="inference sliding overlap (for enrich)")

    # visualization & plotting
    ap.add_argument("--vizdir", default=None, help="folder for training previews (defaults under run dir)")
    ap.add_argument("--viz-interval", type=int, default=5, help="save previews every N epochs")
    ap.add_argument("--plot-interval", type=int, default=1, help="update loss plots every N epochs")
    ap.add_argument("--tb", action="store_true", help="enable TensorBoard summaries if available")

    # Optional utilities
    ap.add_argument("--prep-exr-to-ldr", action="store_true", help="run batch_convert(EXR→PNG) and exit")
    ap.add_argument("--prep-src", default=None, help="source HDR folder for EXR→PNG")
    ap.add_argument("--prep-dst", default=None, help="dest LDR folder for PNG output")
    ap.add_argument("--infer-only", action="store_true", help="skip training, run inference on val set")
    ap.add_argument("--infer-out", default=None, help="folder to write EXRs during inference")
    ap.add_argument("--progress", choices=["bar","dots","none"], default="bar", help="in-epoch progress: tqdm bar, dot logging, or silent")
    ap.add_argument("--metrics-batches", type=int, default=2, help="val batches used to compute PSNR/SSIM per epoch")
    ap.add_argument("--no-cudnn-benchmark", action="store_true",
                    help="disable cuDNN benchmark (avoids some alg bugs)")
    ap.add_argument("--deterministic", action="store_true",
                    help="use deterministic algorithms (slightly slower)")
    ap.add_argument("--metrics-mode", choices=["linear_clip1","linear_pct99","log","tm"],
                    default="linear_pct99", help="how to normalize data for PSNR/SSIM")
    ap.add_argument("--metrics-hi-pct", type=float, default=99.5,
                    help="high percentile for metrics scaling (linear_pct99/log/tm)")
    return ap

def main():
    args = build_argparser().parse_args()
    print_specs()
    set_seed(args.seed)

    # Optional: EXR→PNG prep then exit
    if args.prep_exr_to_ldr:
        if args.prep_src is None or args.prep_dst is None:
            print("--prep-src and --prep-dst are required with --prep-exr-to-ldr", file=sys.stderr)
            sys.exit(2)
        if batch_convert is None:
            print("batch_convert not available. Ensure exr_to_ldr.py is importable.", file=sys.stderr)
            sys.exit(2)
        ensure_dir(args.prep_dst)
        print(f"[Prep] Converting EXR→PNG  src={args.prep_src}  dst={args.prep_dst}")
        batch_convert(args.prep_src, args.prep_dst, fmt="png", exposure=1.0)
        print("[Prep] Done.")
        sys.exit(0)
        
    rc = RunContext(args)
    # route logdir/ckptdir/vizdir through the run
    args.logdir = str(rc.log_dir)
    args.ckptdir = str(rc.ckpt_dir)
    args.vizdir = str(rc.viz_dir)
    rc.write_config(args, extra={"notes": args.notes})

    # Dataset sizes first
    base = HDRDataset(args.ldr, args.hdr, scale_hdr=args.scale_hdr,
                  allow_non_exr=args.allow_non_exr, min_hw=args.min_hw)
    n = len(base)
    print(f"[Dataset] Eligible pairs (≥{args.min_hw}): {n}")

    if n < 2:
        raise RuntimeError(f"Dataset too small: {n} pairs")
    train_n = max(1, int(0.8 * n))
    val_n = max(1, n - train_n)
    #train_ds, val_ds = random_split(dataset, [train_n, val_n])

    # Wrap only the training subset with transform:
    idx_train, idx_val = torch.utils.data.random_split(range(n), [train_n, val_n])

    # Two separate dataset instances so transform affects train only
    train_transform = PairedRandomTiler(patch=args.tile, scale_jitter=(0.9, 1.1))
    dataset_train = HDRDataset(args.ldr, args.hdr, scale_hdr=args.scale_hdr,
                           allow_non_exr=args.allow_non_exr, transform=train_transform, min_hw=args.min_hw)
    val_pad = ValPadToMult4()
    dataset_val   = HDRDataset(args.ldr, args.hdr, scale_hdr=args.scale_hdr,
                           allow_non_exr=args.allow_non_exr, transform=val_pad, min_hw=args.min_hw)
    # Subset them by indices
    train_ds = torch.utils.data.Subset(dataset_train, idx_train.indices if hasattr(idx_train,'indices') else list(idx_train))
    val_ds   = torch.utils.data.Subset(dataset_val,   idx_val.indices   if hasattr(idx_val,'indices')   else list(idx_val))

    # Loaders (num_workers=0 is safest cross-platform; pin_memory improves H2D)
    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, pin_memory=pin_mem, persistent_workers=(args.workers>0))
    val_loader   = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=pin_mem, persistent_workers=(args.workers>0))

    device = args.device
    model = HDRNet()

    if args.infer_only:
        print("[Mode] Inference only")
        ckpt = args.resume if args.resume else get_best_checkpoint(args.ckptdir) or get_latest_checkpoint(args.ckptdir)
        if not ckpt:
            raise RuntimeError("No checkpoint found for inference.")
        load_checkpoint(model, optimizer=None, scaler=None, checkpoint_path=ckpt, map_location=device)
        run_inference(model, val_loader, device=device, out_exr_dir=args.infer_out, scale_hdr=args.scale_hdr,
                      channels_last=args.channels_last)
        print("[Infer] Done.")
        return

    # Train
    train(model, train_loader, val_loader, rc=rc,
          epochs=args.epochs, lr=args.lr, device=device,
          logdir=args.logdir, checkpoint_dir=args.ckptdir,
          resume=args.resume, start_epoch=args.start_epoch,
          grad_clip=args.grad_clip, use_amp=args.amp, grad_accum_steps=args.accum,
          patience=args.patience, channels_last=args.channels_last,
          stop_file=args.stop_file, lock_file=args.lock_file,
          alpha_loss=args.alpha, args=args)

if __name__ == "__main__":
    main()
# EOF