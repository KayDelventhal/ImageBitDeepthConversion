# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: train_8bit2hdr.py

"""
250914init setup
"""

# %%
print(" 1. Imports ")
# ============================================================
# 1. Imports
# ============================================================
import os
# Make sure OpenCV can read EXR before importing cv2
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import imageio.v2 as imageio
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================
# 2. Utility functions
# ============================================================
from exr_to_ldr import srgb_to_linear, load_exr, tone_map

# %%
print(" 3. Dataset class ")
# ============================================================
# 3. Dataset class
# ============================================================
class HDRDataset(Dataset):
    """
    Paired dataset:
    - LDR: 8-bit (JPG/PNG) → linear float
    - HDR: 32-bit EXR
    """
    def __init__(self, ldr_folder, hdr_folder, transform=None, scale_hdr=4.0):
        self.ldr_files = sorted(glob(os.path.join(ldr_folder, "*")))
        self.hdr_files = sorted(glob(os.path.join(hdr_folder, "*")))
        assert len(self.ldr_files) == len(self.hdr_files), "Mismatch in dataset size!"
        self.transform = transform
        self.scale_hdr = scale_hdr

    def __len__(self):
        return len(self.ldr_files)

    def __getitem__(self, idx):
        # LDR → linear
        ldr = cv2.imread(self.ldr_files[idx], cv2.IMREAD_COLOR)
        ldr = cv2.cvtColor(ldr, cv2.COLOR_BGR2RGB) / 255.0
        ldr = srgb_to_linear(ldr)

        # HDR
        hdr = load_exr(self.hdr_files[idx])
        hdr = np.nan_to_num(hdr, nan=0.0, posinf=10.0, neginf=0.0)

        # Normalize HDR by fixed scale
        hdr = hdr / self.scale_hdr

        # To tensors [C,H,W]
        ldr = torch.from_numpy(ldr.transpose(2, 0, 1)).float()
        hdr = torch.from_numpy(hdr.transpose(2, 0, 1)).float()

        if self.transform:
            ldr, hdr = self.transform(ldr, hdr)

        return ldr, hdr

print(" 4. HDRNet model ")
# ============================================================
# 4. HDRNet model
# ============================================================
class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.bn1   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)

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

    def forward(self, x):
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(e1))
        e3 = F.relu(self.enc3(e2))
        b  = self.bottleneck(e3)
        d3 = F.relu(self.dec3(b))
        d2 = F.relu(self.dec2(d3))
        out = self.dec1(d2)
        return out  # linear HDR

# %%
print(" 5. Loss ")
# ============================================================
# 5. Loss
# ============================================================
def hdr_loss(pred, target, alpha=0.5):
    l1 = F.l1_loss(pred, target)
    log_l1 = F.l1_loss(torch.log1p(pred), torch.log1p(target))
    return alpha * l1 + (1 - alpha) * log_l1

print(" 6. Training ")
# ============================================================
# 6. Training
# ============================================================

def train_model(model, train_loader, val_loader, epochs=10, lr=1e-4, device="cuda", *,
                 checkpoint_dir: str = "./checkpoints", resume_from: str | None = None,
                 save_every: int = 1, save_best: bool = True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    start_epoch = 0
    best_val = float("inf")

    # Resume logic
    if resume_from:
        ckpt_path = resume_from
        if resume_from in ("latest", "best"):
            ckpt_path = get_latest_checkpoint(checkpoint_dir) if resume_from == "latest" else get_best_checkpoint(checkpoint_dir)
        if ckpt_path and os.path.exists(ckpt_path):
            data = load_checkpoint(model, optimizer, ckpt_path, map_location=device)
            start_epoch = int(data.get("epoch", 0)) + 1
            best_val = float(data.get("val_loss", best_val))
            print(f"Resumed from {ckpt_path} @ epoch {start_epoch} (best_val={best_val:.4f})")
        else:
            print(f"No checkpoint found for resume_from={resume_from}; starting fresh.")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        for ldr, hdr in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            ldr, hdr = ldr.to(device), hdr.to(device)
            optimizer.zero_grad()
            pred = model(ldr)
            loss = hdr_loss(pred, hdr)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for ldr, hdr in val_loader:
                ldr, hdr = ldr.to(device), hdr.to(device)
                pred = model(ldr)
                val_loss += hdr_loss(pred, hdr).item()
        val_loss /= max(1, len(val_loader))

        print(f"Epoch {epoch+1}: Train {train_loss:.4f} | Val {val_loss:.4f}")

        # Save checkpoints
        should_save = ((epoch + 1) % save_every == 0)
        if should_save:
            path = save_checkpoint(model, optimizer, epoch, train_loss, val_loss, checkpoint_dir, tag=f"e{epoch+1:03d}")
            print("Saved checkpoint:", path)
        if save_best and val_loss < best_val:
            best_val = val_loss
            path = save_checkpoint(model, optimizer, epoch, train_loss, val_loss, checkpoint_dir, tag="best")
            print("Updated best checkpoint:", path)

    return model


# %%
print(" 7. Inference (testing) ")
# ============================================================
# 7. Inference (testing)
# ============================================================
def test_model(model, test_loader, device="cuda", save_folder="results"):
    os.makedirs(save_folder, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i, (ldr, _) in enumerate(test_loader):
            ldr = ldr.to(device)
            pred = model(ldr).cpu().numpy()
            pred = np.clip(pred * 4.0, 0, None)   # undo HDR scale
            pred = pred.transpose(0, 2, 3, 1)     # [B,H,W,C]
            for j in range(pred.shape[0]):
                out_path = os.path.join(save_folder, f"test_{i}_{j}.exr")
                imageio.imwrite(out_path, pred[j].astype(np.float32), format="EXR")
                print("Saved:", out_path)

# %%
print(" 8. Checkpoint utilities ")
# ============================================================
# 8. Checkpoint utilities
# ============================================================

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, checkpoint_dir="./checkpoints", tag=None):
    checkpoint_dir = ensure_dir(checkpoint_dir)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = tag or f"e{epoch:03d}"
    ckpt_path = checkpoint_dir / f"ckpt_{tag}_{ts}.pt"
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "timestamp": ts,
    }, ckpt_path)
    # also update symlinks/files for convenience
    latest_path = checkpoint_dir / "latest.pt"
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "timestamp": ts,
    }, latest_path)
    return str(ckpt_path)

def load_checkpoint(model, optimizer=None, checkpoint_path: str | Path = "./checkpoints/latest.pt", map_location=None):
    checkpoint_path = str(checkpoint_path)
    data = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(data["model_state"])
    if optimizer is not None and "optimizer_state" in data:
        optimizer.load_state_dict(data["optimizer_state"])
    return data

def list_checkpoints(checkpoint_dir="./checkpoints"):
    p = Path(checkpoint_dir)
    files = sorted(p.glob("ckpt_*.pt"))
    return [str(f) for f in files]

def get_latest_checkpoint(checkpoint_dir="./checkpoints"):
    p = Path(checkpoint_dir)
    latest = p / "latest.pt"
    return str(latest) if latest.exists() else None

def get_best_checkpoint(checkpoint_dir="./checkpoints"):
    # choose the ckpt with lowest recorded val_loss
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

# %%
print(" 9. Convert EXR → LDR (png) ")
# ============================================================
# 9. Convert EXR → LDR (png)
# ============================================================
# # Batch convert EXR → LDR (png)
from exr_to_ldr import batch_convert

train_ldr_folder = "./data/train/ldr"
train_hdr_folder = "./data/train/hdr"
os.makedirs(train_ldr_folder, exist_ok=True)
os.makedirs(train_hdr_folder, exist_ok=True)

test_ldr_folder = "./data/test/ldr"
test_hdr_folder = "./data/test/hdr"
os.makedirs(test_ldr_folder, exist_ok=True)
os.makedirs(test_hdr_folder, exist_ok=True)

batch_convert(train_hdr_folder, train_ldr_folder, fmt="png", exposure=1.0)
batch_convert(test_hdr_folder, test_ldr_folder, fmt="png", exposure=1.0)

# %%
print(" 9b. Copy files ")
# Copy from folder every n (e.g 100th) image into other folder
input_folder = r"E:\20241219_KAI_Exports\Mozart_TEST_EP02".replace("\\", "/")
input_folder = r"E:\20241219_KAI_Exports\Mozart_TEST_EP04".replace("\\", "/")
output_folder = r"C:\Users\vfx\OneDrive\_Python_\DEV\ImageBitDeepthConversion\data\train\hdr".replace("\\", "/")
import shutil
os.makedirs(output_folder, exist_ok=True)

n = 100  # copy every 100th image
for i, filename in enumerate(os.listdir(input_folder)):
    if i % n == 0:
        shutil.copy(os.path.join(input_folder, filename), output_folder)

# %%
print(" 9b. Move files n%")
# Move random 10% of files from input_folder to output_folder
import random
import shutil
input_folder = r"C:\Users\vfx\OneDrive\_Python_\DEV\ImageBitDeepthConversion\data\train\hdr".replace("\\", "/")
output_folder = r"C:\Users\vfx\OneDrive\_Python_\DEV\ImageBitDeepthConversion\data\test\hdr".replace("\\", "/")
os.makedirs(output_folder, exist_ok=True)
n = 10  # 10%
all_files = os.listdir(input_folder)
num_to_copy = max(1, len(all_files) // n)
files_to_copy = random.sample(all_files, num_to_copy)
for filename in files_to_copy:
    shutil.move(os.path.join(input_folder, filename), output_folder)

# %%
print(" 10. List available checkpoints ")
# ============================================================
# 10. List available checkpoints
# ============================================================

from pathlib import Path
from pprint import pprint

def show_checkpoints(checkpoint_dir="./checkpoints"):
    files = list_checkpoints(checkpoint_dir)
    print(f"Found {len(files)} checkpoints in {checkpoint_dir}")
    for f in files:
        try:
            d = torch.load(f, map_location="cpu")
            print(f"- {Path(f).name}: epoch={d.get('epoch')}, val={d.get('val_loss')} ts={d.get('timestamp')}")
        except Exception as e:
            print(f"- {Path(f).name}: (unreadable) {e}")

show_checkpoints()

# %%
print(" 9b. Run training")
# ============================================================
# 11. run training
# ============================================================
def run_training(ldr_folder, hdr_folder)
    dataset = HDRDataset(ldr_folder, hdr_folder)
    n = len(dataset)
    train_size = int(0.8 * n)
    val_size   = n - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # On Windows/Jupyter, use single-process loading to avoid worker crashes
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HDRNet()

    model = train_model(model, train_loader, val_loader, epochs=5, device=device)

    test_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
    test_model(model, test_loader, device=device)
    
# Example: resume from latest checkpoint and continue 2 epochs
model = HDRNet()
ldr_folder = "./data/ldr"
hdr_folder = "./data/hdr"

dataset = HDRDataset(ldr_folder, hdr_folder)
n = len(dataset)
train_size = int(0.8 * n)
val_size   = n - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"CUDA or CPU ? - Device: {device}")

model = train_model(model, train_loader, val_loader, epochs=2, device=device,
                    checkpoint_dir="./checkpoints", resume_from="latest", save_every=1)


