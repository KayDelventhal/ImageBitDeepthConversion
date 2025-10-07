#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: exr_to_ldr.py
"""
Convert .exr → LDR (jpg/png) for all files in a directory.
- Prefers imageio for EXR I/O (no OpenCV EXR toggle needed).
- Robust handling of Gray/RGBA EXRs.
- Simple, fast tone mapping (log or Reinhard).
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# EXR → LDR (jpg/png) for all files in a directory (OpenCV-only)

import os
# Ensure EXR reading is enabled in OpenCV before cv2 is imported
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import argparse
import numpy as np
import cv2

def _ensure_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr.astype(np.float32, copy=False)

def load_exr(path: str) -> np.ndarray:
    exr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if exr is None:
        raise RuntimeError(
            f"Failed to read EXR: {path}. "
            "Set OPENCV_IO_ENABLE_OPENEXR=1 in your environment and restart Python."
        )
    if exr.ndim == 3 and exr.shape[-1] >= 3:
        exr = cv2.cvtColor(exr, cv2.COLOR_BGR2RGB)
    return _ensure_rgb(exr)

def srgb_to_linear(img):
    """Convert sRGB [0..1] to linear RGB"""
    img = np.float32(img)
    threshold = 0.04045
    below = img <= threshold
    above = img > threshold
    img[below] = img[below] / 12.92
    img[above] = ((img[above] + 0.055) / 1.055) ** 2.4
    return img

def tone_map(hdr: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    hdr = np.clip(hdr, 0, None) * float(exposure)
    hdr = np.nan_to_num(hdr, nan=0.0, posinf=0.0, neginf=0.0)
    maxv = float(np.max(hdr))
    if maxv <= 0.0 or not np.isfinite(maxv):
        return np.zeros_like(hdr, dtype=np.float32)
    ldr = np.log1p(hdr) / np.log1p(maxv)   # simple log TM
    return np.clip(ldr.astype(np.float32, copy=False), 0.0, 1.0)

def exr_to_ldr(exr_path: str, out_path: str, exposure: float = 1.0) -> None:
    hdr = load_exr(exr_path)
    ldr = tone_map(hdr, exposure=exposure)
    ldr8 = (ldr * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(out_path, cv2.cvtColor(ldr8, cv2.COLOR_RGB2BGR))
    print("Saved:", out_path)

def batch_convert(folder_in: str, folder_out: str, fmt: str = "jpg", exposure: float = 1.0) -> None:
    os.makedirs(folder_out, exist_ok=True)
    files = sorted(f for f in os.listdir(folder_in) if f.lower().endswith(".exr"))
    if not files:
        print(f"No .exr files found in: {folder_in}")
        return
    print(f"files: {len(files)}")
    for f in files:
        in_path = os.path.join(folder_in, f)
        out_path = os.path.join(folder_out, os.path.splitext(f)[0] + f".{fmt}")
        exr_to_ldr(in_path, out_path, exposure=exposure)

def main():
    ap = argparse.ArgumentParser(description="Convert EXR → LDR (jpg/png) for all files in a folder.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--fmt", default="jpg", choices=["jpg", "png"])
    ap.add_argument("--exposure", type=float, default=1.0)
    args = ap.parse_args()
    batch_convert(args.input, args.output, fmt=args.fmt, exposure=args.exposure)

if __name__ == "__main__":
    main()
