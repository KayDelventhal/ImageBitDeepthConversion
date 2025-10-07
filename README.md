# 8-bit to HDR (EXR) Conversion with Deep Learning

## 🎯 Project Goal

Modern AI image generators and mobile cameras often produce **8-bit JPEG/PNG** with limited dynamic range.
For **VFX workflows**, we need **high-fidelity 32-bit EXR** that preserves:

1. **Dynamic Range** (bright highlights & deep shadows)
2. **Color Accuracy** (linearized RGB suitable for compositing)

This project trains a model that maps **LDR (8-bit)** → **HDR (32-bit float EXR)**.

---

## 🔑 Motivation

* AI images look fine but lack usable bit depth.
* In VFX, 8-bit assets **clip** and don’t comp well.
* This tool works as an **HDR extender**: deterministic, linear HDR EXR suitable for pipelines.

---

## 🧩 Approach

### 1) Model

**Residual encoder–decoder with dilated residual blocks (HDRNet-style)**

* Captures **global luminance context**
* **HDR-safe output** via Softplus (≥0, unbounded; no clamping)
* Scales from 8 GB → 48 GB GPUs

### 2) Data

* **Input (LDR):** 8-bit JPG/PNG (sRGB → linear before training)
* **Target (HDR):** 32-bit EXR (linear float RGB)
* **Rule:** pairs **smaller than 1024×1024 are skipped**
* **Training:** random tiles (configurable `--tile`, e.g. 256–1024) + optional scale-jitter
* **Validation:** full frames padded to multiples of 4

### 3) Loss Function (Hybrid HDR Loss)

$$
\mathcal{L}=\alpha\,\mathrm{L1}(y,\hat y)\;+\;(1-\alpha)\,\mathrm{L1}\!\big(\log(1+y),\log(1+\hat y)\big)
$$

* **Linear L1** → absolute radiance accuracy for comp
* **Log-L1** → balances shadows/highlights (tone expansion)
* Default $\alpha=0.5$. Numerically safe (Softplus outputs, small epsilon inside logs).

### 4) Training Features

* **AMP**, gradient clipping, accumulation
* **Early stop** (patience) and **STOP\_TRAINING** file
* **Lock file** to prevent concurrent runs
* **Checkpoints**: best & latest (resumable) **with metadata**
* **Logging**: CSV + JSONL, optional **TensorBoard**
* **Visuals**: periodic side-by-side **Input / Pred / GT** PNGs
* **Auto run folders** under `runs/` with **config snapshot** and **loss plots**
* **Quality metrics**: **PSNR/SSIM** computed every epoch on a few val batches → **CSV + TensorBoard + plot**

### 5) Output

* Model outputs **linear HDR floats**
* Saved to **EXR** (float16/float32)
* For quick viewing, log tone-mapping previews are generated

---

## 🚀 What’s New / Experiment Tracking

Every run is neatly organized:

```
runs/<timestamp>[_tag]/
├─ checkpoints/         # ckpt_eXXX_*.pt, best.pt (via tag "best"), latest.pt
├─ logs/
│  ├─ train_log.csv     # epoch, phase, loss, time, lr, timestamp
│  ├─ train_log.jsonl
│  └─ metrics.csv       # epoch, psnr, ssim, timestamp   ⟵ NEW
├─ viz/                 # epochXXXX_sample*.png previews (Input/Pred/GT)
├─ tb/                  # TensorBoard (if --tb)
├─ loss_curves.png      # auto-updated from train_log.csv
├─ metrics_curves.png   # auto-updated from logs/metrics.csv   ⟵ NEW
├─ config.json          # full CLI args + device info + notes
├─ snapshot_train_8bit2hdr.py
└─ snapshot_exr_to_ldr.py
```

* **Checkpoints include metadata** (`tile`, `overlap`, `scale_hdr`, `alpha_loss`, `notes`).
* **config.json** stores all CLI args & environment → reproducibility.

---

## 📂 Project Structure

```
project_root/
│
├─ data/
│  ├─ train/
│  │  ├─ ldr/       # 8-bit inputs
│  │  └─ hdr/       # linear EXR ground truth
│  └─ test/
│     ├─ ldr/
│     └─ hdr/
│
├─ tools/
│  └─ exr_to_ldr.py  # EXR → PNG/JPG (tone-mapped)
│
├─ train_8bit2hdr.py  # training/inference + logging/visuals/runs + metrics
├─ requirements.txt
└─ README.md
```

---

## 🖥️ Example Usage (single-line, Windows PowerShell / CMD friendly)

### Train (with AMP, workers, run tracking, **metrics**)

```bash
python .\train_8bit2hdr.py --ldr .\data\train\ldr --hdr .\data\train\hdr ^
  --epochs 20 --batch-size 2 --val-batch-size 2 --amp --workers 4 ^
  --tile 512 --run-root .\runs --tag b2_tile512 --tb --metrics-batches 2
```

> `--metrics-batches` controls how many **validation** batches are used for per-epoch PSNR/SSIM (default: 2).
> Metrics are written to `logs/metrics.csv`, plotted to `metrics_curves.png`, and (if `--tb`) to TensorBoard (`metrics/psnr_val`, `metrics/ssim_val`).

### Resume (best)

```bash
python .\train_8bit2hdr.py --ldr .\data\train\ldr --hdr .\data\train\hdr ^
  --resume best --epochs 40 --start-epoch 20 --amp --workers 4 --tile 512 --metrics-batches 2
```

### Early stop (no val improvement for 5 epochs)

```bash
python .\train_8bit2hdr.py --ldr .\data\train\ldr --hdr .\data\train\hdr --patience 5 --metrics-batches 2
```

### Stop gracefully (create this file during training)

```bash
type nul > STOP_TRAINING
```

### Inference (write EXRs from validation split)

```bash
python .\train_8bit2hdr.py --ldr .\data\test\ldr --hdr .\data\test\hdr ^
  --infer-only --resume best --infer-out .\data\output
```

### Prep PNGs from EXRs (helper)

```bash
python .\tools\exr_to_ldr.py --input .\data\train\hdr --output .\data\train\ldr --fmt png --exposure 1.0
```

> Tip (Windows): avoid `--` line breaks; keep commands on **one line** as shown.

---

## 📊 Why This Method?

| Method                    | Pros                           | Cons                                | Verdict             |
| ------------------------- | ------------------------------ | ----------------------------------- | ------------------- |
| **U-Net**                 | Detail preserved               | Identity bias, HDR clamping         | ❌ Not ideal         |
| **HDRNet-style Residual** | HDR-safe, good global exposure | Slightly less micro-detail          | ✅ Best balance      |
| **GAN (HDRGAN)**          | Realistic textures             | Hallucination risk (unsafe for VFX) | ❌ Not deterministic |
| **Transformer (SwinHDR)** | Excellent global context       | Heavy VRAM/data, slow               | ⚠️ R\&D only        |
| **Physics-inspired**      | Physically grounded            | Needs known ISP/camera pipeline     | ⚠️ Limited scope    |

**Choice:** HDRNet-style residual encoder–decoder → **reliable, deterministic, HDR-safe**.

---

## 📈 Visuals & Metrics During Training

* **Previews**: `viz/epochXXXX_sample*.png` (Input LDR, Pred HDR→LDR, GT HDR→LDR)
* **Loss curves**: `loss_curves.png` from `logs/train_log.csv`
* **Metrics**:

  * **Where**: `logs/metrics.csv` (per-epoch), `metrics_curves.png`, TensorBoard scalars (`metrics/psnr_val`, `metrics/ssim_val` if `--tb`)
  * **What**: **PSNR** (dB) and **SSIM** (0–1) on the **first N** validation batches (`--metrics-batches`)
  * **How (current default)**: metrics are computed in the **normalized linear space** (model outputs / `scale_hdr`), with values **clamped to \[0,1]** **for metric stability only**.

    * This **does not** affect training or saved EXRs (they remain unbounded HDR).
    * The clamp prevents a few extremely hot pixels from dominating the score.
  * **Reading the numbers** (rough guide; depends on data/tiling):

    * PSNR: low-20s (early), \~28–35dB (good), >35dB (very good)
    * SSIM: \~0.2–0.4 (early), \~0.6–0.8 (good), >0.85 (very good)

> Want **HDR-aware** metrics that respect values >1? Swap the metric mapping in `eval_metrics()` (e.g., scale by a high percentile like 99.5%, or compute in log-domain). This keeps bright highlights meaningful while avoiding instability. (Planned: CLI switches for modes.)

---

## 🧪 Repro Tips

* Use **`--tile`** to start small (e.g., `256`) then scale up to `512/1024`.
* Keep **`--overlap`** metadata for sliding-window inference (blend seams).
* Store notes via `--tag` and `--notes` so later inference matches training tiling/blending assumptions.

---

## ⚙️ Installation

```bash
git clone https://github.com/KayDelventhal/ImageBitDepthHDR.git
cd ImageBitDepthHDR
pip install -r requirements.txt
```

---

## 🛠️ Troubleshooting & Best Practices

**GPU memory**

* Lower `--batch-size`, reduce `--tile`, enable `--amp`, consider `--accum`.

**EXR precision**

* Float16 is standard in VFX; use float32 for extreme brightness (sun, fire).

**Dataset hygiene**

* Filenames must match across `ldr/` and `hdr/`.
* LDR must be **sRGB→linear**; HDR must be **linear EXR** (no baked gamma).
* Pairs **<1024 px** are skipped by default.

**Training stability**

* Lower LR (`--lr 5e-5`), enable grad clip (`--grad-clip 1.0`), check normalization.

**Viewing**

* Raw EXRs will look dark in standard viewers → use tone-mapped previews or your comp app.

**Workers**

* `--workers` controls DataLoader processes. On Windows, start with `0–2`.
  Increase to `4` if CPU/RAM allow and disk is fast (watch CPU/RAM).

---

## ✅ Summary

A production-oriented, **deterministic** 8-bit → **HDR EXR** converter with:

* Robust training (AMP, grad clip, early stop)
* **Experiment tracking** in `runs/` (config snapshot, loss/metrics plots, TensorBoard)
* **Checkpoints w/ metadata** (tile/overlap/scale/notes)
* Visual previews + clean inference path
* **Objective metrics (PSNR/SSIM)** alongside visuals to track progress numerically
