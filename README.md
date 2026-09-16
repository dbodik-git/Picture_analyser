# Image Rank Stability Analyzer

A Python tool for comparing image variants against a baseline image, with a focus on **LoRA / SVD rank-compression tests**.

It calculates structural, pixel-level and perceptual similarity metrics, reports technical image changes, and can analyze a whole `BASE + rank variants` series in parallel.

## 📋 Features

- **Pair comparison** — compare two images with MSE, PSNR, SSIM, LPIPS and histogram correlation.
- **Technical image statistics** — resolution, aspect ratio, brightness, contrast, saturation, sharpness and black/white clipping.
- **Series analysis** — compare every rank variant against `BASE` automatically.
- **Rank detection** — recognizes names such as `r32.png`, `r16.png`, `r8.png`, `rank16.png` and `R16.png`.
- **Visual Stability Score** — an explicit heuristic for sorting variants by similarity/stability relative to `BASE`.
- **Quality-cliff detection** — reports a large drop in Stability Score between consecutive ranks that are actually present.
- **Parallel processing** — series variants are processed with `ThreadPoolExecutor`.
- **LPIPS model caching** — LPIPS is loaded lazily and cached once per process; a lock prevents several worker threads from loading duplicate copies at startup.
- **Difference visualization** — pair mode can display and save a colorized difference heatmap.
- **Deterministic output** — results are printed in the same order as the collected files, regardless of which worker finishes first.

## 📦 Requirements and Installation

Python **3.8+** is required.

Install the core dependencies:

```bash
pip install numpy opencv-python scikit-image
```

LPIPS is optional. To enable the perceptual metric:

```bash
pip install lpips torch
```

If `lpips` is not installed, LPIPS is skipped and the remaining metrics are still calculated.

### 🛠 GPU / CUDA

LPIPS uses CUDA automatically when a CUDA-enabled PyTorch installation is available. Otherwise it falls back to **CPU**; a GPU is **not required** for the script to run or for LPIPS itself to be calculated.

For a CUDA setup, install a PyTorch build appropriate for your system from the official PyTorch instructions, then install `lpips`.

## 🚀 Usage

### Compare two images

```bash
python Picture_analyser.py image1.png image2.png
```

Do not open the preview window and save the difference heatmap:

```bash
python Picture_analyser.py image1.png image2.png --no-show --save-diff diff.png
```

The old two-file syntax without the explicit `compare` command is also supported.

### Analyze a rank series

```bash
python Picture_analyser.py series ./test
```

Limit the number of worker threads:

```bash
python Picture_analyser.py series ./test --workers 8
```

If the image sizes differ, the comparison can be controlled with:

```bash
--resize-to first
--resize-to second
--resize-to min
```

## 💾 Folder structure

A typical rank-compression test folder looks like this:

```text
folder/
├── base.png          # Reference image
├── r32.png           # Rank 32
├── r24.png           # Rank 24
├── r16.png           # Rank 16
├── r8.png            # Rank 8
├── r4.png            # Rank 4
└── r1.png            # Rank 1
```

The reference is selected by a filename containing `BASE` (case-insensitive). If no such file exists, the first file in the collected/sorted list is used as the reference.

Supported image formats:

`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tif`, `.tiff`

## ⚙️ How it works

### Pair comparison

Each pair is loaded and compared. Technical statistics are calculated from the original image files. Pixel-wise metrics use the selected `--resize-to` policy when the images have different dimensions.

The following metrics are independent measurements:

- **MSE** — pixel error; lower is closer.
- **PSNR** — peak signal-to-noise ratio; higher is closer. Reported in dB.
- **SSIM** — structural similarity; higher is more structurally similar.
- **LPIPS** — learned perceptual distance; **lower is perceptually closer**.
- **Histogram correlation** — similarity of the color distributions; higher is more similar.

No single metric is an objective measure of artistic quality.

### Series comparison

In `series` mode, every non-BASE image is compared **directly to BASE**. Variants are not compared against one another, so a rank cannot score well merely because it resembles the previous rank.

The work is distributed across threads. OpenCV, NumPy and PyTorch perform most heavy computation outside Python's GIL, while the LPIPS model is shared within the process instead of being copied into separate worker processes.

### Visual Stability Score

`Stability Score` is a **heuristic similarity/stability indicator**, not a universal quality score.

The current formula uses:

- **60% SSIM**
- **25% histogram correlation**
- **15% sharpness-change component**

PSNR and LPIPS are reported separately and are **not included in the Stability Score formula**. This keeps the meaning of the existing `STABLE / GOOD / WATCH / WARNING` labels consistent.

Current labels:

| Stability Score | Label | Meaning |
|---:|:---:|---|
| **≥ 90** | `STABLE` | Very stable relative to BASE by this heuristic |
| **≥ 80** | `GOOD` | Good similarity/stability |
| **≥ 70** | `WATCH` | Worth checking visually |
| **< 70** | `WARNING` | Strong deviation; inspect the images |

The analyzer can also report a **quality cliff** when the Stability Score drops by **8 or more points** between consecutive ranks that are actually present in the folder.

## 📊 Series output table

The `series` command prints one row per image variant:

| Column | Description | Better / interpretation |
|---|---|---|
| `Variant` | File name | — |
| `Rank` | Rank extracted from the file name; `-` if none | — |
| `SSIM` | Structural Similarity Index, `0–1` | **Higher = more structurally similar** |
| `PSNR` | Peak Signal-to-Noise Ratio, dB | **Higher = closer pixel-wise** |
| `LPIPS` | Learned perceptual distance | **Lower = perceptually closer**; `n/a` if unavailable |
| `Hist` | 3D BGR histogram correlation | **Higher = more similar color distribution** |
| `Sharp Δ` | Sharpness change of variant relative to BASE | `+` = sharper, `−` = softer; not inherently good or bad |
| `Bright Δ` | Brightness change relative to BASE | `+` = brighter, `−` = darker |
| `Contrast Δ` | Contrast change relative to BASE | `+` = more contrast, `−` = less contrast |
| `Stable` | Visual Stability Score, `0–100` | **Higher = more stable by the heuristic** |
| `Flag` | Text label derived from `Stable` | `STABLE`, `GOOD`, `WATCH` or `WARNING` |

### Important: the metrics answer different questions

For example:

- A high **SSIM** does not prove that a LoRA preserves its concept or style.
- A low **LPIPS** means the image is perceptually close to BASE; it does not mean that the LoRA itself is better.
- A high **Stability Score** does not mean that the image is artistically better.
- `Sharp Δ` is descriptive: a positive value means more measured sharpness, but more sharpness is not automatically better.

For LoRA rank experiments, use the table to find interesting ranges and then make the final decision with **A/B visual inspection** and, when relevant, tests at the intended LoRA weight.

## 🧪 Example rank experiment

If a folder contains:

```text
BASE.png
r32.png
r24.png
r16.png
r8.png
r4.png
r1.png
```

run:

```bash
python Picture_analyser.py series ./folder
```

The resulting table lets you see how similarity changes as rank is reduced. A sudden drop in the Stability Score can indicate a possible degradation point, while SSIM / PSNR / LPIPS provide additional independent measurements.

## ⚠️ Notes and limitations

- The tool compares **images**, not LoRA tensors. It cannot directly determine whether a rank preserves a LoRA's concept, style or identity.
- All series metrics are comparisons against the same BASE image.
- `Stability Score` is intentionally a heuristic and should not be presented as an objective image-quality score.
- LPIPS is optional. Without the `lpips` package, its column is shown as `n/a`.
- LPIPS automatically uses CUDA when available and otherwise uses CPU.
- Thread count can be controlled with `--workers`. More threads are not always faster, especially when LPIPS is running on a single GPU.
- For large image sets, an SSD can improve image-loading performance.
- If a folder contains unrelated images (for example, a manually assembled comparison sheet), they may be included in the series output unless their names are changed or they are moved elsewhere.

## 📄 License

MIT License
