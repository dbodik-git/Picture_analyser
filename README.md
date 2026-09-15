# Image Rank Stability Analyzer

A Python script for automatically comparing image variants against a baseline image. It calculates quality metrics (SSIM, PSNR, LPIPS), analyzes brightness/contrast/sharpness deltas, and evaluates the stability of generation and ranks.

## 📋 Features
- **Batch mode**: Supports files named `BASE.png` (or `.jpg`) for the reference and `r{rank}.png` (e.g., `r32.png`, `r16.png`, `r8.png`) for variants.
- **Multidimensional metrics**: SSIM, PSNR, LPIPS (optional), histogram correlation, sharpness/brightness/contrast delta (%), stability assessment.
- **Optimized Parallelization**: Uses `ThreadPoolExecutor` for efficient CPU/GPU utilization. The LPIPS model is loaded once per process, which saves GPU memory and eliminates duplicate weights.
- **Deterministic output**: Results are displayed in the original file order, not in the order in which threads complete.

## 📦 Requirements and Installation
The script requires Python 3.8+ and the following dependencies:
```bash
pip install numpy opencv-python scikit-image
```
To enable the LPIPS perceptual metric (optional):
```bash
pip install lpips
```
⚠️ If lpips is not installed, the metric will be skipped and a warning will be displayed. The remaining metrics will be calculated as usual.

## 🚀 Usage
Run the script, specifying the path to the folder containing the images:
```bash
python analyze.py --folder path/to/images
```

## 💾 Folder structure:
```
folder/
├── BASE.png          # Reference image (or the first file found)
├── r32.png           # Rank 32 variant
├── r16.png           # Rank 16 variant
└── r8.png            # Rank 8 variant
```

## ⚙️ How it works
The script automatically collects images from the specified folder and identifies the base file based on the “BASE” pattern or the first file found.
Each variant is compared to the baseline in parallel using threads (ThreadPoolExecutor). Since OpenCV and NumPy release the GIL, and the LPIPS model is loaded only once per process, this solution ensures maximum speed without duplicating GPU memory.
All metrics are calculated independently of one another (each variant is compared only to BASE).

## 📊 Output Format
The script outputs a table with the following columns:
```
Field    Description
rank    Rank number of the variant (- if not specified)
ssim    Structural Similarity Index (0–1)
psnr    Peak Signal-to-Noise Ratio (dB)
lpips    Perceptual metric (optional; n/a if not set)
hist_correlation    Histogram correlation
sharpness_delta_pct    Sharpness delta (%)
brightness_delta_pct    Brightness delta (%)
contrast_delta_pct    Contrast Delta (%)
stability_score    Rank Stability Score
stability_label    Textual Stability Assessment
```

## 📝 Notes
A GPU (CUDA) is required for LPIPS to function correctly. The model is loaded into memory once per process.
If metrics are calculated for a large number of files, it is recommended to use an SSD to speed up image reading.
The script uses threads (threading) rather than processes (multiprocessing), which allows the LPIPS model to remain in memory only once and avoids duplicating it during parallel computations.

## 📄 License
MIT License
