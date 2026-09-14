"""
Picture Analyser — анализ и сравнение изображений.

Возможности:
    • сравнение двух изображений: MSE, PSNR, SSIM, LPIPS, корреляция гистограмм;
    • технические характеристики: размер, aspect ratio, яркость,
      контраст, насыщенность, резкость, clipping;
    • анализ серии BASE/rank-вариантов относительно BASE (параллельно);
    • visual stability score и поиск резкого падения между rank;
    • карта различий и удобная текстовая таблица для копирования.

Примеры:
    python main.py image1.png image2.png
    python main.py image1.png image2.png --no-show --save-diff diff.png
    python main.py series ./test
    python main.py series ./test --workers 8

Для пакетного режима имена файлов должны содержать BASE и/или rank,
например: BASE.png, r32.png, r24.png, r16.png, r8.png.

LPIPS — опциональная перцептивная метрика (`pip install lpips`). Если пакет
не установлен, LPIPS просто пропускается (один раз выводится предупреждение),
остальные метрики считаются как обычно.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as structural_ssim


# ---------- Загрузка ----------


def load_image(path: str | Path) -> np.ndarray | None:
    """Загрузить изображение как 3-канальный BGR."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[!] Не удалось загрузить: {path}", file=sys.stderr)
    return img


# ---------- Базовые метрики ----------


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Squared Error. Меньше = ближе."""
    a32 = a.astype(np.float32)
    b32 = b.astype(np.float32)
    diff = a32 - b32
    return float(np.mean(diff * diff))


def psnr(a: np.ndarray, b: np.ndarray, max_value: float = 255.0) -> float:
    """Peak Signal-to-Noise Ratio в дБ. Больше = ближе (inf при полном совпадении)."""
    mse_val = mse(a, b)
    if mse_val == 0:
        return float("inf")
    return float(10.0 * math.log10((max_value ** 2) / mse_val))


def ssim_map(
    gray1: np.ndarray,
    gray2: np.ndarray,
    need_diff_map: bool = True,
) -> tuple[float, np.ndarray | None]:
    """SSIM и (опционально) карта различий, где 255 означает сильное различие.

    `full=True` в scikit-image считает и возвращает полную поточечную карту
    сходства — лишняя аллокация и работа, если она никому не нужна (например,
    в batch-режиме без --save-diff/показа). Передавайте need_diff_map=False,
    чтобы посчитать только скалярный SSIM.
    """
    if not need_diff_map:
        score = structural_ssim(gray1, gray2, full=False, data_range=255)
        return float(score), None

    score, similarity_map = structural_ssim(
        gray1,
        gray2,
        full=True,
        data_range=255,
    )
    diff_map = np.clip((1.0 - similarity_map) * 255.0, 0, 255).astype(np.uint8)
    return float(score), diff_map


def hist_correlation(img1: np.ndarray, img2: np.ndarray) -> float:
    """Корреляция 3D-гистограмм BGR (8x8x8 бинов)."""
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    cv2.normalize(hist1, hist1, alpha=1.0, norm_type=cv2.NORM_L1)
    cv2.normalize(hist2, hist2, alpha=1.0, norm_type=cv2.NORM_L1)
    return float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))


# ---------- LPIPS (опционально) ----------

_LPIPS_MODEL = None
_LPIPS_DEVICE = None
_LPIPS_WARNED = False
# analyse_series() calls _get_lpips_model() from several worker threads at
# once; without a lock, each thread sees "_LPIPS_MODEL is None" before the
# first one finishes and starts its own load — that's what produced the
# repeated "Setting up [LPIPS]..." / "Loading model from..." lines.
_LPIPS_LOCK = threading.Lock()


def _get_lpips_model():
    """Лениво загрузить и закэшировать модель LPIPS (один раз на процесс,
    даже если несколько потоков запросят её одновременно)."""
    global _LPIPS_MODEL, _LPIPS_DEVICE, _LPIPS_WARNED
    if _LPIPS_MODEL is not None:
        return _LPIPS_MODEL

    with _LPIPS_LOCK:
        # Another thread may have finished loading while we were waiting
        # for the lock — re-check before doing the work again.
        if _LPIPS_MODEL is not None:
            return _LPIPS_MODEL

        try:
            import lpips
            import torch
        except ImportError:
            if not _LPIPS_WARNED:
                print(
                    "[!] Пакет 'lpips' не установлен — LPIPS будет пропущен "
                    "(pip install lpips torch, если нужна эта метрика).",
                    file=sys.stderr,
                )
                _LPIPS_WARNED = True
            return None

        _LPIPS_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        with warnings.catch_warnings():
            # Silences lpips's own "Setting up.../Loading model from..."
            # status prints (verbose=False) and torchvision's unrelated
            # 'pretrained' deprecation warning that lpips triggers
            # internally when building the AlexNet backbone.
            warnings.simplefilter("ignore", UserWarning)
            _LPIPS_MODEL = lpips.LPIPS(net="alex", verbose=False).to(_LPIPS_DEVICE).eval()
        return _LPIPS_MODEL


def lpips_distance(img1_bgr: np.ndarray, img2_bgr: np.ndarray) -> float | None:
    """Перцептивное LPIPS-расстояние (меньше = перцептивно ближе).

    Возвращает None, если пакет `lpips` не установлен — вызывающий код должен
    аккуратно обработать этот случай (пропустить метрику, не падать).
    """
    model = _get_lpips_model()
    if model is None:
        return None

    import torch

    def to_tensor(img_bgr: np.ndarray):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        return tensor * 2.0 - 1.0  # LPIPS ожидает вход в диапазоне [-1, 1]

    t1 = to_tensor(img1_bgr).to(_LPIPS_DEVICE)
    t2 = to_tensor(img2_bgr).to(_LPIPS_DEVICE)
    with torch.no_grad():
        dist = model(t1, t2)
    return float(dist.item())


# ---------- Характеристики изображения ----------


def image_stats(img: np.ndarray) -> dict[str, float | int | str]:
    """Рассчитать компактный набор характеристик изображения."""
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    saturation = float(np.mean(hsv[:, :, 1]))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    total = gray.size
    dark_clip = float(np.count_nonzero(gray <= 2) / total * 100.0)
    bright_clip = float(np.count_nonzero(gray >= 253) / total * 100.0)

    return {
        "width": width,
        "height": height,
        "aspect_ratio": width / height,
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
        "sharpness": sharpness,
        "dark_clip_pct": dark_clip,
        "bright_clip_pct": bright_clip,
    }


def format_stats(stats: dict[str, float | int | str]) -> str:
    """Красивый текстовый вывод характеристик."""
    return (
        f"  Размер:               {stats['width']} x {stats['height']}\n"
        f"  Aspect ratio:         {stats['aspect_ratio']:.4f}\n"
        f"  Яркость:              {stats['brightness']:.2f}\n"
        f"  Контраст:             {stats['contrast']:.2f}\n"
        f"  Насыщенность:         {stats['saturation']:.2f}\n"
        f"  Резкость (Laplacian): {stats['sharpness']:.2f}\n"
        f"  Чёрный clipping:      {stats['dark_clip_pct']:.2f}%\n"
        f"  Белый clipping:       {stats['bright_clip_pct']:.2f}%"
    )


# ---------- Приведение размеров ----------


def resize_pair(
    img1: np.ndarray,
    img2: np.ndarray,
    resize_to: str = "first",
) -> tuple[np.ndarray, np.ndarray] | None:
    """Привести пару к совместимому размеру для сравнения (пиксельные метрики)."""
    if img1.shape == img2.shape:
        return img1, img2

    if resize_to == "first":
        target = (img1.shape[1], img1.shape[0])
        img2 = cv2.resize(img2, target, interpolation=cv2.INTER_AREA)
    elif resize_to == "second":
        target = (img2.shape[1], img2.shape[0])
        img1 = cv2.resize(img1, target, interpolation=cv2.INTER_AREA)
    elif resize_to == "min":
        height = min(img1.shape[0], img2.shape[0])
        width = min(img1.shape[1], img2.shape[1])
        img1 = cv2.resize(img1, (width, height), interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, (width, height), interpolation=cv2.INTER_AREA)
    else:
        print("[!] Изображения разных размеров, сравнение невозможно.", file=sys.stderr)
        return None

    return img1, img2


# ---------- Визуализация ----------


def make_heatmap(diff_map: np.ndarray) -> np.ndarray:
    """Раскрасить карту различий."""
    return cv2.applyColorMap(diff_map, cv2.COLORMAP_JET)


def side_by_side(
    img1: np.ndarray,
    img2: np.ndarray,
    diff_map: np.ndarray,
    label_h: int = 30,
) -> np.ndarray:
    """Собрать Image 1 / Image 2 / Difference в одну панель."""
    heat = make_heatmap(diff_map)

    def pad(img: np.ndarray) -> np.ndarray:
        return cv2.copyMakeBorder(
            img,
            label_h,
            0,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(30, 30, 30),
        )

    panels = [pad(img1), pad(img2), pad(heat)]
    labels = ["Image 1", "Image 2", "Difference (hot = more different)"]

    for panel, text in zip(panels, labels):
        cv2.putText(
            panel,
            text,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return np.hstack(panels)


# ---------- Вывод результатов ----------


def verdict_from_ssim(ssim_value: float) -> str:
    """Интерпретация SSIM; это НЕ оценка художественного качества."""
    if ssim_value > 0.95:
        return "очень похожи"
    if ssim_value > 0.80:
        return "похожи"
    if ssim_value > 0.50:
        return "заметно разные"
    return "сильно разные"


def signed_delta(value: float, base: float) -> tuple[float, float]:
    """Вернуть абсолютную и процентную разницу."""
    delta = value - base
    if base == 0:
        return delta, 0.0
    return delta, delta / abs(base) * 100.0


def visual_stability_score(
    ssim_value: float,
    hist_value: float,
    sharpness_delta_pct: float,
) -> float:
    """Сводный индикатор стабильности относительно BASE.

    Это эвристика для сортировки вариантов, а не объективная оценка качества.
    SSIM имеет основной вес, цветовая гистограмма — меньший, а изменение
    резкости мягко штрафуется в обе стороны. Веса не менялись — PSNR/LPIPS
    добавлены как отдельные репортируемые метрики, а не как часть формулы,
    чтобы не менять смысл уже знакомых STABLE/GOOD/WATCH/WARNING вердиктов.
    """
    ssim_component = float(np.clip(ssim_value, 0.0, 1.0))
    hist_component = float(np.clip((hist_value + 1.0) / 2.0, 0.0, 1.0))
    sharp_component = math.exp(-abs(sharpness_delta_pct) / 100.0)
    score = 100.0 * (
        0.60 * ssim_component
        + 0.25 * hist_component
        + 0.15 * sharp_component
    )
    return float(np.clip(score, 0.0, 100.0))


def stability_label(score: float) -> str:
    """Человеческая метка для эвристического stability score."""
    if score >= 90.0:
        return "STABLE"
    if score >= 80.0:
        return "GOOD"
    if score >= 70.0:
        return "WATCH"
    return "WARNING"


def analyse_rank_stability(rows: list[dict]) -> dict:
    """Найти возможный quality cliff между соседними rank.

    Анализируется только фактически присутствующая серия. Никакие отсутствующие
    rank не считаются стабильными автоматически.
    """
    ranked = sorted(
        (row for row in rows if row.get("rank") is not None),
        key=lambda row: int(row["rank"]),
        reverse=True,
    )

    cliffs: list[dict] = []
    for high, low in zip(ranked, ranked[1:]):
        drop = float(high["stability_score"]) - float(low["stability_score"])
        if drop >= 8.0:
            cliffs.append({
                "from_rank": int(high["rank"]),
                "to_rank": int(low["rank"]),
                "score_drop": drop,
            })

    stable_candidates = [
        row for row in ranked
        if float(row["stability_score"]) >= 80.0
    ]

    minimum_stable_rank = None
    if stable_candidates:
        minimum_stable_rank = min(int(row["rank"]) for row in stable_candidates)

    return {
        "cliffs": cliffs,
        "minimum_stable_rank": minimum_stable_rank,
    }


def compare(
    path1: str,
    path2: str,
    show: bool = True,
    save_diff: str | None = None,
    resize_to: str = "first",
) -> dict:
    """Сравнить два изображения и вернуть все рассчитанные метрики."""
    img1 = load_image(path1)
    img2 = load_image(path2)
    if img1 is None or img2 is None:
        return {}

    # Stats are computed on the ORIGINAL files before any resizing, so the
    # reported "Размер" always reflects the real dimensions of each file —
    # not the working-copy size used below for pixel-wise comparison.
    stats1 = image_stats(img1)
    stats2 = image_stats(img2)
    same_shape = img1.shape == img2.shape
    # "Identical" is judged on the originals: two files of different native
    # size can never be identical, regardless of how they're resized for
    # comparison below.
    identical = bool(same_shape and np.array_equal(img1, img2))

    resized = resize_pair(img1, img2, resize_to)
    if resized is None:
        return {}
    cmp1, cmp2 = resized

    gray1 = cv2.cvtColor(cmp1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(cmp2, cv2.COLOR_BGR2GRAY)

    # The diff map (used by --save-diff and the on-screen panel) is the only
    # expensive part of SSIM — skip computing it entirely when nothing will
    # display it.
    need_diff_map = show or bool(save_diff)

    mse_val = mse(gray1, gray2)
    color_mse_val = mse(cmp1, cmp2)
    psnr_val = psnr(gray1, gray2)
    ssim_val, diff_map = ssim_map(gray1, gray2, need_diff_map=need_diff_map)
    hist_val = hist_correlation(cmp1, cmp2)
    lpips_val = lpips_distance(cmp1, cmp2)

    _, sharp_pct = signed_delta(float(stats2["sharpness"]), float(stats1["sharpness"]))
    stability = visual_stability_score(ssim_val, hist_val, sharp_pct)

    results = {
        "mse_gray": mse_val,
        "mse_color": color_mse_val,
        "psnr": psnr_val,
        "ssim": ssim_val,
        "hist_correlation": hist_val,
        "lpips": lpips_val,
        "stability_score": stability,
        "identical": identical,
        "stats1": stats1,
        "stats2": stats2,
    }

    print("=" * 70)
    print(f"  {Path(path1).name}  vs  {Path(path2).name}")
    print("=" * 70)
    if not same_shape:
        print(
            f"  ⚠ Разные исходные размеры ({stats1['width']}x{stats1['height']} vs "
            f"{stats2['width']}x{stats2['height']}); для пиксельных метрик ниже "
            f"изображения были приведены к одному размеру (--resize-to {resize_to})."
        )
    print(f"  Gray MSE:                                      {mse_val:12.2f}")
    print(f"  Color MSE:                                     {color_mse_val:12.2f}")
    print(f"  PSNR:                                          {psnr_val:12.2f} dB")
    print(f"  SSIM:                                           {ssim_val:12.4f}")
    print(f"  Корреляция гистограмм:                         {hist_val:12.4f}")
    if lpips_val is not None:
        print(f"  LPIPS (перцептивное):                          {lpips_val:12.4f}")
    print(f"  Visual stability score:                        {stability:12.2f}/100")

    print("\n  Характеристики Image 1:")
    print(format_stats(stats1))
    print("\n  Характеристики Image 2:")
    print(format_stats(stats2))

    print("\n  Изменения Image 2 относительно Image 1:")
    for key, label in (
        ("brightness", "Яркость"),
        ("contrast", "Контраст"),
        ("saturation", "Насыщенность"),
        ("sharpness", "Резкость"),
        ("dark_clip_pct", "Чёрный clipping"),
        ("bright_clip_pct", "Белый clipping"),
    ):
        delta, percent = signed_delta(float(stats2[key]), float(stats1[key]))
        print(f"  {label:22s}: {delta:+10.2f}  ({percent:+7.2f}%)")

    if identical:
        print("\n  >>> Изображения полностью идентичны <<<")
    else:
        print(f"\n  Вывод по SSIM: изображения {verdict_from_ssim(ssim_val)}")
        print("  ⚠ SSIM — метрика сходства, а не универсальная оценка качества.")
        print("  ℹ Stability score — эвристика для сравнения вариантов, не оценка искусства.")
    print("=" * 70)

    if save_diff:
        output = Path(save_diff)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Save the same colourised heatmap shown in the preview window
        # ("hot = more different"), not the raw grayscale diff map — the
        # saved file should match what you actually looked at.
        cv2.imwrite(str(output), make_heatmap(diff_map))
        print(f"  Карта различий сохранена: {output}")

    if show:
        panel = side_by_side(cmp1, cmp2, diff_map)
        height, width = panel.shape[:2]
        max_width = 1800
        if width > max_width:
            scale = max_width / width
            panel = cv2.resize(panel, (int(width * scale), int(height * scale)))
        cv2.imshow("Image comparison (press any key)", panel)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return results


# ---------- Пакетный анализ серии ----------


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def rank_from_name(path: Path) -> int | None:
    """Извлечь rank из имени вроде r16, rank16, R16."""
    match = re.search(r"(?:^|[_ .-])r(?:ank)?[_ .-]?(\d+)(?:$|[_ .-])", path.stem, re.I)
    if match:
        return int(match.group(1))
    return None


def is_base_name(path: Path) -> bool:
    """Определить BASE/reference по имени файла."""
    return bool(re.search(r"(?:^|[_ .-])base(?:$|[_ .-])", path.stem, re.I))


def collect_images(folder: str | Path) -> list[Path]:
    """Собрать изображения из папки, отсортировав BASE, затем rank по убыванию."""
    folder_path = Path(folder)
    if not folder_path.is_dir():
        print(f"[!] Папка не найдена: {folder}", file=sys.stderr)
        return []

    paths = [
        p for p in folder_path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    def sort_key(path: Path) -> tuple[int, int, str]:
        if is_base_name(path):
            return (0, 0, path.name.lower())
        rank = rank_from_name(path)
        if rank is not None:
            return (1, -rank, path.name.lower())
        return (2, 0, path.name.lower())

    return sorted(paths, key=sort_key)


def _compare_variant_to_base(
    path: Path,
    base_img: np.ndarray,
    base_stats: dict,
    resize_to: str,
) -> dict | None:
    """Посчитать одну строку таблицы серии. Выполняется в рабочем потоке."""
    img = load_image(path)
    if img is None:
        return None

    resized = resize_pair(base_img, img, resize_to)
    if resized is None:
        return None
    base_cmp, img_cmp = resized

    gray_base = cv2.cvtColor(base_cmp, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.cvtColor(img_cmp, cv2.COLOR_BGR2GRAY)
    ssim_value, _ = ssim_map(gray_base, gray_img, need_diff_map=False)
    hist_value = hist_correlation(base_cmp, img_cmp)
    psnr_value = psnr(gray_base, gray_img)
    lpips_value = lpips_distance(base_cmp, img_cmp)
    stats = image_stats(img_cmp)

    _, sharp_pct = signed_delta(float(stats["sharpness"]), float(base_stats["sharpness"]))
    _, bright_pct = signed_delta(float(stats["brightness"]), float(base_stats["brightness"]))
    _, contrast_pct = signed_delta(float(stats["contrast"]), float(base_stats["contrast"]))
    stability = visual_stability_score(ssim_value, hist_value, sharp_pct)

    rank = rank_from_name(path)
    return {
        "path": str(path),
        "name": path.name,
        "rank": rank,
        "ssim": ssim_value,
        "psnr": psnr_value,
        "lpips": lpips_value,
        "hist_correlation": hist_value,
        "sharpness_delta_pct": sharp_pct,
        "brightness_delta_pct": bright_pct,
        "contrast_delta_pct": contrast_pct,
        "stability_score": stability,
        "stability_label": stability_label(stability),
    }


def analyse_series(
    folder: str | Path,
    resize_to: str = "first",
    max_workers: int | None = None,
) -> list[dict]:
    """Сравнить все rank-варианты в папке с BASE.

    Варианты друг от друга не зависят (каждый сравнивается только с BASE),
    поэтому считаются параллельно через ThreadPoolExecutor. Используются
    именно потоки, а не процессы: cv2/numpy/torch — это в основном C/CUDA
    код, освобождающий GIL, а LPIPS-модель (если установлена) при этом
    загружается и живёт в памяти один раз на процесс, а не по разу на
    воркер — с ProcessPoolExecutor каждый процесс держал бы свою копию
    модели (и своё выделение GPU-памяти).
    """
    paths = collect_images(folder)
    if not paths:
        return []

    base = next((path for path in paths if is_base_name(path)), paths[0])
    base_img = load_image(base)
    if base_img is None:
        return []

    base_stats = image_stats(base_img)
    variant_paths = [p for p in paths if p.resolve() != base.resolve()]

    print("=" * 132)
    print(f"  SERIES ANALYSIS: {Path(folder).resolve()}")
    print(f"  BASE: {base.name}")
    print("  Все метрики ниже сравнивают вариант с BASE.")
    print("=" * 132)
    print(
        f"  {'Variant':24s} {'Rank':>5s} {'SSIM':>8s} {'PSNR':>8s} {'LPIPS':>8s} {'Hist':>8s} "
        f"{'Sharp Δ':>10s} {'Bright Δ':>10s} {'Contrast Δ':>11s} {'Stable':>9s} {'Flag':>8s}"
    )
    print("-" * 132)

    worker_count = max_workers or min(32, (os.cpu_count() or 4))
    rows_by_path: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(_compare_variant_to_base, path, base_img, base_stats, resize_to): path
            for path in variant_paths
        }
        for future, path in futures.items():
            row = future.result()
            if row is not None:
                rows_by_path[str(path)] = row

    # Печатаем в исходном (отсортированном) порядке, а не в порядке
    # завершения потоков, чтобы таблица оставалась детерминированной.
    rows: list[dict] = []
    for path in variant_paths:
        row = rows_by_path.get(str(path))
        if row is None:
            continue
        rows.append(row)

        rank_text = str(row["rank"]) if row["rank"] is not None else "-"
        lpips_text = f"{row['lpips']:.4f}" if row["lpips"] is not None else "n/a"
        print(
            f"  {row['name'][:24]:24s} {rank_text:>5s} {row['ssim']:8.4f} "
            f"{row['psnr']:8.2f} {lpips_text:>8s} {row['hist_correlation']:8.4f} "
            f"{row['sharpness_delta_pct']:+9.2f}% {row['brightness_delta_pct']:+9.2f}% "
            f"{row['contrast_delta_pct']:+10.2f}% {row['stability_score']:8.2f} "
            f"{row['stability_label']:>8s}"
        )

    analysis = analyse_rank_stability(rows)

    print("-" * 132)
    if analysis["cliffs"]:
        print("  ⚠ QUALITY CLIFF: резкое падение stability score между:")
        for cliff in analysis["cliffs"]:
            print(
                f"      r{cliff['from_rank']} → r{cliff['to_rank']}: "
                f"-{cliff['score_drop']:.2f} points"
            )
    else:
        print("  ✓ Резкого quality cliff среди присутствующих rank не обнаружено.")

    if analysis["minimum_stable_rank"] is not None:
        print(
            f"  ★ Минимальный стабильный rank по этой эвристике: "
            f"r{analysis['minimum_stable_rank']}"
        )
    else:
        print("  ⚠ Ни один присутствующий rank не достиг stability score 80/100.")

    print("  ℹ Stable ≥90, Good ≥80, Watch ≥70; это эвристические пороги.")
    print("  ℹ SSIM/Hist показывают сходство с BASE; это не рейтинг художественного качества.")
    print("  ℹ Stability score помогает сортировать варианты, но финальное решение делаем по A/B-картинкам.")
    print("=" * 132)
    return rows


# ---------- CLI ----------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Анализ и сравнение изображений для тестов Krea2/LoRA/SVD."
    )
    subparsers = parser.add_subparsers(dest="command")

    pair = subparsers.add_parser("compare", help="сравнить два изображения")
    pair.add_argument("image1")
    pair.add_argument("image2")
    pair.add_argument("--no-show", action="store_true", help="не открывать окно")
    pair.add_argument("--save-diff", metavar="PATH", help="сохранить карту различий")
    pair.add_argument(
        "--resize-to",
        choices=["first", "second", "min"],
        default="first",
        help="к какому размеру приводить изображения",
    )

    series = subparsers.add_parser("series", help="сравнить серию изображений с BASE")
    series.add_argument("folder", help="папка с BASE/rank изображениями")
    series.add_argument(
        "--resize-to",
        choices=["first", "second", "min"],
        default="first",
        help="к какому размеру приводить изображения",
    )
    series.add_argument(
        "--workers",
        type=int,
        default=None,
        help="число потоков для сравнения вариантов (по умолчанию: авто)",
    )

    return parser


def main() -> None:
    # Backward-compat shim for the old two-positional-args syntax shown in the
    # module docstring: `python main.py image1.png image2.png [flags]`.
    # argparse's subparsers positional is the *only* top-level positional, so
    # if we parsed argv as-is it would try to match "image1.png" against the
    # {compare,series} choices and hard-exit with "invalid choice" before any
    # fallback code could run. Rewriting argv up front (inserting "compare")
    # is the fix — patching this up after parse_args() is too late.
    argv = sys.argv[1:]
    if argv and not argv[0].startswith("-") and argv[0] not in ("compare", "series"):
        argv = ["compare", *argv]

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "compare":
        compare(
            args.image1,
            args.image2,
            show=not args.no_show,
            save_diff=args.save_diff,
            resize_to=args.resize_to,
        )
    elif args.command == "series":
        analyse_series(args.folder, resize_to=args.resize_to, max_workers=args.workers)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()