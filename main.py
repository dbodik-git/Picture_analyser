"""
Сравнение двух изображений: MSE, SSIM, корреляция гистограмм
и визуальная карта различий.
 ***********************************
Запуск:
    python compare.py image1.jpg image2.jpg
    python compare.py image1.jpg image2.jpg --no-show --save-diff diff.png
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


# ---------- Загрузка ----------

def load_image(path: str) -> np.ndarray | None:
    """Загружает изображение, приводя его к 3-канальному BGR."""
    # IMREAD_COLOR отбрасывает альфа-канал (проблема с PNG решена)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[!] Не удалось загрузить: {path}", file=sys.stderr)
    return img


# ---------- Метрики ----------

def mse(a: np.ndarray, b: np.ndarray) -> float:
    """MSE на float32 — быстрее и без переполнения."""
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(diff * diff))


def ssim_map(gray1: np.ndarray, gray2: np.ndarray):
    """SSIM + карта. Возвращаем и значение, и карту различий (1 = различие)."""
    score, sim_map = ssim(gray1, gray2, full=True, data_range=255)
    diff_map = ((1.0 - sim_map) * 255).astype(np.uint8)  # инвертируем!
    return float(score), diff_map


def hist_correlation(img1: np.ndarray, img2: np.ndarray) -> float:
    """Корреляция 3D-гистограмм (8x8x8 бинов)."""
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    cv2.normalize(hist1, hist1, alpha=1.0, norm_type=cv2.NORM_L1)
    cv2.normalize(hist2, hist2, alpha=1.0, norm_type=cv2.NORM_L1)
    return float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))


# ---------- Визуализация ----------

def make_heatmap(diff_map: np.ndarray) -> np.ndarray:
    """Раскрашивает карту различий, чтобы было видно, ГДЕ отличаются."""
    return cv2.applyColorMap(diff_map, cv2.COLORMAP_JET)


def side_by_side(img1, img2, diff_map, label_h=30) -> np.ndarray:
    """Собирает три изображения в одну панель с подписями."""
    heat = make_heatmap(diff_map)

    def pad(img):
        # добавляем сверху полоску для подписи
        return cv2.copyMakeBorder(
            img, label_h, 0, 0, 0,
            cv2.BORDER_CONSTANT, value=(30, 30, 30),
        )

    a, b, c = pad(img1), pad(img2), pad(heat)

    for img, text in ((a, "Image 1"), (b, "Image 2"), (c, "Difference (hot = more different)")):
        cv2.putText(img, text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)

    return np.hstack([a, b, c])


# ---------- Основная функция ----------

def compare(path1: str, path2: str,
            show: bool = True,
            save_diff: str | None = None,
            resize_to: str = "first") -> dict:
    """
    Сравнивает два изображения.

    resize_to: 'first'  — приводить второе к размеру первого
               'second' — наоборот
               'min'    — оба к минимальному общему размеру
               None     — не менять (тогда при разных размерах будет ошибка SSIM)
    """
    img1 = load_image(path1)
    img2 = load_image(path2)
    if img1 is None or img2 is None:
        return {}

    # --- Приведение размеров ---
    if img1.shape != img2.shape:
        if resize_to == "first":
            target = (img1.shape[1], img1.shape[0])
            img2 = cv2.resize(img2, target, interpolation=cv2.INTER_AREA)
        elif resize_to == "second":
            target = (img2.shape[1], img2.shape[0])
            img1 = cv2.resize(img1, target, interpolation=cv2.INTER_AREA)
        elif resize_to == "min":
            h = min(img1.shape[0], img2.shape[0])
            w = min(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (w, h), interpolation=cv2.INTER_AREA)
            img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)
        else:
            print("[!] Изображения разных размеров, SSIM невозможен.", file=sys.stderr)
            return {}

    # --- Градации серого ---
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # --- Метрики ---
    mse_val = mse(gray1, gray2)
    ssim_val, diff_map = ssim_map(gray1, gray2)
    hist_val = hist_correlation(img1, img2)

    results = {
        "mse": mse_val,
        "ssim": ssim_val,
        "hist_correlation": hist_val,
        "identical": mse_val == 0.0,
    }

    # --- Отчёт ---
    print("=" * 50)
    print(f"  {Path(path1).name}  vs  {Path(path2).name}")
    print("=" * 50)
    print(f"  MSE  (меньше = лучше, 0 = идентичны):       {mse_val:12.2f}")
    print(f"  SSIM (1 = идентичны):                       {ssim_val:12.4f}")
    print(f"  Корреляция гистограмм (1 = идентичны):      {hist_val:12.4f}")

    if results["identical"]:
        print("\n  >>> Изображения полностью идентичны <<<")
    else:
        verdict = (
            "очень похожи" if ssim_val > 0.95 else
            "похожи"       if ssim_val > 0.80 else
            "заметно разные" if ssim_val > 0.50 else
            "сильно разные"
        )
        print(f"\n  Вывод: изображения {verdict} (по SSIM)")
    print("=" * 50)

    # --- Сохранение карты различий ---
    if save_diff:
        cv2.imwrite(save_diff, diff_map)
        print(f"  Карта различий сохранена: {save_diff}")

    # --- Показ ---
    if show:
        panel = side_by_side(img1, img2, diff_map)
        # уменьшаем, если слишком широко для экрана
        h, w = panel.shape[:2]
        max_w = 1800
        if w > max_w:
            scale = max_w / w
            panel = cv2.resize(panel, (int(w * scale), int(h * scale)))
        cv2.imshow("Image comparison (press any key)", panel)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return results


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Сравнение двух изображений.")
    parser.add_argument("image1")
    parser.add_argument("image2")
    parser.add_argument("--no-show", action="store_true",
                        help="не открывать окно (для серверов без GUI)")
    parser.add_argument("--save-diff", metavar="PATH",
                        help="сохранить карту различий в файл")
    parser.add_argument("--resize-to", choices=["first", "second", "min"],
                        default="first",
                        help="к какому размеру приводить изображения")
    args = parser.parse_args()

    compare(
        args.image1,
        args.image2,
        show=not args.no_show,
        save_diff=args.save_diff,
        resize_to=args.resize_to,
    )


if __name__ == "__main__":
    main()