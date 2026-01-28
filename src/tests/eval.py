#!/usr/bin/env python3
"""
Evaluate segmentation masks against ground truth.

Usage:
  python -m src.tests.eval --gt_dir path/to/gt --pred_dir path/to/pred
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import tifffile as tiff
except ImportError:  # pragma: no cover - fallback when tifffile isn't available
    tiff = None
    try:
        import imageio.v3 as iio  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install tifffile or imageio to read .tif masks.") from exc


def _read_tif(path: Path) -> np.ndarray:
    if tiff is not None:
        return tiff.imread(str(path))
    return iio.imread(path)


def _to_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask after squeeze, got shape {arr.shape}")
    return arr


def _metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    # Soft overlap metrics without binarization.
    overlap = np.minimum(gt, pred).sum()
    sum_gt = gt.sum()
    sum_pred = pred.sum()

    denom_iou = sum_gt + sum_pred - overlap
    iou = overlap / denom_iou if denom_iou > 0 else 1.0

    denom_dice = sum_gt + sum_pred
    dice = (2 * overlap) / denom_dice if denom_dice > 0 else 1.0

    precision = overlap / sum_pred if sum_pred > 0 else 1.0
    recall = overlap / sum_gt if sum_gt > 0 else 1.0

    denom_f1 = precision + recall
    f1 = (2 * precision * recall) / denom_f1 if denom_f1 > 0 else 0.0

    mae = np.abs(gt - pred).mean()

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mae": float(mae),
    }


def _collect_files(gt_dir: Path) -> list[Path]:
    files = []
    for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        files.extend(gt_dir.glob(ext))
    return sorted(files)


def evaluate_masks(gt_dir: Path, pred_dir: Path, per_image: bool) -> None:
    gt_files = _collect_files(gt_dir)
    if not gt_files:
        raise FileNotFoundError(f"No .tif/.tiff files found in {gt_dir}")

    per_image_rows: list[tuple[str, dict[str, float]]] = []
    agg = {k: [] for k in ("iou", "dice", "precision", "recall", "f1", "mae")}

    for gt_path in gt_files:
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction mask: {pred_path}")

        gt = _to_2d(_read_tif(gt_path))
        pred = _to_2d(_read_tif(pred_path))

        if gt.shape != pred.shape:
            raise ValueError(
                f"Shape mismatch for {gt_path.name}: gt {gt.shape} vs pred {pred.shape}"
            )

        m = _metrics(gt, pred)
        per_image_rows.append((gt_path.name, m))
        for k, v in m.items():
            agg[k].append(v)

    means = {k: float(np.mean(v)) for k, v in agg.items()}

    if per_image:
        for name, m in per_image_rows:
            print(
                f"{name}\t"
                f"iou={m['iou']:.4f}\t"
                f"dice={m['dice']:.4f}\t"
                f"precision={m['precision']:.4f}\t"
                f"recall={m['recall']:.4f}\t"
                f"f1={m['f1']:.4f}\t"
                f"mae={m['mae']:.4f}"
            )

    print("Mean metrics over images")
    print(
        f"iou={means['iou']:.4f} "
        f"dice={means['dice']:.4f} "
        f"precision={means['precision']:.4f} "
        f"recall={means['recall']:.4f} "
        f"f1={means['f1']:.4f} "
        f"mae={means['mae']:.4f}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation masks against ground truth."
    )
    parser.add_argument(
        "--gt_dir",
        required=True,
        type=Path,
        help="Folder with ground-truth grayscale .tif masks.",
    )
    parser.add_argument(
        "--pred_dir",
        required=True,
        type=Path,
        help="Folder with predicted grayscale .tif masks.",
    )
    parser.add_argument(
        "--per_image",
        action="store_true",
        help="Print per-image metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    evaluate_masks(args.gt_dir, args.pred_dir, args.per_image)


if __name__ == "__main__":
    main()
