#!/usr/bin/env python3
"""
Evaluate labeled instance segmentation masks against ground truth.

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


def _match_instances(
    gt: np.ndarray, pred: np.ndarray, iou_threshold: float
) -> tuple[list[tuple[int, int, float]], np.ndarray, np.ndarray, np.ndarray]:
    gt_labels = np.unique(gt)
    pred_labels = np.unique(pred)

    # Always include background label 0 if missing.
    if gt_labels.size == 0 or gt_labels[0] != 0:
        gt_labels = np.insert(gt_labels, 0, 0)
    if pred_labels.size == 0 or pred_labels[0] != 0:
        pred_labels = np.insert(pred_labels, 0, 0)

    gt_flat = gt.ravel()
    pred_flat = pred.ravel()

    gt_idx = np.searchsorted(gt_labels, gt_flat)
    pred_idx = np.searchsorted(pred_labels, pred_flat)

    n_gt = len(gt_labels)
    n_pred = len(pred_labels)

    inter = np.bincount(gt_idx * n_pred + pred_idx, minlength=n_gt * n_pred)
    inter = inter.reshape((n_gt, n_pred)).astype(np.float32)

    area_gt = inter.sum(axis=1)
    area_pred = inter.sum(axis=0)

    pairs: list[tuple[int, int, float]] = []
    for gi in range(1, n_gt):
        for pi in range(1, n_pred):
            intersection = inter[gi, pi]
            if intersection <= 0:
                continue
            union = area_gt[gi] + area_pred[pi] - intersection
            if union <= 0:
                continue
            iou = intersection / union
            if iou >= iou_threshold:
                pairs.append((gi, pi, float(iou)))

    pairs.sort(key=lambda x: x[2], reverse=True)

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for gi, pi, iou in pairs:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        matches.append((gi, pi, iou))

    return matches, inter, gt_labels, pred_labels


def _metrics(gt: np.ndarray, pred: np.ndarray, iou_threshold: float) -> dict[str, float]:
    matches, inter, gt_labels_all, pred_labels_all = _match_instances(
        gt, pred, iou_threshold
    )

    gt_labels = gt_labels_all[gt_labels_all != 0]
    pred_labels = pred_labels_all[pred_labels_all != 0]

    num_gt = len(gt_labels)
    num_pred = len(pred_labels)
    num_match = len(matches)

    if num_gt == 0 and num_pred == 0:
        return {
            "iou": 1.0,
            "dice": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "mae": 0.0,
        }

    ious: list[float] = []
    dices: list[float] = []

    if num_match > 0:
        area_gt = inter.sum(axis=1)
        area_pred = inter.sum(axis=0)

        for gi_idx, pi_idx, iou in matches:
            intersection = inter[gi_idx, pi_idx]
            denom = area_gt[gi_idx] + area_pred[pi_idx]
            dice = (2 * intersection) / denom if denom > 0 else 1.0
            ious.append(iou)
            dices.append(float(dice))

    precision = num_match / num_pred if num_pred > 0 else 1.0
    recall = num_match / num_gt if num_gt > 0 else 1.0
    denom_f1 = precision + recall
    f1 = (2 * precision * recall) / denom_f1 if denom_f1 > 0 else 0.0

    mean_iou = float(np.mean(ious)) if ious else 0.0
    mean_dice = float(np.mean(dices)) if dices else 0.0

    # Map matched predicted instances to GT labels to make MAE less arbitrary.
    pred_mapped = np.zeros_like(pred)
    for gi_idx, pi_idx, _ in matches:
        gt_label = gt_labels_all[gi_idx]
        pred_label = pred_labels_all[pi_idx]
        pred_mapped[pred == pred_label] = gt_label

    mae = np.abs(gt.astype(np.float32) - pred_mapped.astype(np.float32)).mean()

    return {
        "iou": mean_iou,
        "dice": mean_dice,
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


def evaluate_masks(gt_dir: Path, pred_dir: Path, iou_threshold: float, per_image: bool) -> None:
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

        m = _metrics(gt, pred, iou_threshold)
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
        "--iou_threshold",
        type=float,
        default=0.8,
        help="IoU threshold to match GT and predicted instances.",
    )
    parser.add_argument(
        "--per_image",
        action="store_true",
        help="Print per-image metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    evaluate_masks(args.gt_dir, args.pred_dir, args.iou_threshold, args.per_image)


if __name__ == "__main__":
    main()
