#!/usr/bin/env python3
"""
Evaluate labeled instance segmentation masks against ground truth.

Usage:
  python -m src.models.tests.eval --gt_dir path/to/gt --pred_dir path/to/pred
"""

from __future__ import annotations

import argparse
import csv
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


def _write_summary_csv(agg: dict[str, list[float]], output_path: Path) -> None:
    """Write per-metric summary statistics across evaluated images."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["metric", "mean", "median", "min", "max", "std"],
        )
        writer.writeheader()
        for metric, values in agg.items():
            value_array = np.asarray(values, dtype=float)
            writer.writerow(
                {
                    "metric": metric,
                    "mean": float(np.mean(value_array)),
                    "median": float(np.median(value_array)),
                    "min": float(np.min(value_array)),
                    "max": float(np.max(value_array)),
                    "std": float(np.std(value_array)),
                }
            )


def _instance_failure_records(
    gt: np.ndarray,
    pred: np.ndarray,
    iou_threshold: float,
    image_name: str,
) -> tuple[list[dict[str, float | str | bool]], int]:
    """Return GT-instance morphology and matching results for failure analysis."""
    matches, inter, gt_labels_all, pred_labels_all = _match_instances(
        gt, pred, iou_threshold
    )
    n_labels = len(gt_labels_all)
    flat_label_indices = np.searchsorted(gt_labels_all, gt.ravel())
    foreground = flat_label_indices != 0
    pixel_indices = np.flatnonzero(foreground)
    row_coords, col_coords = np.divmod(pixel_indices, gt.shape[1])
    instance_indices = flat_label_indices[foreground]

    areas = np.bincount(instance_indices, minlength=n_labels).astype(float)
    sum_rows = np.bincount(
        instance_indices, weights=row_coords, minlength=n_labels
    )
    sum_cols = np.bincount(
        instance_indices, weights=col_coords, minlength=n_labels
    )
    sum_rows_sq = np.bincount(
        instance_indices, weights=row_coords ** 2, minlength=n_labels
    )
    sum_cols_sq = np.bincount(
        instance_indices, weights=col_coords ** 2, minlength=n_labels
    )
    sum_row_cols = np.bincount(
        instance_indices, weights=row_coords * col_coords, minlength=n_labels
    )

    matched_by_gt = {gt_idx: (pred_idx, iou) for gt_idx, pred_idx, iou in matches}
    records: list[dict[str, float | str | bool]] = []
    for gt_idx in range(1, n_labels):
        area = areas[gt_idx]
        mean_row = sum_rows[gt_idx] / area
        mean_col = sum_cols[gt_idx] / area
        var_row = max(sum_rows_sq[gt_idx] / area - mean_row ** 2, 0.0)
        var_col = max(sum_cols_sq[gt_idx] / area - mean_col ** 2, 0.0)
        covariance = sum_row_cols[gt_idx] / area - mean_row * mean_col
        eig_discriminant = np.sqrt((var_row - var_col) ** 2 + 4 * covariance ** 2)
        major_eigenvalue = (var_row + var_col + eig_discriminant) / 2
        minor_eigenvalue = (var_row + var_col - eig_discriminant) / 2
        eccentricity = (
            float(np.sqrt(max(0.0, 1 - minor_eigenvalue / major_eigenvalue)))
            if major_eigenvalue > 0
            else 0.0
        )

        matched = gt_idx in matched_by_gt
        record: dict[str, float | str | bool] = {
            "image": image_name,
            "area_px": float(area),
            "eccentricity": eccentricity,
            "matched": matched,
        }
        if matched:
            pred_idx, iou = matched_by_gt[gt_idx]
            intersection = inter[gt_idx, pred_idx]
            dice = (2 * intersection) / (areas[gt_idx] + inter[:, pred_idx].sum())
            record.update({"iou": float(iou), "dice": float(dice)})
        records.append(record)

    false_positives = len(pred_labels_all[pred_labels_all != 0]) - len(matches)
    return records, false_positives


def _summary_stats(values: list[float]) -> dict[str, float | str]:
    """Return CSV-safe summary statistics, leaving unavailable values blank."""
    if not values:
        return {"mean": "", "median": "", "min": "", "max": "", "std": ""}

    value_array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(value_array)),
        "median": float(np.median(value_array)),
        "min": float(np.min(value_array)),
        "max": float(np.max(value_array)),
        "std": float(np.std(value_array)),
    }


def _write_failure_mode_csv(
    records: list[dict[str, float | str | bool]],
    false_positives: int,
    output_path: Path,
) -> None:
    """Write performance metrics stratified by GT cell size and eccentricity."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    areas = np.asarray([float(record["area_px"]) for record in records])
    size_lower, size_upper = (
        np.quantile(areas, [1 / 3, 2 / 3]) if len(areas) else (0, 0)
    )

    def size_class(area: float) -> str:
        if area <= size_lower:
            return "small"
        if area <= size_upper:
            return "medium"
        return "large"

    def eccentricity_class(eccentricity: float) -> str:
        if eccentricity < 0.5:
            return "round"
        if eccentricity < 0.8:
            return "intermediate"
        return "elongated"

    class_definitions = {
        ("size", "small"): f"area_px <= {size_lower:.2f}",
        ("size", "medium"): f"{size_lower:.2f} < area_px <= {size_upper:.2f}",
        ("size", "large"): f"area_px > {size_upper:.2f}",
        ("eccentricity", "round"): "eccentricity < 0.5",
        ("eccentricity", "intermediate"): "0.5 <= eccentricity < 0.8",
        ("eccentricity", "elongated"): "eccentricity >= 0.8",
    }

    groups: dict[tuple[str, str], list[dict[str, float | str | bool]]] = {}
    for record in records:
        group_keys = (
            ("size", size_class(float(record["area_px"]))),
            ("eccentricity", eccentricity_class(float(record["eccentricity"]))),
        )
        for group_key in group_keys:
            groups.setdefault(group_key, []).append(record)

    fieldnames = [
        "stratifier", "class", "class_definition", "n_ground_truth", "n_matched", "n_missed",
        "recall", "iou_mean", "iou_median", "iou_min", "iou_max", "iou_std",
        "dice_mean", "dice_median", "dice_min", "dice_max", "dice_std",
        "global_false_positives",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for (stratifier, class_name), group_records in groups.items():
            matched_records = [record for record in group_records if record["matched"]]
            iou_stats = _summary_stats(
                [float(record["iou"]) for record in matched_records]
            )
            dice_stats = _summary_stats(
                [float(record["dice"]) for record in matched_records]
            )
            writer.writerow(
                {
                    "stratifier": stratifier,
                    "class": class_name,
                    "class_definition": class_definitions[(stratifier, class_name)],
                    "n_ground_truth": len(group_records),
                    "n_matched": len(matched_records),
                    "n_missed": len(group_records) - len(matched_records),
                    "recall": len(matched_records) / len(group_records),
                    **{f"iou_{key}": value for key, value in iou_stats.items()},
                    **{f"dice_{key}": value for key, value in dice_stats.items()},
                    "global_false_positives": false_positives,
                }
            )


def evaluate_masks(
    gt_dir: Path,
    pred_dir: Path,
    iou_threshold: float,
    per_image: bool,
    summary_csv: Path,
    failure_mode_csv: Path,
) -> None:
    gt_files = _collect_files(gt_dir)
    if not gt_files:
        raise FileNotFoundError(f"No .tif/.tiff files found in {gt_dir}")

    per_image_rows: list[tuple[str, dict[str, float]]] = []
    agg = {k: [] for k in ("iou", "dice", "precision", "recall", "f1", "mae")}
    failure_records: list[dict[str, float | str | bool]] = []
    false_positives = 0

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
        image_records, image_false_positives = _instance_failure_records(
            gt, pred, iou_threshold, gt_path.name
        )
        per_image_rows.append((gt_path.name, m))
        failure_records.extend(image_records)
        false_positives += image_false_positives
        for k, v in m.items():
            agg[k].append(v)

    means = {k: float(np.mean(v)) for k, v in agg.items()}
    _write_summary_csv(agg, summary_csv)
    _write_failure_mode_csv(failure_records, false_positives, failure_mode_csv)

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
    print(f"Summary CSV saved to {summary_csv}")
    print(f"Failure-mode CSV saved to {failure_mode_csv}")


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
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=Path("mask_evaluation_summary.csv"),
        help="Path for per-metric summary statistics CSV.",
    )
    parser.add_argument(
        "--failure_mode_csv",
        type=Path,
        default=Path("mask_failure_modes.csv"),
        help=(
            "Path for size- and eccentricity-stratified instance metrics CSV. "
            "Size classes use dataset area tertiles; eccentricity classes are "
            "round (<0.5), intermediate (<0.8), and elongated (>=0.8)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    evaluate_masks(
        args.gt_dir,
        args.pred_dir,
        args.iou_threshold,
        args.per_image,
        args.summary_csv,
        args.failure_mode_csv,
    )


if __name__ == "__main__":
    main()
