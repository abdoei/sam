import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.metrics import binary_mask_metrics

ROOT = Path(__file__).resolve().parents[2]


def read_binary_mask(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Mask file not found: {path}")
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 0


def read_ground_truth_index(index_path: Path) -> dict[int, dict[str, Any]]:
    with index_path.open(newline="", encoding="utf-8") as index_file:
        reader = csv.DictReader(index_file)
        required = {"ann_id", "image_id", "category_id", "mask_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Ground-truth index is missing columns: {', '.join(sorted(missing))}")

        ground_truth: dict[int, dict[str, Any]] = {}
        for row in reader:
            ann_id = int(row["ann_id"])
            if ann_id in ground_truth:
                raise ValueError(
                    f"Duplicate annotation ID in ground-truth index: {ann_id}")
            ground_truth[ann_id] = {
                "image_id": int(row["image_id"]),
                "category_id": int(row["category_id"]),
                "path": index_path.parent / row["mask_path"],
            }
    return ground_truth


def evaluate_manifest(
    manifest_path: Path,
    ground_truth_index_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, int | float | None]]:
    ground_truth = read_ground_truth_index(ground_truth_index_path)
    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        reader = csv.DictReader(manifest_file)
        required = {"ann_id", "prediction_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            missing_columns = ", ".join(sorted(missing))
            raise ValueError(
                f"Prediction manifest is missing columns: {missing_columns}")

        result_rows: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        for row in reader:
            ann_id = int(row["ann_id"])
            if ann_id < 1:
                raise ValueError(f"Annotation ID must be positive: {ann_id}")
            if ann_id in seen_ids:
                raise ValueError(
                    f"Duplicate annotation ID in prediction manifest: {ann_id}")
            if ann_id not in ground_truth:
                raise ValueError(
                    f"Annotation ID {ann_id} is absent from the ground-truth index")
            seen_ids.add(ann_id)

            prediction_path = Path(row["prediction_path"])
            if not prediction_path.is_absolute():
                prediction_path = manifest_path.parent / prediction_path
            target = ground_truth[ann_id]
            metrics = binary_mask_metrics(
                read_binary_mask(prediction_path),
                read_binary_mask(target["path"]),
            )
            elapsed_text = row.get("elapsed_seconds", "").strip()
            elapsed_seconds = float(elapsed_text) if elapsed_text else None
            if elapsed_seconds is not None and (
                not math.isfinite(elapsed_seconds) or elapsed_seconds < 0
            ):
                raise ValueError(
                    f"Annotation time must be finite and non-negative for ID {ann_id}")
            result_rows.append(
                {
                    "ann_id": ann_id,
                    "image_id": target["image_id"],
                    "category_id": target["category_id"],
                    **metrics,
                    "elapsed_seconds": elapsed_seconds,
                }
            )

    if not result_rows:
        raise ValueError("Prediction manifest contains no annotations")

    timed_rows = [
        row["elapsed_seconds"] for row in result_rows if row["elapsed_seconds"] is not None
    ]
    summary: dict[str, int | float | None] = {
        "count": len(result_rows),
        "mean_iou": statistics.fmean(row["iou"] for row in result_rows),
        "mean_dice": statistics.fmean(row["dice"] for row in result_rows),
        "timed_count": len(timed_rows),
        "mean_annotation_seconds": statistics.fmean(timed_rows) if timed_rows else None,
        "median_annotation_seconds": statistics.median(timed_rows) if timed_rows else None,
    }
    return result_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate downloaded binary masks against exported COCO instance masks."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT / "data/evaluation/predictions.csv",
        help="CSV columns: ann_id,prediction_path; optional elapsed_seconds",
    )
    parser.add_argument(
        "--gt-index",
        type=Path,
        default=ROOT / "data/gt_masks/gt_index.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/evaluation/per_instance_metrics.csv",
    )
    args = parser.parse_args()

    result_rows, summary = evaluate_manifest(args.predictions, args.gt_index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "ann_id",
                "image_id",
                "category_id",
                "iou",
                "dice",
                "elapsed_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(result_rows)

    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(
        summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Per-instance results: {args.output}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
