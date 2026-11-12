"""Compare the SAM2 condition with the manual polygon baseline.

Both conditions are scored by the same evaluator against the same COCO ground truth,
so the only thing that differs is how the mask was produced. This script reports
quality and annotation time side by side, and restricts the timing comparison to
annotations both conditions covered: a mean taken over different sets of objects
would compare the objects as much as the methods.
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from app.coco import EVALUATION_METHODS, evaluation_dir
from scripts.evaluate_masks import evaluate_manifest

ROOT = Path(__file__).resolve().parents[2]


def _summary_for(method: str, gt_index: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    directory = evaluation_dir(method)
    manifest = directory / "predictions.csv"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"No {method} predictions at {manifest}. Annotate some objects in that mode first."
        )
    rows, summary = evaluate_manifest(manifest, gt_index)
    return summary, {row["ann_id"]: row for row in rows}


def _paired_summary(
    sam2: dict[int, dict[str, Any]],
    polygon: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    shared = sorted(set(sam2) & set(polygon))
    if not shared:
        return {"count": 0}
    sam2_time = [
        sam2[ann_id]["elapsed_seconds"]
        for ann_id in shared
        if sam2[ann_id]["elapsed_seconds"] is not None
    ]
    polygon_time = [
        polygon[ann_id]["elapsed_seconds"]
        for ann_id in shared
        if polygon[ann_id]["elapsed_seconds"] is not None
    ]
    timed = min(len(sam2_time), len(polygon_time))
    result: dict[str, Any] = {
        "count": len(shared),
        "mean_iou_sam2": statistics.fmean(sam2[ann_id]["iou"] for ann_id in shared),
        "mean_iou_polygon": statistics.fmean(polygon[ann_id]["iou"] for ann_id in shared),
        "mean_dice_sam2": statistics.fmean(sam2[ann_id]["dice"] for ann_id in shared),
        "mean_dice_polygon": statistics.fmean(polygon[ann_id]["dice"] for ann_id in shared),
        "timed_count": timed,
    }
    if timed:
        result["mean_seconds_sam2"] = statistics.fmean(sam2_time[:timed])
        result["mean_seconds_polygon"] = statistics.fmean(polygon_time[:timed])
        result["mean_seconds_saved"] = (
            result["mean_seconds_polygon"] - result["mean_seconds_sam2"]
        )
        if result["mean_seconds_polygon"] > 0:
            result["time_reduction_percent"] = (
                result["mean_seconds_saved"] / result["mean_seconds_polygon"] * 100
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare SAM2 annotation against the manual polygon baseline."
    )
    parser.add_argument(
        "--gt-index",
        type=Path,
        default=ROOT / "data/gt_masks/gt_index.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/comparison.json",
        help="Where to write the combined summary",
    )
    args = parser.parse_args()

    summaries: dict[str, Any] = {}
    rows_by_method: dict[str, dict[int, dict[str, Any]]] = {}
    for method in sorted(EVALUATION_METHODS):
        summaries[method], rows_by_method[method] = _summary_for(method, args.gt_index)

    comparison = {
        "methods": summaries,
        "paired": _paired_summary(rows_by_method["sam2"], rows_by_method["polygon"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
