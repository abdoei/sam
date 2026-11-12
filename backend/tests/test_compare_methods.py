import csv

import pytest

from app.coco import evaluation_dir
from scripts.compare_methods import _paired_summary
from scripts.compare_methods import main as compare_main


def _write_condition(root, method: str, rows: list[tuple[int, float, float]]) -> None:
    """Write a pretending manifest for one method: (ann_id, iou, seconds)."""
    directory = evaluation_dir(method)
    (directory / "predictions").mkdir(parents=True, exist_ok=True)
    with (directory / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["ann_id", "prediction_path", "elapsed_seconds"]
        )
        writer.writeheader()
        for ann_id, _iou, seconds in rows:
            writer.writerow(
                {
                    "ann_id": ann_id,
                    "prediction_path": f"predictions/{ann_id}.png",
                    "elapsed_seconds": seconds,
                }
            )


def test_paired_summary_only_uses_annotations_both_methods_cover() -> None:
    sam2 = {
        1: {"iou": 0.9, "dice": 0.95, "elapsed_seconds": 4.0},
        2: {"iou": 0.8, "dice": 0.9, "elapsed_seconds": 6.0},
        3: {"iou": 0.7, "dice": 0.8, "elapsed_seconds": 5.0},
    }
    polygon = {
        1: {"iou": 0.7, "dice": 0.8, "elapsed_seconds": 20.0},
        2: {"iou": 0.6, "dice": 0.75, "elapsed_seconds": 30.0},
        4: {"iou": 0.5, "dice": 0.6, "elapsed_seconds": 25.0},
    }

    paired = _paired_summary(sam2, polygon)

    assert paired["count"] == 2, "annotation 3 and 4 are not comparable"
    assert paired["mean_iou_sam2"] == pytest.approx(0.85)
    assert paired["mean_iou_polygon"] == pytest.approx(0.65)
    assert paired["mean_seconds_saved"] == pytest.approx(20.0)
    assert paired["time_reduction_percent"] == pytest.approx(80.0)


def test_paired_summary_reports_no_overlap_honestly() -> None:
    paired = _paired_summary(
        {1: {"iou": 0.9, "dice": 0.9, "elapsed_seconds": 1.0}},
        {2: {"iou": 0.9, "dice": 0.9, "elapsed_seconds": 2.0}},
    )

    assert paired == {"count": 0}


def test_compare_main_fails_clearly_when_a_condition_is_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("COCO_EVALUATION_DIR", str(tmp_path / "evaluation"))
    monkeypatch.setenv("COCO_GT_DIR", str(tmp_path / "gt_masks"))
    index = tmp_path / "gt_index.csv"
    index.write_text("ann_id,image_id,category_id,mask_path\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="No .* predictions"):
        monkeypatch.setattr(
            "sys.argv", ["compare", "--gt-index", str(index), "--output", str(tmp_path / "o.json")]
        )
        compare_main()
