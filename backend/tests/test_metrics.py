import csv

import numpy as np
import pytest
from PIL import Image

from app.metrics import binary_mask_metrics
from scripts.evaluate_masks import evaluate_manifest


def test_binary_mask_metrics_reports_iou_and_dice() -> None:
    prediction = np.array([[1, 1], [0, 0]], dtype=bool)
    ground_truth = np.array([[1, 0], [1, 0]], dtype=bool)

    assert binary_mask_metrics(prediction, ground_truth) == {
        "iou": 1 / 3, "dice": 0.5}


def test_binary_mask_metrics_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="dimensions do not match"):
        binary_mask_metrics(np.zeros((2, 2)), np.zeros((3, 2)))


def test_evaluate_manifest_joins_coco_annotation_ids(tmp_path) -> None:
    ground_truth_path = tmp_path / "gt" / "123.png"
    prediction_path = tmp_path / "pred.png"
    ground_truth_path.parent.mkdir()
    mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    Image.fromarray(mask).save(ground_truth_path)
    Image.fromarray(mask).save(prediction_path)

    index_path = tmp_path / "gt" / "gt_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as index_file:
        writer = csv.DictWriter(
            index_file,
            fieldnames=["ann_id", "image_id",
                        "category_id", "area", "mask_path"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "ann_id": 123,
                "image_id": 45,
                "category_id": 7,
                "area": 2,
                "mask_path": "123.png",
            }
        )

    manifest_path = tmp_path / "predictions.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=["ann_id", "prediction_path", "elapsed_seconds"],
        )
        writer.writeheader()
        writer.writerow(
            {"ann_id": 123, "prediction_path": "pred.png", "elapsed_seconds": "18.5"}
        )

    rows, summary = evaluate_manifest(manifest_path, index_path)

    assert rows[0]["ann_id"] == 123
    assert rows[0]["image_id"] == 45
    assert rows[0]["iou"] == 1.0
    assert summary["mean_dice"] == 1.0
    assert summary["mean_annotation_seconds"] == 18.5
