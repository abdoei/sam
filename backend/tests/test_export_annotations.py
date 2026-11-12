import csv
import json
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as mask_utils

from app.coco import evaluation_dir
from scripts.export_annotations import _mask_to_rle, export


def _write_condition(tmp_path, method: str, mask: np.ndarray, elapsed: float) -> None:
    directory = evaluation_dir(method)
    (directory / "predictions").mkdir(parents=True, exist_ok=True)
    (directory / "records").mkdir(parents=True, exist_ok=True)
    ann_id, image_id = 12, 7
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(
        directory / "predictions" / f"{ann_id}.png"
    )
    (directory / "records" / f"{ann_id}.json").write_text(
        json.dumps(
            {
                "image_id": image_id,
                "ann_id": ann_id,
                "category_id": 3,
                "elapsed_seconds": elapsed,
                "iou": 0.75,
                "dice": 0.86,
                "prediction_path": f"predictions/{ann_id}.png",
                "prompts": {"points": [{"x": 1, "y": 1, "label": 1}], "box": None},
            }
        ),
        encoding="utf-8",
    )


def _subset(tmp_path, monkeypatch) -> None:
    subset_dir = tmp_path / "coco_subset"
    annotations_dir = subset_dir / "annotations"
    annotations_dir.mkdir(parents=True)
    subset = {
        "info": {},
        "licenses": [],
        "images": [{"id": 7, "file_name": "a.jpg", "width": 8, "height": 8}],
        "annotations": [
            {"id": 12, "image_id": 7, "category_id": 3, "area": 4, "iscrowd": 0}
        ],
        "categories": [{"id": 3, "name": "object"}],
    }
    (annotations_dir / "subset.json").write_text(json.dumps(subset), encoding="utf-8")
    monkeypatch.setenv("COCO_SUBSET_JSON", str(annotations_dir / "subset.json"))
    monkeypatch.setenv("COCO_GT_DIR", str(tmp_path / "gt_masks"))
    monkeypatch.setenv("COCO_EVALUATION_DIR", str(tmp_path / "evaluation"))


def _block(size: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((8, 8), dtype=bool),
        np.ones((8, 8), dtype=bool),
        _block(8, 1, 1, 3, 3),
        _block(8, 0, 0, 8, 2),
    ],
)
def test_rle_round_trips_through_pycocotools(mask) -> None:
    # The whole point of emitting COCO RLE is that other tools can read it back.
    rle = _mask_to_rle(mask)

    decoded = mask_utils.decode(rle).astype(bool)

    assert np.array_equal(decoded, mask)


def test_rle_round_trips_around_a_hole() -> None:
    mask = _block(12, 2, 2, 10, 10)
    mask[5:7, 5:7] = False

    decoded = mask_utils.decode(_mask_to_rle(mask)).astype(bool)

    assert np.array_equal(decoded, mask)


def test_export_writes_one_archive_per_condition(tmp_path, monkeypatch) -> None:
    _subset(tmp_path, monkeypatch)
    mask = _block(8, 1, 1, 3, 3)
    _write_condition(tmp_path, "sam2", mask, 5.0)
    _write_condition(tmp_path, "polygon", mask, 20.0)

    written = export(["sam2", "polygon"], tmp_path / "exports", "run")

    names = sorted(path.name for path in written)
    assert names == ["run-comparison.json", "run-polygon.zip", "run-sam2.zip"]

    with ZipFile(tmp_path / "exports" / "run-sam2.zip") as archive:
        contents = set(archive.namelist())
        assert contents == {
            "manifest.json",
            "predictions.json",
            "metrics.csv",
            "masks/12.png",
        }
        predictions = json.loads(archive.read("predictions.json"))
        manifest = json.loads(archive.read("manifest.json"))
        metrics = list(csv.DictReader(archive.read("metrics.csv").decode().splitlines()))

    assert predictions["annotations"][0]["id"] == 12
    assert predictions["annotations"][0]["iscrowd"] == 0
    assert predictions["images"][0]["id"] == 7
    assert "categories" in predictions and "licenses" in predictions
    # The segmentation must decode back to the mask that was measured.
    decoded = mask_utils.decode(predictions["annotations"][0]["segmentation"]).astype(bool)
    assert np.array_equal(decoded, mask)
    assert predictions["annotations"][0]["area"] == 4.0
    assert manifest["method"] == "sam2"
    assert manifest["annotation_count"] == 1
    assert manifest["summary"]["count"] == 1
    assert metrics[0]["elapsed_seconds"] == "5.0"

    comparison = json.loads((tmp_path / "exports" / "run-comparison.json").read_text())
    assert comparison["sam2"]["mean_iou"] == pytest.approx(0.75)
    assert comparison["polygon"]["mean_annotation_seconds"] == pytest.approx(20.0)


def test_export_refuses_a_condition_with_no_records(tmp_path, monkeypatch) -> None:
    _subset(tmp_path, monkeypatch)
    _write_condition(tmp_path, "sam2", _block(8, 1, 1, 3, 3), 5.0)

    with pytest.raises(FileNotFoundError, match="polygon"):
        export(["sam2", "polygon"], tmp_path / "exports", "run")


def test_export_rejects_a_mask_of_the_wrong_size(tmp_path, monkeypatch) -> None:
    _subset(tmp_path, monkeypatch)
    _write_condition(tmp_path, "sam2", np.zeros((4, 4), dtype=bool), 5.0)

    with pytest.raises(ValueError, match="but image 7 is"):
        export(["sam2"], tmp_path / "exports", "run")
