import csv
import json
import sys

import pytest
from PIL import Image

from scripts.export_gt_masks import main as export_gt_masks
from scripts.prepare_coco_subset import main as prepare_subset
from scripts.prepare_coco_subset import select_image_ids


def test_select_image_ids_is_seeded_and_ignores_crowd_annotations() -> None:
    images = [{"id": image_id} for image_id in range(1, 6)]
    annotations = [
        {"image_id": 1, "iscrowd": 0},
        {"image_id": 2, "iscrowd": 1},
        {"image_id": 3, "iscrowd": 0},
        {"image_id": 4, "iscrowd": 0},
        {"image_id": 4, "iscrowd": 0},
        {"image_id": 5, "iscrowd": 0},
    ]

    selected = select_image_ids(images, annotations, 2, 42, 1, 1)

    assert selected == select_image_ids(images, annotations, 2, 42, 1, 1)
    assert selected == sorted(selected)
    assert 2 not in selected
    assert len(selected) == 2


def test_select_image_ids_rejects_impossible_sample_size() -> None:
    with pytest.raises(ValueError, match="cannot select"):
        select_image_ids([{"id": 1}], [], 1, 42, 1, 8)


def test_select_image_ids_rejects_invalid_instance_range() -> None:
    with pytest.raises(ValueError, match="instance limits"):
        select_image_ids([], [], 1, 42, 3, 2)


def test_subset_and_ground_truth_export_scripts_run_end_to_end(tmp_path, monkeypatch) -> None:
    source_images = tmp_path / "raw" / "val2017"
    source_images.mkdir(parents=True)
    Image.new("RGB", (4, 4), color="white").save(source_images / "sample.png")
    annotations_path = tmp_path / "raw" / "instances.json"
    annotations_path.write_text(
        json.dumps(
            {
                "info": {},
                "licenses": [],
                "images": [
                    {
                        "id": 7,
                        "file_name": "sample.png",
                        "width": 4,
                        "height": 4,
                    }
                ],
                "annotations": [
                    {
                        "id": 12,
                        "image_id": 7,
                        "category_id": 3,
                        "segmentation": [[1, 1, 3, 1, 3, 3, 1, 3]],
                        "area": 4,
                        "bbox": [1, 1, 2, 2],
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 3, "name": "object", "supercategory": "object"}],
            }
        ),
        encoding="utf-8",
    )
    subset_dir = tmp_path / "subset"
    ids_path = tmp_path / "selected_image_ids.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_coco_subset.py",
            "--annotations",
            str(annotations_path),
            "--images",
            str(source_images),
            "--output",
            str(subset_dir),
            "--ids-output",
            str(ids_path),
            "--count",
            "1",
        ],
    )

    prepare_subset()

    assert json.loads(ids_path.read_text(encoding="utf-8")) == [7]
    assert (subset_dir / "images" / "sample.png").is_file()

    mask_dir = tmp_path / "masks"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_gt_masks.py",
            "--annotations",
            str(subset_dir / "annotations" / "subset.json"),
            "--output",
            str(mask_dir),
        ],
    )

    export_gt_masks()

    assert (mask_dir / "7" / "12.png").is_file()
    with (mask_dir / "gt_index.csv").open(newline="", encoding="utf-8") as index_file:
        rows = list(csv.DictReader(index_file))
    assert len(rows) == 1
    assert rows[0]["ann_id"] == "12"
