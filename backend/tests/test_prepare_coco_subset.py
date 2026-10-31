import pytest

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
