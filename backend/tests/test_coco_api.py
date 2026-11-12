import base64
import csv
import json
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def create_coco_fixture(tmp_path, monkeypatch) -> bytes:
    subset_dir = tmp_path / "coco_subset"
    annotations_dir = subset_dir / "annotations"
    images_dir = subset_dir / "images"
    gt_dir = tmp_path / "gt_masks" / "7"
    annotations_dir.mkdir(parents=True)
    images_dir.mkdir()
    gt_dir.mkdir(parents=True)

    subset = {
        "images": [
            {"id": 7, "file_name": "sample.jpg", "width": 4, "height": 4}
        ],
        "annotations": [
            {
                "id": 12,
                "image_id": 7,
                "category_id": 3,
                "area": 4,
                "bbox": [1, 1, 2, 2],
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 3, "name": "object"}],
    }
    (annotations_dir / "subset.json").write_text(json.dumps(subset), encoding="utf-8")
    Image.new("RGB", (4, 4), color="white").save(
        images_dir / "sample.jpg", format="JPEG")

    mask = Image.new("L", (4, 4))
    for x, y in ((1, 1), (2, 1), (1, 2), (2, 2)):
        mask.putpixel((x, y), 255)
    mask.save(gt_dir / "12.png", format="PNG")

    output = BytesIO()
    mask.save(output, format="PNG")
    monkeypatch.setenv("COCO_SUBSET_JSON", str(
        annotations_dir / "subset.json"))
    monkeypatch.setenv("COCO_GT_DIR", str(tmp_path / "gt_masks"))
    monkeypatch.setenv("COCO_EVALUATION_DIR", str(tmp_path / "evaluation"))
    return output.getvalue()


def test_coco_image_listing_and_serving(tmp_path, monkeypatch) -> None:
    create_coco_fixture(tmp_path, monkeypatch)

    listing = client.get("/api/coco/images")
    image = client.get("/api/coco/images/7")

    assert listing.status_code == 200
    assert listing.json()["images"][0]["annotations"] == [
        {
            "id": 12,
            "category_id": 3,
            "category_name": "object",
            "area": 4,
            "bbox": [1.0, 1.0, 2.0, 2.0],
            "label": "object #1",
            "centroid": {"x": 1.5, "y": 1.5},
        }
    ]
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/jpeg"
    assert image.content.startswith(b"\xff\xd8")


def test_save_annotation_computes_metrics_and_writes_manifest(tmp_path, monkeypatch) -> None:
    mask_png = create_coco_fixture(tmp_path, monkeypatch)
    payload = {
        "image_id": 7,
        "annotation_id": 12,
        "elapsed_seconds": 18.5,
        "mask_png_base64": base64.b64encode(mask_png).decode("ascii"),
        "prompt": {"points": [{"x": 1, "y": 1, "label": 1}]},
    }

    response = client.post("/api/annotations", json=payload)

    assert response.status_code == 200
    assert response.json()["iou"] == 1
    assert response.json()["dice"] == 1
    assert response.json()["elapsed_seconds"] == 18.5
    manifest_path = tmp_path / "evaluation" / "predictions.csv"
    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        rows = list(csv.DictReader(manifest_file))
    assert rows == [
        {
            "ann_id": "12",
            "prediction_path": "predictions/12.png",
            "elapsed_seconds": "18.5",
        }
    ]
    assert (tmp_path / "evaluation/records/12.json").is_file()
    assert (tmp_path / "evaluation/predictions/12.png").is_file()


def test_annotations_of_one_category_get_distinct_labels_and_centroids(
    tmp_path, monkeypatch
) -> None:
    subset_dir = tmp_path / "coco_subset"
    annotations_dir = subset_dir / "annotations"
    annotations_dir.mkdir(parents=True)
    gt_dir = tmp_path / "gt_masks" / "9"
    gt_dir.mkdir(parents=True)

    subset = {
        "images": [{"id": 9, "file_name": "people.jpg", "width": 12, "height": 4}],
        "annotations": [
            {"id": 103, "image_id": 9, "category_id": 1, "area": 4, "iscrowd": 0},
            {"id": 101, "image_id": 9, "category_id": 1, "area": 4, "iscrowd": 0},
            {"id": 102, "image_id": 9, "category_id": 1, "area": 4, "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "person"}],
    }
    (annotations_dir / "subset.json").write_text(json.dumps(subset), encoding="utf-8")

    # One solid 2x2 block per annotation, at increasing x.
    for annotation_id, x_start in ((101, 0), (102, 5), (103, 10)):
        mask = Image.new("L", (12, 4))
        for x in range(x_start, x_start + 2):
            for y in (1, 2):
                mask.putpixel((x, y), 255)
        mask.save(gt_dir / f"{annotation_id}.png", format="PNG")

    monkeypatch.setenv("COCO_SUBSET_JSON", str(annotations_dir / "subset.json"))
    monkeypatch.setenv("COCO_GT_DIR", str(tmp_path / "gt_masks"))
    monkeypatch.setenv("COCO_EVALUATION_DIR", str(tmp_path / "evaluation"))

    annotations = client.get("/api/coco/images").json()["images"][0]["annotations"]

    assert [annotation["id"] for annotation in annotations] == [101, 102, 103]
    assert [annotation["label"] for annotation in annotations] == [
        "person #1",
        "person #2",
        "person #3",
    ]
    assert [annotation["centroid"] for annotation in annotations] == [
        {"x": 0.5, "y": 1.5},
        {"x": 5.5, "y": 1.5},
        {"x": 10.5, "y": 1.5},
    ]


def test_annotation_without_ground_truth_mask_has_no_centroid(tmp_path, monkeypatch) -> None:
    subset_dir = tmp_path / "coco_subset"
    annotations_dir = subset_dir / "annotations"
    annotations_dir.mkdir(parents=True)
    subset = {
        "images": [{"id": 4, "file_name": "x.jpg", "width": 4, "height": 4}],
        "annotations": [
            {"id": 55, "image_id": 4, "category_id": 1, "area": 4, "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "person"}],
    }
    (annotations_dir / "subset.json").write_text(json.dumps(subset), encoding="utf-8")
    monkeypatch.setenv("COCO_SUBSET_JSON", str(annotations_dir / "subset.json"))
    monkeypatch.setenv("COCO_GT_DIR", str(tmp_path / "gt_masks"))
    monkeypatch.setenv("COCO_EVALUATION_DIR", str(tmp_path / "evaluation"))

    annotation = client.get("/api/coco/images").json()["images"][0]["annotations"][0]

    assert annotation["label"] == "person #1"
    assert annotation["centroid"] is None


def test_ground_truth_mask_is_served_for_its_annotation(tmp_path, monkeypatch) -> None:
    create_coco_fixture(tmp_path, monkeypatch)

    response = client.get("/api/coco/images/7/masks/12")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")


def test_ground_truth_mask_rejects_image_annotation_mismatch(tmp_path, monkeypatch) -> None:
    create_coco_fixture(tmp_path, monkeypatch)

    # Annotation 12 belongs to image 7, so there is no mask at this path.
    response = client.get("/api/coco/images/8/masks/12")

    assert response.status_code == 404


def test_ground_truth_mask_reports_missing_mask(tmp_path, monkeypatch) -> None:
    subset_dir = tmp_path / "coco_subset"
    annotations_dir = subset_dir / "annotations"
    annotations_dir.mkdir(parents=True)
    subset = {
        "images": [{"id": 4, "file_name": "x.jpg", "width": 4, "height": 4}],
        "annotations": [
            {"id": 55, "image_id": 4, "category_id": 1, "area": 4, "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "person"}],
    }
    (annotations_dir / "subset.json").write_text(json.dumps(subset), encoding="utf-8")
    monkeypatch.setenv("COCO_SUBSET_JSON", str(annotations_dir / "subset.json"))
    monkeypatch.setenv("COCO_GT_DIR", str(tmp_path / "gt_masks"))
    monkeypatch.setenv("COCO_EVALUATION_DIR", str(tmp_path / "evaluation"))

    response = client.get("/api/coco/images/4/masks/55")

    assert response.status_code == 404
    assert "export_gt_masks" in response.json()["detail"]


def test_save_annotation_rejects_annotation_for_another_image(tmp_path, monkeypatch) -> None:
    mask_png = create_coco_fixture(tmp_path, monkeypatch)
    payload = {
        "image_id": 8,
        "annotation_id": 12,
        "elapsed_seconds": 1,
        "mask_png_base64": base64.b64encode(mask_png).decode("ascii"),
        "prompt": {"points": [{"x": 1, "y": 1, "label": 1}]},
    }

    response = client.post("/api/annotations", json=payload)

    assert response.status_code == 422


def test_save_polygon_scores_against_ground_truth_and_writes_its_own_directory(
    tmp_path, monkeypatch
) -> None:
    create_coco_fixture(tmp_path, monkeypatch)
    # The fixture's annotation 12 covers the 2x2 block at (1,1)-(2,2).
    payload = {
        "image_id": 7,
        "annotation_id": 12,
        "elapsed_seconds": 9.5,
        "points": [[1, 1], [3, 1], [3, 3], [1, 3]],
        "mode": "polygon",
    }

    response = client.post("/api/annotations/polygon", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["method"] == "polygon"
    assert body["mode"] == "polygon"
    assert body["drawn_points"] == 4
    assert body["outline"]["vertex_count"] == 4
    assert body["elapsed_seconds"] == 9.5
    assert 0 < body["iou"] <= 1

    # The polygon result lands beside the SAM2 directory, never inside it.
    polygon_dir = tmp_path / "evaluation_polygon"
    assert (polygon_dir / "predictions/12.png").is_file()
    assert (polygon_dir / "records/12.json").is_file()
    with (polygon_dir / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["ann_id"] for row in rows] == ["12"]
    assert not (tmp_path / "evaluation").exists(), "SAM2 results must stay untouched"


def test_save_polygon_simplifies_freehand_traces(tmp_path, monkeypatch) -> None:
    create_coco_fixture(tmp_path, monkeypatch)
    # A closed freehand trace around the same square, densely sampled on each edge.
    trace = []
    for step in range(20):
        trace.append([1 + 0.1 * step, 1.0])
        trace.append([3.0, 1 + 0.1 * step])
        trace.append([3 - 0.1 * step, 3.0])
        trace.append([1.0, 3 - 0.1 * step])
    payload = {
        "image_id": 7,
        "annotation_id": 12,
        "elapsed_seconds": 4.0,
        "points": trace,
        "mode": "freehand",
    }

    body = client.post("/api/annotations/polygon", json=payload).json()

    assert body["mode"] == "freehand"
    assert body["drawn_points"] == len(trace)
    assert body["outline"]["vertex_count"] < len(trace) / 4, "stored trace is simplified"
    # The fixture's ground truth is 4x4 px, so a one-pixel boundary difference
    # already costs a large share of IoU; the point is that the outline still
    # overlaps the target at all rather than being empty or inverted.
    assert body["iou"] > 0.3, "the simplified outline still covers the target"


def test_save_polygon_rejects_points_outside_the_image(tmp_path, monkeypatch) -> None:
    create_coco_fixture(tmp_path, monkeypatch)
    payload = {
        "image_id": 7,
        "annotation_id": 12,
        "elapsed_seconds": 1,
        "points": [[0, 0], [40, 0], [40, 40]],
        "mode": "polygon",
    }

    response = client.post("/api/annotations/polygon", json=payload)

    assert response.status_code == 422
    assert "outside" in response.json()["detail"]


def test_save_polygon_rejects_annotation_for_another_image(tmp_path, monkeypatch) -> None:
    create_coco_fixture(tmp_path, monkeypatch)
    payload = {
        "image_id": 8,
        "annotation_id": 12,
        "elapsed_seconds": 1,
        "points": [[0, 0], [3, 0], [3, 3]],
        "mode": "polygon",
    }

    response = client.post("/api/annotations/polygon", json=payload)

    assert response.status_code == 422


def test_polygon_parts_are_scored_as_their_union(tmp_path, monkeypatch) -> None:
    create_coco_fixture(tmp_path, monkeypatch)
    # The fixture's annotation 12 is the 2x2 block at (1,1)-(2,2). The first part
    # covers it; the second part is a disconnected piece elsewhere in the image.
    payload = {
        "image_id": 7,
        "annotation_id": 12,
        "elapsed_seconds": 7.0,
        "parts": [
            [[1, 1], [3, 1], [3, 3], [1, 3]],
            [[0, 0], [1, 0], [1, 1], [0, 1]],
        ],
        "mode": "polygon",
    }

    body = client.post("/api/annotations/polygon", json=payload).json()

    assert body["outline"]["part_count"] == 2
    assert body["drawn_points"] == 8
    # Both parts count towards the prediction: the extra piece sits outside the
    # ground truth, so it lowers IoU rather than being ignored.
    single = client.post(
        "/api/annotations/polygon",
        json={**payload, "parts": [payload["parts"][0]]},
    ).json()
    assert body["iou"] < single["iou"]


def test_polygon_rejects_empty_payload(tmp_path, monkeypatch) -> None:
    create_coco_fixture(tmp_path, monkeypatch)
    payload = {"image_id": 7, "annotation_id": 12, "elapsed_seconds": 1.0}

    response = client.post("/api/annotations/polygon", json=payload)

    assert response.status_code == 422
