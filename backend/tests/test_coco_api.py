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
