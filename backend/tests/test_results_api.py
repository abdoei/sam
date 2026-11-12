import json
from zipfile import ZipFile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def _write_results(tmp_path, method: str, rows: list[dict]) -> None:
    """Write records the way the app does, into a given evaluation directory."""
    directory = tmp_path / ("evaluation_polygon" if method == "polygon" else "evaluation")
    (directory / "records").mkdir(parents=True, exist_ok=True)
    (directory / "predictions").mkdir(parents=True, exist_ok=True)
    # A real 8x8 mask: the exporter decodes these, so a stub header would fail there
    # rather than exercising the endpoint.
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    for row in rows:
        ann_id = row["ann_id"]
        (directory / "records" / f"{ann_id}.json").write_text(
            json.dumps(row), encoding="utf-8"
        )
        Image.fromarray(mask, mode="L").save(directory / "predictions" / f"{ann_id}.png")


def _env(tmp_path, monkeypatch) -> None:
    subset_dir = tmp_path / "coco_subset" / "annotations"
    subset_dir.mkdir(parents=True)
    (subset_dir / "subset.json").write_text(
        json.dumps(
            {
                "info": {},
                "licenses": [],
                "images": [
                    {"id": 7, "file_name": "a.jpg", "width": 8, "height": 8},
                    {"id": 8, "file_name": "b.jpg", "width": 8, "height": 8},
                ],
                "annotations": [],
                "categories": [{"id": 3, "name": "chair"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("COCO_SUBSET_JSON", str(subset_dir / "subset.json"))
    monkeypatch.setenv("COCO_EVALUATION_DIR", str(tmp_path / "evaluation"))
    monkeypatch.setenv("COCO_GT_DIR", str(tmp_path / "gt_masks"))


def _record(ann_id: int, image_id: int, iou: float, seconds: float | None) -> dict:
    return {
        "ann_id": ann_id,
        "image_id": image_id,
        "category_id": 3,
        "iou": iou,
        "dice": iou,
        "elapsed_seconds": seconds,
        "prediction_path": f"predictions/{ann_id}.png",
    }


def test_results_are_scoped_to_the_requested_image(tmp_path, monkeypatch) -> None:
    _env(tmp_path, monkeypatch)
    _write_results(
        tmp_path,
        "sam2",
        [_record(1, 7, 0.8, 10.0), _record(2, 8, 0.6, 20.0)],
    )

    body = client.get("/api/results", params={"image_id": 7}).json()

    assert [row["ann_id"] for row in body["methods"][0]["results"]] == [1]
    assert body["totals"]["scope"]["count"] == 1
    # Everything is still counted, so the panel can show both scopes.
    assert body["totals"]["all"]["count"] == 2


def test_results_report_means_and_handle_missing_times(tmp_path, monkeypatch) -> None:
    _env(tmp_path, monkeypatch)
    _write_results(
        tmp_path,
        "sam2",
        [_record(1, 7, 0.8, 10.0), _record(2, 7, 0.4, None)],
    )

    totals = client.get("/api/results", params={"image_id": 7}).json()["totals"]["scope"]

    assert totals["count"] == 2
    assert totals["mean_iou"] == pytest.approx(0.6)
    assert totals["timed_count"] == 1
    assert totals["mean_seconds"] == pytest.approx(10.0)


def test_results_include_both_conditions(tmp_path, monkeypatch) -> None:
    _env(tmp_path, monkeypatch)
    _write_results(tmp_path, "sam2", [_record(1, 7, 0.8, 10.0)])
    _write_results(tmp_path, "polygon", [_record(1, 7, 0.5, 30.0)])

    body = client.get("/api/results", params={"image_id": 7}).json()

    assert {entry["method"] for entry in body["methods"]} == {"sam2", "polygon"}


def test_results_are_empty_but_wellformed_when_nothing_is_annotated(
    tmp_path, monkeypatch
) -> None:
    _env(tmp_path, monkeypatch)

    body = client.get("/api/results").json()

    assert body["methods"] == []
    assert body["totals"]["scope"] == {"count": 0}
    assert body["errors"] == []


def test_export_serves_a_zip_matching_the_cli(tmp_path, monkeypatch) -> None:
    _env(tmp_path, monkeypatch)
    _write_results(tmp_path, "sam2", [_record(1, 7, 0.8, 10.0)])

    response = client.get("/api/export", params={"methods": "sam2"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert "attachment" in response.headers["content-disposition"]
    archive_path = tmp_path / "served.zip"
    archive_path.write_bytes(response.content)
    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        assert {"manifest.json", "predictions.json", "metrics.csv"} <= names
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["method"] == "sam2"
    assert manifest["annotation_count"] == 1


def test_export_bundles_both_conditions_into_one_download(tmp_path, monkeypatch) -> None:
    _env(tmp_path, monkeypatch)
    _write_results(tmp_path, "sam2", [_record(1, 7, 0.8, 10.0)])
    _write_results(tmp_path, "polygon", [_record(1, 7, 0.5, 30.0)])

    response = client.get("/api/export")

    assert response.status_code == 200
    archive_path = tmp_path / "bundle.zip"
    archive_path.write_bytes(response.content)
    with ZipFile(archive_path) as archive:
        names = archive.namelist()
    # One download containing both per-condition archives.
    assert any(name.endswith("sam2.zip") for name in names)
    assert any(name.endswith("polygon.zip") for name in names)


def test_export_refuses_when_a_condition_has_no_results(tmp_path, monkeypatch) -> None:
    _env(tmp_path, monkeypatch)
    _write_results(tmp_path, "sam2", [_record(1, 7, 0.8, 10.0)])

    response = client.get("/api/export", params={"methods": "sam2,polygon"})

    assert response.status_code == 409
    assert "polygon" in response.json()["detail"]


def test_export_rejects_an_unknown_method(tmp_path, monkeypatch) -> None:
    _env(tmp_path, monkeypatch)

    response = client.get("/api/export", params={"methods": "magic"})

    assert response.status_code == 422
