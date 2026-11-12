import csv
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.report import paired_comparison, summarise
from scripts.results_dataset import (
    load_archive,
    load_directory,
    merge,
    to_rows,
)


def _write_records(directory: Path, records: list[dict]) -> Path:
    (directory / "records").mkdir(parents=True, exist_ok=True)
    for record in records:
        (directory / "records" / f"{record['ann_id']}.json").write_text(
            json.dumps(record), encoding="utf-8"
        )
    return directory


def _record(ann_id: int, iou: float, dice: float, seconds: float | None, image_id: int = 1):
    return {
        "ann_id": ann_id,
        "image_id": image_id,
        "category_id": 3,
        "iou": iou,
        "dice": dice,
        "elapsed_seconds": seconds,
    }


def test_load_directory_recovers_the_method_from_its_name(tmp_path) -> None:
    _write_records(tmp_path / "evaluation", [_record(1, 0.9, 0.95, 4.0)])
    _write_records(tmp_path / "evaluation_polygon", [_record(1, 0.7, 0.8, 20.0)])

    assert {r.method for r in load_directory(tmp_path / "evaluation")} == {"sam2"}
    assert {r.method for r in load_directory(tmp_path / "evaluation_polygon")} == {"polygon"}


def test_merge_combines_sources_without_duplicating_identical_records(tmp_path) -> None:
    first = _write_records(tmp_path / "evaluation", [_record(1, 0.9, 0.95, 4.0)])
    # The same run, exported and re-read: identical measurements must collapse.
    copy = _write_records(tmp_path / "evaluation_copy", [_record(1, 0.9, 0.95, 4.0)])
    polygon = _write_records(tmp_path / "evaluation_polygon", [_record(1, 0.7, 0.8, 20.0)])

    dataset = merge([first, copy, polygon])

    assert len(dataset.results) == 2, "the duplicate sam2 record collapses"
    assert len(dataset.sources) == 3


def test_merge_refuses_conflicting_records_instead_of_picking_one(tmp_path) -> None:
    first = _write_records(tmp_path / "evaluation", [_record(1, 0.9, 0.95, 4.0)])
    conflicting = _write_records(tmp_path / "evaluation_rerun", [_record(1, 0.4, 0.5, 9.0)])

    with pytest.raises(ValueError, match="Conflicting records for sam2 annotation 1"):
        merge([first, conflicting])


def test_merge_rejects_a_record_missing_required_fields(tmp_path) -> None:
    broken = tmp_path / "evaluation"
    (broken / "records").mkdir(parents=True)
    (broken / "records" / "1.json").write_text(json.dumps({"ann_id": 1}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing image_id"):
        load_directory(broken)


def test_load_archive_reads_provenance_and_image_sizes(tmp_path) -> None:
    archive_path = tmp_path / "run-sam2.zip"
    predictions = {
        "images": [{"id": 5, "width": 64, "height": 48}],
        "categories": [{"id": 3, "name": "object"}],
        "annotations": [
            {
                "id": 9,
                "image_id": 5,
                "category_id": 3,
                "area": 120.0,
                "sam2_annotation": {
                    "method": "sam2",
                    "iou": 0.81,
                    "dice": 0.89,
                    "elapsed_seconds": 6.5,
                },
            }
        ],
    }
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("predictions.json", json.dumps(predictions))
        archive.writestr("manifest.json", json.dumps({"method": "sam2"}))

    results = load_archive(archive_path)

    assert len(results) == 1
    result = results[0]
    assert (result.method, result.ann_id, result.category_name) == ("sam2", 9, "object")
    assert result.iou == pytest.approx(0.81)
    assert result.image_width == 64
    assert result.mask_area == 120.0


def test_archive_without_predictions_is_rejected(tmp_path) -> None:
    path = tmp_path / "broken.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", "{}")

    with pytest.raises(ValueError, match="no predictions.json"):
        load_archive(path)


def _dataset_from(tmp_path, sam2: list[dict], polygon: list[dict]):
    sources = []
    if sam2:
        sources.append(_write_records(tmp_path / "evaluation", sam2))
    if polygon:
        sources.append(_write_records(tmp_path / "evaluation_polygon", polygon))
    return merge(sources)


def test_paired_comparison_only_counts_shared_annotations(tmp_path) -> None:
    dataset = _dataset_from(
        tmp_path,
        sam2=[_record(1, 0.9, 0.95, 4.0), _record(2, 0.8, 0.9, 6.0), _record(3, 0.7, 0.8, 5.0)],
        polygon=[_record(1, 0.7, 0.8, 20.0), _record(2, 0.6, 0.75, 30.0)],
    )

    pair = paired_comparison(dataset)

    assert pair is not None
    assert pair["count"] == 2
    assert pair["mean_iou_sam2"] == pytest.approx(0.85)
    assert pair["mean_iou_polygon"] == pytest.approx(0.65)
    assert pair["mean_seconds_saved"] == pytest.approx(20.0)
    assert pair["time_reduction_percent"] == pytest.approx(80.0)
    assert pair["speedup_factor"] == pytest.approx(5.0)
    assert pair["wins_sam2"] == 2


def test_paired_comparison_is_absent_without_both_conditions(tmp_path) -> None:
    dataset = _dataset_from(tmp_path, sam2=[_record(1, 0.9, 0.95, 4.0)], polygon=[])

    assert paired_comparison(dataset) is None
    assert "paired" not in summarise(dataset)


def test_paired_comparison_is_absent_when_no_annotation_is_shared(tmp_path) -> None:
    dataset = _dataset_from(
        tmp_path,
        sam2=[_record(1, 0.9, 0.95, 4.0)],
        polygon=[_record(2, 0.7, 0.8, 20.0)],
    )

    assert paired_comparison(dataset) is None


def test_summary_reports_distribution_and_per_category(tmp_path, monkeypatch) -> None:
    # Category names live in the subset, not in the records.
    subset_dir = tmp_path / "coco_subset" / "annotations"
    subset_dir.mkdir(parents=True)
    (subset_dir / "subset.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "a.jpg", "width": 8, "height": 8}],
                "annotations": [],
                "categories": [{"id": 3, "name": "chair"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("COCO_SUBSET_JSON", str(subset_dir / "subset.json"))

    dataset = _dataset_from(
        tmp_path,
        sam2=[_record(1, 0.9, 0.95, 4.0), _record(2, 0.5, 0.6, 8.0)],
        polygon=[],
    )

    summary = summarise(dataset)
    stats = summary["methods"]["sam2"]

    assert stats["iou"]["count"] == 2
    assert stats["iou"]["mean"] == pytest.approx(0.7)
    assert stats["iou"]["min"] == pytest.approx(0.5)
    assert stats["seconds"]["mean"] == pytest.approx(6.0)
    assert stats["categories"] == ["chair"], "names come from the subset, not the record"


def test_to_rows_is_csv_ready(tmp_path) -> None:
    dataset = _dataset_from(tmp_path, sam2=[_record(1, 0.9123456, 0.95, 4.0)], polygon=[])
    rows = to_rows(dataset)

    assert rows[0]["iou"] == pytest.approx(0.912346)
    assert set(rows[0]) >= {"method", "ann_id", "iou", "elapsed_seconds", "source"}
    # Round-trips through the CSV writer used by the report.
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    assert "sam2" in buffer.getvalue()


def test_report_merges_a_directory_with_an_archive(tmp_path, monkeypatch) -> None:
    # The real hand-off case: local results plus a teammate's exported archive.
    import sys

    from scripts.report import main as report_main

    directory = _write_records(tmp_path / "evaluation", [_record(1, 0.9, 0.95, 4.0)])
    archive_path = tmp_path / "mate-polygon.zip"
    predictions = {
        "images": [{"id": 1, "width": 8, "height": 8}],
        "categories": [{"id": 3, "name": "object"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 3,
                "area": 4.0,
                "sam2_annotation": {
                    "method": "polygon",
                    "iou": 0.7,
                    "dice": 0.8,
                    "elapsed_seconds": 20.0,
                },
            }
        ],
    }
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("predictions.json", json.dumps(predictions))
        archive.writestr("manifest.json", json.dumps({"method": "polygon"}))

    out = tmp_path / "report"
    monkeypatch.setattr(
        sys, "argv", ["report", str(directory), str(archive_path), "--out", str(out)]
    )
    report_main()

    summary = json.loads((out / "summary.json").read_text())
    assert set(summary["methods"]) == {"sam2", "polygon"}
    assert summary["paired"]["count"] == 1
    assert summary["paired"]["mean_seconds_saved"] == pytest.approx(16.0)
    assert (out / "summary.md").is_file()
    assert (out / "results.csv").is_file()
    # Both raster formats are produced, and non-empty.
    for name in ("quality_distribution", "time_distribution", "time_vs_quality"):
        png = out / f"{name}.png"
        svg = out / f"{name}.svg"
        assert png.is_file() and png.stat().st_size > 1000, name
        assert svg.is_file() and svg.stat().st_size > 1000, name
    assert (out / "paired_comparison.png").is_file()
