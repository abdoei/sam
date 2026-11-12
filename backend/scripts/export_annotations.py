"""Package annotation results into a portable archive for hand-off.

Each annotation condition is written as its own zip so a teammate can open one file
and have everything needed to reproduce or inspect the masks: the prediction masks,
a COCO-format predictions file, per-instance metrics, and a manifest recording where
the numbers came from.

The JSON follows the COCO annotation schema rather than a bespoke shape, so it can
be loaded by pycocotools, Supervisely, CVAT or any other COCO-aware tool without a
converter. RLE is used for the segmentation field because the masks are dense and
polygonising them would alter what was actually measured.
"""

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from app.coco import EVALUATION_METHODS, evaluation_dir, load_subset

ROOT = Path(__file__).resolve().parents[2]
EXPORT_VERSION = 1


def _read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 0


def _mask_to_rle(mask: np.ndarray) -> dict[str, Any]:
    """COCO compressed RLE, as pycocotools' encode produces.

    Compressed RLE (a base64 counts string, not a list) is what COCO tools actually
    load: pycocotools' decode rejects an uncompressed counts list, so an export that
    used one would not open in the very tools this format exists for.
    """
    fortran_mask = np.asfortranarray(mask.astype(np.uint8))
    encoded = mask_utils.encode(fortran_mask)
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(value) for value in encoded["size"]], "counts": counts}


def _read_records(directory: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((directory / "records").glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _method_payload(
    method: str,
    subset: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[tuple[str, bytes]], list[str]]:
    """Build the COCO predictions, metric rows, mask files and skipped records."""
    directory = evaluation_dir(method)
    records = _read_records(directory)
    if not records:
        return None, [], [], []

    categories = {
        int(category["id"]): category for category in subset["categories"]
    }
    image_by_id = {int(image["id"]): image for image in subset["images"]}

    predictions: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    mask_files: list[tuple[str, bytes]] = []
    images_used: set[int] = set()
    skipped: list[str] = []

    for record in sorted(records, key=lambda item: int(item["ann_id"])):
        ann_id = int(record["ann_id"])
        image_id = int(record["image_id"])
        mask_path = directory / "predictions" / f"{ann_id}.png"
        if not mask_path.is_file():
            skipped.append(f"ann {ann_id}: no mask file")
            continue
        mask = _read_mask(mask_path)
        height, width = mask.shape
        image = image_by_id.get(image_id)
        if image is None:
            # Left behind by an image removed from the subset; exporting it would
            # produce a reference to an image the recipient does not have.
            skipped.append(f"ann {ann_id}: image {image_id} is not in the subset")
            continue
        if (height, width) != (int(image["height"]), int(image["width"])):
            raise ValueError(
                f"Mask for annotation {ann_id} is {width}x{height}, "
                f"but image {image_id} is {image['width']}x{image['height']}"
            )

        images_used.add(image_id)
        category = categories.get(int(record["category_id"]), {})
        predictions.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": int(record["category_id"]),
                "segmentation": _mask_to_rle(mask),
                "area": float(int(mask.sum())),
                "bbox": _bbox(mask),
                "iscrowd": 0,
                # Provenance specific to this project, in a namespaced field so the
                # rest of the document stays valid COCO.
                "sam2_annotation": {
                    "method": method,
                    "iou": record["iou"],
                    "dice": record["dice"],
                    "elapsed_seconds": record["elapsed_seconds"],
                    "prompt": record.get("prompts"),
                },
            }
        )
        metric_rows.append(
            {
                "ann_id": ann_id,
                "image_id": image_id,
                "category_id": int(record["category_id"]),
                "category_name": category.get("name", "unknown"),
                "iou": record["iou"],
                "dice": record["dice"],
                "elapsed_seconds": record["elapsed_seconds"],
            }
        )
        mask_files.append(
            (f"masks/{ann_id}.png", mask_path.read_bytes())
        )

    if not predictions:
        return None, [], [], skipped

    payload = {
        "info": {
            "description": f"SAM2 annotation condition: {method}",
            "version": EXPORT_VERSION,
            "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        },
        "licenses": subset.get("licenses", []),
        "categories": [
            categories[category_id] for category_id in sorted(categories)
        ],
        "images": [
            image_by_id[image_id] for image_id in sorted(images_used)
        ],
        "annotations": predictions,
    }
    return payload, metric_rows, mask_files, skipped


def _bbox(mask: np.ndarray) -> list[float]:
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x0, x1 = int(columns.min()), int(columns.max())
    y0, y1 = int(rows.min()), int(rows.max())
    return [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]


def _metrics_csv(rows: list[dict[str, Any]]) -> bytes:
    fieldnames = [
        "ann_id",
        "image_id",
        "category_id",
        "category_name",
        "iou",
        "dice",
        "elapsed_seconds",
    ]
    handle = StringIO()
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    times = [row["elapsed_seconds"] for row in rows if row["elapsed_seconds"] is not None]
    return {
        "count": len(rows),
        "mean_iou": float(np.mean([row["iou"] for row in rows])),
        "mean_dice": float(np.mean([row["dice"] for row in rows])),
        "timed_count": len(times),
        "mean_annotation_seconds": float(np.mean(times)) if times else None,
    }


def export(
    methods: list[str],
    output_dir: Path,
    stem: str,
) -> list[Path]:
    subset = load_subset()
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    summaries: dict[str, Any] = {}

    for method in methods:
        if method not in EVALUATION_METHODS:
            raise ValueError(
                f"Unknown method {method!r}; expected one of "
                f"{', '.join(sorted(EVALUATION_METHODS))}"
            )
        built = _method_payload(method, subset)
        payload, metric_rows, mask_files, skipped = built
        if payload is None:
            detail = f"; skipped: {'; '.join(skipped)}" if skipped else ""
            raise FileNotFoundError(
                f"No usable annotated records for method {method!r} in "
                f"{evaluation_dir(method)}{detail}"
            )
        summaries[method] = _summary(metric_rows)

        manifest = {
            "export_version": EXPORT_VERSION,
            "method": method,
            "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "annotation_count": len(payload["annotations"]),
            "image_count": len(payload["images"]),
            "summary": summaries[method],
            "skipped_records": skipped,
            "contents": {
                "predictions.json": "COCO-format annotations with RLE segmentation",
                "metrics.csv": "Per-instance IoU, Dice and annotation time",
                "masks/<ann_id>.png": "Binary prediction masks, one per annotation",
                "manifest.json": "This file",
            },
            "notes": (
                "segmentation uses COCO compressed RLE, readable with "
                "pycocotools.mask.decode; use frPyObjects if a polygon form is required."
            ),
        }

        archive_path = output_dir / f"{stem}-{method}.zip"
        with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
            archive.writestr("predictions.json", json.dumps(payload) + "\n")
            archive.writestr("metrics.csv", _metrics_csv(metric_rows))
            for name, content in mask_files:
                archive.writestr(name, content)
        written.append(archive_path)

    if len(summaries) > 1:
        combined = output_dir / f"{stem}-comparison.json"
        combined.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
        written.append(combined)

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export annotated masks as a portable archive: COCO predictions, "
            "per-instance metrics and masks in one zip per condition."
        )
    )
    parser.add_argument(
        "--method",
        action="append",
        choices=sorted(EVALUATION_METHODS),
        help="Condition to export; repeat to export several (default: all with results)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data/exports",
        help="Directory to write the archives into",
    )
    parser.add_argument(
        "--stem",
        default=datetime.now(UTC).strftime("annotations-%Y%m%d-%H%M%S"),
        help="Filename prefix, so repeated exports do not overwrite each other",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = args.method
    if not methods:
        methods = sorted(
            method
            for method in EVALUATION_METHODS
            if (evaluation_dir(method) / "records").is_dir()
            and any((evaluation_dir(method) / "records").glob("*.json"))
        )
        if not methods:
            raise SystemExit(
                "No annotated records found for any condition; annotate something first."
            )

    written = export(methods, args.out, args.stem)
    for path in written:
        print(f"{path}  ({path.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
