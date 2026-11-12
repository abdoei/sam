"""Load and merge annotation results from evaluation directories or exported zips.

Results arrive from several places — this machine's evaluation directories, a
teammate's export archive, an older run kept for comparison. They share a record
format, so they can be merged into one dataset, but only if loading refuses to
silently reconcile disagreements: two records for the same annotation and method
that measured different masks would corrupt any statistic computed from them.

Duplicate handling follows one rule: identical measurements collapse (the same run
exported twice), conflicting ones are an error listing where they came from.
"""

import csv
import json
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile

REQUIRED_RECORD_FIELDS = ("ann_id", "image_id", "category_id", "iou", "dice")


@dataclass
class AnnotationResult:
    ann_id: int
    image_id: int
    category_id: int
    category_name: str
    method: str
    iou: float
    dice: float
    elapsed_seconds: float | None
    source: str
    image_width: int | None = None
    image_height: int | None = None
    mask_area: float | None = None

    def identity(self) -> tuple[str, int]:
        return (self.method, self.ann_id)

    def signature(self) -> tuple[float, float, float | None]:
        """What must agree for two records to be the same measurement."""
        return (
            round(self.iou, 9),
            round(self.dice, 9),
            None if self.elapsed_seconds is None else round(self.elapsed_seconds, 6),
        )


@dataclass
class Dataset:
    results: list[AnnotationResult] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def by_method(self) -> dict[str, list[AnnotationResult]]:
        grouped: dict[str, list[AnnotationResult]] = {}
        for result in self.results:
            grouped.setdefault(result.method, []).append(result)
        return grouped

    def methods(self) -> list[str]:
        return sorted(self.by_method())


def _record_to_result(
    record: dict[str, Any],
    method: str,
    source: str,
    category_name: str | None = None,
) -> AnnotationResult:
    missing = [name for name in REQUIRED_RECORD_FIELDS if name not in record]
    if missing:
        raise ValueError(f"{source}: record is missing {', '.join(missing)}")
    elapsed = record.get("elapsed_seconds")
    return AnnotationResult(
        ann_id=int(record["ann_id"]),
        image_id=int(record["image_id"]),
        category_id=int(record["category_id"]),
        category_name=category_name or str(record.get("category_name", "unknown")),
        method=method,
        iou=float(record["iou"]),
        dice=float(record["dice"]),
        elapsed_seconds=None if elapsed is None else float(elapsed),
        source=source,
    )


def load_directory(directory: Path, method: str | None = None) -> list[AnnotationResult]:
    """Read a live evaluation directory (records/*.json plus optional manifest)."""
    records_dir = directory / "records"
    if not records_dir.is_dir():
        raise FileNotFoundError(f"No records directory in {directory}")

    resolved_method = method
    if resolved_method is None:
        # The polygon directory is named evaluation_polygon; the SAM2 one carries no
        # suffix, so the directory name is the only signal available.
        resolved_method = "polygon" if directory.name.endswith("_polygon") else "sam2"

    # Records store only a category id. Names come from the subset, which is the
    # source of truth, falling back to the manifest when it carries them.
    category_names = _subset_category_names()
    manifest = directory / "predictions.csv"
    if manifest.is_file():
        with manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if "category_name" in row and row.get("category_name"):
                    category_names[int(row["ann_id"])] = row["category_name"]

    results: list[AnnotationResult] = []
    for path in sorted(records_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        results.append(
            _record_to_result(
                record,
                resolved_method,
                f"{directory.name}/{path.name}",
                category_names.get(int(record.get("category_id", 0))),
            )
        )
    return results


def _subset_category_names() -> dict[int, str]:
    """Category id to name, from the COCO subset when it is available."""
    try:
        from app.coco import load_subset

        return {
            int(category["id"]): str(category["name"])
            for category in load_subset()["categories"]
        }
    except (FileNotFoundError, ValueError, KeyError):
        return {}


def load_archive(path: Path) -> list[AnnotationResult]:
    """Read one exported zip, using its predictions.json and image metadata."""
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        if "predictions.json" not in names:
            raise ValueError(f"{path.name} has no predictions.json")
        predictions = json.loads(archive.read("predictions.json"))

        method = "unknown"
        if "manifest.json" in names:
            method = json.loads(archive.read("manifest.json")).get("method", "unknown")

        categories = {
            int(category["id"]): category.get("name", "unknown")
            for category in predictions.get("categories", [])
        }
        sizes: dict[int, tuple[int, int]] = {
            int(image["id"]): (int(image["width"]), int(image["height"]))
            for image in predictions.get("images", [])
        }

        results: list[AnnotationResult] = []
        for annotation in predictions.get("annotations", []):
            ann_id = int(annotation["id"])
            provenance = annotation.get("sam2_annotation", {})
            category_id = int(annotation["category_id"])
            image_id = int(annotation["image_id"])
            width, height = sizes.get(image_id, (None, None))
            results.append(
                AnnotationResult(
                    ann_id=ann_id,
                    image_id=image_id,
                    category_id=category_id,
                    category_name=categories.get(category_id, "unknown"),
                    method=str(provenance.get("method", method)),
                    iou=float(provenance["iou"]),
                    dice=float(provenance["dice"]),
                    elapsed_seconds=(
                        None
                        if provenance.get("elapsed_seconds") is None
                        else float(provenance["elapsed_seconds"])
                    ),
                    source=f"{path.name}:{ann_id}",
                    image_width=width,
                    image_height=height,
                    mask_area=annotation.get("area"),
                )
            )
        return results


def load_csv_manifest(path: Path, method: str) -> list[AnnotationResult]:
    """Read a metrics CSV (as written by the exporter) as a fallback source."""
    text = path.read_text(encoding="utf-8")
    results: list[AnnotationResult] = []
    for row in csv.DictReader(StringIO(text)):
        if not row.get("iou"):
            continue
        results.append(_record_to_result(row, method, f"{path.name}:{row.get('ann_id')}"))
    return results


def load_source(path: Path) -> list[AnnotationResult]:
    """Load results from a zip archive, an evaluation directory or a metrics CSV."""
    if path.is_dir():
        return load_directory(path)
    if path.suffix == ".zip":
        return load_archive(path)
    if path.suffix == ".csv":
        method = "polygon" if "polygon" in path.name else "sam2"
        return load_csv_manifest(path, method)
    raise ValueError(f"Unsupported source: {path} (expected a directory, .zip or .csv)")


def merge(sources: list[Path]) -> Dataset:
    """Merge several sources, refusing to silently reconcile conflicting records."""
    dataset = Dataset()
    seen: dict[tuple[str, int], AnnotationResult] = {}

    for path in sources:
        results = load_source(path)
        dataset.sources.append(str(path))
        for result in results:
            key = result.identity()
            existing = seen.get(key)
            if existing is None:
                seen[key] = result
                dataset.results.append(result)
                continue
            if existing.signature() == result.signature():
                # The same run exported twice; keep one copy.
                continue
            raise ValueError(
                f"Conflicting records for {result.method} annotation {result.ann_id}: "
                f"{existing.source} reports IoU {existing.iou:.4f} in "
                f"{existing.elapsed_seconds}s, while {result.source} reports IoU "
                f"{result.iou:.4f} in {result.elapsed_seconds}s. Export one of them, or "
                f"load them separately, rather than merging incomparable measurements."
            )

    dataset.results.sort(key=lambda item: (item.method, item.ann_id))
    return dataset


def to_rows(dataset: Dataset) -> list[dict[str, Any]]:
    return [
        {
            "method": result.method,
            "ann_id": result.ann_id,
            "image_id": result.image_id,
            "category_id": result.category_id,
            "category_name": result.category_name,
            "iou": round(result.iou, 6),
            "dice": round(result.dice, 6),
            "elapsed_seconds": (
                None if result.elapsed_seconds is None else round(result.elapsed_seconds, 6)
            ),
            "source": result.source,
        }
        for result in dataset.results
    ]
