import csv
import json
import os
import tempfile
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.metrics import binary_mask_metrics
from app.prompts import PromptPayload

ROOT = Path(__file__).resolve().parents[2]
_save_lock = Lock()


def subset_path() -> Path:
    return Path(
        os.getenv(
            "COCO_SUBSET_JSON",
            str(ROOT / "data/coco_subset/annotations/subset.json"),
        )
    ).expanduser()


def ground_truth_dir() -> Path:
    return Path(
        os.getenv("COCO_GT_DIR", str(ROOT / "data/gt_masks"))
    ).expanduser()


def evaluation_dir() -> Path:
    return Path(
        os.getenv("COCO_EVALUATION_DIR", str(ROOT / "data/evaluation"))
    ).expanduser()


def load_subset() -> dict[str, Any]:
    path = subset_path()
    try:
        with path.open(encoding="utf-8") as subset_file:
            subset = json.load(subset_file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"COCO subset not found at {path}. Run `uv run python -m "
            "scripts.download_coco_subset` from backend/."
        ) from exc
    if not isinstance(subset, dict) or not all(
        key in subset for key in ("images", "annotations", "categories")
    ):
        raise ValueError(f"COCO subset JSON is missing required fields: {path}")
    return subset


def image_records() -> list[dict[str, Any]]:
    subset = load_subset()
    categories = {
        int(category["id"]): category["name"] for category in subset["categories"]
    }
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in subset["annotations"]:
        if int(annotation.get("iscrowd", 0)):
            continue
        image_id = int(annotation["image_id"])
        annotations_by_image.setdefault(image_id, []).append(
            {
                "id": int(annotation["id"]),
                "category_id": int(annotation["category_id"]),
                "category_name": categories.get(int(annotation["category_id"]), "unknown"),
                "area": int(annotation["area"]),
            }
        )

    return [
        {
            "id": int(image["id"]),
            "file_name": image["file_name"],
            "width": int(image["width"]),
            "height": int(image["height"]),
            "annotations": annotations_by_image.get(int(image["id"]), []),
        }
        for image in sorted(subset["images"], key=lambda item: int(item["id"]))
    ]


def find_image(image_id: int) -> dict[str, Any]:
    for image in image_records():
        if image["id"] == image_id:
            return image
    raise KeyError(f"COCO image ID {image_id} was not found in the selected subset")


def image_path(image: dict[str, Any]) -> Path:
    file_name = str(image["file_name"])
    if Path(file_name).name != file_name or file_name in {"", ".", ".."}:
        raise ValueError(f"Unsafe COCO image filename: {file_name!r}")
    return subset_path().parent.parent / "images" / file_name


def find_annotation(annotation_id: int) -> tuple[dict[str, Any], dict[str, Any]]:
    subset = load_subset()
    image_by_id = {int(image["id"]): image for image in subset["images"]}
    for annotation in subset["annotations"]:
        if int(annotation["id"]) != annotation_id or int(annotation.get("iscrowd", 0)):
            continue
        image = image_by_id.get(int(annotation["image_id"]))
        if image is None:
            break
        return annotation, image
    raise KeyError(f"COCO annotation ID {annotation_id} was not found in the selected subset")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".part",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        temporary_file.write(content)
    temporary_path.replace(path)


def _write_prediction_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".part",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        writer = csv.DictWriter(
            temporary_file,
            fieldnames=["ann_id", "prediction_path", "elapsed_seconds"],
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def save_and_evaluate_annotation(
    image_id: int,
    annotation_id: int,
    elapsed_seconds: float,
    mask_png: bytes,
    prompt: PromptPayload,
) -> dict[str, Any]:
    target_annotation, target_image = find_annotation(annotation_id)
    if int(target_annotation["image_id"]) != image_id:
        raise ValueError(
            f"Annotation {annotation_id} belongs to image "
            f"{target_annotation['image_id']}, not {image_id}"
        )

    try:
        with Image.open(ground_truth_dir() / str(image_id) / f"{annotation_id}.png") as image:
            ground_truth = np.asarray(image.convert("L"), dtype=np.uint8) > 0
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Ground-truth mask for annotation {annotation_id} is missing. "
            "Run `uv run python -m scripts.export_gt_masks` from backend/."
        ) from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Ground-truth mask for annotation {annotation_id} is invalid") from exc

    try:
        with Image.open(BytesIO(mask_png)) as image:
            if image.format != "PNG":
                raise ValueError("Prediction mask must be encoded as PNG")
            prediction = np.asarray(image.convert("L"), dtype=np.uint8) > 0
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Prediction mask is not a valid image") from exc

    expected_shape = (int(target_image["height"]), int(target_image["width"]))
    if prediction.shape != expected_shape:
        raise ValueError(
            f"Prediction mask must be {expected_shape[1]}x{expected_shape[0]}, "
            f"received {prediction.shape[1]}x{prediction.shape[0]}"
        )
    if ground_truth.shape != expected_shape:
        raise ValueError(
            f"Ground-truth mask dimensions do not match COCO image {image_id}"
        )

    metrics = binary_mask_metrics(prediction, ground_truth)
    prediction_path = evaluation_dir() / "predictions" / f"{annotation_id}.png"
    record_path = evaluation_dir() / "records" / f"{annotation_id}.json"
    manifest_path = evaluation_dir() / "predictions.csv"
    record = {
        "image_id": image_id,
        "ann_id": annotation_id,
        "category_id": int(target_annotation["category_id"]),
        "elapsed_seconds": elapsed_seconds,
        **metrics,
        "prediction_path": prediction_path.relative_to(evaluation_dir()).as_posix(),
        "prompts": prompt.model_dump(),
    }

    with _save_lock:
        _atomic_write(prediction_path, mask_png)
        _atomic_write(
            record_path,
            (json.dumps(record, indent=2) + "\n").encode("utf-8"),
        )
        existing: list[dict[str, Any]] = []
        if manifest_path.is_file():
            with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
                existing = list(csv.DictReader(manifest_file))
        existing = [row for row in existing if int(row["ann_id"]) != annotation_id]
        existing.append(
            {
                "ann_id": annotation_id,
                "prediction_path": f"predictions/{annotation_id}.png",
                "elapsed_seconds": elapsed_seconds,
            }
        )
        existing.sort(key=lambda row: int(row["ann_id"]))
        _write_prediction_manifest(existing, manifest_path)

    return {
        **record,
        "saved": True,
        "prediction_path": str(prediction_path),
        "manifest_path": str(manifest_path),
    }
