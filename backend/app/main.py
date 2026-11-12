import base64
import binascii
import json
import tempfile
from pathlib import Path
from typing import Any, Literal
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from starlette.concurrency import run_in_threadpool

from app.coco import (
    EVALUATION_METHODS,
    evaluation_dir,
    find_annotation,
    find_image,
    ground_truth_dir,
    image_path,
    image_records,
    save_and_evaluate_annotation,
)
from app.polygon import (
    parts_mask_png,
    polygon_summary,
    simplify_polyline,
    validate_polygon,
)
from app.prompts import PromptPayload
from app.segmentation import MAX_IMAGE_BYTES, model_status, segment_image
from app.video import (
    MAX_VIDEO_FRAME_BYTES,
    MAX_VIDEO_FRAMES,
    VideoKeyframePrompt,
    track_video_frames,
)

MAX_PREDICTION_BYTES = 15 * 1024 * 1024
MAX_VIDEO_UPLOAD_BYTES = MAX_VIDEO_FRAMES * MAX_VIDEO_FRAME_BYTES
# Freehand traces are stored with this RDP tolerance, in image pixels.
POLYGON_SIMPLIFY_TOLERANCE = 1.5

app = FastAPI(title="SAM2 Interactive Annotation API", version="0.1.0")


class SaveAnnotationPayload(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    image_id: int = Field(gt=0)
    annotation_id: int = Field(gt=0)
    elapsed_seconds: float = Field(ge=0)
    # Total model time across every generation trial for this object, and how many
    # trials were run. Zero means the mask was produced without timing data.
    inference_seconds: float = Field(default=0.0, ge=0)
    inference_runs: int = Field(default=0, ge=0)
    mask_png_base64: str = Field(min_length=1, max_length=20 * 1024 * 1024)
    prompt: PromptPayload


class SavePolygonPayload(BaseModel):
    """A polygon or freehand annotation submitted as the manual baseline.

    `parts` holds one ring per visible piece of the object, so something split by
    an occluder is described completely and scored as the union of its pieces.
    `points` is still accepted as shorthand for a single ring.
    """

    model_config = ConfigDict(allow_inf_nan=False)

    image_id: int = Field(gt=0)
    annotation_id: int = Field(gt=0)
    elapsed_seconds: float = Field(ge=0)
    parts: list[list[list[float]]] = Field(default_factory=list)
    points: list[list[float]] = Field(default_factory=list)
    mode: Literal["polygon", "freehand"] = "polygon"

    @model_validator(mode="after")
    def require_rings(self) -> "SavePolygonPayload":
        if not self.parts and not self.points:
            raise ValueError("Provide at least one part or a points ring")
        if self.parts and self.points:
            raise ValueError("Provide either parts or points, not both")
        return self

    def rings(self) -> list[list[list[float]]]:
        return self.parts if self.parts else [self.points]


@app.get("/api/health")
def health() -> dict[str, object]:
    return {"status": "ok", "model": model_status()}


@app.get("/api/coco/images")
def coco_images() -> dict[str, Any]:
    try:
        return {"images": image_records()}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=503, detail=f"COCO subset is invalid: {exc}") from exc


@app.get("/api/coco/images/{image_id}")
def coco_image(image_id: int) -> FileResponse:
    try:
        record = find_image(image_id)
        path = image_path(record)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"COCO image {image_id} is missing from the local subset",
        )
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/results")
def list_results(image_id: int | None = None) -> dict[str, Any]:
    """Every annotated object, optionally restricted to one image.

    Returns per-object metrics plus roll-ups: one for the requested scope and one
    across every recorded annotation, computed from the full records rather than
    from the filtered rows.
    """
    from scripts.results_dataset import load_directory

    methods: list[dict[str, Any]] = []
    errors: list[str] = []
    scoped: list[dict[str, Any]] = []
    everything: list[dict[str, Any]] = []
    for method in sorted(EVALUATION_METHODS):
        directory = evaluation_dir(method)
        if not (directory / "records").is_dir():
            continue
        try:
            results = load_directory(directory, method)
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"{method}: {exc}")
            continue
        if not results:
            continue
        rows = [
            {
                "ann_id": result.ann_id,
                "image_id": result.image_id,
                "category_name": result.category_name,
                "iou": result.iou,
                "dice": result.dice,
                "elapsed_seconds": result.elapsed_seconds,
            }
            for result in results
        ]
        everything.extend(rows)
        visible = [
            row for row in rows if image_id is None or row["image_id"] == image_id
        ]
        scoped.extend(visible)
        methods.append({"method": method, "results": visible, "all_count": len(rows)})

    return {
        "image_id": image_id,
        "methods": [entry for entry in methods if entry["results"]],
        "totals": {
            "scope": _summarise_results(scoped),
            "all": _summarise_results(everything),
        },
        "errors": errors,
    }


def _summarise_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count and mean IoU/Dice/time over a set of results."""
    if not rows:
        return {"count": 0}
    ious = [row["iou"] for row in rows]
    dices = [row["dice"] for row in rows]
    times = [row["elapsed_seconds"] for row in rows if row["elapsed_seconds"] is not None]
    return {
        "count": len(rows),
        "mean_iou": round(sum(ious) / len(ious), 4),
        "mean_dice": round(sum(dices) / len(dices), 4),
        "mean_seconds": round(sum(times) / len(times), 3) if times else None,
        "timed_count": len(times),
    }


@app.get("/api/export")
def download_export(methods: str | None = None) -> Response:
    """Build the same archive the CLI writes and serve it as a download."""
    from scripts.export_annotations import export

    requested = [value for value in (methods or "").split(",") if value]
    unknown = [value for value in requested if value not in EVALUATION_METHODS]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown method(s): {', '.join(unknown)}",
        )
    if not requested:
        requested = sorted(EVALUATION_METHODS)

    with tempfile.TemporaryDirectory(prefix="sam2-export-") as directory:
        try:
            written = export(requested, Path(directory), "annotations")
        except FileNotFoundError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        archives = [path for path in written if path.suffix == ".zip"]
        if not archives:
            raise HTTPException(status_code=409, detail="Nothing to export")
        if len(archives) == 1:
            payload = archives[0].read_bytes()
            filename = archives[0].name
        else:
            # Several conditions: hand back one bundle rather than forcing the
            # browser to manage multiple downloads.
            bundle = Path(directory) / "annotations-all.zip"
            with ZipFile(bundle, "w", ZIP_DEFLATED) as archive:
                for path in written:
                    archive.write(path, arcname=path.name)
            payload = bundle.read_bytes()
            filename = bundle.name

    return Response(
        content=payload,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/results/masks/{image_id}/{annotation_id}.png")
def saved_mask(image_id: int, annotation_id: int, method: str = "sam2") -> FileResponse:
    """Serve a saved prediction mask so the results panel can preview it."""
    if method not in EVALUATION_METHODS:
        raise HTTPException(status_code=422, detail=f"Unknown method: {method}")
    path = evaluation_dir(method) / "predictions" / f"{annotation_id}.png"
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"No saved {method} mask for annotation {annotation_id}",
        )
    return FileResponse(path, media_type="image/png")


@app.get("/api/coco/images/{image_id}/masks/{annotation_id}")
def coco_ground_truth_mask(image_id: int, annotation_id: int) -> FileResponse:
    """Serve one annotation's ground-truth mask as a PNG overlay reference."""
    try:
        find_image(image_id)
        find_annotation(annotation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    path = ground_truth_dir() / str(image_id) / f"{annotation_id}.png"
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Ground-truth mask for annotation {annotation_id} is missing. "
                "Run `uv run python -m scripts.export_gt_masks` from backend/."
            ),
        )
    return FileResponse(path, media_type="image/png")


@app.post("/api/segment")
async def segment(
    image: UploadFile = File(...),
    prompt: str = Form(...),
) -> dict[str, int | float | str]:
    try:
        parsed_prompt = PromptPayload.model_validate_json(prompt)
    except ValidationError as exc:
        errors = exc.errors(include_context=False, include_url=False)
        raise HTTPException(status_code=422, detail=errors) from exc

    image_bytes = await image.read(MAX_IMAGE_BYTES + 1)
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 15 MiB upload limit")

    return await run_in_threadpool(segment_image, image_bytes, parsed_prompt)


@app.post("/api/video/track")
async def track_video(
    frames: list[UploadFile] = File(...),
    keyframes: str = Form(...),
) -> dict[str, Any]:
    if len(frames) > MAX_VIDEO_FRAMES:
        raise HTTPException(
            status_code=413,
            detail=f"Video exceeds the {MAX_VIDEO_FRAMES}-frame sampling limit",
        )
    try:
        keyframe_data = json.loads(keyframes)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="Invalid video prompt data") from exc
    if not isinstance(keyframe_data, list):
        raise HTTPException(status_code=422, detail="Video prompts must be a list")
    try:
        parsed_keyframes = [VideoKeyframePrompt.model_validate(item) for item in keyframe_data]
    except ValidationError as exc:
        errors = exc.errors(include_context=False, include_url=False)
        raise HTTPException(status_code=422, detail=errors) from exc

    frame_bytes: list[bytes] = []
    total_bytes = 0
    for frame_index, upload in enumerate(frames):
        data = await upload.read(MAX_VIDEO_FRAME_BYTES + 1)
        if len(data) > MAX_VIDEO_FRAME_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Sampled frame {frame_index} exceeds the 2 MiB limit",
            )
        total_bytes += len(data)
        if total_bytes > MAX_VIDEO_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Sampled video frames exceed the upload limit",
            )
        frame_bytes.append(data)

    return await run_in_threadpool(track_video_frames, frame_bytes, parsed_keyframes)


@app.post("/api/annotations")
async def save_annotation(payload: SaveAnnotationPayload) -> dict[str, Any]:
    try:
        image = find_image(payload.image_id)
        find_annotation(payload.annotation_id)
        encoded_mask = base64.b64decode(payload.mask_png_base64, validate=True)
    except (KeyError, binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if len(encoded_mask) > MAX_PREDICTION_BYTES:
        raise HTTPException(status_code=413, detail="Prediction mask exceeds the 15 MiB limit")
    if not Path(image_path(image)).is_file():
        raise HTTPException(
            status_code=503,
            detail=f"COCO image {payload.image_id} is missing from the local subset",
        )

    try:
        return await run_in_threadpool(
            save_and_evaluate_annotation,
            payload.image_id,
            payload.annotation_id,
            payload.elapsed_seconds,
            encoded_mask,
            payload.prompt,
            "sam2",
            payload.inference_seconds,
            payload.inference_runs,
        )
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/annotations/polygon")
async def save_polygon_annotation(payload: SavePolygonPayload) -> dict[str, Any]:
    """Score a polygon or freehand outline against the same ground truth as SAM2.

    The outline is rasterised server-side so both conditions are measured from
    exactly the same mask comparison, and the result is written to the polygon
    evaluation directory.
    """
    try:
        image = find_image(payload.image_id)
        find_annotation(payload.annotation_id)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    width, height = int(image["width"]), int(image["height"])
    rings = payload.rings()
    try:
        # Each ring is validated on its own, and a freehand ring is simplified
        # separately: simplifying across parts would join disjoint pieces.
        stored_rings: list[list[list[float]]] = []
        for ring in rings:
            validated = validate_polygon(ring, width, height)
            if payload.mode == "freehand":
                simplified = simplify_polyline(validated, POLYGON_SIMPLIFY_TOLERANCE)
                # Simplification can flatten a thin trace below a usable ring; fall
                # back to the drawn outline rather than discarding the work.
                validated = simplified if len(simplified) >= 3 else validated
            stored_rings.append([[x, y] for x, y in validated])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        mask_png = parts_mask_png(stored_rings, width, height)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # The polygon condition has no SAM2 prompt: the outlines are the annotation,
    # so the record stores them instead of inventing a point or box.
    prompt = {"mode": payload.mode, "parts": stored_rings}
    try:
        result = await run_in_threadpool(
            save_and_evaluate_annotation,
            payload.image_id,
            payload.annotation_id,
            payload.elapsed_seconds,
            mask_png,
            prompt,
            "polygon",
        )
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        **result,
        "mode": payload.mode,
        "outline": polygon_summary(
            [[(x, y) for x, y in ring] for ring in stored_rings]
        ),
        "drawn_points": sum(len(ring) for ring in rings),
    }
