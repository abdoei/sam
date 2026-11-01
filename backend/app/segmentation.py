import base64
import io
import os
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError

from app.prompts import PromptPayload

DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
prediction_lock = Lock()


def model_status() -> dict[str, Any]:
    from importlib.util import find_spec

    checkpoint = os.getenv("SAM2_CHECKPOINT", "")
    installed = find_spec("sam2") is not None
    checkpoint_exists = bool(checkpoint) and Path(checkpoint).expanduser().is_file()
    return {
        "ready": installed and checkpoint_exists,
        "installed": installed,
        "checkpoint_configured": checkpoint_exists,
        "config": os.getenv("SAM2_CONFIG", DEFAULT_CONFIG),
        "device": os.getenv("SAM2_DEVICE", "cpu"),
    }


@lru_cache(maxsize=1)
def _load_predictor(config: str, checkpoint: str, device: str) -> Any:
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "SAM2 is not installed. Install the optional backend dependency "
                "with `uv sync --extra sam2`."
            ),
        ) from exc

    model = build_sam2(config, checkpoint, device=device)
    return SAM2ImagePredictor(model)


@lru_cache(maxsize=1)
def _load_video_predictor(config: str, checkpoint: str, device: str) -> Any:
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "SAM2 video support is unavailable. Install the optional backend "
                "dependency with `uv sync --extra sam2`."
            ),
        ) from exc

    return build_sam2_video_predictor(config, checkpoint, device=device)


def get_predictor() -> Any:
    config = os.getenv("SAM2_CONFIG", DEFAULT_CONFIG)
    checkpoint_setting = os.getenv("SAM2_CHECKPOINT", "")
    if not checkpoint_setting:
        raise HTTPException(
            status_code=503,
            detail=(
                "SAM2 checkpoint is not configured. Set SAM2_CHECKPOINT to the "
                "downloaded weights file."
            ),
        )

    checkpoint = str(Path(checkpoint_setting).expanduser())
    if not Path(checkpoint).is_file():
        raise HTTPException(
            status_code=503,
            detail=f"SAM2 checkpoint file does not exist: {checkpoint}",
        )

    return _load_predictor(config, checkpoint, os.getenv("SAM2_DEVICE", "cpu"))


def get_video_predictor() -> Any:
    config = os.getenv("SAM2_CONFIG", DEFAULT_CONFIG)
    checkpoint_setting = os.getenv("SAM2_CHECKPOINT", "")
    if not checkpoint_setting:
        raise HTTPException(
            status_code=503,
            detail=(
                "SAM2 checkpoint is not configured. Set SAM2_CHECKPOINT to the "
                "downloaded weights file."
            ),
        )

    checkpoint = str(Path(checkpoint_setting).expanduser())
    if not Path(checkpoint).is_file():
        raise HTTPException(
            status_code=503,
            detail=f"SAM2 checkpoint file does not exist: {checkpoint}",
        )

    return _load_video_predictor(config, checkpoint, os.getenv("SAM2_DEVICE", "cpu"))


def segment_image(
    image_bytes: bytes,
    prompt: PromptPayload,
) -> dict[str, int | float | str]:
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail=f"Image exceeds the {MAX_IMAGE_PIXELS:,}-pixel limit",
                )
            image = source.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc

    width, height = image.size
    for point in prompt.points:
        if point.x >= width or point.y >= height:
            raise HTTPException(
                status_code=422,
                detail=f"Point coordinates must be within the {width}x{height} image",
            )
    if prompt.box is not None:
        x0, y0, x1, y1 = prompt.box
        if x1 <= x0 or y1 <= y0:
            raise HTTPException(status_code=422, detail="Box must have positive width and height")
        if x1 > width or y1 > height:
            raise HTTPException(
                status_code=422,
                detail=f"Box coordinates must be within the {width}x{height} image",
            )

    point_coords = (
        np.asarray([[point.x, point.y] for point in prompt.points], dtype=np.float32)
        if prompt.points
        else None
    )
    point_labels = (
        np.asarray([point.label for point in prompt.points], dtype=np.int32)
        if prompt.points
        else None
    )
    box = np.asarray(prompt.box, dtype=np.float32) if prompt.box is not None else None

    with prediction_lock:
        predictor = get_predictor()
        import torch
        
        print("DEVICE USED: ", predictor.device)

        with torch.inference_mode():
            predictor.set_image(np.array(image, copy=True))
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True,
            )

    best_index = int(np.argmax(scores))
    mask = np.asarray(masks[best_index], dtype=np.uint8) * 255
    mask_buffer = io.BytesIO()
    Image.fromarray(mask).save(mask_buffer, format="PNG")
    

    return {
        "width": width,
        "height": height,
        "score": float(scores[best_index]),
        "mask_png_base64": base64.b64encode(mask_buffer.getvalue()).decode("ascii"),
    }
