import base64
import io
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from app.prompts import PromptPayload
from app.segmentation import get_video_predictor, prediction_lock

MAX_VIDEO_FRAMES = 30
MAX_VIDEO_OBJECTS = 20
MAX_VIDEO_FRAME_BYTES = 2 * 1024 * 1024
MAX_VIDEO_FRAME_DIMENSION = 1280
MAX_VIDEO_FRAME_PIXELS = 921_600


class VideoKeyframePrompt(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    frame_index: int = Field(ge=0, lt=MAX_VIDEO_FRAMES)
    object_id: int = Field(ge=1)
    prompt: PromptPayload


def _encode_mask(mask_logits: Any) -> str:
    if hasattr(mask_logits, "detach"):
        mask_array = mask_logits.detach().to("cpu").numpy()
    else:
        mask_array = np.asarray(mask_logits)
    binary_mask = np.asarray(mask_array).squeeze() > 0
    output = io.BytesIO()
    Image.fromarray(binary_mask.astype(np.uint8) * 255).save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def track_video_frames(
    frame_bytes: list[bytes],
    keyframes: list[VideoKeyframePrompt],
) -> dict[str, Any]:
    if not frame_bytes:
        raise HTTPException(status_code=422, detail="Upload at least one sampled video frame")
    if len(frame_bytes) > MAX_VIDEO_FRAMES:
        raise HTTPException(
            status_code=413,
            detail=f"Video exceeds the {MAX_VIDEO_FRAMES}-frame sampling limit",
        )
    if not keyframes:
        raise HTTPException(status_code=422, detail="Add a point or box prompt on a video frame")

    decoded_frames: list[Image.Image] = []
    image_size: tuple[int, int] | None = None
    for frame_index, frame in enumerate(frame_bytes):
        if len(frame) > MAX_VIDEO_FRAME_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Sampled frame {frame_index} exceeds the 2 MiB limit",
            )
        try:
            with Image.open(io.BytesIO(frame)) as image:
                width, height = image.size
                if (
                    max(width, height) > MAX_VIDEO_FRAME_DIMENSION
                    or width * height > MAX_VIDEO_FRAME_PIXELS
                ):
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Video frames must have a longest edge of at most "
                            f"{MAX_VIDEO_FRAME_DIMENSION} pixels and at most "
                            f"{MAX_VIDEO_FRAME_PIXELS:,} total pixels"
                        ),
                    )
                decoded = image.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Sampled frame {frame_index} is not a valid image",
            ) from exc

        width, height = decoded.size
        if image_size is not None and decoded.size != image_size:
            raise HTTPException(
                status_code=422,
                detail="All sampled video frames must have the same dimensions",
            )
        image_size = decoded.size
        decoded_frames.append(decoded)

    assert image_size is not None
    width, height = image_size
    prompt_keys = {(item.object_id, item.frame_index) for item in keyframes}
    if len(prompt_keys) != len(keyframes):
        raise HTTPException(
            status_code=422,
            detail="Only one prompt set per object is allowed on each frame",
        )
    object_ids = {item.object_id for item in keyframes}
    if len(object_ids) > MAX_VIDEO_OBJECTS:
        raise HTTPException(
            status_code=413,
            detail=f"Video tracking supports at most {MAX_VIDEO_OBJECTS} objects",
        )
    for keyframe in keyframes:
        if keyframe.frame_index >= len(decoded_frames):
            raise HTTPException(
                status_code=422,
                detail=f"Prompt frame {keyframe.frame_index} is outside the sampled video",
            )
        prompt = keyframe.prompt
        if any(point.x >= width or point.y >= height for point in prompt.points):
            raise HTTPException(
                status_code=422,
                detail=f"Prompt coordinates must be within the {width}x{height} video frame",
            )
        if prompt.box is not None:
            x0, y0, x1, y1 = prompt.box
            if x1 <= x0 or y1 <= y0 or x1 > width or y1 > height:
                raise HTTPException(
                    status_code=422,
                    detail=f"Box coordinates must be within the {width}x{height} video frame",
                )

    masks: dict[str, list[str | None]] = {
        str(object_id): [None] * len(decoded_frames) for object_id in object_ids
    }
    with tempfile.TemporaryDirectory(prefix="sam2-video-") as directory:
        frames_dir = Path(directory)
        for frame_index, image in enumerate(decoded_frames):
            image.save(frames_dir / f"{frame_index:05d}.jpg", format="JPEG", quality=90)

        with prediction_lock:
            predictor = get_video_predictor()
            import torch
            print("DEVICE USED: ", predictor.device)

            with torch.inference_mode():
                for object_id in sorted(object_ids):
                    object_keyframes = sorted(
                        (item for item in keyframes if item.object_id == object_id),
                        key=lambda item: item.frame_index,
                    )
                    inference_state = predictor.init_state(
                        video_path=str(frames_dir),
                        offload_video_to_cpu=True,
                        offload_state_to_cpu=True,
                    )
                    for keyframe in object_keyframes:
                        points = keyframe.prompt.points
                        point_coords = [[point.x, point.y] for point in points] if points else None
                        point_labels = [point.label for point in points] if points else None
                        predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=keyframe.frame_index,
                            obj_id=object_id,
                            points=point_coords,
                            labels=point_labels,
                            box=keyframe.prompt.box,
                        )

                    earliest_prompt = object_keyframes[0].frame_index
                    if earliest_prompt > 0:
                        for frame_index, tracked_object_ids, logits in predictor.propagate_in_video(
                            inference_state,
                            start_frame_idx=earliest_prompt,
                            reverse=True,
                        ):
                            if frame_index > earliest_prompt or object_id not in tracked_object_ids:
                                continue
                            object_index = list(tracked_object_ids).index(object_id)
                            masks[str(object_id)][frame_index] = _encode_mask(logits[object_index])

                    for frame_index, tracked_object_ids, logits in predictor.propagate_in_video(
                        inference_state,
                        start_frame_idx=earliest_prompt,
                    ):
                        if object_id not in tracked_object_ids:
                            continue
                        object_index = list(tracked_object_ids).index(object_id)
                        masks[str(object_id)][frame_index] = _encode_mask(logits[object_index])

    return {
        "width": width,
        "height": height,
        "frame_count": len(decoded_frames),
        "mask_png_base64": masks,
    }
