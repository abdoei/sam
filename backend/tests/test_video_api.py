import base64
import io
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app import video
from app.main import app

client = TestClient(app)


def jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), color=color).save(buffer, format="JPEG")
    return buffer.getvalue()


class FakeVideoPredictor:
    def __init__(self) -> None:
        self.seeded_prompts: list[tuple[int, int]] = []

    def init_state(self, video_path: str, **_kwargs) -> dict[str, int]:
        paths = sorted(Path(video_path).glob("*.jpg"))
        assert [path.name for path in paths] == ["00000.jpg", "00001.jpg"]
        return {"num_frames": len(paths)}

    def add_new_points_or_box(self, inference_state, frame_idx, **_kwargs) -> None:
        self.seeded_prompts.append((frame_idx, _kwargs["obj_id"]))

    def propagate_in_video(
        self,
        inference_state: dict[str, int],
        start_frame_idx: int,
        reverse: bool = False,
    ):
        frame_range = (
            range(start_frame_idx, -1, -1)
            if reverse
            else range(start_frame_idx, inference_state["num_frames"])
        )
        for frame_index in frame_range:
            yield frame_index, [1, 2], np.ones((2, 1, 6, 8), dtype=np.float32)


def test_video_track_returns_masks_for_both_sides_of_prompt(monkeypatch) -> None:
    predictor = FakeVideoPredictor()
    monkeypatch.setattr(video, "get_video_predictor", lambda: predictor)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))

    response = client.post(
        "/api/video/track",
        files=[
            ("frames", ("frame-0.jpg", jpeg_bytes((0, 0, 0)), "image/jpeg")),
            ("frames", ("frame-1.jpg", jpeg_bytes((255, 255, 255)), "image/jpeg")),
        ],
        data={
            "keyframes": json.dumps(
                [
                    {
                        "frame_index": 0,
                        "object_id": 1,
                        "prompt": {"points": [{"x": 4, "y": 3, "label": 1}]},
                    },
                    {
                        "frame_index": 1,
                        "object_id": 2,
                        "prompt": {"box": [1, 1, 6, 5]},
                    },
                ]
            )
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["frame_count"] == 2
    assert result["width"] == 8
    assert result["height"] == 6
    assert predictor.seeded_prompts == [(0, 1), (1, 2)]
    assert set(result["mask_png_base64"]) == {"1", "2"}
    for object_masks in result["mask_png_base64"].values():
        assert all(object_masks)
        with Image.open(io.BytesIO(base64.b64decode(object_masks[0]))) as mask:
            assert mask.size == (8, 6)
            assert mask.getpixel((0, 0)) == 255


def test_video_track_rejects_empty_prompt_list() -> None:
    response = client.post(
        "/api/video/track",
        files=[("frames", ("frame.jpg", jpeg_bytes((10, 20, 30)), "image/jpeg"))],
        data={"keyframes": "[]"},
    )

    assert response.status_code == 422
    assert "Add a point or box prompt" in response.json()["detail"]
