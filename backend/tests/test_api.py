import io
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app import segmentation
from app.main import app

client = TestClient(app)


def image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 12), color=(30, 50, 70)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_reports_model_configuration(monkeypatch) -> None:
    monkeypatch.delenv("SAM2_CHECKPOINT", raising=False)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model"]["checkpoint_configured"] is False


def test_segment_requires_a_prompt() -> None:
    response = client.post(
        "/api/segment",
        files={"image": ("sample.png", image_bytes(), "image/png")},
        data={"prompt": '{"points": [], "box": null}'},
    )

    assert response.status_code == 422


def test_segment_rejects_point_outside_image_before_model_loading(monkeypatch) -> None:
    monkeypatch.delenv("SAM2_CHECKPOINT", raising=False)
    response = client.post(
        "/api/segment",
        files={"image": ("sample.png", image_bytes(), "image/png")},
        data={"prompt": '{"points": [{"x": 16, "y": 5, "label": 1}]}'},
    )

    assert response.status_code == 422
    assert "within the 16x12 image" in response.json()["detail"]


def test_segment_reports_missing_checkpoint(monkeypatch) -> None:
    monkeypatch.delenv("SAM2_CHECKPOINT", raising=False)
    response = client.post(
        "/api/segment",
        files={"image": ("sample.png", image_bytes(), "image/png")},
        data={"prompt": '{"points": [{"x": 4, "y": 5, "label": 1}]}'},
    )

    assert response.status_code == 503
    assert "SAM2 checkpoint is not configured" in response.json()["detail"]


def test_segment_passes_writable_image_array_to_predictor(monkeypatch) -> None:
    observed: dict[str, bool] = {}

    class Predictor:
        def set_image(self, image: np.ndarray) -> None:
            observed["writable"] = image.flags.writeable

        def predict(self, **_kwargs):
            return np.zeros((1, 12, 16), dtype=np.uint8), np.array([0.5]), None

    monkeypatch.setattr(segmentation, "get_predictor", Predictor)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))

    response = client.post(
        "/api/segment",
        files={"image": ("sample.png", image_bytes(), "image/png")},
        data={"prompt": '{"points": [{"x": 4, "y": 5, "label": 1}]}'},
    )

    assert response.status_code == 200
    assert observed["writable"] is True
