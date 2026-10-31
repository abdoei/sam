import io

from fastapi.testclient import TestClient
from PIL import Image

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
