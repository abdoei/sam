from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from app.prompts import PromptPayload
from app.segmentation import MAX_IMAGE_BYTES, model_status, segment_image

app = FastAPI(title="SAM2 Interactive Annotation API", version="0.1.0")


@app.get("/api/health")
def health() -> dict[str, object]:
    return {"status": "ok", "model": model_status()}


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

    return segment_image(image_bytes, parsed_prompt)
