---
name: SAM2 Annotation Project
description: "Use when implementing, testing, documenting, or evaluating this repository's local SAM 2 COCO annotation application, including FastAPI, React/TypeScript, uv, pnpm, milestone work, or report updates."
---

# SAM2 Annotation Project Agent

You work on this repository's two-person Digital Image Processing project: a
local, full-stack image/video annotation tool using SAM 2.1, COCO, and measured
evaluation. Deliver working changes in the repository; do not stop at general
advice when implementation is requested.

## Project constraints

- Backend: Python 3.11/3.12, FastAPI, managed with `uv`.
- Frontend: React, TypeScript, and Vite, managed with `pnpm`.
- Segmentation runs locally with the frozen SAM 2.1 Hiera-Tiny checkpoint.
  Do not add hosted inference dependencies or send images to external services.
- Video mode accepts bounded MP4 clips, samples frames in the browser, and uses
  the SAM 2 video predictor for multiple independently prompted objects. Keep
  object prompts and masks separate, allow prompt edits, and preserve explicit
  duration, frame-count, image-dimension, object-count, and payload-size limits.
  Do not claim video quality metrics without a scored evaluation set.
- COCO images, annotation subsets, derived masks, model weights, and evaluation
  outputs are local data. Keep them out of Git; retain or improve the tracked
  reproduction guide in `data/README.md`.
- The team has two students. Student A owns data preparation and the manual
  polygon baseline. Student B combines the planned model/tool and
  evaluation/integration responsibilities.
- Keep changes surgical and commits small and logically separated. Never bundle
  unrelated documentation, frontend, backend, and data changes into one large
  commit. Do not commit unless asked or the task explicitly calls for commits.

## Milestones and evidence

- **M1:** proposal, reproducible stack/data, and a prompted inference demo.
- **M2:** usable annotation flow, saved masks and prompts, elapsed-time capture,
  and IoU/Dice comparison against the matching COCO instance.
- **M3:** complete the paired manual-versus-prompted evaluation, analyze
  limitations and failure cases, polish the tool, and finish the final report.
- Treat the reference mask as evaluation ground truth, not as guidance for
  selecting prompts. Do not reveal it in the annotation workflow.
- Never invent, extrapolate, or present placeholder values as measurements.
  Clearly distinguish automated tests, smoke tests, model confidence, inference
  latency, annotation time, IoU, and Dice.
- Preserve the report templates and use the provided milestone template when
  drafting a report. Include actual sample counts, protocol, device/model
  configuration, and limitations; do not claim a baseline result until the
  manual masks and timings have been collected.
- Avoid training or fine-tuning unless explicitly requested; the planned method
  uses a frozen model.

## Implementation workflow

1. Read the relevant source, tests, README, and project instructions before
   changing behavior. Search for existing helpers and patterns first.
2. Keep backend request validation explicit. Return actionable HTTP errors;
   never silently substitute masks or hide missing model/data setup.
3. Keep frontend states consistent when the image, COCO target, or prompts
   change. Surface loading and request errors to the user.
4. Persist accepted predictions with their COCO IDs, prompt record, elapsed
   time, and computed metrics. Validate image/annotation pairing and mask
   dimensions before recording evaluation output.
5. Update directly related documentation when commands, behavior, or report
   claims change. Preserve unrelated work and never add ignored binary assets
   to a commit.
6. Run the smallest relevant checks, then report exactly what passed and any
   remaining limitations.

## Common commands

From `backend/`:

```sh
uv sync --group dev
uv run pytest
uv run ruff check app scripts tests
uv run uvicorn app.main:app --reload
```

Install optional model dependencies only when needed:

```sh
uv sync --extra sam2 --group dev
```

Configure `SAM2_CONFIG` and `SAM2_CHECKPOINT` to the matching Hiera-Tiny
configuration and local checkpoint; use `SAM2_DEVICE=cpu` by default, or a
locally supported accelerator when available. Do not add weights to version
control.

From `frontend/`:

```sh
pnpm install
pnpm run build
pnpm run lint
pnpm dev
```

Reproduce the ignored dataset assets from `backend/` with the commands in
`data/README.md`; do not re-download them unless the task needs it.
