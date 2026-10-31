# SAM2 Interactive Annotation

A small, local web application for image annotation with SAM2 point and box prompts.
The backend is Python/FastAPI and the frontend is React, TypeScript, and Vite. Model
inference runs locally; images and prompts are not sent to a hosted segmentation
service.

## Project stages

- **M1 — Proposal and inference demo:** set up the reproducible stack, inspect a
  COCO subset, and run prompted SAM2 inference in the browser.
- **M2 — Working annotation tool:** save annotations, measure per-image annotation
  time, and compare predicted masks with COCO ground truth using IoU.
- **M3 — Evaluation and polish:** complete the time-savings and IoU analysis, add
  usability improvements and a small user test, and package the demo/report.

The team has two members: Student A owns the COCO data preparation and manual
baseline; Student B combines the catalog's Student B (model/tool) and Student C
(evaluation/integration) responsibilities.

## Requirements

- Python 3.11 or 3.12 managed with [uv](https://docs.astral.sh/uv/)
- Node.js 20+ and [pnpm](https://pnpm.io/)
- The SAM2 checkpoint and matching configuration (downloaded separately; model
  weights are intentionally not stored in Git)

## Run locally

In one terminal:

```sh
cd backend
uv sync --group dev
uv run uvicorn app.main:app --reload
```

In another terminal:

```sh
cd frontend
pnpm install
pnpm dev
```

Open the URL printed by Vite. The frontend proxies `/api` requests to
`http://127.0.0.1:8000`.

To install SAM2 in the Python environment, run `uv sync --extra sam2 --group dev`
from `backend`. After downloading the matching SAM2.1 Hiera-Tiny checkpoint, set
the model configuration and local checkpoint path before starting the backend:

```sh
export SAM2_CONFIG=configs/sam2.1/sam2.1_hiera_t.yaml
export SAM2_CHECKPOINT="$HOME/models/sam2.1_hiera_tiny.pt"
export SAM2_DEVICE=cpu
```

The default configuration targets the small SAM2.1 Hiera-Tiny model; do not pair
a checkpoint with a different model configuration. Uploads are limited to 15 MiB
and 25 million pixels to keep local inference within reasonable memory bounds.

Until the checkpoint is installed and configured, the API reports that inference
is unavailable and returns an explicit error for segmentation requests; it does
not generate substitute masks.

## COCO data layout

The official-source, 300-image COCO download and checkpoint reproduction steps
are documented in [`data/README.md`](data/README.md). From `backend/`, run:

```sh
uv run python -m scripts.download_coco_subset
uv run python -m scripts.export_gt_masks
```

To inspect the downloaded subset:

```sh
uv run python -m scripts.inspect_coco data/coco_subset/annotations/subset.json
```

The subset downloader selects 300 images from COCO 2017 val using seed 42 and
1–8 non-crowd instances per image; it fetches only those JPEGs, not the full
val2017 image archive. The selected IDs are recorded in
`docs/selected_image_ids.json`. All image, annotation, mask, and checkpoint
artifacts are ignored by Git. Read the source image licenses before redistribution.

The inspection and local-copy scripts are still available for an already-downloaded
full COCO structure:

```sh
uv run python -m scripts.inspect_coco
uv run python -m scripts.prepare_coco_subset
```

For evaluation, download each accepted mask using its COCO annotation ID and create
`data/evaluation/predictions.csv`:

```csv
ann_id,prediction_path,elapsed_seconds
123456,123456-sam2-mask.png,18.42
```

`elapsed_seconds` is the interactive timer from the first prompt through accepting
the mask. Use the same start/stop definition for the manual-polygon baseline. Then,
from `backend/`, run:

```sh
uv run python -m scripts.evaluate_masks
```

The script writes per-instance IoU and Dice along with aggregate means and any
recorded annotation times. It rejects duplicate/unknown annotation IDs and
dimension mismatches rather than silently skipping samples.

## Reproducibility

The subset selection seed and filtering rules are explicit in the scripts. Keep
the selected image IDs, annotation split, software lockfiles, model configuration,
and checkpoint identity with any reported result. Do not report placeholder or
unmeasured timing/quality values. Commit `docs/selected_image_ids.json` after
generating the subset, but do not commit the COCO images or derived masks.

The COCO subset and mask-export workflow follows the public artifact layout shared
by the Student A project at
[Almaz-Elbery/dip-sam2-annotation](https://github.com/Almaz-Elbery/dip-sam2-annotation/tree/main/scripts);
the scripts in this repository are independently adapted for command-line use.

## Report

Use [`docs/m1_proposal.tex`](docs/m1_proposal.tex) as the report-writing template.
The report should disclose AI-assisted code scaffolding in line with the course
instructions, and all measurements and claims must be verified by the team.
