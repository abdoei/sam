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
from `backend`. Set `SAM2_CHECKPOINT` to the downloaded checkpoint file and
`SAM2_CONFIG` to the matching SAM2 configuration name/path before starting the
backend. The default configuration targets the small SAM2.1 Hiera-T model; do not
pair a checkpoint with a different model configuration.

Until the checkpoint is installed and configured, the API reports that inference
is unavailable and returns an explicit error for segmentation requests; it does
not generate substitute masks.

## COCO data layout

Download the COCO 2017 validation images and annotations from the official
[COCO download page](https://cocodataset.org/#download), then extract them without
committing the dataset:

```text
data/
└── raw/
    ├── val2017/
    │   └── 000000000139.jpg
    └── annotations/
        └── instances_val2017.json
```

The preprocessing scripts use this layout. They deterministically select 300
images (seed 42) with 1–8 non-crowd instances, write the chosen COCO image IDs,
and export binary ground-truth masks. Read the source dataset's license and
individual image terms before sharing any data or demo output; the dataset is not
included in this repository.

## Reproducibility

The subset selection seed and filtering rules are explicit in the scripts. Keep
the selected image IDs, annotation split, software lockfiles, model configuration,
and checkpoint identity with any reported result. Do not report placeholder or
unmeasured timing/quality values.

The COCO subset and mask-export workflow follows the public artifact layout shared
by the Student A project at
[Almaz-Elbery/dip-sam2-annotation](https://github.com/Almaz-Elbery/dip-sam2-annotation/tree/main/scripts);
the scripts in this repository are independently adapted for command-line use.

## Report

Use [`docs/m1_proposal.tex`](docs/m1_proposal.tex) as the report-writing template.
The report should disclose AI-assisted code scaffolding in line with the course
instructions, and all measurements and claims must be verified by the team.
