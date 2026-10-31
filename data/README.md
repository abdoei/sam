# Local data and model assets

All downloaded images, annotations, masks, and model weights in this directory are
local-only and ignored by Git. This file is tracked so teammates can reproduce the
assets without putting binary datasets or checkpoints in the repository.

## COCO 2017 validation subset

The downloader uses the official COCO 2017 validation instance annotations and
downloads only the selected JPEGs from the COCO public S3 bucket. It does **not**
download the full 5,000-image validation archive. It uses the TLS-valid path-style
S3 URL for the same official `images.cocodataset.org` bucket.

From `backend/`:

```sh
uv run python -m scripts.download_coco_subset
uv run python -m scripts.export_gt_masks
```

By default, the script selects 300 images from `val2017`, with 1–8 non-crowd
instances per image, using seed `42`. Candidate IDs are sorted before sampling;
the chosen IDs are written to the tracked `docs/selected_image_ids.json`. The
official annotation archive is downloaded to a temporary file, parsed, and removed.
The selected images and subset annotations are placed here:

```text
data/coco_subset/
├── annotations/subset.json
└── images/
    └── <COCO file_name>.jpg
```

The sampling rules can be adjusted explicitly, for example:

```sh
uv run python -m scripts.download_coco_subset \
  --count 300 --seed 42 --min-instances 1 --max-instances 8 --workers 8
```

The command is resumable: existing JPEGs with valid JPEG signatures are reused;
failed/partial downloads are removed and failed files are named in the error. To
use a previously downloaded official `instances_val2017.json`, pass
`--annotations /path/to/instances_val2017.json`.

The exported non-crowd ground-truth masks are written under `data/gt_masks/`.
Check each source image's license metadata before redistribution. Do not upload
COCO source images or derived masks to a public demo or commit them to Git.

## SAM2.1 Hiera-Tiny checkpoint

The downloaded checkpoint is the official Meta SAM 2.1 Hiera-Tiny model. It
matches the application's `configs/sam2.1/sam2.1_hiera_t.yaml` configuration.
Download it from the official public file host:

```sh
mkdir -p data/checkpoints
curl --fail --location --retry 3 \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt \
  --output data/checkpoints/sam2.1_hiera_tiny.pt
shasum -a 256 data/checkpoints/sam2.1_hiera_tiny.pt
```

The verified SHA-256 for the downloaded checkpoint is
`7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`. The
checkpoint is intentionally not tracked in Git. For local inference, install the
optional model dependency from `backend/` and point the app to this file:

```sh
uv sync --extra sam2 --group dev
export SAM2_CONFIG=configs/sam2.1/sam2.1_hiera_t.yaml
export SAM2_CHECKPOINT="$PWD/../data/checkpoints/sam2.1_hiera_tiny.pt"
export SAM2_DEVICE=cpu
uv run uvicorn app.main:app --reload
```

SAM2 uses a separate software/model license; review Meta's SAM 2 repository and
checkpoint terms before use. Do not substitute checkpoints from another model
size without updating `SAM2_CONFIG`.
