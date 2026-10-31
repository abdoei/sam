import argparse
import json
import tempfile
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from scripts.prepare_coco_subset import select_image_ids

ROOT = Path(__file__).resolve().parents[2]
COCO_S3_ROOT = "https://s3.us-east-1.amazonaws.com/images.cocodataset.org"
ANNOTATIONS_URL = f"{COCO_S3_ROOT}/annotations/annotations_trainval2017.zip"
IMAGE_URL = f"{COCO_S3_ROOT}/val2017/{{file_name}}"
ANNOTATIONS_MEMBER = "annotations/instances_val2017.json"
USER_AGENT = "sam2-interactive-annotation/0.1 (COCO research subset downloader)"
MAX_DOWNLOAD_ATTEMPTS = 3


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_name(destination.name + ".part")
    try:
        for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
            try:
                with urllib.request.urlopen(_request(url), timeout=60) as response:
                    with partial_path.open("wb") as partial_file:
                        while chunk := response.read(1024 * 1024):
                            partial_file.write(chunk)
                partial_path.replace(destination)
                return
            except (OSError, urllib.error.URLError):
                partial_path.unlink(missing_ok=True)
                if attempt == MAX_DOWNLOAD_ATTEMPTS:
                    raise
    finally:
        partial_path.unlink(missing_ok=True)


def load_annotations(annotation_path: Path | None, archive_path: Path) -> dict[str, Any]:
    if annotation_path is not None:
        with annotation_path.open(encoding="utf-8") as annotation_file:
            return json.load(annotation_file)

    _download_file(ANNOTATIONS_URL, archive_path)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            with archive.open(ANNOTATIONS_MEMBER) as annotation_file:
                return json.load(annotation_file)
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise RuntimeError(
            f"Could not read {ANNOTATIONS_MEMBER} from official COCO archive {archive_path}"
        ) from exc
    finally:
        archive_path.unlink(missing_ok=True)


def download_image(image: dict[str, Any], output_dir: Path) -> Path:
    file_name = str(image["file_name"])
    if Path(file_name).name != file_name or file_name in {"", ".", ".."}:
        raise ValueError(f"Unsafe COCO image filename: {file_name!r}")

    destination = output_dir / file_name
    if destination.is_file() and destination.stat().st_size > 0:
        with destination.open("rb") as image_file:
            if image_file.read(3) == b"\xff\xd8\xff":
                return destination
        destination.unlink()

    _download_file(IMAGE_URL.format(file_name=file_name), destination)
    with destination.open("rb") as image_file:
        if image_file.read(3) != b"\xff\xd8\xff":
            destination.unlink(missing_ok=True)
            raise ValueError(f"Downloaded COCO image is not a JPEG: {destination}")
    return destination


def download_subset(
    coco: dict[str, Any],
    output_dir: Path,
    ids_output: Path,
    count: int,
    seed: int,
    min_instances: int,
    max_instances: int,
    workers: int,
) -> tuple[int, int]:
    ids = select_image_ids(
        images=coco["images"],
        annotations=coco["annotations"],
        sample_size=count,
        seed=seed,
        min_instances=min_instances,
        max_instances=max_instances,
    )
    selected_ids = set(ids)
    selected_images = [image for image in coco["images"] if image["id"] in selected_ids]
    image_dir = output_dir / "images"
    annotation_dir = output_dir / "annotations"
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    ids_output.parent.mkdir(parents=True, exist_ok=True)

    subset = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
        "images": selected_images,
        "annotations": [
            annotation
            for annotation in coco["annotations"]
            if annotation["image_id"] in selected_ids
        ],
    }

    failures: list[tuple[str, Exception]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_image, image, image_dir): image["file_name"]
            for image in selected_images
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                failures.append((futures[future], exc))

    if failures:
        details = "; ".join(f"{name}: {error}" for name, error in failures[:5])
        raise RuntimeError(
            f"Failed to download {len(failures)} of {len(selected_images)} COCO images. "
            f"Successful files are retained so the command can be rerun. {details}"
        )

    ids_output.write_text(json.dumps(ids, indent=2) + "\n", encoding="utf-8")
    subset_path = annotation_dir / "subset.json"
    subset_path.write_text(json.dumps(subset), encoding="utf-8")
    return len(selected_images), len(subset["annotations"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a deterministic COCO val2017 image subset directly from the official source. "
            "Only the COCO annotations archive and selected images are downloaded."
        )
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        help="Optional existing instances_val2017.json file; otherwise fetch the official archive.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/coco_subset",
    )
    parser.add_argument(
        "--ids-output",
        type=Path,
        default=ROOT / "docs/selected_image_ids.json",
    )
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-instances", type=int, default=1)
    parser.add_argument("--max-instances", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be at least one")

    (ROOT / "data").mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="coco-annotations-",
        suffix=".zip",
        dir=ROOT / "data",
        delete=False,
    ) as archive_file:
        archive_path = Path(archive_file.name)
    archive_path.unlink()

    try:
        coco = load_annotations(args.annotations, archive_path)
    finally:
        archive_path.unlink(missing_ok=True)

    image_count, annotation_count = download_subset(
        coco=coco,
        output_dir=args.output,
        ids_output=args.ids_output,
        count=args.count,
        seed=args.seed,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
        workers=args.workers,
    )
    print(f"Selected images: {image_count}")
    print(f"Selected annotations: {annotation_count}")
    print(f"COCO image subset: {args.output}")
    print(f"Selected IDs: {args.ids_output}")


if __name__ == "__main__":
    main()
