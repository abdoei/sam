import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def select_image_ids(
    images: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    sample_size: int,
    seed: int,
    min_instances: int,
    max_instances: int,
) -> list[int]:
    if sample_size < 1:
        raise ValueError("sample_size must be greater than zero")
    if min_instances < 0 or max_instances < min_instances:
        raise ValueError("instance limits must satisfy 0 <= min_instances <= max_instances")

    non_crowd_counts = Counter(
        int(annotation["image_id"])
        for annotation in annotations
        if int(annotation.get("iscrowd", 0)) == 0
    )
    candidates = sorted(
        int(image["id"])
        for image in images
        if min_instances <= non_crowd_counts[int(image["id"])] <= max_instances
    )
    if len(candidates) < sample_size:
        raise ValueError(
            f"Only {len(candidates)} images meet the filtering rules; "
            f"cannot select {sample_size}"
        )

    return sorted(random.Random(seed).sample(candidates, sample_size))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a deterministic COCO image subset and copy its source images."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=ROOT / "data/raw/annotations/instances_val2017.json",
    )
    parser.add_argument("--images", type=Path, default=ROOT / "data/raw/val2017")
    parser.add_argument("--output", type=Path, default=ROOT / "data/coco_subset")
    parser.add_argument("--ids-output", type=Path, default=ROOT / "docs/selected_image_ids.json")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-instances", type=int, default=1)
    parser.add_argument("--max-instances", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.annotations.open(encoding="utf-8") as annotation_file:
        coco = json.load(annotation_file)

    selected_ids = select_image_ids(
        images=coco["images"],
        annotations=coco["annotations"],
        sample_size=args.count,
        seed=args.seed,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
    )
    selected = set(selected_ids)
    selected_images = [image for image in coco["images"] if image["id"] in selected]
    missing_images = [
        image["file_name"]
        for image in selected_images
        if not (args.images / image["file_name"]).is_file()
    ]
    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} selected COCO images are missing from {args.images}; "
            f"first missing file: {missing_images[0]}"
        )

    output_annotations = args.output / "annotations"
    output_images = args.output / "images"
    output_annotations.mkdir(parents=True, exist_ok=True)
    output_images.mkdir(parents=True, exist_ok=True)
    args.ids_output.parent.mkdir(parents=True, exist_ok=True)

    subset = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
        "images": selected_images,
        "annotations": [
            annotation
            for annotation in coco["annotations"]
            if annotation["image_id"] in selected
        ],
    }
    for image in selected_images:
        shutil.copy2(args.images / image["file_name"], output_images / image["file_name"])

    args.ids_output.write_text(json.dumps(selected_ids, indent=2) + "\n", encoding="utf-8")
    (output_annotations / "subset.json").write_text(
        json.dumps(subset), encoding="utf-8"
    )
    print(f"Selected images: {len(selected_images)}")
    print(f"Selected annotations: {len(subset['annotations'])}")
    print(f"Image IDs: {args.ids_output}")
    print(f"Subset: {args.output}")


if __name__ == "__main__":
    main()
