import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a concise summary of a COCO instance-annotation JSON file."
    )
    parser.add_argument(
        "annotations",
        nargs="?",
        type=Path,
        default=ROOT / "data/raw/annotations/instances_val2017.json",
    )
    args = parser.parse_args()

    with args.annotations.open(encoding="utf-8") as annotation_file:
        coco: dict[str, Any] = json.load(annotation_file)

    categories = {
        int(category["id"]): category["name"] for category in coco.get("categories", [])
    }
    print(f"File: {args.annotations}")
    print(f"Top-level keys: {', '.join(sorted(coco))}")
    print(f"Images: {len(coco.get('images', []))}")
    print(f"Annotations: {len(coco.get('annotations', []))}")
    print(f"Categories: {len(categories)}")
    print("Sample image records:")
    for image in coco.get("images", [])[:3]:
        print(
            f"  id={image['id']} file={image['file_name']} "
            f"size={image['width']}x{image['height']}"
        )
    print("Sample categories:")
    for category_id, name in list(categories.items())[:5]:
        print(f"  {category_id}: {name}")


if __name__ == "__main__":
    main()
