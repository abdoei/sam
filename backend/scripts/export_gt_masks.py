import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export non-crowd COCO instance masks as individual PNG files."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=ROOT / "data/coco_subset/annotations/subset.json",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data/gt_masks")
    args = parser.parse_args()

    coco = COCO(str(args.annotations))
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, int | str]] = []
    skipped_crowd = 0
    skipped_empty = 0

    for annotation in coco.dataset["annotations"]:
        if annotation["iscrowd"] == 1:
            skipped_crowd += 1
            continue

        mask = coco.annToMask(annotation)
        if not np.any(mask):
            skipped_empty += 1
            continue

        mask_path = (
            args.output / str(annotation["image_id"]) / f"{annotation['id']}.png"
        )
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
        rows.append(
            {
                "ann_id": annotation["id"],
                "image_id": annotation["image_id"],
                "category_id": annotation["category_id"],
                "area": annotation["area"],
                "mask_path": mask_path.relative_to(args.output).as_posix(),
            }
        )

    index_path = args.output / "gt_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as index_file:
        writer = csv.DictWriter(
            index_file,
            fieldnames=["ann_id", "image_id", "category_id", "area", "mask_path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Masks saved: {len(rows)}")
    print(f"Skipped crowd: {skipped_crowd}")
    print(f"Skipped empty: {skipped_empty}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
