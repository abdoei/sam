"""Merge annotation results and produce charts and a written summary.

Reads any mix of evaluation directories, exported archives and metrics CSVs, then
reports quality, speed and the paired difference between conditions. Charts are
saved as PNG and SVG so they can go straight into the report, and the summary is
written as JSON and Markdown.

Statistics are computed on what the data supports: the paired comparison only uses
annotations both conditions annotated, because comparing means over different
object sets would compare the objects rather than the methods.
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from app.coco import EVALUATION_METHODS, evaluation_dir  # noqa: E402
from scripts.results_dataset import Dataset, merge, to_rows  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
METHOD_STYLES = {
    "sam2": {"color": "#2563eb", "label": "SAM2 prompting"},
    "polygon": {"color": "#d97706", "label": "Manual polygon"},
}


def _style(method: str) -> dict[str, str]:
    return METHOD_STYLES.get(method, {"color": "#6b7280", "label": method})


def _describe(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "stdev": statistics.pstdev(ordered) if len(ordered) > 1 else 0.0,
        "min": ordered[0],
        "max": ordered[-1],
        "p10": ordered[max(0, int(round(0.10 * (len(ordered) - 1))))],
        "p90": ordered[max(0, int(round(0.90 * (len(ordered) - 1))))],
    }


def summarise(dataset: Dataset) -> dict[str, Any]:
    summary: dict[str, Any] = {"sources": dataset.sources, "methods": {}}
    for method, results in sorted(dataset.by_method().items()):
        ious = [result.iou for result in results]
        dices = [result.dice for result in results]
        times = [
            result.elapsed_seconds
            for result in results
            if result.elapsed_seconds is not None
        ]
        by_category: dict[str, list[float]] = {}
        for result in results:
            by_category.setdefault(result.category_name, []).append(result.iou)
        summary["methods"][method] = {
            "images": len({result.image_id for result in results}),
            "categories": sorted(by_category),
            "iou": _describe(ious),
            "dice": _describe(dices),
            "seconds": _describe(times),
            "per_category": {
                name: _describe(values) for name, values in sorted(by_category.items())
            },
        }

    pair = paired_comparison(dataset)
    if pair:
        summary["paired"] = pair
    return summary


def paired_comparison(dataset: Dataset) -> dict[str, Any] | None:
    grouped = dataset.by_method()
    if "sam2" not in grouped or "polygon" not in grouped:
        return None
    sam2 = {result.ann_id: result for result in grouped["sam2"]}
    polygon = {result.ann_id: result for result in grouped["polygon"]}
    shared = sorted(set(sam2) & set(polygon))
    if not shared:
        return None

    iou_sam2 = [sam2[ann_id].iou for ann_id in shared]
    iou_poly = [polygon[ann_id].iou for ann_id in shared]
    timed = [
        ann_id
        for ann_id in shared
        if sam2[ann_id].elapsed_seconds is not None
        and polygon[ann_id].elapsed_seconds is not None
    ]
    result: dict[str, Any] = {
        "count": len(shared),
        "mean_iou_sam2": statistics.fmean(iou_sam2),
        "mean_iou_polygon": statistics.fmean(iou_poly),
        "iou_difference": statistics.fmean(iou_sam2) - statistics.fmean(iou_poly),
        "wins_sam2": sum(1 for a, b in zip(iou_sam2, iou_poly) if a > b),
        "wins_polygon": sum(1 for a, b in zip(iou_sam2, iou_poly) if b > a),
        "ties": sum(1 for a, b in zip(iou_sam2, iou_poly) if a == b),
        "timed_count": len(timed),
    }
    if timed:
        seconds_sam2 = [sam2[ann_id].elapsed_seconds for ann_id in timed]
        seconds_poly = [polygon[ann_id].elapsed_seconds for ann_id in timed]
        mean_sam2 = statistics.fmean(seconds_sam2)
        mean_poly = statistics.fmean(seconds_poly)
        result.update(
            {
                "mean_seconds_sam2": mean_sam2,
                "mean_seconds_polygon": mean_poly,
                "mean_seconds_saved": mean_poly - mean_sam2,
                "time_reduction_percent": (
                    (mean_poly - mean_sam2) / mean_poly * 100 if mean_poly else None
                ),
                "speedup_factor": (mean_poly / mean_sam2) if mean_sam2 else None,
            }
        )
    return result


def _save(figure, output_dir: Path, name: str) -> list[str]:
    paths = []
    for suffix in ("png", "svg"):
        path = output_dir / f"{name}.{suffix}"
        figure.savefig(path, dpi=170, bbox_inches="tight")
        paths.append(str(path))
    plt.close(figure)
    return paths


def make_charts(dataset: Dataset, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = dataset.by_method()
    written: list[str] = []
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    # 1. IoU and Dice distributions side by side.
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for axis, metric in zip(axes, ("iou", "dice")):
        data, labels, colors = [], [], []
        for method, results in sorted(grouped.items()):
            data.append([getattr(result, metric) for result in results])
            labels.append(f"{_style(method)['label']}\n(n={len(results)})")
            colors.append(_style(method)["color"])
        box = axis.boxplot(data, patch_artist=True, medianprops={"color": "#111827"})
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        axis.set_xticklabels(labels, fontsize=8)
        axis.set_title(metric.upper(), fontsize=10, loc="left")
        axis.set_ylim(0, 1.02)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Mask agreement with COCO ground truth", fontsize=11, x=0.02, ha="left", y=1.04)
    written += _save(figure, output_dir, "quality_distribution")

    # 2. Annotation time distributions.
    timed: dict[str, list[float]] = {}
    for method, results in sorted(dataset.by_method().items()):
        values = [r.elapsed_seconds for r in results if r.elapsed_seconds is not None]
        if values:
            timed[method] = [value for value in values if value is not None]
    if timed:
        figure, axis = plt.subplots(figsize=(5.2, 3.2))
        data = list(timed.values())
        labels = [
            f"{_style(method)['label']}\n(n={len(values)})" for method, values in timed.items()
        ]
        box = axis.boxplot(data, patch_artist=True, medianprops={"color": "#111827"})
        for patch, method in zip(box["boxes"], timed):
            patch.set_facecolor(_style(method)["color"])
            patch.set_alpha(0.55)
        axis.set_xticklabels(labels, fontsize=8)
        axis.set_ylabel("seconds per object")
        axis.set_title("Annotation time", fontsize=10, loc="left")
        axis.grid(axis="y", alpha=0.25)
        written += _save(figure, output_dir, "time_distribution")

    # 3. Time versus quality.
    if timed:
        figure, axis = plt.subplots(figsize=(5.6, 3.6))
        for method, results in sorted(grouped.items()):
            points = [r for r in results if r.elapsed_seconds is not None]
            if not points:
                continue
            axis.scatter(
                [float(r.elapsed_seconds) for r in points if r.elapsed_seconds is not None],
                [r.iou for r in points],
                s=34,
                alpha=0.75,
                color=_style(method)["color"],
                edgecolor="white",
                linewidth=0.6,
                label=_style(method)["label"],
            )
        axis.set_xlabel("annotation time (s)")
        axis.set_ylabel("IoU")
        axis.set_ylim(0, 1.02)
        axis.set_title("Time versus quality", fontsize=10, loc="left")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
        written += _save(figure, output_dir, "time_vs_quality")

    # 4. Mean IoU per category.
    categories = sorted({result.category_name for result in dataset.results})
    if categories and len(grouped) > 1:
        figure, axis = plt.subplots(figsize=(max(6, 0.34 * len(categories) * len(grouped)), 3.2))
        width = 0.8 / max(1, len(grouped))
        for offset, (method, results) in enumerate(sorted(grouped.items())):
            by_category: dict[str, list[float]] = {}
            for result in results:
                by_category.setdefault(result.category_name, []).append(result.iou)
            positions = [
                index + offset * width - (len(grouped) - 1) * width / 2
                for index in range(len(categories))
            ]
            axis.bar(
                positions,
                [statistics.fmean(by_category.get(name, [0.0])) or 0.0 for name in categories],
                width=width,
                label=_style(method)["label"],
                color=_style(method)["color"],
                alpha=0.85,
            )
        axis.set_xticks(range(len(categories)))
        axis.set_xticklabels(categories, rotation=45, ha="right", fontsize=7)
        axis.set_ylabel("mean IoU")
        axis.set_ylim(0, 1.02)
        axis.set_title("Quality by category", fontsize=10, loc="left")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
        written += _save(figure, output_dir, "per_category")

    # 5. Paired comparison, when both conditions cover the same annotations.
    pair = paired_comparison(dataset)
    if pair and pair["count"]:
        figure, axis = plt.subplots(figsize=(5.4, 3.4))
        inds = [0, 1]
        axis.bar(
            inds,
            [pair["mean_iou_polygon"], pair["mean_iou_sam2"]],
            color=[_style("polygon")["color"], _style("sam2")["color"]],
            alpha=0.85,
            width=0.55,
        )
        axis.set_xticks(inds)
        axis.set_xticklabels([_style("polygon")["label"], _style("sam2")["label"]], fontsize=8)
        axis.set_ylabel("mean IoU")
        axis.set_ylim(0, 1.02)
        axis.set_title(
            f"Paired comparison on {pair['count']} shared annotations", fontsize=10, loc="left"
        )
        for index, value in enumerate([pair["mean_iou_polygon"], pair["mean_iou_sam2"]]):
            axis.text(index, value + 0.02, f"{value:.3f}", ha="center", fontsize=8)
        written += _save(figure, output_dir, "paired_comparison")

    return written


def _markdown(summary: dict[str, Any]) -> str:
    lines = ["# Annotation results", ""]
    lines.append("Sources: " + ", ".join(f"`{source}`" for source in summary["sources"]))
    lines.append("")
    for method, stats in summary["methods"].items():
        label = _style(method)["label"]
        iou = stats["iou"]
        seconds = stats["seconds"]
        lines += [
            f"## {label}",
            "",
            f"- Annotations: **{iou.get('count', 0)}** across {stats['images']} images",
            f"- IoU: mean **{iou.get('mean', 0):.3f}**, median {iou.get('median', 0):.3f}, "
            f"p10–p90 {iou.get('p10', 0):.3f}–{iou.get('p90', 0):.3f}",
            f"- Dice: mean **{stats['dice'].get('mean', 0):.3f}**",
        ]
        if seconds.get("count"):
            lines.append(
                f"- Time: mean **{seconds['mean']:.1f}s**, median {seconds['median']:.1f}s, "
                f"min {seconds['min']:.1f}s, max {seconds['max']:.1f}s"
            )
        lines.append("")
    pair = summary.get("paired")
    if pair:
        lines += [
            "## Paired comparison",
            "",
            f"Restricted to the {pair['count']} annotations both conditions covered.",
            "",
            f"- Mean IoU: SAM2 {pair['mean_iou_sam2']:.3f} vs polygon "
            f"{pair['mean_iou_polygon']:.3f} (difference {pair['iou_difference']:+.3f})",
            f"- SAM2 scored higher on {pair['wins_sam2']}, polygon on "
            f"{pair['wins_polygon']}, {pair['ties']} ties",
        ]
        if pair.get("timed_count"):
            lines += [
                f"- Mean time: SAM2 {pair['mean_seconds_sam2']:.1f}s vs polygon "
                f"{pair['mean_seconds_polygon']:.1f}s",
                f"- Saved **{pair['mean_seconds_saved']:.1f}s per object** "
                f"({pair['time_reduction_percent']:.0f}% less), "
                f"{pair['speedup_factor']:.2f}x faster",
            ]
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge annotation results from directories, archives or CSVs and produce "
            "charts plus a written summary."
        )
    )
    parser.add_argument(
        "sources",
        nargs="*",
        type=Path,
        help="Evaluation directories, exported .zip archives or metrics .csv files. "
        "Defaults to the live evaluation directories that have records.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data/reports",
        help="Directory for charts, summary.json and summary.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = args.sources
    if not sources:
        sources = [
            evaluation_dir(method)
            for method in sorted(EVALUATION_METHODS)
            if (evaluation_dir(method) / "records").is_dir()
            and any((evaluation_dir(method) / "records").glob("*.json"))
        ]
        if not sources:
            raise SystemExit("No results found; annotate something or pass a source path.")

    missing = [path for path in sources if not path.exists()]
    if missing:
        raise SystemExit("Source not found: " + ", ".join(str(path) for path in missing))

    dataset = merge(sources)
    if not dataset.results:
        raise SystemExit("The sources contain no usable results.")

    args.out.mkdir(parents=True, exist_ok=True)
    summary = summarise(dataset)
    charts = make_charts(dataset, args.out)

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.out / "summary.md").write_text(_markdown(summary), encoding="utf-8")
    (args.out / "results.csv").write_text(
        _rows_csv(dataset), encoding="utf-8"
    )

    print(_markdown(summary))
    print("Charts:")
    for path in charts:
        print(f"  {path}")


def _rows_csv(dataset: Dataset) -> str:
    import csv
    from io import StringIO

    rows = to_rows(dataset)
    if not rows:
        return ""
    handle = StringIO()
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


if __name__ == "__main__":
    main()
