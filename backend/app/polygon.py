"""Rasterise polygon annotations into binary masks.

The polygon baseline is scored with the same IoU/Dice comparison as the SAM2
condition, so its geometry is converted to a mask here rather than in the browser:
the frontend sends the outline it drew and the backend decides which pixels that
outline covers.
"""

from typing import Any

import numpy as np
from PIL import Image, ImageDraw

MAX_POLYGON_POINTS = 20_000


def validate_polygon(
    points: list[list[float]] | list[tuple[float, float]],
    width: int,
    height: int,
) -> list[tuple[float, float]]:
    """Return the polygon as (x, y) pairs, rejecting anything unusable."""
    if not isinstance(points, list):
        raise ValueError("Polygon points must be a list of [x, y] pairs")
    if len(points) < 3:
        raise ValueError("A polygon needs at least three points")
    if len(points) > MAX_POLYGON_POINTS:
        raise ValueError(f"Polygon exceeds the {MAX_POLYGON_POINTS}-point limit")

    cleaned: list[tuple[float, float]] = []
    for index, point in enumerate(points):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"Polygon point {index} must be an [x, y] pair")
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Polygon point {index} is not numeric") from exc
        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError(f"Polygon point {index} is not finite")
        if x < 0 or y < 0 or x > width - 1 or y > height - 1:
            raise ValueError(
                f"Polygon point {index} ({x:g}, {y:g}) is outside the "
                f"{width}x{height} image"
            )
        cleaned.append((x, y))
    return cleaned


def polygon_area(points: list[tuple[float, float]]) -> float:
    """Shoelace area; used to reject degenerate outlines before rasterising."""
    total = 0.0
    for index, (x0, y0) in enumerate(points):
        x1, y1 = points[(index + 1) % len(points)]
        total += x0 * y1 - x1 * y0
    return abs(total) / 2


def rasterize_polygon(
    points: list[list[float]] | list[tuple[float, float]],
    width: int,
    height: int,
) -> np.ndarray:
    """Fill a polygon into a boolean mask of the given size."""
    cleaned = validate_polygon(points, width, height)
    if polygon_area(cleaned) <= 0:
        raise ValueError("Polygon has zero area; draw a shape enclosing some pixels")

    canvas = Image.new("L", (width, height), 0)
    ImageDraw.Draw(canvas).polygon(cleaned, fill=255, outline=255)
    return np.asarray(canvas, dtype=np.uint8) > 0


def polygon_mask_png(
    points: list[list[float]] | list[tuple[float, float]],
    width: int,
    height: int,
) -> bytes:
    """Encode the polygon's mask as an 8-bit PNG, matching SAM2's mask format."""
    mask = rasterize_polygon(points, width, height)
    from io import BytesIO

    buffer = BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def simplify_polyline(
    points: list[list[float]] | list[tuple[float, float]],
    tolerance: float,
) -> list[tuple[float, float]]:
    """Reduce a freehand trace with Ramer-Douglas-Peucker.

    A freehand drag produces hundreds of samples; storing and scoring all of them
    is slow and hides how much effort the trace actually took. Simplification
    keeps the stored shape faithful while reporting its true vertex count.
    """
    if tolerance < 0:
        raise ValueError("Simplification tolerance must not be negative")
    if len(points) < 3:
        return [(float(x), float(y)) for x, y in points]

    coords = [(float(p[0]), float(p[1])) for p in points]
    keep = {0, len(coords) - 1}
    stack = [(0, len(coords) - 1)]
    while stack:
        start, end = stack.pop()
        x0, y0 = coords[start]
        x1, y1 = coords[end]
        segment = np.hypot(x1 - x0, y1 - y0)
        best_index, best_distance = -1, tolerance
        for index in range(start + 1, end):
            x, y = coords[index]
            if segment == 0:
                distance = float(np.hypot(x - x0, y - y0))
            else:
                distance = float(
                    abs((x1 - x0) * (y0 - y) - (x0 - x) * (y1 - y0)) / segment
                )
            if distance > best_distance:
                best_index, best_distance = index, distance
        if best_index != -1:
            keep.add(best_index)
            stack.append((start, best_index))
            stack.append((best_index, end))
    return [coords[index] for index in sorted(keep)]


def polygon_summary(points: list[tuple[float, float]]) -> dict[str, Any]:
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    return {
        "vertex_count": len(points),
        "area": round(polygon_area(points), 2),
        "bbox": [min(xs), min(ys), max(xs), max(ys)],
    }
