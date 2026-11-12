import pytest

from app.polygon import (
    MAX_POLYGON_PARTS,
    polygon_area,
    polygon_mask_png,
    polygon_summary,
    rasterize_parts,
    rasterize_polygon,
    simplify_polyline,
    validate_polygon,
)


def test_polygon_rasterises_the_enclosed_pixels_only() -> None:
    mask = rasterize_polygon([[1, 1], [5, 1], [5, 5], [1, 5]], 8, 8)

    assert mask.shape == (8, 8)
    assert mask[2, 2] and mask[4, 4]
    assert not mask[0, 0], "outside the square"
    assert not mask[6, 6], "outside the square"


def test_polygon_area_is_close_to_the_mask_it_fills() -> None:
    points = validate_polygon([[1, 1], [5, 1], [5, 5], [1, 5]], 8, 8)

    assert polygon_area(points) == pytest.approx(16)
    # PIL fills boundary pixels too, so the mask covers the 4x4 region plus its
    # outline; the two agree to within the boundary, which is what matters for IoU.
    filled = rasterize_polygon(points, 8, 8).sum()
    assert 16 <= filled <= 25


def test_polygon_rejects_out_of_bounds_points() -> None:
    with pytest.raises(ValueError, match="outside"):
        validate_polygon([[0, 0], [20, 0], [20, 20]], 8, 8)


def test_polygon_rejects_too_few_points() -> None:
    with pytest.raises(ValueError, match="at least three"):
        validate_polygon([[0, 0], [4, 4]], 8, 8)


def test_polygon_rejects_negative_coordinates() -> None:
    with pytest.raises(ValueError, match="outside"):
        validate_polygon([[-2, 0], [4, 0], [4, 4]], 8, 8)


def test_polygon_rejects_non_numeric_points() -> None:
    with pytest.raises(ValueError, match="not numeric"):
        validate_polygon([[0, 0], ["x", 0], [4, 4]], 8, 8)


def test_degenerate_polygon_is_rejected_before_rasterising() -> None:
    with pytest.raises(ValueError, match="zero area"):
        rasterize_polygon([[1, 1], [2, 2], [3, 3]], 8, 8)


def test_mask_png_matches_the_rasterised_mask() -> None:
    from io import BytesIO

    import numpy as np
    from PIL import Image

    png = polygon_mask_png([[1, 1], [6, 1], [6, 6], [1, 6]], 8, 8)

    assert png.startswith(b"\x89PNG")
    with Image.open(BytesIO(png)) as image:
        decoded = np.asarray(image.convert("L")) > 0
    assert np.array_equal(decoded, rasterize_polygon([[1, 1], [6, 1], [6, 6], [1, 6]], 8, 8))


def test_simplification_drops_collinear_points() -> None:
    # A straight trace with many samples reduces to its endpoints.
    straight = [[float(x), 5.0] for x in range(0, 21)]

    simplified = simplify_polyline(straight, 1.0)

    assert len(simplified) < len(straight)
    assert simplified[0] == (0.0, 5.0)
    assert simplified[-1] == (20.0, 5.0)


def test_simplification_keeps_a_real_corner() -> None:
    corner = [[0.0, 0.0], [5.0, 0.0], [10.0, 0.0], [10.0, 10.0]]

    simplified = simplify_polyline(corner, 0.5)

    assert (10.0, 0.0) in simplified
    assert len(simplified) == 3


def test_simplification_with_zero_tolerance_drops_only_collinear_points() -> None:
    # Points on the same straight line are removed; any deviation survives even
    # with no tolerance, which is what keeps a freehand corner intact.
    points = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 2.0]]

    simplified = simplify_polyline(points, 0)

    assert (1.0, 0.0) not in simplified, "the collinear midpoint is dropped"
    assert (2.0, 0.0) in simplified, "the corner is kept"
    assert simplified[0] == (0.0, 0.0) and simplified[-1] == (2.0, 2.0)


def test_simplification_rejects_negative_tolerance() -> None:
    with pytest.raises(ValueError, match="tolerance"):
        simplify_polyline([[0.0, 0.0], [1.0, 1.0]], -1)


def test_disconnected_parts_are_unioned_not_replaced() -> None:
    # Two 2x2 blocks far apart, like a pill visible either side of an occluder.
    left = [[1, 1], [3, 1], [3, 3], [1, 3]]
    right = [[10, 10], [13, 10], [13, 13], [10, 13]]

    both = rasterize_parts([left, right], 16, 16)

    assert both[2, 2], "the first part is kept"
    assert both[11, 11], "the second part is added"
    assert both.sum() > rasterize_parts([left], 16, 16).sum(), "the union is larger"


def test_parts_mask_equals_the_union_of_individual_masks() -> None:
    import numpy as np

    left = [[1, 1], [4, 1], [4, 4], [1, 4]]
    right = [[8, 8], [12, 8], [12, 12], [8, 12]]

    combined = rasterize_parts([left, right], 16, 16)
    expected = rasterize_polygon(left, 16, 16) | rasterize_polygon(right, 16, 16)

    assert np.array_equal(combined, expected)


def test_an_invalid_part_is_reported_by_number() -> None:
    good = [[1, 1], [3, 1], [3, 3]]
    bad = [[1, 1], [3, 1]]  # two points cannot enclose anything

    with pytest.raises(ValueError, match="Part 2"):
        rasterize_parts([good, bad], 8, 8)


def test_a_zero_area_part_is_rejected() -> None:
    with pytest.raises(ValueError, match="Part 1 has zero area"):
        rasterize_parts([[[1, 1], [2, 2], [3, 3]]], 8, 8)


def test_part_limit_is_enforced() -> None:
    ring = [[1, 1], [3, 1], [3, 3]]
    with pytest.raises(ValueError, match="part limit"):
        rasterize_parts([ring] * (MAX_POLYGON_PARTS + 1), 16, 16)


def test_summary_describes_all_parts() -> None:
    left = [(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)]
    right = [(10.0, 10.0), (13.0, 10.0), (13.0, 13.0), (10.0, 13.0)]

    summary = polygon_summary([left, right])

    assert summary["part_count"] == 2
    assert summary["vertex_count"] == 8
    assert summary["area"] == pytest.approx(13.0)
    assert summary["bbox"] == [1.0, 1.0, 13.0, 13.0]
