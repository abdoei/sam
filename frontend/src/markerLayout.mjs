const MIN_MARKER_RADIUS = 8;
const MAX_MARKER_RADIUS = 14;

/** Smallest marker radius, in on-screen pixels, that stays legible. */
export const MIN_MARKER_SCREEN_RADIUS = 9;

export function markerRadiusFor(imageWidth) {
  return Math.max(MIN_MARKER_RADIUS, Math.min(MAX_MARKER_RADIUS, imageWidth / 55));
}

function clampToBounds(value, minimum, maximum) {
  if (maximum < minimum) return (minimum + maximum) / 2;
  return Math.max(minimum, Math.min(maximum, value));
}

/**
 * Place a numbered marker for each centroid without overlapping.
 *
 * Targets of the same category can sit within a few pixels of one another, so a
 * marker drawn exactly on its centroid would hide its neighbours. The first
 * centroid keeps its position and each later one is pushed radially outward,
 * re-checking every marker already placed, until it clears all of them; `moved`
 * records whether a leader line back to the true centroid is needed. Iteration
 * order is the caller's, so the layout is stable for a given image.
 *
 * When `bounds` is given, markers are kept fully inside the image: a marker on an
 * object touching the edge would otherwise be half clipped and unreadable.
 */
export function layoutMarkers(centroids, radius, bounds) {
  if (!(radius > 0)) {
    throw new RangeError("Marker radius must be positive.");
  }
  const minimumGap = radius * 2 + 1;
  const placed = [];
  const place = (x, y) =>
    bounds
      ? [
          clampToBounds(x, radius, bounds.width - radius),
          clampToBounds(y, radius, bounds.height - radius),
        ]
      : [x, y];

  return centroids.map(({ x: originX, y: originY }) => {
    let [x, y] = place(originX, originY);

    // Each pass resolves the nearest collision; repeat until a full pass finds
    // none, because escaping one marker can push into another.
    for (let pass = 0; pass < 500; pass += 1) {
      let nearest = null;
      let nearestGap = Infinity;
      for (const other of placed) {
        const gap = Math.hypot(other.x - x, other.y - y);
        if (gap < minimumGap && gap < nearestGap) {
          nearest = other;
          nearestGap = gap;
        }
      }
      if (!nearest) break;

      const dx = x - nearest.x;
      const dy = y - nearest.y;
      const distance = Math.hypot(dx, dy);
      if (distance < 0.01) {
        // Exactly coincident: any direction works, so use a fixed one.
        x += minimumGap;
        y -= minimumGap;
      } else {
        const push = minimumGap - distance;
        x += (dx / distance) * push;
        y += (dy / distance) * push;
      }
      [x, y] = place(x, y);
    }

    placed.push({ x, y });
    return {
      x,
      y,
      originX,
      originY,
      moved: Math.hypot(x - originX, y - originY) > radius,
    };
  });
}
