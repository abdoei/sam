import assert from "node:assert/strict";
import test from "node:test";

import { layoutMarkers, markerRadiusFor } from "../src/markerLayout.mjs";

test("marker radius stays within legible bounds across image sizes", () => {
  assert.equal(markerRadiusFor(320), 8, "small images use the floor");
  assert.equal(markerRadiusFor(2400), 14, "large images use the ceiling");
  assert.ok(markerRadiusFor(480) > markerRadiusFor(320));
});

test("distant centroids keep their exact positions", () => {
  const placed = layoutMarkers(
    [
      { x: 10, y: 10 },
      { x: 200, y: 10 },
      { x: 100, y: 300 },
    ],
    10,
  );

  placed.forEach((marker) => {
    assert.equal(marker.x, marker.originX);
    assert.equal(marker.y, marker.originY);
    assert.equal(marker.moved, false);
  });
});

test("crowded centroids are separated and flagged as moved", () => {
  // Five people whose centroids sit within a few pixels, as in COCO image 1761.
  const centroids = [
    { x: 13.56, y: 621.75 },
    { x: 7.6, y: 621.18 },
    { x: 18.02, y: 620.65 },
    { x: 37.86, y: 629.24 },
    { x: 23.67, y: 621.14 },
  ];

  const placed = layoutMarkers(centroids, 8);
  const minimumGap = 8 * 2 + 1;

  for (let index = 0; index < placed.length; index += 1) {
    for (let other = index + 1; other < placed.length; other += 1) {
      const gap = Math.hypot(
        placed[index].x - placed[other].x,
        placed[index].y - placed[other].y,
      );
      assert.ok(
        gap >= minimumGap - 1e-9,
        `markers ${index} and ${other} overlap (gap ${gap})`,
      );
    }
  }

  assert.equal(placed[0].moved, false, "the first marker stays on its centroid");
  assert.ok(
    placed.slice(1).some((marker) => marker.moved),
    "later markers are pushed clear",
  );
});

test("exactly coincident centroids do not collapse onto each other", () => {
  const placed = layoutMarkers(
    [
      { x: 50, y: 50 },
      { x: 50, y: 50 },
      { x: 50, y: 50 },
    ],
    8,
  );

  assert.equal(new Set(placed.map((marker) => `${marker.x},${marker.y}`)).size, 3);
});

test("a non-positive radius is rejected", () => {
  assert.throws(() => layoutMarkers([{ x: 0, y: 0 }], 0), RangeError);
});

test("markers on edge-touching objects stay fully inside the image", () => {
  // The persons in COCO image 1761 sit on the bottom edge; an unclamped marker
  // there is half cut off and unreadable.
  const bounds = { width: 427, height: 640 };
  const placed = layoutMarkers(
    [
      { x: 13.56, y: 621.75 },
      { x: 7.6, y: 639 },
      { x: 3, y: 637 },
    ],
    10,
    bounds,
  );

  placed.forEach((marker, index) => {
    assert.ok(marker.x >= 10 && marker.x <= 417, `marker ${index} x in bounds`);
    assert.ok(marker.y >= 10 && marker.y <= 630, `marker ${index} y in bounds`);
  });
});
