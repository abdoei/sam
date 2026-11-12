import assert from "node:assert/strict";
import test from "node:test";

import { zoomViewportFor } from "../src/targetZoom.mjs";

const IMAGE = { width: 427, height: 640 };

test("frames a small target with margin around it", () => {
  // A person in COCO image 1761: 9x14 px near the bottom edge.
  const view = zoomViewportFor([7, 614, 9, 14], IMAGE);

  assert.ok(view, "a small object should zoom");
  assert.ok(view.width < IMAGE.width, "viewport is narrower than the image");
  // A 14px object is magnified, but the crop keeps enough surroundings to
  // recognise it rather than filling the canvas with a few blurred pixels.
  assert.ok(view.height >= 160, `expected surroundings, got ${view.height}`);
  assert.ok(view.height < IMAGE.height / 2, "still a genuine crop");
  assert.ok(IMAGE.height / view.height > 1.5, "and a real magnification");
});

test("the viewport contains the target with margin on every side", () => {
  const bbox = [200, 300, 40, 60];
  const view = zoomViewportFor(bbox, IMAGE);
  const [x, y, width, height] = bbox;

  assert.ok(view.x <= x, "left margin present");
  assert.ok(view.y <= y, "top margin present");
  assert.ok(view.x + view.width >= x + width, "right margin present");
  assert.ok(view.y + view.height >= y + height, "bottom margin present");
  // The margin is generous: at least the box's own size on each axis.
  assert.ok(view.width >= width * 2);
  assert.ok(view.height >= height * 2);
});

test("never runs outside the image at an edge", () => {
  const view = zoomViewportFor([0, 0, 30, 30], IMAGE);

  assert.equal(view.x, 0);
  assert.equal(view.y, 0);
  assert.ok(view.x + view.width <= IMAGE.width);
  assert.ok(view.y + view.height <= IMAGE.height);
});

test("clamps against the far edge too", () => {
  const view = zoomViewportFor([400, 600, 27, 40], IMAGE);

  assert.ok(view.x >= 0);
  assert.ok(view.y >= 0);
  assert.equal(view.x + view.width, IMAGE.width);
  assert.equal(view.y + view.height, IMAGE.height);
});

test("returns null when the object already fills the image", () => {
  assert.equal(zoomViewportFor([0, 0, 427, 640], IMAGE), null);
});

test("returns null for missing or invalid input", () => {
  assert.equal(zoomViewportFor(null, IMAGE), null);
  assert.equal(zoomViewportFor([1, 2, 3, 4], null), null);
  assert.equal(zoomViewportFor([1, 2, 0, 4], IMAGE), null, "zero width");
  assert.equal(zoomViewportFor([1, 2, 3, -4], IMAGE), null, "negative height");
  assert.equal(zoomViewportFor([1, 2, 3], IMAGE), null, "short bbox");
});

test("a larger margin fraction widens the viewport", () => {
  const bbox = [100, 100, 40, 40];
  const narrow = zoomViewportFor(bbox, IMAGE, 0.5);
  const wide = zoomViewportFor(bbox, IMAGE, 3);

  assert.ok(wide.width > narrow.width);
  assert.ok(wide.height > narrow.height);
});
