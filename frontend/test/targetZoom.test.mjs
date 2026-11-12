import assert from "node:assert/strict";
import test from "node:test";

import { zoomViewportFor } from "../src/targetZoom.mjs";

const IMAGE = { width: 427, height: 640 };
const WIDE_IMAGE = { width: 640, height: 480 };

function containsFully(view, bbox) {
  const [x, y, width, height] = bbox;
  return (
    view.x <= x &&
    view.y <= y &&
    view.x + view.width >= x + width &&
    view.y + view.height >= y + height
  );
}

test("a tiny target is magnified but keeps recognisable surroundings", () => {
  // A person in COCO image 1761: 9x14 px near the bottom edge.
  const bbox = [7, 614, 9, 14];
  const view = zoomViewportFor(bbox, IMAGE);

  assert.ok(view, "a small object should zoom");
  assert.ok(view.height < IMAGE.height / 2, "a genuine crop");
  assert.ok(IMAGE.height / view.height > 2, "and real magnification");
  assert.ok(containsFully(view, bbox), "the target stays inside the view");
});

test("a large target still zooms instead of framing the whole image", () => {
  // COCO image 33114: the parking meter spans 72x120 of a 640x480 photo. Before
  // the aspect fix every non-tiny target clamped back to the full frame, so this
  // row and the airplane row produced identical zooms.
  const meter = [245.93, 336.36, 72.27, 119.73];
  const view = zoomViewportFor(meter, WIDE_IMAGE);

  assert.ok(view, "a mid-sized object should zoom");
  assert.ok(view.width < WIDE_IMAGE.width, "narrower than the full image");
  assert.ok(view.height < WIDE_IMAGE.height, "shorter than the full image");
  assert.ok(containsFully(view, meter));
  assert.ok(meter[3] / view.height > 0.4, `fill ${meter[3] / view.height}`);
});

test("declines to zoom a target that already dominates the image", () => {
  // The airplane spans 340 of 640 px, so any crop that contains it is barely
  // smaller than the photograph; showing the frame unchanged is the honest result.
  const airplane = [163.76, 172.1, 339.92, 104.15];
  assert.equal(zoomViewportFor(airplane, WIDE_IMAGE), null);
});

test("targets of similar relative size are framed at similar magnification", () => {
  const meter = [245.93, 336.36, 72.27, 119.73];
  const smallMeter = [224.12, 344.72, 15.37, 21.57];

  const meterView = zoomViewportFor(meter, WIDE_IMAGE);
  const smallView = zoomViewportFor(smallMeter, WIDE_IMAGE);

  assert.ok(meterView && smallView);
  assert.notDeepEqual(meterView, smallView, "different targets must frame differently");
  // The tiny meter hits the minimum viewport, so it is magnified harder; the
  // mid-sized one is framed at the target fill fraction.
  assert.ok(smallView.width <= meterView.width);
  assert.ok(containsFully(meterView, meter));
  assert.ok(containsFully(smallView, smallMeter));
});

test("the viewport contains the target", () => {
  const bbox = [200, 300, 40, 60];
  assert.ok(containsFully(zoomViewportFor(bbox, IMAGE), bbox));
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
