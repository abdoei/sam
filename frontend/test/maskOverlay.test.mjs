import assert from "node:assert/strict";
import test from "node:test";
import { compositeMaskPixels } from "../src/maskOverlay.mjs";

test("composites distinct object masks without erasing earlier masks", () => {
  const composite = new Uint8ClampedArray(8);
  const firstMask = new Uint8ClampedArray([
    255, 255, 255, 255,
    0, 0, 0, 0,
  ]);
  const secondMask = new Uint8ClampedArray([
    0, 0, 0, 0,
    255, 255, 255, 255,
  ]);

  compositeMaskPixels(composite, firstMask, "#ff0000");
  compositeMaskPixels(composite, secondMask, "#0000ff");

  assert.deepEqual(Array.from(composite), [
    255, 0, 0, 120,
    0, 0, 255, 120,
  ]);
});

test("layers overlapping masks with the later object on top", () => {
  const composite = new Uint8ClampedArray(4);
  const foreground = new Uint8ClampedArray([255, 255, 255, 255]);

  compositeMaskPixels(composite, foreground, "#ff0000");
  compositeMaskPixels(composite, foreground, "#0000ff");

  assert.ok(composite[0] > 0);
  assert.equal(composite[1], 0);
  assert.ok(composite[2] > composite[0]);
  assert.ok(composite[3] > 120);
});
