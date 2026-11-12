import assert from "node:assert/strict";
import test from "node:test";

import { filterResults, groupByImage, sortResults } from "../src/resultsTable.mts";

const rows = [
  { ann_id: 1, image_id: 10, category_name: "person", iou: 0.4, dice: 0.5, elapsed_seconds: 30, method: "sam2" },
  { ann_id: 2, image_id: 10, category_name: "car", iou: 0.9, dice: 0.9, elapsed_seconds: null, method: "sam2" },
  { ann_id: 3, image_id: 11, category_name: "person", iou: 0.7, dice: 0.8, elapsed_seconds: 10, method: "polygon" },
];

test("sorting by IoU puts the best mask first", () => {
  assert.deepEqual(
    sortResults(rows, "iou").map((row) => row.ann_id),
    [2, 3, 1],
  );
});

test("sorting by time keeps untimed rows last instead of treating them as instant", () => {
  assert.deepEqual(
    sortResults(rows, "time").map((row) => row.ann_id),
    [3, 1, 2],
  );
});

test("sorting by category falls back to annotation id", () => {
  assert.deepEqual(
    sortResults(rows, "category").map((row) => row.ann_id),
    [2, 1, 3],
  );
});

test("filters combine condition, category and image id", () => {
  assert.deepEqual(
    filterResults(rows, { method: "sam2" }).map((row) => row.ann_id),
    [1, 2],
  );
  assert.deepEqual(
    filterResults(rows, { category: "person" }).map((row) => row.ann_id),
    [1, 3],
  );
  assert.deepEqual(
    filterResults(rows, { category: "person", method: "polygon" }).map(
      (row) => row.ann_id,
    ),
    [3],
  );
  assert.deepEqual(
    filterResults(rows, { imageQuery: "11" }).map((row) => row.ann_id),
    [3],
  );
});

test("groups keep every annotation for an image together", () => {
  const groups = groupByImage(rows);
  assert.equal(groups.length, 2);
  const [first, second] = groups;
  assert.equal(first[0], 10);
  assert.deepEqual(first[1].map((row) => row.ann_id), [1, 2]);
  assert.equal(second[0], 11);
});
