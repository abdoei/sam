/**
 * Filtering, sorting and grouping for the results table.
 *
 * Kept as plain JavaScript so the behaviour can be tested in node without a DOM,
 * the same way the zoom and mask-overlay logic is tested.
 */

export type SortKey = "iou" | "time" | "category";

export type Filters = {
  method?: string;
  category?: string;
  imageQuery?: string;
};

export function filterResults<T extends {
  image_id: number;
  category_name: string;
  method: string;
}>(rows: T[], filters: Filters): T[] {
  return rows.filter((row) => {
    if (filters.method && filters.method !== "all" && row.method !== filters.method) {
      return false;
    }
    if (
      filters.category &&
      filters.category !== "all" &&
      row.category_name !== filters.category
    ) {
      return false;
    }
    if (filters.imageQuery && !String(row.image_id).includes(filters.imageQuery.trim())) {
      return false;
    }
    return true;
  });
}

export function sortResults<T extends {
  ann_id: number;
  category_name: string;
  iou: number;
  elapsed_seconds: number | null;
}>(rows: T[], key: SortKey): T[] {
  const copy = [...rows];
  if (key === "iou") {
    copy.sort((a, b) => b.iou - a.iou);
  } else if (key === "time") {
    // Untimed rows sort last rather than as zero, so an older record without a
    // measurement cannot look like the fastest annotation.
    copy.sort(
      (a, b) =>
        (a.elapsed_seconds ?? Number.POSITIVE_INFINITY) -
        (b.elapsed_seconds ?? Number.POSITIVE_INFINITY),
    );
  } else {
    copy.sort(
      (a, b) => a.category_name.localeCompare(b.category_name) || a.ann_id - b.ann_id,
    );
  }
  return copy;
}

export function groupByImage<T extends { image_id: number }>(
  rows: T[],
): [number, T[]][] {
  const groups = new Map<number, T[]>();
  for (const row of rows) {
    const group = groups.get(row.image_id);
    if (group) group.push(row);
    else groups.set(row.image_id, [row]);
  }
  return [...groups.entries()].sort((a, b) => b[1].length - a[1].length);
}
