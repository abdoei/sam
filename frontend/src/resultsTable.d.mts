export type SortKey = "iou" | "time" | "category";

export type Filters = {
  method?: string;
  category?: string;
  imageQuery?: string;
};

export function filterResults<
  T extends { image_id: number; category_name: string; method: string },
>(rows: T[], filters: Filters): T[];

export function sortResults<
  T extends {
    ann_id: number;
    category_name: string;
    iou: number;
    elapsed_seconds: number | null;
  },
>(rows: T[], key: SortKey): T[];

export function groupByImage<T extends { image_id: number }>(
  rows: T[],
): [number, T[]][];
