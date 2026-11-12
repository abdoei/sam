import { useMemo, useState } from "react";

import { filterResults, groupByImage, sortResults, type SortKey } from "./resultsTable.mts";

export type ResultRow = {
  ann_id: number;
  image_id: number;
  category_name: string;
  iou: number;
  dice: number;
  elapsed_seconds: number | null;
  /** Which condition produced it, needed to address its saved mask. */
  method: string;
};

export type ResultSummary = {
  count: number;
  mean_iou?: number;
  mean_dice?: number;
  mean_seconds?: number | null;
  timed_count?: number;
};

export type ResultsResponse = {
  image_id: number | null;
  methods: { method: string; results: Omit<ResultRow, "method">[]; all_count: number }[];
  totals: { scope: ResultSummary; all: ResultSummary };
  errors: string[];
};

const methodLabel = (method: string) =>
  method === "polygon" ? "Manual polygon" : "SAM2";

function formatSeconds(seconds: number | null): string {
  return seconds == null ? "—" : `${seconds.toFixed(1)}s`;
}

type Props = {
  results: ResultsResponse | null;
  loading: boolean;
  isExporting: boolean;
  onExport: () => void;
  onPreview: (row: ResultRow) => void;
};

/**
 * Full-page results browsing.
 *
 * The panel this replaces listed only the current image; the graded question is
 * what the whole annotation run achieved, so the page groups by image and lets the
 * table be filtered by condition and category and sorted by IoU, category or time.
 */
export function ResultsPage({
  results,
  loading,
  isExporting,
  onExport,
  onPreview,
}: Props) {
  const [methodFilter, setMethodFilter] = useState("all");
  const [categoryFilter, setCategoryFilter] = useState("all");
  const [sortKey, setSortKey] = useState<SortKey>("iou");
  const [query, setQuery] = useState("");

  const rows = useMemo(() => {
    if (!results) return [] as ResultRow[];
    const flat: ResultRow[] = results.methods.flatMap((entry) =>
      entry.results.map((row) => ({ ...row, method: entry.method })),
    );
    return filterResults(flat, { method: methodFilter, category: categoryFilter, imageQuery: query });
  }, [results, methodFilter, categoryFilter, query]);

  const sorted = useMemo(() => sortResults(rows, sortKey), [rows, sortKey]);

  const byImage = useMemo(() => groupByImage(sorted), [sorted]);

  const categories = useMemo(() => {
    const seen = new Set<string>();
    for (const entry of results?.methods ?? []) {
      for (const row of entry.results) seen.add(row.category_name);
    }
    return [...seen].sort();
  }, [results]);

  const shown = rows.length;
  const total = results?.totals.all.count ?? 0;

  return (
    <section className="results-page">
      <div className="results-heading">
        <div>
          <p className="eyebrow">ANNOTATION RESULTS</p>
          <h1>What the annotation run produced</h1>
          <p className="results-intro">
            Every saved annotation across the COCO subset, scored against the
            ground-truth mask. Click a row to inspect the mask that was stored.
          </p>
        </div>
        <button
          className="primary-button export-button"
          disabled={isExporting || !total}
          onClick={onExport}
          type="button"
        >
          {isExporting ? "Packaging…" : "Export (.zip)"}
        </button>
      </div>

      {results && total > 0 ? (
        <>
          <div className="results-summary">
            <div className="summary-tile">
              <span className="summary-label">Annotations</span>
              <strong>{results.totals.all.count}</strong>
              <span className="summary-sub">
                across {new Set(
                  results.methods.flatMap((m) =>
                    m.results.map((r) => r.image_id),
                  ),
                ).size}{" "}
                images
              </span>
            </div>
            <div className="summary-tile">
              <span className="summary-label">Mean IoU</span>
              <strong>{results.totals.all.mean_iou?.toFixed(3) ?? "—"}</strong>
              <span className="summary-sub">
                Dice {results.totals.all.mean_dice?.toFixed(3) ?? "—"}
              </span>
            </div>
            <div className="summary-tile">
              <span className="summary-label">Mean time</span>
              <strong>
                {results.totals.all.mean_seconds != null
                  ? `${results.totals.all.mean_seconds.toFixed(1)}s`
                  : "—"}
              </strong>
              <span className="summary-sub">
                {results.totals.all.timed_count ?? 0} timed
              </span>
            </div>
            {results.methods.map((entry) => (
              <div className="summary-tile" key={entry.method}>
                <span className="summary-label">{methodLabel(entry.method)}</span>
                <strong>{entry.all_count}</strong>
                <span className="summary-sub">annotations</span>
              </div>
            ))}
          </div>

          <div className="results-controls">
            <label>
              <span>Condition</span>
              <select
                value={methodFilter}
                onChange={(event) => setMethodFilter(event.target.value)}
              >
                <option value="all">All</option>
                {results.methods.map((entry) => (
                  <option key={entry.method} value={entry.method}>
                    {methodLabel(entry.method)}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span>Category</span>
              <select
                value={categoryFilter}
                onChange={(event) => setCategoryFilter(event.target.value)}
              >
                <option value="all">All</option>
                {categories.map((category) => (
                  <option key={category} value={category}>
                    {category}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span>Sort by</span>
              <select
                value={sortKey}
                onChange={(event) => setSortKey(event.target.value as SortKey)}
              >
                <option value="iou">Best IoU first</option>
                <option value="time">Slowest first</option>
                <option value="category">Category</option>
              </select>
            </label>
            <label>
              <span>Image ID</span>
              <input
                inputMode="numeric"
                placeholder="Filter…"
                value={query}
                onChange={(event) =>
                  setQuery(event.target.value.replace(/[^0-9]/g, ""))
                }
              />
            </label>
            <span className="results-count">
              {shown} of {total} shown
            </span>
          </div>

          {byImage.length === 0 ? (
            <p className="helper-text">No annotations match these filters.</p>
          ) : (
            <div className="results-groups">
              {byImage.map(([imageId, group]) => (
                <article className="results-group" key={imageId}>
                  <p className="results-group-heading">
                    Image {imageId}
                    <span>{group.length} annotations</span>
                  </p>
                  <table className="results-table">
                    <thead>
                      <tr>
                        <th>Category</th>
                        <th>Annotation</th>
                        <th>Condition</th>
                        <th className="numeric">IoU</th>
                        <th className="numeric">Dice</th>
                        <th className="numeric">Time</th>
                      </tr>
                    </thead>
                    <tbody>
                      {group.map((row) => (
                        <tr
                          key={`${row.method}-${row.ann_id}`}
                          onClick={() => onPreview(row)}
                        >
                          <td>{row.category_name}</td>
                          <td className="dim">#{row.ann_id}</td>
                          <td className="dim">{methodLabel(row.method)}</td>
                          <td className="numeric">
                            <span
                              className={`iou-pill ${
                                row.iou >= 0.75
                                  ? "good"
                                  : row.iou >= 0.5
                                    ? "fair"
                                    : "poor"
                              }`}
                            >
                              {row.iou.toFixed(3)}
                            </span>
                          </td>
                          <td className="numeric">{row.dice.toFixed(3)}</td>
                          <td className="numeric">
                            {formatSeconds(row.elapsed_seconds)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </article>
              ))}
            </div>
          )}
        </>
      ) : (
        <p className="helper-text">
          {loading
            ? "Loading results…"
            : "Nothing saved yet. Accept a mask on the Annotate page and it will appear here."}
        </p>
      )}

      {results && results.errors.length > 0 && (
        <section className="protocol-note">
          <span className="note-icon" aria-hidden="true">
            !
          </span>
          <p>{results.errors.join(" · ")}</p>
        </section>
      )}
    </section>
  );
}
