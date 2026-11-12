import { useEffect, useRef, useState } from "react";

import { drawMaskOverImage } from "./maskPreview.mts";
import type { ResultRow } from "./ResultsPage";

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Could not load ${src}`));
    image.src = src;
  });
}

/**
 * Saved mask preview, drawn over the source image.
 *
 * A stored mask on its own is a white shape on black: it cannot show whether the
 * annotation captured the right object. Compositing it over the image does, which
 * is what a reviewer is actually checking.
 */
export function SavedMaskPreview({
  row,
  onClose,
}: {
  row: ResultRow;
  onClose: () => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "failed">("loading");
  const [showRaw, setShowRaw] = useState(false);

  const maskUrl = `/api/results/masks/${row.image_id}/${row.ann_id}.png?method=${row.method}`;
  const imageUrl = `/api/coco/images/${row.image_id}`;

  useEffect(() => {
    let active = true;

    Promise.all([loadImage(imageUrl), loadImage(maskUrl)])
      .then(([image, mask]) => {
        if (!active) return;
        const canvas = canvasRef.current;
        const context = canvas?.getContext("2d");
        if (!canvas || !context) return;
        canvas.width = image.naturalWidth;
        canvas.height = image.naturalHeight;
        const drawn = drawMaskOverImage(context, image, mask);
        setStatus(drawn ? "ready" : "failed");
      })
      .catch(() => {
        if (active) setStatus("failed");
      });

    return () => {
      active = false;
    };
  }, [imageUrl, maskUrl]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <aside className="mask-preview-overlay" role="dialog" aria-label="Saved mask">
      <div className="mask-preview-dialog">
        <div className="card-heading">
          <div>
            <p className="eyebrow">SAVED MASK</p>
            <h2>
              {row.category_name} #{row.ann_id}
            </h2>
          </div>
          <button className="small-action-button" onClick={onClose} type="button">
            Close
          </button>
        </div>

        {showRaw ? (
          <img
            alt={`Stored mask for annotation ${row.ann_id}`}
            className="saved-mask-preview"
            src={maskUrl}
          />
        ) : (
          <canvas className="saved-mask-preview" ref={canvasRef} />
        )}

        <div className="mask-preview-footer">
          <p className="helper-text">
            Image {row.image_id} ·{" "}
            {row.method === "polygon" ? "Manual polygon" : "SAM2"} · IoU{" "}
            {row.iou.toFixed(3)} · Dice {row.dice.toFixed(3)}
            {status === "failed" && " · mask could not be overlaid"}
          </p>
          <button
            className="small-action-button"
            onClick={() => setShowRaw((current) => !current)}
            type="button"
          >
            {showRaw ? "Show over image" : "Show mask only"}
          </button>
        </div>
      </div>
    </aside>
  );
}
