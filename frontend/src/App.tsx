import {
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type MouseEvent,
  type PointerEvent,
} from "react";
import "./App.css";

type Tool = "positive" | "negative" | "box";

type PointPrompt = {
  x: number;
  y: number;
  label: 0 | 1;
};

type BoxPrompt = {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
};

type ImageSize = {
  width: number;
  height: number;
};

type HealthResponse = {
  status: string;
  model: {
    ready: boolean;
    installed: boolean;
    checkpoint_configured: boolean;
    config: string;
    device: string;
  };
};

type SegmentResponse = {
  width: number;
  height: number;
  score: number;
  mask_png_base64: string;
};

type CocoAnnotation = {
  id: number;
  category_id: number;
  category_name: string;
  area: number;
};

type CocoImage = {
  id: number;
  file_name: string;
  width: number;
  height: number;
  annotations: CocoAnnotation[];
};

type SavedAnnotation = {
  saved: boolean;
  iou: number;
  dice: number;
  elapsed_seconds: number;
  manifest_path: string;
};

const toolLabels: Record<Tool, string> = {
  positive: "Add object point",
  negative: "Add background point",
  box: "Draw a box",
};

function errorMessage(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (detail) return JSON.stringify(detail);
  return "The request failed. Check that the backend is running and try again.";
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [annotationId, setAnnotationId] = useState("");
  const [cocoImages, setCocoImages] = useState<CocoImage[]>([]);
  const [selectedCocoImageId, setSelectedCocoImageId] = useState<number | null>(
    null,
  );
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageSize, setImageSize] = useState<ImageSize | null>(null);
  const [tool, setTool] = useState<Tool>("positive");
  const [points, setPoints] = useState<PointPrompt[]>([]);
  const [box, setBox] = useState<BoxPrompt | null>(null);
  const [draftBox, setDraftBox] = useState<BoxPrompt | null>(null);
  const [mask, setMask] = useState<SegmentResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [datasetError, setDatasetError] = useState<string | null>(null);
  const [savedAnnotation, setSavedAnnotation] =
    useState<SavedAnnotation | null>(null);
  const [isLoadingCocoImage, setIsLoadingCocoImage] = useState(false);
  const [isSavingAnnotation, setIsSavingAnnotation] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [inferenceMs, setInferenceMs] = useState<number | null>(null);
  const [annotationSeconds, setAnnotationSeconds] = useState<number | null>(
    null,
  );
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const boxStart = useRef<{ x: number; y: number } | null>(null);
  const annotationStart = useRef<number | null>(null);

  useEffect(() => {
    let active = true;
    fetch("/api/health")
      .then(async (response) => {
        if (!response.ok)
          throw new Error(`Backend health check failed (${response.status})`);
        return (await response.json()) as HealthResponse;
      })
      .then((result) => {
        if (active) {
          setHealth(result);
          setHealthError(null);
        }
      })
      .catch((reason: unknown) => {
        if (active) {
          setHealth(null);
          setHealthError(
            reason instanceof Error ? reason.message : String(reason),
          );
        }
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    let active = true;
    fetch("/api/coco/images")
      .then(async (response) => {
        const body = await response.json();
        if (!response.ok) throw new Error(errorMessage(body.detail));
        return body as { images: CocoImage[] };
      })
      .then((result) => {
        if (active) {
          setCocoImages(result.images);
          setDatasetError(null);
        }
      })
      .catch((reason: unknown) => {
        if (active) {
          setCocoImages([]);
          setDatasetError(
            reason instanceof Error ? reason.message : String(reason),
          );
        }
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    return () => {
      if (imageUrl) URL.revokeObjectURL(imageUrl);
    };
  }, [imageUrl]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageSize) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    let active = true;
    context.clearRect(0, 0, canvas.width, canvas.height);

    const drawPrompts = () => {
      if (!active) return;
      if (box || draftBox) {
        const selectedBox = draftBox ?? box;
        if (selectedBox) {
          context.strokeStyle = draftBox ? "#fbbf24" : "#f8fafc";
          context.lineWidth = Math.max(2, imageSize.width / 400);
          context.setLineDash(draftBox ? [8, 6] : []);
          context.strokeRect(
            selectedBox.x0,
            selectedBox.y0,
            selectedBox.x1 - selectedBox.x0,
            selectedBox.y1 - selectedBox.y0,
          );
          context.setLineDash([]);
        }
      }

      points.forEach((point) => {
        const color = point.label === 1 ? "#22c55e" : "#ef4444";
        context.beginPath();
        context.arc(
          point.x,
          point.y,
          Math.max(7, imageSize.width / 120),
          0,
          Math.PI * 2,
        );
        context.fillStyle = color;
        context.fill();
        context.lineWidth = Math.max(2, imageSize.width / 500);
        context.strokeStyle = "#ffffff";
        context.stroke();
      });
    };

    if (mask) {
      const maskImage = new Image();
      maskImage.onload = () => {
        if (!active) return;
        const source = document.createElement("canvas");
        source.width = imageSize.width;
        source.height = imageSize.height;
        const sourceContext = source.getContext("2d");
        if (!sourceContext) return;
        sourceContext.drawImage(
          maskImage,
          0,
          0,
          imageSize.width,
          imageSize.height,
        );
        const pixels = sourceContext.getImageData(
          0,
          0,
          imageSize.width,
          imageSize.height,
        );
        for (let index = 0; index < pixels.data.length; index += 4) {
          const foreground = pixels.data[index] > 0;
          pixels.data[index] = 45;
          pixels.data[index + 1] = 212;
          pixels.data[index + 2] = 144;
          pixels.data[index + 3] = foreground ? 115 : 0;
        }
        context.putImageData(pixels, 0, 0);
        drawPrompts();
      };
      maskImage.src = `data:image/png;base64,${mask.mask_png_base64}`;
    } else {
      drawPrompts();
    }

    return () => {
      active = false;
    };
  }, [box, draftBox, imageSize, mask, points]);

  function imageCoordinates(event: { clientX: number; clientY: number }) {
    const canvas = canvasRef.current;
    if (!canvas || !imageSize) return null;
    const bounds = canvas.getBoundingClientRect();
    return {
      x: Math.max(
        0,
        Math.min(
          imageSize.width - 1,
          ((event.clientX - bounds.left) * imageSize.width) / bounds.width,
        ),
      ),
      y: Math.max(
        0,
        Math.min(
          imageSize.height - 1,
          ((event.clientY - bounds.top) * imageSize.height) / bounds.height,
        ),
      ),
    };
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const nextFile = event.currentTarget.files?.[0] ?? null;
    setFile(nextFile);
    setSelectedCocoImageId(null);
    setAnnotationId("");
    setImageUrl(nextFile ? URL.createObjectURL(nextFile) : null);
    setImageSize(null);
    setMask(null);
    setPoints([]);
    setBox(null);
    setDraftBox(null);
    boxStart.current = null;
    setError(null);
    setInferenceMs(null);
    setAnnotationSeconds(null);
    setSavedAnnotation(null);
    annotationStart.current = null;
  }

  async function loadCocoImage(imageId: number) {
    const record = cocoImages.find((image) => image.id === imageId);
    if (!record) return;
    setIsLoadingCocoImage(true);
    setDatasetError(null);
    try {
      const response = await fetch(`/api/coco/images/${imageId}`);
      if (!response.ok) {
        const body = await response.json();
        throw new Error(errorMessage(body.detail));
      }
      const blob = await response.blob();
      const nextFile = new File([blob], record.file_name, {
        type: blob.type || "image/jpeg",
      });
      setFile(nextFile);
      setSelectedCocoImageId(imageId);
      setAnnotationId(record.annotations[0]?.id.toString() ?? "");
      setImageUrl(URL.createObjectURL(nextFile));
      setImageSize(null);
      setPoints([]);
      setBox(null);
      setDraftBox(null);
      boxStart.current = null;
      setMask(null);
      setError(null);
      setInferenceMs(null);
      setAnnotationSeconds(null);
      setSavedAnnotation(null);
      annotationStart.current = null;
    } catch (reason: unknown) {
      setDatasetError(
        reason instanceof Error ? reason.message : String(reason),
      );
    } finally {
      setIsLoadingCocoImage(false);
    }
  }

  function startAnnotation() {
    if (annotationStart.current === null)
      annotationStart.current = performance.now();
    setAnnotationSeconds(null);
  }

  function handleCanvasClick(event: MouseEvent<HTMLCanvasElement>) {
    if (tool === "box") return;
    const coordinate = imageCoordinates(event);
    if (!coordinate) return;
    startAnnotation();
    setPoints((current) => [
      ...current,
      { ...coordinate, label: tool === "positive" ? 1 : 0 },
    ]);
    setMask(null);
    setSavedAnnotation(null);
    setError(null);
  }

  function handleBoxStart(event: PointerEvent<HTMLCanvasElement>) {
    if (tool !== "box") return;
    const coordinate = imageCoordinates(event);
    if (!coordinate) return;
    startAnnotation();
    boxStart.current = coordinate;
    setDraftBox({
      x0: coordinate.x,
      y0: coordinate.y,
      x1: coordinate.x,
      y1: coordinate.y,
    });
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function handleBoxMove(event: PointerEvent<HTMLCanvasElement>) {
    const start = boxStart.current;
    const coordinate = imageCoordinates(event);
    if (!start || !coordinate) return;
    setDraftBox({
      x0: Math.min(start.x, coordinate.x),
      y0: Math.min(start.y, coordinate.y),
      x1: Math.max(start.x, coordinate.x),
      y1: Math.max(start.y, coordinate.y),
    });
  }

  function handleBoxEnd(event: PointerEvent<HTMLCanvasElement>) {
    const start = boxStart.current;
    const coordinate = imageCoordinates(event);
    if (!start || !coordinate) return;
    boxStart.current = null;
    setDraftBox(null);
    const nextBox = {
      x0: Math.min(start.x, coordinate.x),
      y0: Math.min(start.y, coordinate.y),
      x1: Math.max(start.x, coordinate.x),
      y1: Math.max(start.y, coordinate.y),
    };
    if (nextBox.x1 - nextBox.x0 < 3 || nextBox.y1 - nextBox.y0 < 3) return;
    setBox(nextBox);
    setMask(null);
    setSavedAnnotation(null);
    setError(null);
  }

  function clearPrompts() {
    boxStart.current = null;
    setPoints([]);
    setBox(null);
    setDraftBox(null);
    setMask(null);
    setError(null);
    setInferenceMs(null);
    setAnnotationSeconds(null);
    setSavedAnnotation(null);
    annotationStart.current = null;
  }

  async function runSegmentation() {
    if (!file || (!points.length && !box)) return;
    setIsRunning(true);
    setError(null);
    const started = performance.now();
    const form = new FormData();
    form.append("image", file);
    form.append(
      "prompt",
      JSON.stringify({
        points,
        box: box ? [box.x0, box.y0, box.x1, box.y1] : null,
      }),
    );
    try {
      const response = await fetch("/api/segment", {
        method: "POST",
        body: form,
      });
      const body = (await response.json()) as
        | SegmentResponse
        | { detail?: unknown };
      if (!response.ok) {
        throw new Error(errorMessage("detail" in body ? body.detail : null));
      }
      if (!("mask_png_base64" in body)) {
        throw new Error("The backend response did not contain a mask.");
      }
      setMask(body);
      setInferenceMs(performance.now() - started);
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setIsRunning(false);
    }
  }

  function downloadMask() {
    if (!mask) return;
    const link = document.createElement("a");
    link.href = `data:image/png;base64,${mask.mask_png_base64}`;
    const name =
      annotationId.trim() || file?.name.replace(/\.[^.]+$/, "") || "mask";
    link.download = `${name}-sam2-mask.png`;
    link.click();
  }

  async function acceptAnnotation() {
    if (annotationStart.current === null) return;
    const elapsedSeconds = (performance.now() - annotationStart.current) / 1000;
    if (selectedCocoImageId !== null && annotationId) {
      setIsSavingAnnotation(true);
      setError(null);
      try {
        const response = await fetch("/api/annotations", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image_id: selectedCocoImageId,
            annotation_id: Number(annotationId),
            elapsed_seconds: elapsedSeconds,
            mask_png_base64: mask?.mask_png_base64,
            prompt: {
              points,
              box: box ? [box.x0, box.y0, box.x1, box.y1] : null,
            },
          }),
        });
        const body = (await response.json()) as
          | SavedAnnotation
          | { detail?: unknown };
        if (!response.ok) {
          throw new Error(errorMessage("detail" in body ? body.detail : null));
        }
        if (!("iou" in body) || !("dice" in body)) {
          throw new Error(
            "The backend response did not contain evaluation metrics.",
          );
        }
        setSavedAnnotation(body);
        setAnnotationSeconds(elapsedSeconds);
        annotationStart.current = null;
      } catch (reason: unknown) {
        setError(reason instanceof Error ? reason.message : String(reason));
      } finally {
        setIsSavingAnnotation(false);
      }
    } else {
      setAnnotationSeconds(elapsedSeconds);
      annotationStart.current = null;
    }
  }

  const canRun = Boolean(file && (points.length || box) && !isRunning);
  const modelStatus = healthError
    ? "API unavailable"
    : health?.model.ready
      ? "Model ready"
      : health
        ? "Checkpoint needed"
        : "Checking API";

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#" aria-label="SAM2 Annotator home">
          <span className="brand-mark">S</span>
          <span>
            SAM2 <strong>Annotator</strong>
          </span>
        </a>
        <div
          className={`service-status ${health?.model.ready ? "is-ready" : ""}`}
        >
          <span className="status-dot" />
          {modelStatus}
        </div>
      </header>

      <section className="intro">
        <div>
          <p className="eyebrow">INTERACTIVE SEGMENTATION · MILESTONE 2</p>
          <h1>Label with a few prompts.</h1>
          <p className="subtitle">
            Add object or background clicks, or draw a box. SAM2 generates a
            local mask for review.
          </p>
        </div>
        <a
          className="demo-link"
          href="https://sam2.metademolab.com/demo"
          target="_blank"
          rel="noreferrer"
        >
          View official demo <span aria-hidden="true">↗</span>
        </a>
      </section>

      <div className="workspace">
        <section className="canvas-panel" aria-label="Image annotation canvas">
          <div className="canvas-toolbar">
            <div className="tool-group" role="group" aria-label="Prompt tool">
              {(["positive", "negative", "box"] as const).map((nextTool) => (
                <button
                  className={`tool-button ${tool === nextTool ? "selected" : ""}`}
                  key={nextTool}
                  onClick={() => setTool(nextTool)}
                  type="button"
                  aria-pressed={tool === nextTool}
                >
                  <span className={`tool-icon ${nextTool}`} aria-hidden="true">
                    {nextTool === "positive"
                      ? "+"
                      : nextTool === "negative"
                        ? "−"
                        : "□"}
                  </span>
                  {toolLabels[nextTool]}
                </button>
              ))}
            </div>
            <label className="coco-sample-select">
              <span>COCO sample</span>
              <select
                aria-label="Choose a COCO validation image"
                disabled={!cocoImages.length || isLoadingCocoImage}
                onChange={(event) => {
                  if (event.currentTarget.value)
                    void loadCocoImage(Number(event.currentTarget.value));
                }}
                value={selectedCocoImageId ?? ""}
              >
                <option value="">
                  {cocoImages.length ? "Select image…" : "Dataset unavailable"}
                </option>
                {cocoImages.map((image) => (
                  <option key={image.id} value={image.id}>
                    {image.id} · {image.file_name}
                  </option>
                ))}
              </select>
            </label>
            <button
              className="text-button"
              onClick={clearPrompts}
              type="button"
              disabled={!points.length && !box && !mask}
            >
              Clear
            </button>
          </div>

          <div className={`image-stage ${imageUrl ? "has-image" : ""}`}>
            {imageUrl ? (
              <div className="media-layer">
                <img
                  alt="Selected image for segmentation"
                  className="source-image"
                  onLoad={(event) => {
                    const target = event.currentTarget;
                    const size = {
                      width: target.naturalWidth,
                      height: target.naturalHeight,
                    };
                    setImageSize(size);
                    if (canvasRef.current) {
                      canvasRef.current.width = size.width;
                      canvasRef.current.height = size.height;
                    }
                  }}
                  ref={imageRef}
                  src={imageUrl}
                />
                <canvas
                  aria-label="Click to add prompts or drag to draw a box"
                  className={`prompt-canvas cursor-${tool}`}
                  onClick={handleCanvasClick}
                  onPointerDown={handleBoxStart}
                  onPointerMove={handleBoxMove}
                  onPointerUp={handleBoxEnd}
                  ref={canvasRef}
                />
              </div>
            ) : (
              <label className="upload-prompt">
                <input
                  accept="image/png,image/jpeg,image/webp"
                  onChange={handleFileChange}
                  type="file"
                />
                <span className="upload-icon" aria-hidden="true">
                  ↑
                </span>
                <strong>Choose an image to begin</strong>
                <span>PNG, JPG, or WebP · up to 15 MiB</span>
                <span className="upload-button">Browse files</span>
              </label>
            )}
          </div>

          {imageUrl && (
            <div className="image-footer">
              <div>
                <strong>{file?.name}</strong>
                {imageSize && (
                  <span>
                    {imageSize.width} × {imageSize.height}px
                  </span>
                )}
              </div>
              <label className="change-image">
                Change image
                <input
                  accept="image/png,image/jpeg,image/webp"
                  onChange={handleFileChange}
                  type="file"
                />
              </label>
            </div>
          )}
        </section>

        <aside className="side-panel">
          <section className="card prompt-card">
            <div className="card-heading">
              <div>
                <p className="eyebrow">PROMPTS</p>
                <h2>Guide the mask</h2>
              </div>
              <span className="count-badge">
                {points.length + (box ? 1 : 0)}
              </span>
            </div>
            <p className="helper-text">
              {tool === "box"
                ? "Drag on the image to draw a box around the object."
                : tool === "positive"
                  ? "Click the object to add a positive point."
                  : "Click the background to add a negative point."}
            </p>
            <div className="legend">
              <span>
                <i className="legend-dot positive-dot" /> Object
              </span>
              <span>
                <i className="legend-dot negative-dot" /> Background
              </span>
              <span>
                <i className="legend-box" /> Box
              </span>
            </div>
            {selectedCocoImageId !== null ? (
              <label className="annotation-id-field">
                Target COCO object <span>(ground truth hidden)</span>
                <select
                  onChange={(event) => {
                    setAnnotationId(event.currentTarget.value);
                    setPoints([]);
                    setBox(null);
                    setDraftBox(null);
                    setMask(null);
                    setInferenceMs(null);
                    setAnnotationSeconds(null);
                    setSavedAnnotation(null);
                    annotationStart.current = null;
                  }}
                  value={annotationId}
                >
                  {cocoImages
                    .find((image) => image.id === selectedCocoImageId)
                    ?.annotations.map((annotation) => (
                      <option key={annotation.id} value={annotation.id}>
                        {annotation.category_name} · #{annotation.id}
                      </option>
                    ))}
                </select>
              </label>
            ) : (
              <label className="annotation-id-field">
                COCO annotation ID <span>(for naming downloads)</span>
                <input
                  inputMode="numeric"
                  min="1"
                  onChange={(event) =>
                    setAnnotationId(event.currentTarget.value)
                  }
                  placeholder="Optional"
                  type="number"
                  value={annotationId}
                />
              </label>
            )}
            {datasetError && (
              <p className="error-message" role="status">
                COCO samples unavailable: {datasetError}
              </p>
            )}
            {isLoadingCocoImage && (
              <p className="helper-text">Loading COCO image…</p>
            )}
            {health && !health.model.ready && (
              <p className="setup-hint" role="status">
                {!health.model.installed
                  ? "Install SAM2 in the backend environment to enable inference."
                  : !health.model.checkpoint_configured
                    ? "Set SAM2_CHECKPOINT to your downloaded model weights."
                    : "Check the configured SAM2 model and checkpoint."}
              </p>
            )}
            <button
              className="primary-button"
              disabled={!canRun}
              onClick={runSegmentation}
              type="button"
            >
              {isRunning ? (
                <>
                  <span className="spinner" /> Running SAM2…
                </>
              ) : (
                "Generate mask"
              )}
            </button>
            {error && (
              <p className="error-message" role="alert">
                {error}
              </p>
            )}
            {healthError && (
              <p className="error-message" role="status">
                {healthError}
              </p>
            )}
          </section>

          <section className="card result-card">
            <div className="card-heading">
              <div>
                <p className="eyebrow">RESULT</p>
                <h2>Current mask</h2>
              </div>
              {mask && <span className="result-tag">SAM2</span>}
            </div>
            {mask ? (
              <>
                <div className="result-summary">
                  <p>
                    SAM2 confidence score{" "}
                    <strong>{mask.score.toFixed(3)}</strong>
                  </p>
                  {annotationSeconds === null && (
                    <button
                      className="secondary-button"
                      disabled={isSavingAnnotation}
                      onClick={() => void acceptAnnotation()}
                      type="button"
                    >
                      {isSavingAnnotation ? "Saving…" : "Accept annotation"}
                    </button>
                  )}
                  <button
                    className="secondary-button"
                    onClick={downloadMask}
                    type="button"
                  >
                    Download PNG
                  </button>
                </div>
                {annotationSeconds !== null && (
                  <p className="runtime-note">
                    Total annotation time: {annotationSeconds.toFixed(2)}s (from
                    first prompt to acceptance)
                  </p>
                )}
                {savedAnnotation && (
                  <p className="evaluation-metrics">
                    Ground-truth comparison · IoU{" "}
                    <strong>{savedAnnotation.iou.toFixed(3)}</strong>
                    {" · "}Dice{" "}
                    <strong>{savedAnnotation.dice.toFixed(3)}</strong>
                    <span>Saved to the local evaluation manifest.</span>
                  </p>
                )}
              </>
            ) : (
              <p className="empty-result">
                Your generated mask will appear over the image.
              </p>
            )}
            {inferenceMs !== null && (
              <p className="runtime-note">
                Inference: {(inferenceMs / 1000).toFixed(2)}s (not total
                annotation time)
              </p>
            )}
          </section>

          <section className="protocol-note">
            <span className="note-icon" aria-hidden="true">
              i
            </span>
            <p>
              <strong>Evaluation plan</strong>
              Compare per-instance mask IoU with COCO ground truth and measure
              total annotation time separately from inference latency. The timer
              starts at the first prompt and stops when you accept the mask.
            </p>
          </section>
        </aside>
      </div>

      <footer className="page-footer">
        <span>Local inference · Images are processed on this machine</span>
        <span>
          Backend: {health?.model.device ?? "—"} ·{" "}
          {health?.model.config ?? "SAM2.1 Hiera-T"}
        </span>
      </footer>
    </main>
  );
}

export default App;
