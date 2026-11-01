import {
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
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

type VideoFramePrompts = {
  points: PointPrompt[];
  box: BoxPrompt | null;
};

type TrackedObject = {
  id: number;
  name: string;
  color: string;
};

type VideoTrackResponse = {
  width: number;
  height: number;
  frame_count: number;
  mask_png_base64: Record<string, Array<string | null>>;
};

const TRACK_OBJECT_COLORS = [
  "#22c55e",
  "#38bdf8",
  "#f59e0b",
  "#e879f9",
  "#fb7185",
  "#a3e635",
  "#818cf8",
  "#2dd4bf",
];
const INITIAL_TRACKED_OBJECT: TrackedObject = {
  id: 1,
  name: "Object 1",
  color: TRACK_OBJECT_COLORS[0],
};
const VIDEO_FRAME_INTERVAL_SECONDS = 0.5;
const MAX_VIDEO_DURATION_SECONDS = 15;
const MAX_VIDEO_FILE_BYTES = 100 * 1024 * 1024;
const MAX_VIDEO_FRAME_DIMENSION = 1280;

function monotonicNow(): number {
  return performance.now();
}

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

async function extractVideoFrames(
  file: File,
  startTime: number,
): Promise<{
  frames: Blob[];
  width: number;
  height: number;
  interval: number;
  duration: number;
  sourceDuration: number;
}> {
  if (file.size > MAX_VIDEO_FILE_BYTES) {
    throw new Error("Video mode supports MP4 files up to 100 MiB.");
  }
  const url = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.preload = "auto";
  video.muted = true;
  video.src = url;

  try {
    await new Promise<void>((resolve, reject) => {
      video.onloadedmetadata = () => resolve();
      video.onerror = () => reject(new Error("Could not read the selected MP4 video."));
    });
    if (!Number.isFinite(video.duration) || video.duration <= 0) {
      throw new Error("The selected video has no readable duration.");
    }
    const scale = Math.min(
      1,
      MAX_VIDEO_FRAME_DIMENSION / video.videoWidth,
      MAX_VIDEO_FRAME_DIMENSION / video.videoHeight,
      Math.sqrt(921600 / (video.videoWidth * video.videoHeight)),
    );
    const width = Math.max(1, Math.round(video.videoWidth * scale));
    const height = Math.max(1, Math.round(video.videoHeight * scale));
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Could not create a canvas to sample video frames.");
    if (video.readyState < 2) {
      await new Promise<void>((resolve, reject) => {
        video.onloadeddata = () => resolve();
        video.onerror = () => reject(new Error("Could not decode the selected video."));
      });
    }

    const duration = Math.min(
      MAX_VIDEO_DURATION_SECONDS,
      video.duration - startTime,
    );
    const frameCount = Math.ceil(duration / VIDEO_FRAME_INTERVAL_SECONDS);
    const frames: Blob[] = [];
    for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
      const frameTime = Math.min(
        startTime + frameIndex * VIDEO_FRAME_INTERVAL_SECONDS,
        Math.max(0, video.duration - 0.01),
      );
      if (Math.abs(video.currentTime - frameTime) > 0.001) {
        await new Promise<void>((resolve, reject) => {
          const cleanup = () => {
            clearTimeout(timeout);
            video.removeEventListener("seeked", onSeeked);
            video.removeEventListener("error", onError);
          };
          const onSeeked = () => {
            cleanup();
            resolve();
          };
          const onError = () => {
            cleanup();
            reject(new Error(`Could not decode video frame ${frameIndex}.`));
          };
          const timeout = window.setTimeout(() => {
            cleanup();
            reject(new Error(`Timed out while decoding video frame ${frameIndex}.`));
          }, 10000);
          video.addEventListener("seeked", onSeeked, { once: true });
          video.addEventListener("error", onError, { once: true });
          video.currentTime = frameTime;
        });
      }
      context.drawImage(video, 0, 0, width, height);
      const frame = await new Promise<Blob>((resolve, reject) => {
        canvas.toBlob(
          (blob) => (blob ? resolve(blob) : reject(new Error("Could not encode a sampled frame."))),
          "image/jpeg",
          0.82,
        );
      });
      frames.push(frame);
    }
    return {
      frames,
      width,
      height,
      interval: VIDEO_FRAME_INTERVAL_SECONDS,
      duration,
      sourceDuration: video.duration,
    };
  } finally {
    video.removeAttribute("src");
    video.load();
    URL.revokeObjectURL(url);
  }
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [showAbout, setShowAbout] = useState(false);
  const [mediaMode, setMediaMode] = useState<"image" | "video">("image");
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoFrames, setVideoFrames] = useState<Blob[]>([]);
  const [videoDuration, setVideoDuration] = useState<number | null>(null);
  const [videoSegmentStart, setVideoSegmentStart] = useState(0);
  const [selectedVideoSegmentStart, setSelectedVideoSegmentStart] = useState(0);
  const [videoSegmentDuration, setVideoSegmentDuration] = useState(0);
  const [videoFrameInterval, setVideoFrameInterval] = useState(
    VIDEO_FRAME_INTERVAL_SECONDS,
  );
  const [currentVideoFrame, setCurrentVideoFrame] = useState(0);
  const [videoPrompts, setVideoPrompts] = useState<
    Record<number, Record<number, VideoFramePrompts>>
  >({});
  const [imagePrompts, setImagePrompts] = useState<
    Record<number, VideoFramePrompts>
  >({});
  const [imageMasks, setImageMasks] = useState<Record<number, SegmentResponse>>(
    {},
  );
  const [trackedObjects, setTrackedObjects] = useState<TrackedObject[]>([
    INITIAL_TRACKED_OBJECT,
  ]);
  const [activeTrackedObjectId, setActiveTrackedObjectId] = useState(1);
  const [videoMasks, setVideoMasks] = useState<
    Record<string, Array<string | null>>
  >({});
  const [isPreparingVideo, setIsPreparingVideo] = useState(false);
  const [isVideoPlaying, setIsVideoPlaying] = useState(false);
  const [annotationId, setAnnotationId] = useState("");
  const [cocoImages, setCocoImages] = useState<CocoImage[]>([]);
  const [selectedCocoImageId, setSelectedCocoImageId] = useState<number | null>(
    null,
  );
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageSize, setImageSize] = useState<ImageSize | null>(null);
  const [tool, setTool] = useState<Tool>("positive");
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
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const boxStart = useRef<{ x: number; y: number } | null>(null);
  const annotationStart = useRef<number | null>(null);
  const activeImagePrompts = imagePrompts[activeTrackedObjectId] ?? {
    points: [],
    box: null,
  };
  const activeVideoFramePrompts =
    videoPrompts[activeTrackedObjectId]?.[currentVideoFrame] ?? {
      points: [],
      box: null,
    };
  const activeObjectPrompts =
    mediaMode === "video" ? activeVideoFramePrompts : activeImagePrompts;

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
    return () => {
      if (videoUrl) URL.revokeObjectURL(videoUrl);
    };
  }, [videoUrl]);

  useEffect(() => {
    if (!imageSize || !canvasRef.current) return;
    canvasRef.current.width = imageSize.width;
    canvasRef.current.height = imageSize.height;
  }, [imageSize, mediaMode, videoUrl]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageSize) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    let active = true;
    context.clearRect(0, 0, canvas.width, canvas.height);
    const drawBox = (selectedBox: BoxPrompt, color: string) => {
      context.strokeStyle = color;
      context.lineWidth = Math.max(2, imageSize.width / 400);
      context.strokeRect(
        selectedBox.x0,
        selectedBox.y0,
        selectedBox.x1 - selectedBox.x0,
        selectedBox.y1 - selectedBox.y0,
      );
    };
    const drawPoints = (promptPoints: PointPrompt[], positiveColor: string) => {
      promptPoints.forEach((point) => {
        context.beginPath();
        context.arc(
          point.x,
          point.y,
          Math.max(7, imageSize.width / 120),
          0,
          Math.PI * 2,
        );
        context.fillStyle = point.label === 1 ? positiveColor : "#ef4444";
        context.fill();
        context.lineWidth = Math.max(2, imageSize.width / 500);
        context.strokeStyle = "#ffffff";
        context.stroke();
      });
    };
    const drawPrompts = () => {
      if (!active) return;
      if (mediaMode === "video") {
        trackedObjects.forEach((object) => {
          const prompt = videoPrompts[object.id]?.[currentVideoFrame];
          if (!prompt) return;
          if (prompt.box) drawBox(prompt.box, object.color);
          drawPoints(prompt.points, object.color);
        });
        if (draftBox) drawBox(draftBox, "#fbbf24");
      } else {
        trackedObjects.forEach((object) => {
          const prompt = imagePrompts[object.id];
          if (!prompt) return;
          if (prompt.box) drawBox(prompt.box, object.color);
          drawPoints(prompt.points, object.color);
        });
        if (draftBox) drawBox(draftBox, "#fbbf24");
      }
    };

    const masks =
      mediaMode === "video"
        ? trackedObjects.flatMap((object) => {
            const encoded = videoMasks[object.id]?.[currentVideoFrame];
            return encoded ? [{ encoded, color: object.color }] : [];
          })
        : trackedObjects.flatMap((object) => {
            const result = imageMasks[object.id];
            return result
              ? [{ encoded: result.mask_png_base64, color: object.color }]
              : [];
          });

    void Promise.all(
      masks.map(
        ({ encoded, color }) =>
          new Promise<{ image: HTMLImageElement; color: string }>(
            (resolve, reject) => {
              const image = new Image();
              image.onload = () => resolve({ image, color });
              image.onerror = reject;
              image.src = `data:image/png;base64,${encoded}`;
            },
          ),
      ),
    )
      .then((loadedMasks) => {
        if (!active) return;
        loadedMasks.forEach(({ image, color }) => {
          const source = document.createElement("canvas");
          source.width = imageSize.width;
          source.height = imageSize.height;
          const sourceContext = source.getContext("2d");
          if (!sourceContext) return;
          sourceContext.drawImage(image, 0, 0, imageSize.width, imageSize.height);
          const pixels = sourceContext.getImageData(
            0,
            0,
            imageSize.width,
            imageSize.height,
          );
          const red = Number.parseInt(color.slice(1, 3), 16);
          const green = Number.parseInt(color.slice(3, 5), 16);
          const blue = Number.parseInt(color.slice(5, 7), 16);
          for (let index = 0; index < pixels.data.length; index += 4) {
            const foreground = pixels.data[index] > 0;
            pixels.data[index] = red;
            pixels.data[index + 1] = green;
            pixels.data[index + 2] = blue;
            pixels.data[index + 3] = foreground ? 120 : 0;
          }
          context.putImageData(pixels, 0, 0);
        });
        drawPrompts();
      })
      .catch((reason: unknown) => {
        if (active) {
          setError(
            reason instanceof Error
              ? `Could not draw the returned mask: ${reason.message}`
              : "Could not draw a returned mask.",
          );
        }
      });

    return () => {
      active = false;
    };
  }, [
    currentVideoFrame,
    draftBox,
    imageMasks,
    imagePrompts,
    imageSize,
    mediaMode,
    trackedObjects,
    videoMasks,
    videoPrompts,
  ]);

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
    setImagePrompts({});
    setImageMasks({});
    setVideoPrompts({});
    setVideoMasks({});
    setTrackedObjects([INITIAL_TRACKED_OBJECT]);
    setActiveTrackedObjectId(INITIAL_TRACKED_OBJECT.id);
    setDraftBox(null);
    boxStart.current = null;
    setError(null);
    setInferenceMs(null);
    setAnnotationSeconds(null);
    setSavedAnnotation(null);
    annotationStart.current = null;
  }

  async function handleVideoChange(event: ChangeEvent<HTMLInputElement>) {
    const nextFile = event.currentTarget.files?.[0] ?? null;
    if (!nextFile) return;
    if (
      nextFile.type !== "video/mp4" &&
      !nextFile.name.toLowerCase().endsWith(".mp4")
    ) {
      setError("Video mode currently supports MP4 files.");
      return;
    }

    videoRef.current?.pause();
    setIsVideoPlaying(false);
    setMediaMode("video");
    setVideoFile(nextFile);
    setVideoFrames([]);
    setVideoDuration(null);
    setVideoSegmentStart(0);
    setSelectedVideoSegmentStart(0);
    setVideoSegmentDuration(0);
    setVideoMasks({});
    setVideoPrompts({});
    setImagePrompts({});
    setImageMasks({});
    setMask(null);
    setTrackedObjects([INITIAL_TRACKED_OBJECT]);
    setActiveTrackedObjectId(INITIAL_TRACKED_OBJECT.id);
    setCurrentVideoFrame(0);
    setImageSize(null);
    setError(null);
    setAnnotationSeconds(null);
    setInferenceMs(null);
    annotationStart.current = null;
    const nextUrl = URL.createObjectURL(nextFile);
    setVideoUrl(nextUrl);
    setIsPreparingVideo(true);
    try {
      const sampled = await extractVideoFrames(nextFile, 0);
      setVideoFrames(sampled.frames);
      setImageSize({ width: sampled.width, height: sampled.height });
      setVideoFrameInterval(sampled.interval);
      setVideoDuration(sampled.sourceDuration);
      setVideoSegmentStart(0);
      setSelectedVideoSegmentStart(0);
      setVideoSegmentDuration(sampled.duration);
      if (canvasRef.current) {
        canvasRef.current.width = sampled.width;
        canvasRef.current.height = sampled.height;
      }
    } catch (reason: unknown) {
      setVideoFrames([]);
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setIsPreparingVideo(false);
    }
  }

  async function sampleVideoSegment(fileToSample: File, startTime: number) {
    videoRef.current?.pause();
    setIsVideoPlaying(false);
    setIsPreparingVideo(true);
    setError(null);
    try {
      const sampled = await extractVideoFrames(fileToSample, startTime);
      setVideoFrames(sampled.frames);
      setImageSize({ width: sampled.width, height: sampled.height });
      setVideoFrameInterval(sampled.interval);
      setVideoSegmentStart(startTime);
      setSelectedVideoSegmentStart(startTime);
      setVideoSegmentDuration(sampled.duration);
      setVideoMasks({});
      setVideoPrompts({});
      setCurrentVideoFrame(0);
      setAnnotationSeconds(null);
      setInferenceMs(null);
      annotationStart.current = null;
      if (canvasRef.current) {
        canvasRef.current.width = sampled.width;
        canvasRef.current.height = sampled.height;
      }
      if (videoRef.current) videoRef.current.currentTime = startTime;
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setIsPreparingVideo(false);
    }
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
      setImagePrompts({});
      setImageMasks({});
      setVideoPrompts({});
      setVideoMasks({});
      setTrackedObjects([INITIAL_TRACKED_OBJECT]);
      setActiveTrackedObjectId(INITIAL_TRACKED_OBJECT.id);
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
      annotationStart.current = monotonicNow();
    setAnnotationSeconds(null);
  }

  function handleCanvasPointerDown(event: PointerEvent<HTMLCanvasElement>) {
    if (tool === "box") {
      handleBoxStart(event);
      return;
    }
    if (event.button !== 0) return;
    const coordinate = imageCoordinates(event);
    if (!coordinate) return;
    startAnnotation();
    if (mediaMode === "video") {
      videoRef.current?.pause();
      setIsVideoPlaying(false);
      setVideoPrompts((current) => {
        const prompt = current[activeTrackedObjectId]?.[currentVideoFrame] ?? {
          points: [],
          box: null,
        };
        return {
          ...current,
          [activeTrackedObjectId]: {
            ...current[activeTrackedObjectId],
            [currentVideoFrame]: {
            ...prompt,
            points: [
              ...prompt.points,
              { ...coordinate, label: tool === "positive" ? 1 : 0 },
            ],
            },
          },
        };
      });
      setVideoMasks({});
      setError(null);
      return;
    }
    setImagePrompts((current) => {
      const prompt = current[activeTrackedObjectId] ?? { points: [], box: null };
      return {
        ...current,
        [activeTrackedObjectId]: {
          ...prompt,
          points: [
            ...prompt.points,
            { ...coordinate, label: tool === "positive" ? 1 : 0 },
          ],
        },
      };
    });
    setImageMasks({});
    setMask(null);
    setSavedAnnotation(null);
    setError(null);
  }

  function handleBoxStart(event: PointerEvent<HTMLCanvasElement>) {
    if (tool !== "box") return;
    const coordinate = imageCoordinates(event);
    if (!coordinate) return;
    if (mediaMode === "video") {
      videoRef.current?.pause();
      setIsVideoPlaying(false);
    }
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
    if (mediaMode === "video") {
      setVideoPrompts((current) => ({
        ...current,
        [activeTrackedObjectId]: {
          ...current[activeTrackedObjectId],
          [currentVideoFrame]: {
            ...(current[activeTrackedObjectId]?.[currentVideoFrame] ?? {
              points: [],
              box: null,
            }),
            box: nextBox,
          },
        },
      }));
      setVideoMasks({});
      setError(null);
      return;
    }
    setImagePrompts((current) => ({
      ...current,
      [activeTrackedObjectId]: {
        ...(current[activeTrackedObjectId] ?? { points: [], box: null }),
        box: nextBox,
      },
    }));
    setImageMasks({});
    setMask(null);
    setSavedAnnotation(null);
    setError(null);
  }

  function addTrackedObject() {
    if (trackedObjects.length >= 20) {
      setError("Video tracking supports at most 20 objects.");
      return;
    }
    const id = Math.max(0, ...trackedObjects.map((object) => object.id)) + 1;
    const object: TrackedObject = {
      id,
      name: `Object ${id}`,
      color: TRACK_OBJECT_COLORS[(id - 1) % TRACK_OBJECT_COLORS.length],
    };
    setTrackedObjects((current) => [...current, object]);
    setActiveTrackedObjectId(id);
    setMask(null);
    setError(null);
  }

  function removeTrackedObject(objectId: number) {
    if (trackedObjects.length <= 1) {
      setError("At least one tracked object is required.");
      return;
    }
    const remaining = trackedObjects.filter((object) => object.id !== objectId);
    setTrackedObjects(remaining);
    if (activeTrackedObjectId === objectId) {
      setActiveTrackedObjectId(remaining[0].id);
    }
    setVideoPrompts((current) => {
      const next = { ...current };
      delete next[objectId];
      return next;
    });
    setVideoMasks((current) => {
      const next = { ...current };
      delete next[objectId];
      return next;
    });
    setImagePrompts((current) => {
      const next = { ...current };
      delete next[objectId];
      return next;
    });
    setImageMasks((current) => {
      const next = { ...current };
      delete next[objectId];
      return next;
    });
    if (activeTrackedObjectId === objectId) {
      setMask(imageMasks[remaining[0].id] ?? null);
    }
    setError(null);
  }

  function removeVideoPoint(pointIndex: number) {
    if (mediaMode === "image") {
      setImagePrompts((current) => {
        const prompt = current[activeTrackedObjectId];
        if (!prompt) return current;
        const points = prompt.points.filter((_, index) => index !== pointIndex);
        const nextPrompts = { ...current };
        if (!points.length && !prompt.box) {
          delete nextPrompts[activeTrackedObjectId];
        } else {
          nextPrompts[activeTrackedObjectId] = { ...prompt, points };
        }
        return nextPrompts;
      });
      setImageMasks({});
      setMask(null);
      return;
    }
    setVideoPrompts((current) => {
      const framePrompts = current[activeTrackedObjectId]?.[currentVideoFrame];
      if (!framePrompts) return current;
      const points = framePrompts.points.filter((_, index) => index !== pointIndex);
      const nextFramePrompts = { ...framePrompts, points };
      const nextObjectPrompts = { ...current[activeTrackedObjectId] };
      if (!points.length && !nextFramePrompts.box) {
        delete nextObjectPrompts[currentVideoFrame];
      } else {
        nextObjectPrompts[currentVideoFrame] = nextFramePrompts;
      }
      return {
        ...current,
        [activeTrackedObjectId]: nextObjectPrompts,
      };
    });
    setVideoMasks({});
  }

  function removeVideoBox() {
    if (mediaMode === "image") {
      setImagePrompts((current) => {
        const prompt = current[activeTrackedObjectId];
        if (!prompt) return current;
        const nextPrompts = { ...current };
        if (!prompt.points.length) {
          delete nextPrompts[activeTrackedObjectId];
        } else {
          nextPrompts[activeTrackedObjectId] = { ...prompt, box: null };
        }
        return nextPrompts;
      });
      setImageMasks({});
      setMask(null);
      return;
    }
    setVideoPrompts((current) => {
      const framePrompts = current[activeTrackedObjectId]?.[currentVideoFrame];
      if (!framePrompts) return current;
      const nextObjectPrompts = { ...current[activeTrackedObjectId] };
      if (!framePrompts.points.length) {
        delete nextObjectPrompts[currentVideoFrame];
      } else {
        nextObjectPrompts[currentVideoFrame] = { ...framePrompts, box: null };
      }
      return {
        ...current,
        [activeTrackedObjectId]: nextObjectPrompts,
      };
    });
    setVideoMasks({});
  }

  function clearPrompts() {
    boxStart.current = null;
    setDraftBox(null);
    if (mediaMode === "video") {
      setVideoPrompts((current) => {
        const next = { ...current };
        const objectPrompts = { ...next[activeTrackedObjectId] };
        delete objectPrompts[currentVideoFrame];
        next[activeTrackedObjectId] = objectPrompts;
        return next;
      });
      setVideoMasks({});
      setError(null);
      setInferenceMs(null);
      setAnnotationSeconds(null);
      annotationStart.current = null;
      return;
    }
    setImagePrompts((current) => {
      const next = { ...current };
      delete next[activeTrackedObjectId];
      return next;
    });
    setImageMasks((current) => {
      const next = { ...current };
      delete next[activeTrackedObjectId];
      return next;
    });
    setMask(null);
    setError(null);
    setInferenceMs(null);
    setAnnotationSeconds(null);
    setSavedAnnotation(null);
    annotationStart.current = null;
  }

  async function runSegmentation() {
    if (mediaMode === "video") {
      await runVideoTracking();
      return;
    }
    if (!file) return;
    const promptedObjects = Object.entries(imagePrompts).filter(
      ([, prompt]) => prompt.points.length > 0 || prompt.box !== null,
    );
    if (!promptedObjects.length) return;
    setIsRunning(true);
    setError(null);
    setImageMasks({});
    setMask(null);
    const started = monotonicNow();
    try {
      const results = await Promise.all(
        promptedObjects.map(async ([objectId, prompt]) => {
          const form = new FormData();
          form.append("image", file);
          form.append(
            "prompt",
            JSON.stringify({
              points: prompt.points,
              box: prompt.box
                ? [prompt.box.x0, prompt.box.y0, prompt.box.x1, prompt.box.y1]
                : null,
            }),
          );
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
          return [Number(objectId), body] as const;
        }),
      );
      const masksByObject = Object.fromEntries(results) as Record<
        number,
        SegmentResponse
      >;
      setImageMasks(masksByObject);
      setMask(masksByObject[activeTrackedObjectId] ?? null);
      setInferenceMs(monotonicNow() - started);
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setIsRunning(false);
    }
  }

  async function runVideoTracking() {
    const keyframes = Object.entries(videoPrompts).flatMap(
      ([objectId, objectPrompts]) =>
        Object.entries(objectPrompts).map(([frameIndex, prompt]) => ({
          frame_index: Number(frameIndex),
          object_id: Number(objectId),
          prompt: {
            points: prompt.points,
            box: prompt.box
              ? [prompt.box.x0, prompt.box.y0, prompt.box.x1, prompt.box.y1]
              : null,
          },
        })),
    ).filter(({ prompt }) => prompt.points.length > 0 || prompt.box !== null);
    if (!videoFrames.length || !keyframes.length) return;
    const promptedObjectIds = new Set(keyframes.map(({ object_id }) => object_id));

    setIsRunning(true);
    setError(null);
    const started = monotonicNow();
    const form = new FormData();
    videoFrames.forEach((frame, frameIndex) => {
      form.append("frames", frame, `frame-${frameIndex.toString().padStart(5, "0")}.jpg`);
    });
    form.append("keyframes", JSON.stringify(keyframes));
    try {
      const response = await fetch("/api/video/track", {
        method: "POST",
        body: form,
      });
      const body = (await response.json()) as
        | VideoTrackResponse
        | { detail?: unknown };
      if (!response.ok) {
        throw new Error(errorMessage("detail" in body ? body.detail : null));
      }
      if (
        !("mask_png_base64" in body) ||
        Object.keys(body.mask_png_base64).length !== promptedObjectIds.size ||
        Object.values(body.mask_png_base64).some(
          (masks) => masks.length !== videoFrames.length,
        )
      ) {
        throw new Error("The video tracker returned an incomplete mask sequence.");
      }
      setVideoMasks(body.mask_png_base64);
      setInferenceMs(monotonicNow() - started);
      if (annotationStart.current !== null) {
        setAnnotationSeconds((monotonicNow() - annotationStart.current) / 1000);
        annotationStart.current = null;
      }
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setIsRunning(false);
    }
  }

  function downloadVideoMasks() {
    Object.entries(videoMasks).forEach(([objectId, objectMasks]) => {
      const objectName =
        trackedObjects.find((object) => object.id === Number(objectId))?.name ??
        `Object-${objectId}`;
      objectMasks.forEach((encodedMask, frameIndex) => {
        if (!encodedMask) return;
        const link = document.createElement("a");
        link.href = `data:image/png;base64,${encodedMask}`;
        link.download = `${videoFile?.name.replace(/\.mp4$/i, "") ?? "video"}-${objectName.replace(/\s+/g, "-").toLowerCase()}-mask-${frameIndex.toString().padStart(5, "0")}.png`;
        link.click();
      });
    });
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
    const elapsedSeconds = (monotonicNow() - annotationStart.current) / 1000;
    if (selectedCocoImageId !== null && annotationId) {
      const prompts = imagePrompts[activeTrackedObjectId] ?? {
        points: [],
        box: null,
      };
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
              points: prompts.points,
              box: prompts.box
                ? [
                    prompts.box.x0,
                    prompts.box.y0,
                    prompts.box.x1,
                    prompts.box.y1,
                  ]
                : null,
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

  const canRun = Boolean(
    file &&
      Object.values(imagePrompts).some(
        (prompt) => prompt.points.length || prompt.box,
      ) &&
      !isRunning,
  );
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
        <nav className="topbar-nav" aria-label="Main navigation">
          <button
            aria-current={!showAbout ? "page" : undefined}
            className={`nav-button ${!showAbout ? "selected" : ""}`}
            onClick={() => setShowAbout(false)}
            type="button"
          >
            Annotate
          </button>
          <button
            aria-current={showAbout ? "page" : undefined}
            className={`nav-button ${showAbout ? "selected" : ""}`}
            onClick={() => {
              videoRef.current?.pause();
              setIsVideoPlaying(false);
              setShowAbout(true);
            }}
            type="button"
          >
            About
          </button>
        </nav>
        <div
          className={`service-status ${health?.model.ready ? "is-ready" : ""}`}
        >
          <span className="status-dot" />
          {modelStatus}
        </div>
      </header>

      <section className="about-page" hidden={!showAbout}>
          <div className="about-heading">
            <p className="eyebrow">ABOUT THIS PROJECT</p>
            <h1>Interactive segmentation, made local.</h1>
            <p className="subtitle">
              A two-person Digital Image Processing project exploring
              prompt-guided object annotation with SAM 2.1.
            </p>
          </div>

          <section className="card about-project">
            <p className="eyebrow">THE PROJECT</p>
            <h2>SAM2 Annotator</h2>
            <p>
              This application lets an annotator add multiple objects to a COCO
              image or video, give each object its own positive or negative
              points and rectangles, and inspect separate color-coded masks.
              Video mode propagates each prompted object through the sampled
              clip. A COCO reference mask stays hidden during prompting and is
              used only to calculate IoU and Dice after image annotation
              acceptance.
            </p>
            <div className="about-facts">
              <span><strong>Model</strong>SAM 2.1 Hiera-Tiny</span>
              <span><strong>Dataset</strong>COCO 2017 val2017 subset</span>
              <span><strong>Stack</strong>FastAPI · React · TypeScript</span>
            </div>
          </section>

          <section className="about-team" aria-labelledby="team-heading">
            <p className="eyebrow">THE TEAM</p>
            <h2 id="team-heading">Project contributors</h2>
            <div className="student-grid">
              <article className="card student-card">
                <span className="student-badge">A</span>
                <p className="eyebrow">STUDENT A · 2301498</p>
                <h3>Abdelhafeez Aref Abdelhafeez Almaz</h3>
                <p>COCO data preparation and manual-baseline work.</p>
              </article>
              <article className="card student-card">
                <span className="student-badge">B</span>
                <p className="eyebrow">STUDENT B · 2501925</p>
                <h3>Abdelrahman Eid Ashour Mohamed</h3>
                <p>SAM 2 integration, annotation tool, and evaluation workflow.</p>
              </article>
            </div>
          </section>
      </section>

      <div className="annotator-view" hidden={showAbout}>
      <section className="intro">
        <div>
          <p className="eyebrow">INTERACTIVE SEGMENTATION · MILESTONE 2</p>
          <h1>Label with a few prompts.</h1>
          <p className="subtitle">
            Add object or background clicks, or draw a box. SAM2 generates a
            local mask for review.
          </p>
        </div>
      </section>

      <div className="workspace">
        <section className="canvas-panel" aria-label="Image annotation canvas">
          <div className="canvas-toolbar">
            <div className="mode-group" role="group" aria-label="Annotation mode">
              <button
                aria-pressed={mediaMode === "image"}
                className={`mode-button ${mediaMode === "image" ? "selected" : ""}`}
                onClick={() => {
                  videoRef.current?.pause();
                  setIsVideoPlaying(false);
                  setMediaMode("image");
                }}
                type="button"
              >
                Image
              </button>
              <button
                aria-pressed={mediaMode === "video"}
                className={`mode-button ${mediaMode === "video" ? "selected" : ""}`}
                onClick={() => setMediaMode("video")}
                type="button"
              >
                Video
              </button>
            </div>
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
            {mediaMode === "image" ? (
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
            ) : (
              <label className="video-upload-button">
                {isPreparingVideo ? "Preparing…" : videoFile?.name ?? "Choose MP4"}
                <input
                  accept="video/mp4,.mp4"
                  disabled={isPreparingVideo || isRunning}
                  onChange={(event) => void handleVideoChange(event)}
                  type="file"
                />
              </label>
            )}
            <button
              className="text-button"
              onClick={clearPrompts}
              type="button"
              disabled={
                mediaMode === "video"
                  ? !videoPrompts[activeTrackedObjectId]?.[currentVideoFrame] &&
                    !Object.keys(videoMasks).length
                  : !imagePrompts[activeTrackedObjectId] &&
                    !imageMasks[activeTrackedObjectId] &&
                    !mask
              }
            >
              Clear
            </button>
          </div>

          <div
            className={`image-stage ${mediaMode === "video" ? "video-stage" : ""} ${(mediaMode === "image" ? imageUrl : videoUrl) ? "has-image" : ""}`}
          >
            {mediaMode === "image" && imageUrl ? (
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
                  onPointerDown={handleCanvasPointerDown}
                  onPointerMove={handleBoxMove}
                  onPointerUp={handleBoxEnd}
                  ref={canvasRef}
                />
              </div>
            ) : mediaMode === "image" ? (
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
            ) : videoUrl ? (
              <div className="media-layer">
                <video
                  className="source-image source-video"
                  onEnded={() => setIsVideoPlaying(false)}
                  onTimeUpdate={(event) => {
                    const time = event.currentTarget.currentTime;
                    const segmentEnd = videoSegmentStart + videoSegmentDuration;
                    if (time >= segmentEnd && !event.currentTarget.paused) {
                      event.currentTarget.pause();
                      event.currentTarget.currentTime = Math.max(
                        videoSegmentStart,
                        segmentEnd - 0.01,
                      );
                      setIsVideoPlaying(false);
                    }
                    const frameIndex = Math.min(
                      videoFrames.length - 1,
                      Math.round(
                        Math.max(0, time - videoSegmentStart) /
                          videoFrameInterval,
                      ),
                    );
                    if (frameIndex >= 0) setCurrentVideoFrame(frameIndex);
                  }}
                  ref={videoRef}
                  src={videoUrl}
                />
                <canvas
                  aria-label="Click to add video tracking prompts"
                  className={`prompt-canvas cursor-${tool}`}
                  onPointerDown={handleCanvasPointerDown}
                  onPointerMove={handleBoxMove}
                  onPointerUp={handleBoxEnd}
                  ref={canvasRef}
                />
              </div>
            ) : (
              <label className="upload-prompt">
                <input
                  accept="video/mp4,.mp4"
                  onChange={(event) => void handleVideoChange(event)}
                  type="file"
                />
                <span className="upload-icon" aria-hidden="true">
                  ▶
                </span>
                <strong>Choose a video to track</strong>
                  <span>
                    MP4 · track up to 15 seconds from any point · sampled at 2
                    frames/second
                  </span>
                  <span className="upload-button">Browse videos</span>
              </label>
            )}
          </div>

          {mediaMode === "video" && videoUrl && (
            <div className="video-segment-control">
              <label>
                Tracking segment starts at{" "}
                <strong>{selectedVideoSegmentStart.toFixed(1)}s</strong>
                {videoDuration !== null && (
                  <input
                    aria-label="Choose tracking segment start"
                    disabled={
                      isPreparingVideo ||
                      isRunning ||
                      videoDuration <= MAX_VIDEO_DURATION_SECONDS
                    }
                    max={Math.max(
                      0,
                      Math.floor(
                        (videoDuration - MAX_VIDEO_DURATION_SECONDS) * 2,
                      ) / 2,
                    )}
                    min={0}
                    onChange={(event) => {
                      const startTime = Number(event.currentTarget.value);
                      setSelectedVideoSegmentStart(startTime);
                      videoRef.current?.pause();
                      setIsVideoPlaying(false);
                      if (videoRef.current) {
                        videoRef.current.currentTime = startTime;
                      }
                    }}
                    step={0.5}
                    type="range"
                    value={selectedVideoSegmentStart}
                  />
                )}
              </label>
              <span>
                Active segment starts at {videoSegmentStart.toFixed(1)}s ·{" "}
                Tracking{" "}
                {Math.min(
                  MAX_VIDEO_DURATION_SECONDS,
                  videoDuration ?? 0,
                ).toFixed(1)}
                s of {(videoDuration ?? 0).toFixed(1)}s
              </span>
              {videoFile && videoDuration !== null &&
                videoDuration > MAX_VIDEO_DURATION_SECONDS && (
                  <button
                    className="secondary-button"
                    disabled={
                      isPreparingVideo ||
                      isRunning ||
                      selectedVideoSegmentStart === videoSegmentStart
                    }
                    onClick={() =>
                      void sampleVideoSegment(
                        videoFile,
                        selectedVideoSegmentStart,
                      )
                    }
                    type="button"
                  >
                    {isPreparingVideo ? "Sampling…" : "Apply tracking segment"}
                  </button>
                )}
            </div>
          )}

          {mediaMode === "video" && videoUrl && (
            <div className="video-controls">
              <button
                className="secondary-button"
                onClick={() => {
                  const player = videoRef.current;
                  if (!player) return;
                  if (player.paused) {
                    player.play()
                      .then(() => setIsVideoPlaying(true))
                      .catch((reason: unknown) =>
                        setError(
                          reason instanceof Error ? reason.message : String(reason),
                        ),
                      );
                  } else {
                    player.pause();
                    setIsVideoPlaying(false);
                  }
                }}
                type="button"
              >
                {isVideoPlaying ? "Pause" : "Play"}
              </button>
              <input
                aria-label="Seek video frame"
                disabled={!videoFrames.length}
                max={Math.max(0, videoFrames.length - 1)}
                min={0}
                onChange={(event) => {
                  const frameIndex = Number(event.currentTarget.value);
                  setCurrentVideoFrame(frameIndex);
                  if (videoRef.current) {
                    videoRef.current.currentTime =
                      videoSegmentStart + frameIndex * videoFrameInterval;
                  }
                }}
                step={1}
                type="range"
                value={currentVideoFrame}
              />
              <span>
                Frame {videoFrames.length ? currentVideoFrame + 1 : 0} /{" "}
                {videoFrames.length}
              </span>
            </div>
          )}

          {mediaMode === "image" && imageUrl && (
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
          {mediaMode === "video" && videoUrl && (
            <div className="image-footer">
              <div>
                <strong>{videoFile?.name}</strong>
                <span>
                  {videoFrames.length} tracking frames · frame {currentVideoFrame + 1}
                </span>
              </div>
              <span className="video-footnote">
                Add prompts for each object, then track them through the clip.
              </span>
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
                {activeObjectPrompts.points.length +
                  (activeObjectPrompts.box ? 1 : 0)}
              </span>
            </div>
            <p className="helper-text">
              {mediaMode === "video"
                ? `Add prompts for ${trackedObjects.find((object) => object.id === activeTrackedObjectId)?.name ?? "this object"} at frame ${currentVideoFrame + 1}.`
                : tool === "box"
                  ? "Drag on the image to draw a rectangle around the selected object."
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
            <>
              <div className="tracking-object-toolbar">
                  <span>{mediaMode === "video" ? "TRACKED OBJECTS" : "IMAGE OBJECTS"}</span>
                  <button
                    className="small-action-button"
                    disabled={trackedObjects.length >= 20 || isRunning}
                    onClick={addTrackedObject}
                    type="button"
                  >
                    + Add object
                  </button>
              </div>
              <div className="tracking-object-list">
                {trackedObjects.map((object) => {
                  const objectPrompts = imagePrompts[object.id];
                  const objectPromptCount =
                    mediaMode === "video"
                      ? Object.values(videoPrompts[object.id] ?? {}).filter(
                          (prompt) => prompt.points.length || prompt.box,
                        ).length
                      : objectPrompts
                        ? objectPrompts.points.length +
                          (objectPrompts.box ? 1 : 0)
                        : 0;
                  return (
                    <div className="tracking-object-row" key={object.id}>
                      <button
                        aria-pressed={activeTrackedObjectId === object.id}
                        className={`tracking-object-button ${activeTrackedObjectId === object.id ? "selected" : ""}`}
                        disabled={isRunning}
                        onClick={() => {
                          setActiveTrackedObjectId(object.id);
                          if (mediaMode === "image") {
                            setMask(imageMasks[object.id] ?? null);
                          }
                        }}
                        type="button"
                      >
                        <i style={{ backgroundColor: object.color }} />
                        {object.name}
                        <span>
                          {mediaMode === "video"
                            ? `${objectPromptCount} frames`
                            : `${objectPromptCount} prompts`}
                        </span>
                      </button>
                      <button
                        aria-label={`Remove ${object.name}`}
                        className="remove-object-button"
                        disabled={trackedObjects.length <= 1 || isRunning}
                        onClick={() => removeTrackedObject(object.id)}
                        type="button"
                      >
                        ×
                      </button>
                    </div>
                  );
                })}
              </div>
              {mediaMode === "video" && (
                <p className="video-prompt-count">
                  Editing{" "}
                  {trackedObjects.find(
                    (object) => object.id === activeTrackedObjectId,
                  )?.name ?? "object"}{" "}
                  at frame {currentVideoFrame + 1}
                  <span>
                    {Object.values(
                      videoPrompts[activeTrackedObjectId] ?? {},
                    ).filter((prompt) => prompt.points.length || prompt.box).length}{" "}
                    prompted frames
                  </span>
                </p>
              )}
              {(activeObjectPrompts.points.length > 0 ||
                activeObjectPrompts.box) && (
                <div className="video-prompt-list">
                  {activeObjectPrompts.points.map((point, index) => (
                      <div className="video-prompt-row" key={`point-${index}`}>
                        <span>
                          <i
                            className={`legend-dot ${point.label ? "positive-dot" : "negative-dot"}`}
                          />
                          {point.label ? "Object point" : "Background point"}{" "}
                          {index + 1}
                        </span>
                        <button
                          aria-label={`Remove ${point.label ? "object" : "background"} point ${index + 1}`}
                          className="remove-prompt-button"
                          disabled={isRunning}
                          onClick={() => removeVideoPoint(index)}
                          type="button"
                        >
                          Remove
                        </button>
                      </div>
                    ))}
                  {activeObjectPrompts.box && (
                      <div className="video-prompt-row">
                        <span>
                          <i className="legend-box" />
                          Rectangle
                        </span>
                        <button
                          aria-label="Remove rectangle"
                          className="remove-prompt-button"
                          disabled={isRunning}
                          onClick={removeVideoBox}
                          type="button"
                        >
                          Remove
                        </button>
                      </div>
                  )}
                </div>
              )}
              {mediaMode === "image" &&
                (selectedCocoImageId !== null ? (
                  <label className="annotation-id-field">
                    Target COCO object <span>(ground truth hidden)</span>
                    <select
                      onChange={(event) => {
                        setAnnotationId(event.currentTarget.value);
                        setImagePrompts({});
                        setImageMasks({});
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
                ))}
            </>
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
              disabled={
                mediaMode === "video"
                  ? !videoFrames.length ||
                    !Object.values(videoPrompts).some((objectPrompts) =>
                      Object.values(objectPrompts).some(
                        (prompt) => prompt.points.length || prompt.box,
                      ),
                    ) ||
                    isRunning ||
                    isPreparingVideo
                  : !canRun
              }
              onClick={runSegmentation}
              type="button"
            >
              {isRunning ? (
                <>
                  <span className="spinner" />
                  {mediaMode === "video" ? "Tracking video…" : "Running SAM2…"}
                </>
              ) : (
                mediaMode === "video"
                  ? Object.keys(videoMasks).length
                    ? "Re-track video"
                    : "Track objects through video"
                  : "Generate mask"
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
              {(mask || Object.keys(videoMasks).length > 0) && (
                <span className="result-tag">SAM2</span>
              )}
            </div>
            {mediaMode === "video" && Object.keys(videoMasks).length ? (
              <div className="result-summary">
                <p>
                  Tracked{" "}
                  <strong>{Object.keys(videoMasks).length} objects</strong>
                  {" · "}
                  <strong>
                    {Math.max(
                      0,
                      ...Object.values(videoMasks).map((masks) =>
                        masks.filter(Boolean).length,
                      ),
                    )}{" "}
                    frames each
                  </strong>
                </p>
                <button
                  className="secondary-button"
                  onClick={downloadVideoMasks}
                  type="button"
                >
                  Download mask frames
                </button>
              </div>
            ) : mediaMode === "image" && mask ? (
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
      </div>
    </main>
  );
}

export default App;
