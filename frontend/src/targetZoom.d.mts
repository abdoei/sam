export type Viewport = { x: number; y: number; width: number; height: number };

export function zoomViewportFor(
  bbox: [number, number, number, number] | null | undefined,
  imageSize: { width: number; height: number } | null | undefined,
): Viewport | null;
