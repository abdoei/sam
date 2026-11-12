export function markerRadiusFor(imageWidth: number): number;
export const MIN_MARKER_SCREEN_RADIUS: number;

export function layoutMarkers(
  centroids: Array<{ x: number; y: number }>,
  radius: number,
  bounds?: { width: number; height: number },
): Array<{
  x: number;
  y: number;
  originX: number;
  originY: number;
  moved: boolean;
}>;
