/** Margin around a target's bounding box, as a fraction of the box's size. */
const ZOOM_MARGIN_FRACTION = 4;

/** A viewport smaller than this many image pixels is more blur than detail. */
const MIN_VIEWPORT_PIXELS = 160;

/**
 * Viewport that frames one COCO object, in image pixels.
 *
 * `bbox` is COCO's [x, y, width, height]. The viewport is the box plus a generous
 * margin so the annotator still sees some surroundings, and is clamped to the
 * image so zooming near an edge never shows empty space. Returns null when the
 * object is missing, or when framing it would show essentially the whole image,
 * where zooming gains nothing.
 */
export function zoomViewportFor(bbox, imageSize, marginFraction = ZOOM_MARGIN_FRACTION) {
  if (!bbox || !imageSize) return null;
  const [x, y, width, height] = bbox;
  if (![x, y, width, height].every((value) => Number.isFinite(value))) return null;
  if (width <= 0 || height <= 0 || imageSize.width <= 0 || imageSize.height <= 0) {
    return null;
  }

  const side = Math.max(
    MIN_VIEWPORT_PIXELS,
    Math.max(width, height) * (1 + marginFraction * 2),
  );
  const viewWidth = Math.min(imageSize.width, side);
  const viewHeight = Math.min(imageSize.height, side);
  if (viewWidth >= imageSize.width * 0.98 && viewHeight >= imageSize.height * 0.98) {
    return null;
  }

  const centreX = x + width / 2;
  const centreY = y + height / 2;

  return {
    x: Math.max(0, Math.min(imageSize.width - viewWidth, centreX - viewWidth / 2)),
    y: Math.max(0, Math.min(imageSize.height - viewHeight, centreY - viewHeight / 2)),
    width: viewWidth,
    height: viewHeight,
  };
}
