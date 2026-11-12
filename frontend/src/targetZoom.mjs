/**
 * Share of the viewport a target's bounding box should occupy.
 *
 * The zoom is chosen so every object looks about the same size on screen: a tiny
 * parking meter and a whole airplane both fill this fraction of the view. Sizing
 * from the box alone failed at both extremes — a large object framed the entire
 * photograph (no zoom at all), a tiny one produced a few blurred pixels.
 */
const TARGET_FILL_FRACTION = 0.5;

/** A viewport smaller than this many image pixels is more blur than detail. */
const MIN_VIEWPORT_PIXELS = 120;

/**
 * Viewport that frames one COCO object, in image pixels.
 *
 * `bbox` is COCO's [x, y, width, height]. The viewport is sized so the object
 * fills TARGET_FILL_FRACTION of it, then clamped to the image so it never shows
 * space outside the photograph. Returns null when the object is missing, or when
 * the framing would show essentially the whole image, where zooming gains nothing.
 */
export function zoomViewportFor(bbox, imageSize) {
  if (!bbox || !imageSize) return null;
  const [x, y, width, height] = bbox;
  if (![x, y, width, height].every((value) => Number.isFinite(value))) return null;
  if (width <= 0 || height <= 0 || imageSize.width <= 0 || imageSize.height <= 0) {
    return null;
  }

  // Size the short side first, then let the long side follow the image's aspect:
  // a square crop of a wide photograph overruns the short side and clamps
  // straight back to the whole frame, which is why large targets never zoomed.
  const side = Math.max(
    MIN_VIEWPORT_PIXELS,
    Math.max(width, height) / TARGET_FILL_FRACTION,
  );
  const imageAspect = imageSize.width / imageSize.height;
  const viewWidth =
    imageAspect >= 1
      ? Math.min(imageSize.width, side * imageAspect)
      : Math.min(imageSize.width, side);
  const viewHeight =
    imageAspect >= 1
      ? Math.min(imageSize.height, side)
      : Math.min(imageSize.height, side / imageAspect);
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
