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
 * Smallest useful magnification. A crop must remove at least this share of the
 * image on its long axis to be worth showing: below it the "zoom" is a few pixels
 * of trimming that reads as no change at all. A large object is zoomed as far as
 * the frame allows rather than declining entirely, because even a modest crop
 * confirms which target the row refers to.
 */
const MIN_ZOOM_REDUCTION = 0.05;

/**
 * Viewport that frames one COCO object, in image pixels.
 *
 * `bbox` is COCO's [x, y, width, height]. The viewport is sized so the object
 * fills TARGET_FILL_FRACTION of it, then clamped to the image so it never shows
 * space outside the photograph. Returns null only when the object is missing or
 * when even the tightest possible crop would trim almost nothing from the frame.
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
  const targetFillSide = Math.max(width, height) / TARGET_FILL_FRACTION;
  const imageAspect = imageSize.width / imageSize.height;
  const boundsFrom = (shortSide) => ({
    width:
      imageAspect >= 1
        ? Math.min(imageSize.width, shortSide * imageAspect)
        : Math.min(imageSize.width, shortSide),
    height:
      imageAspect >= 1
        ? Math.min(imageSize.height, shortSide)
        : Math.min(imageSize.height, shortSide / imageAspect),
  });

  let { width: viewWidth, height: viewHeight } = boundsFrom(
    Math.max(MIN_VIEWPORT_PIXELS, targetFillSide),
  );

  // A target that fills the frame at the preferred magnification would clamp to
  // the whole photograph. Rather than declining, shrink the crop as far as the
  // box allows, so a large object still zooms as much as its size permits.
  const shortSideLimit = Math.min(imageSize.width, imageSize.height);
  if (
    viewWidth > imageSize.width * (1 - MIN_ZOOM_REDUCTION) &&
    viewHeight > imageSize.height * (1 - MIN_ZOOM_REDUCTION)
  ) {
    const fitted = boundsFrom(Math.min(shortSideLimit, Math.max(width, height)));
    viewWidth = fitted.width;
    viewHeight = fitted.height;
  }

  if (
    viewWidth > imageSize.width * (1 - MIN_ZOOM_REDUCTION) &&
    viewHeight > imageSize.height * (1 - MIN_ZOOM_REDUCTION)
  ) {
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
