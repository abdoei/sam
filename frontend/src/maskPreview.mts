/**
 * Compose a saved mask over the image it belongs to, for the results preview.
 *
 * A raw stored mask is a white blob on black: it shows what was captured but not
 * whether it captured the right object. Overlaying keeps the image readable while
 * making the mask's boundary obvious.
 */

export type MaskTint = {
  r: number;
  g: number;
  b: number;
};

/** Cyan matches the reference-mask colour used on the annotate canvas. */
export const SAVED_MASK_TINT: MaskTint = { r: 34, g: 211, b: 238 };

/**
 * Draw `image` and tint every pixel where `mask` is set.
 *
 * Returns false when the mask is smaller than the image, which means the stored
 * mask does not belong to this image and tinting would land in the wrong place.
 */
export function drawMaskOverImage(
  context: CanvasRenderingContext2D,
  image: CanvasImageSource & { width: number; height: number },
  mask: CanvasImageSource & { width: number; height: number },
  tint: MaskTint = SAVED_MASK_TINT,
  opacity = 0.45,
): boolean {
  const width = image.width;
  const height = image.height;
  if (!width || !height) return false;
  if (mask.width < width || mask.height < height) return false;

  context.clearRect(0, 0, width, height);
  context.drawImage(image, 0, 0, width, height);

  const scratch = document.createElement("canvas");
  scratch.width = width;
  scratch.height = height;
  const scratchContext = scratch.getContext("2d");
  if (!scratchContext) return false;
  scratchContext.drawImage(mask, 0, 0, width, height);

  const pixels = scratchContext.getImageData(0, 0, width, height);
  const data = pixels.data;
  for (let index = 0; index < data.length; index += 4) {
    if (data[index] > 127) {
      data[index] = tint.r;
      data[index + 1] = tint.g;
      data[index + 2] = tint.b;
      data[index + 3] = 255;
    } else {
      data[index + 3] = 0;
    }
  }
  scratchContext.putImageData(pixels, 0, 0);

  context.globalAlpha = opacity;
  context.drawImage(scratch, 0, 0, width, height);
  context.globalAlpha = 1;

  // Outline the mask edge so the boundary stays legible over busy areas.
  const edge = scratchContext.createImageData(width, height);
  const source = pixels.data;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = (y * width + x) * 4;
      if (source[index + 3] === 0) continue;
      const left = x > 0 ? source[index - 4 + 3] : 0;
      const right = x < width - 1 ? source[index + 4 + 3] : 0;
      const up = y > 0 ? source[index - width * 4 + 3] : 0;
      const down = y < height - 1 ? source[index + width * 4 + 3] : 0;
      if (left && right && up && down) continue;
      edge.data[index] = tint.r;
      edge.data[index + 1] = tint.g;
      edge.data[index + 2] = tint.b;
      edge.data[index + 3] = 255;
    }
  }
  scratchContext.putImageData(edge, 0, 0);
  context.drawImage(scratch, 0, 0, width, height);

  return true;
}
