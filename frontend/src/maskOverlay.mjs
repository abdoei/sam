const MASK_OPACITY = 120 / 255;

export function compositeMaskPixels(compositePixels, maskPixels, color) {
  if (compositePixels.length !== maskPixels.length) {
    throw new RangeError(
      "Mask and composite pixel buffers must have equal lengths.",
    );
  }

  const red = Number.parseInt(color.slice(1, 3), 16);
  const green = Number.parseInt(color.slice(3, 5), 16);
  const blue = Number.parseInt(color.slice(5, 7), 16);

  for (let index = 0; index < maskPixels.length; index += 4) {
    if (maskPixels[index] === 0) continue;

    const destinationAlpha = compositePixels[index + 3] / 255;
    const outputAlpha = MASK_OPACITY + destinationAlpha * (1 - MASK_OPACITY);
    const destinationWeight =
      (destinationAlpha * (1 - MASK_OPACITY)) / outputAlpha;
    const sourceWeight = MASK_OPACITY / outputAlpha;

    compositePixels[index] =
      red * sourceWeight + compositePixels[index] * destinationWeight;
    compositePixels[index + 1] =
      green * sourceWeight + compositePixels[index + 1] * destinationWeight;
    compositePixels[index + 2] =
      blue * sourceWeight + compositePixels[index + 2] * destinationWeight;
    compositePixels[index + 3] = outputAlpha * 255;
  }
}
