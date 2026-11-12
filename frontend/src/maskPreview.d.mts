export type MaskTint = {
  r: number;
  g: number;
  b: number;
};

export const SAVED_MASK_TINT: MaskTint;

export function drawMaskOverImage(
  context: CanvasRenderingContext2D,
  image: CanvasImageSource & { width: number; height: number },
  mask: CanvasImageSource & { width: number; height: number },
  tint?: MaskTint,
  opacity?: number,
): boolean;
