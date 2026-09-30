import type { CleanMask, MaskParams, RoiGeometry } from "./forearmMask";
import { effectiveRoiGeometry } from "./forearmMask";
import type { FrameResult } from "./frameTypes";

/** Miniatura (lado mayor = maxSide) para la hoja de contactos. */
export function makeThumb(bitmap: ImageBitmap, maxSide = 240) {
  const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(bitmap.width * scale));
  canvas.height = Math.max(1, Math.round(bitmap.height * scale));
  canvas.getContext("2d")?.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  return canvas;
}

/**
 * "overlay": interior cian translúcido + borde amarillo.
 * "alpha": blanco con alpha 255 donde hay máscara, 0 donde no.
 */
export function maskToCanvas(
  mask: Uint8Array,
  w: number,
  h: number,
  mode: "overlay" | "alpha"
) {
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;

  const context = canvas.getContext("2d");
  if (!context) return canvas;

  const image = context.createImageData(w, h);
  const d = image.data;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const p = y * w + x;
      if (!mask[p]) continue;

      const q = p * 4;

      if (mode === "alpha") {
        d[q] = d[q + 1] = d[q + 2] = 255;
        d[q + 3] = 255;
        continue;
      }

      const edge =
        x === 0 ||
        y === 0 ||
        x === w - 1 ||
        y === h - 1 ||
        !mask[p - 1] ||
        !mask[p + 1] ||
        !mask[p - w] ||
        !mask[p + w];

      if (edge) {
        d[q] = 255;
        d[q + 1] = 230;
        d[q + 2] = 0;
        d[q + 3] = 255;
      } else {
        d[q] = 0;
        d[q + 1] = 255;
        d[q + 2] = 255;
        d[q + 3] = 110;
      }
    }
  }

  context.putImageData(image, 0, 0);
  return canvas;
}

/** Escala la máscara a la resolución de la foto y la vuelve a binarizar (bordes suaves, sin grises). */
export function upscaleMaskAlpha(
  mask: Uint8Array,
  w: number,
  h: number,
  outW: number,
  outH: number
) {
  const small = maskToCanvas(mask, w, h, "alpha");
  const out = document.createElement("canvas");
  out.width = outW;
  out.height = outH;

  const context = out.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("No se pudo crear el canvas de exportación");

  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(small, 0, 0, outW, outH);

  const image = context.getImageData(0, 0, outW, outH);
  const d = image.data;

  for (let i = 0; i < d.length; i += 4) {
    const on = d[i + 3] > 127;
    d[i] = d[i + 1] = d[i + 2] = 255;
    d[i + 3] = on ? 255 : 0;
  }

  context.putImageData(image, 0, 0);
  return out;
}

/** Blanco (255) = antebrazo, negro (0) = ignorar. */
export function alphaToBinaryCanvas(alpha: HTMLCanvasElement) {
  const out = document.createElement("canvas");
  out.width = alpha.width;
  out.height = alpha.height;

  const context = out.getContext("2d");
  if (!context) throw new Error("No se pudo crear el canvas de máscara");

  context.fillStyle = "#000";
  context.fillRect(0, 0, out.width, out.height);
  context.drawImage(alpha, 0, 0);

  return out;
}

export function canvasToBlob(
  canvas: HTMLCanvasElement,
  type: string,
  quality?: number
) {
  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      blob => (blob ? resolve(blob) : reject(new Error("toBlob devolvió null"))),
      type,
      quality
    );
  });
}

export function drawRoiOutline(
  context: CanvasRenderingContext2D,
  geo: RoiGeometry,
  lineWidth: number,
  color: string,
  showAxis = true
) {
  context.save();
  context.translate(geo.cx, geo.cy);
  context.rotate(geo.angle);
  context.strokeStyle = color;
  context.lineWidth = lineWidth;
  context.setLineDash([lineWidth * 3, lineWidth * 2]);
  context.strokeRect(-geo.halfL, -geo.halfT, geo.halfL * 2, geo.halfT * 2);

  if (showAxis) {
    context.setLineDash([]);
    context.beginPath();
    context.moveTo(-geo.halfL, 0);
    context.lineTo(geo.halfL, 0);
    context.stroke();
  }

  context.restore();
}

/** Miniatura con máscara + ROI dibujados, como data URL (para la hoja de contactos). */
export function renderPreview(
  frame: FrameResult,
  cleaned: CleanMask,
  params: MaskParams
) {
  const thumb = frame.thumb;
  if (!thumb || !frame.forearmRoi) return "";

  const canvas = document.createElement("canvas");
  canvas.width = thumb.width;
  canvas.height = thumb.height;

  const context = canvas.getContext("2d");
  if (!context) return "";

  context.drawImage(thumb, 0, 0);
  context.drawImage(
    maskToCanvas(cleaned.mask, cleaned.width, cleaned.height, "overlay"),
    0,
    0,
    canvas.width,
    canvas.height
  );

  drawRoiOutline(
    context,
    effectiveRoiGeometry(frame.forearmRoi, canvas.width, canvas.height, params),
    1.5,
    "#ff40ff",
    false
  );

  return canvas.toDataURL("image/jpeg", 0.8);
}