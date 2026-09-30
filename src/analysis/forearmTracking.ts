import { roiGeometry } from "./forearmRoi";
import type { ForearmRoi } from "./forearmRoi";

export type GrayFrame = {
  data: Uint8Array;
  width: number;
  height: number;
};

export type TrackingResult = {
  roi: ForearmRoi;
  /** Desplazamiento en píxeles de la imagen de análisis (GrayFrame). */
  dx: number;
  dy: number;
  /** Error medio absoluto (0-255) del mejor candidato, sin penalización. */
  error: number;
};

const SEARCH_RADIUS = 24;
const SEARCH_STEP = 2;
const SAMPLE_STEP = 4;
const MIN_SAMPLES = 20;
const DISPLACEMENT_PENALTY = 0.1; // sesga hacia no moverse en zonas sin textura
const MAX_ERROR = 35; // por encima se considera tracking perdido

type Sample = { x: number; y: number; value: number };

/** Puntos de la imagen previa dentro del ROI ROTADO. */
function collectSamples(frame: GrayFrame, roi: ForearmRoi): Sample[] {
  const g = roiGeometry(roi, frame.width, frame.height);
  const cos = Math.cos(g.angle);
  const sin = Math.sin(g.angle);
  const halfL = g.length / 2;
  const halfT = g.thickness / 2;

  const samples: Sample[] = [];

  for (let u = -halfL; u <= halfL; u += SAMPLE_STEP) {
    for (let v = -halfT; v <= halfT; v += SAMPLE_STEP) {
      const x = Math.round(g.cx + u * cos - v * sin);
      const y = Math.round(g.cy + u * sin + v * cos);

      if (x < 0 || y < 0 || x >= frame.width || y >= frame.height) {
        continue;
      }

      samples.push({ x, y, value: frame.data[y * frame.width + x] });
    }
  }

  return samples;
}

function errorAt(
  samples: Sample[],
  current: GrayFrame,
  dx: number,
  dy: number
) {
  let error = 0;
  let count = 0;

  for (const s of samples) {
    const qx = s.x + dx;
    const qy = s.y + dy;

    if (qx < 0 || qy < 0 || qx >= current.width || qy >= current.height) {
      continue;
    }

    error += Math.abs(s.value - current.data[qy * current.width + qx]);
    count++;
  }

  return count >= MIN_SAMPLES ? error / count : Infinity;
}

export function trackForearmRoi(
  previous: ForearmRoi,
  previousFrame: GrayFrame,
  currentFrame: GrayFrame
): TrackingResult | null {
  const samples = collectSamples(previousFrame, previous);

  if (samples.length < MIN_SAMPLES) return null;

  let bestDx = 0;
  let bestDy = 0;
  let bestScore = Infinity;
  let bestError = Infinity;

  const evaluate = (dx: number, dy: number) => {
    const error = errorAt(samples, currentFrame, dx, dy);
    const score = error + DISPLACEMENT_PENALTY * Math.hypot(dx, dy);

    if (score < bestScore) {
      bestScore = score;
      bestError = error;
      bestDx = dx;
      bestDy = dy;
    }
  };

  // Búsqueda gruesa
  for (let dy = -SEARCH_RADIUS; dy <= SEARCH_RADIUS; dy += SEARCH_STEP) {
    for (let dx = -SEARCH_RADIUS; dx <= SEARCH_RADIUS; dx += SEARCH_STEP) {
      evaluate(dx, dy);
    }
  }

  // Refinado a 1 px alrededor del mejor
  const coarseDx = bestDx;
  const coarseDy = bestDy;

  for (let dy = coarseDy - 1; dy <= coarseDy + 1; dy++) {
    for (let dx = coarseDx - 1; dx <= coarseDx + 1; dx++) {
      evaluate(dx, dy);
    }
  }

  if (!Number.isFinite(bestError) || bestError > MAX_ERROR) return null;

  const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

  return {
    roi: {
      ...previous,
      center: {
        x: clamp01(previous.center.x + bestDx / currentFrame.width),
        y: clamp01(previous.center.y + bestDy / currentFrame.height)
      },
      source: "tracking"
    },
    dx: bestDx,
    dy: bestDy,
    error: bestError
  };
}