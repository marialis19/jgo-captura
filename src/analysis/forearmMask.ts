import type { ForearmRoi } from "./forearmRoi";

/**
 * Máscara limpia del antebrazo = (persona ∩ ROI orientado) + limpieza morfológica
 * + componente conexa más grande + relleno de huecos.
 * Todo en la resolución de trabajo de la máscara de persona (no en la foto original).
 */

export type MaskParams = {
  /** Multiplica el grosor del ROI. >1 deja que el borde lo defina la silueta de la persona. */
  thicknessScale: number;
  /**
   * Multiplica el largo de los ROI ESTIMADOS desde la mano (no los de pose), extendiéndolos
   * desde la muñeca hacia el codo.
   */
  lengthScale: number;
  /** Fracción del largo del ROI que se recorta en CADA extremo (evita mano / brazo superior). */
  endTrim: number;
  /** Radio (px de resolución de trabajo) de apertura + cierre. 0 = sin limpieza. */
  morphRadius: number;
  /** Invierte la máscara de persona (por si el modelo devuelve la polaridad al revés). */
  invertPerson: boolean;
};

export const DEFAULT_MASK_PARAMS: MaskParams = {
  thicknessScale: 1.25,
  lengthScale: 1.6,
  endTrim: 0.05,
  morphRadius: 2,
  invertPerson: false
};

export type MaskStats = {
  roiPixels: number;
  maskPixels: number;
  /** maskPixels / roiPixels (0..1) */
  coverage: number;
  /** componentes significativas antes de quedarse con la mayor */
  fragments: number;
  /** fracción (0..1) de los bordes largos del ROI tocados por la máscara */
  edgeContact: number;
  flags: string[];
  /** true si no hay avisos bloqueantes (vacía / cobertura baja / fragmentada) */
  ok: boolean;
};

export type CleanMask = {
  mask: Uint8Array;
  width: number;
  height: number;
  stats: MaskStats;
};

export type RoiGeometry = {
  cx: number;
  cy: number;
  angle: number;
  cos: number;
  sin: number;
  halfL: number;
  halfT: number;
};

const MIN_COVERAGE = 0.25;
const EDGE_BAND = 1.5;
const EDGE_CONTACT_WARN = 0.5;

/** ROI efectivo (con recorte de extremos y escala de grosor) en píxeles de width × height. */
export function effectiveRoiGeometry(
  roi: ForearmRoi,
  width: number,
  height: number,
  params: Pick<MaskParams, "thicknessScale" | "lengthScale" | "endTrim">
): RoiGeometry {
  const trim = Math.min(0.45, Math.max(0, params.endTrim));
  const cos = Math.cos(roi.angle);
  const sin = Math.sin(roi.angle);

  // Extiende desde la muñeca (el ROI apunta de muñeca a codo) solo si el largo era estimado.
  const baseLength = roi.length * width;
  const length = baseLength * (roi.measured ? 1 : params.lengthScale);
  const shift = (length - baseLength) / 2;

  return {
    cx: roi.center.x * width + cos * shift,
    cy: roi.center.y * height + sin * shift,
    angle: roi.angle,
    cos,
    sin,
    halfL: (length * (1 - 2 * trim)) / 2,
    halfT: (roi.thickness * width * params.thicknessScale) / 2
  };
}

type Bbox = { x0: number; x1: number; y0: number; y1: number };

function roiBbox(g: RoiGeometry, width: number, height: number): Bbox {
  const ex = Math.abs(g.cos) * g.halfL + Math.abs(g.sin) * g.halfT;
  const ey = Math.abs(g.sin) * g.halfL + Math.abs(g.cos) * g.halfT;

  return {
    x0: Math.max(0, Math.floor(g.cx - ex)),
    x1: Math.min(width - 1, Math.ceil(g.cx + ex)),
    y0: Math.max(0, Math.floor(g.cy - ey)),
    y1: Math.min(height - 1, Math.ceil(g.cy + ey))
  };
}

export function rasterizeRoi(g: RoiGeometry, width: number, height: number) {
  const mask = new Uint8Array(width * height);
  const box = roiBbox(g, width, height);
  let pixels = 0;

  for (let y = box.y0; y <= box.y1; y++) {
    for (let x = box.x0; x <= box.x1; x++) {
      const dx = x + 0.5 - g.cx;
      const dy = y + 0.5 - g.cy;
      const u = dx * g.cos + dy * g.sin;
      const v = -dx * g.sin + dy * g.cos;

      if (Math.abs(u) <= g.halfL && Math.abs(v) <= g.halfT) {
        mask[y * width + x] = 1;
        pixels++;
      }
    }
  }

  return { mask, pixels, box };
}

/** Erosión/dilatación con kernel cuadrado (separable). Fuera de imagen: neutro al erosionar, 0 al dilatar. */
function boxFilter(
  src: Uint8Array,
  w: number,
  h: number,
  r: number,
  mode: "erode" | "dilate"
): Uint8Array {
  if (r <= 0) return src;

  const tmp = new Uint8Array(w * h);
  const out = new Uint8Array(w * h);
  const ps = new Int32Array(Math.max(w, h) + 1);

  const decide = (count: number, size: number) =>
    mode === "dilate" ? (count > 0 ? 1 : 0) : count === size ? 1 : 0;

  for (let y = 0; y < h; y++) {
    const row = y * w;
    ps[0] = 0;
    for (let x = 0; x < w; x++) ps[x + 1] = ps[x] + src[row + x];

    for (let x = 0; x < w; x++) {
      const lo = Math.max(0, x - r);
      const hi = Math.min(w - 1, x + r);
      tmp[row + x] = decide(ps[hi + 1] - ps[lo], hi - lo + 1);
    }
  }

  for (let x = 0; x < w; x++) {
    ps[0] = 0;
    for (let y = 0; y < h; y++) ps[y + 1] = ps[y] + tmp[y * w + x];

    for (let y = 0; y < h; y++) {
      const lo = Math.max(0, y - r);
      const hi = Math.min(h - 1, y + r);
      out[y * w + x] = decide(ps[hi + 1] - ps[lo], hi - lo + 1);
    }
  }

  return out;
}

function labelComponents(mask: Uint8Array, w: number, h: number) {
  const labels = new Int32Array(w * h);
  const queue = new Int32Array(w * h);
  const sizes: number[] = [0];
  let count = 0;

  for (let i = 0; i < w * h; i++) {
    if (!mask[i] || labels[i]) continue;

    count++;
    let head = 0;
    let tail = 0;
    let size = 0;

    queue[tail++] = i;
    labels[i] = count;

    while (head < tail) {
      const p = queue[head++];
      const x = p % w;
      size++;

      if (x > 0 && mask[p - 1] && !labels[p - 1]) {
        labels[p - 1] = count;
        queue[tail++] = p - 1;
      }
      if (x < w - 1 && mask[p + 1] && !labels[p + 1]) {
        labels[p + 1] = count;
        queue[tail++] = p + 1;
      }
      if (p >= w && mask[p - w] && !labels[p - w]) {
        labels[p - w] = count;
        queue[tail++] = p - w;
      }
      if (p < w * (h - 1) && mask[p + w] && !labels[p + w]) {
        labels[p + w] = count;
        queue[tail++] = p + w;
      }
    }

    sizes.push(size);
  }

  return { labels, sizes, count };
}

function fillHoles(mask: Uint8Array, w: number, h: number) {
  const outside = new Uint8Array(w * h);
  const queue = new Int32Array(w * h);
  let head = 0;
  let tail = 0;

  const push = (p: number) => {
    if (!mask[p] && !outside[p]) {
      outside[p] = 1;
      queue[tail++] = p;
    }
  };

  for (let x = 0; x < w; x++) {
    push(x);
    push((h - 1) * w + x);
  }
  for (let y = 0; y < h; y++) {
    push(y * w);
    push(y * w + w - 1);
  }

  while (head < tail) {
    const p = queue[head++];
    const x = p % w;

    if (x > 0) push(p - 1);
    if (x < w - 1) push(p + 1);
    if (p >= w) push(p - w);
    if (p < w * (h - 1)) push(p + w);
  }

  const out = mask.slice();

  for (let i = 0; i < w * h; i++) {
    if (!mask[i] && !outside[i]) out[i] = 1;
  }

  return out;
}

export function cleanForearmMask(
  person: Uint8Array,
  width: number,
  height: number,
  roi: ForearmRoi,
  params: MaskParams
): CleanMask {
  const geo = effectiveRoiGeometry(roi, width, height, params);
  const { mask: roiMask, pixels: roiPixels, box } = rasterizeRoi(
    geo,
    width,
    height
  );

  const invert = params.invertPerson ? 1 : 0;
  const total = width * height;

  // 1) persona ∩ ROI
  let m: Uint8Array = new Uint8Array(total);

  for (let i = 0; i < total; i++) {
    m[i] = roiMask[i] && (person[i] ^ invert) ? 1 : 0;
  }

  // 2) apertura (quita motas) + cierre (tapa grietas)
  const r = Math.round(params.morphRadius);

  if (r > 0) {
    m = boxFilter(boxFilter(m, width, height, r, "erode"), width, height, r, "dilate");
    m = boxFilter(boxFilter(m, width, height, r, "dilate"), width, height, r, "erode");

    const clipped = new Uint8Array(total);
    for (let i = 0; i < total; i++) clipped[i] = m[i] & roiMask[i];
    m = clipped;
  }

  // 3) quedarse con la componente más grande
  const { labels, sizes, count } = labelComponents(m, width, height);

  let best = 0;
  for (let k = 1; k <= count; k++) {
    if (sizes[k] > sizes[best]) best = k;
  }

  const minFragment = Math.max(0.1 * sizes[best], 0.0005 * total);
  let fragments = 0;
  for (let k = 1; k <= count; k++) {
    if (sizes[k] >= minFragment) fragments++;
  }

  const largest = new Uint8Array(total);
  if (best > 0) {
    for (let i = 0; i < total; i++) largest[i] = labels[i] === best ? 1 : 0;
  }

  // 4) rellenar huecos internos
  const mask = fillHoles(largest, width, height);

  // Estadísticas
  let maskPixels = 0;
  let contact = 0;

  for (let y = box.y0; y <= box.y1; y++) {
    for (let x = box.x0; x <= box.x1; x++) {
      if (!mask[y * width + x]) continue;

      maskPixels++;

      const dx = x + 0.5 - geo.cx;
      const dy = y + 0.5 - geo.cy;
      const u = dx * geo.cos + dy * geo.sin;
      const v = -dx * geo.sin + dy * geo.cos;

      if (Math.abs(u) <= geo.halfL && Math.abs(v) >= geo.halfT - EDGE_BAND) {
        contact++;
      }
    }
  }

  const band = 2 * (2 * geo.halfL) * EDGE_BAND;
  const edgeContact = band > 0 ? Math.min(1, contact / band) : 0;
  const coverage = roiPixels > 0 ? maskPixels / roiPixels : 0;

  const blocking: string[] = [];
  const warnings: string[] = [];

  if (maskPixels === 0) blocking.push("vacía");
  else if (coverage < MIN_COVERAGE) blocking.push("cobertura baja");
  if (fragments > 1) blocking.push("fragmentada");
  if (edgeContact > EDGE_CONTACT_WARN) warnings.push("recortada por ROI");

  return {
    mask,
    width,
    height,
    stats: {
      roiPixels,
      maskPixels,
      coverage,
      fragments,
      edgeContact,
      flags: [...blocking, ...warnings],
      ok: blocking.length === 0
    }
  };
}