export type Point = {
  x: number;
  y: number;
};

/**
 * ROI orientado del antebrazo.
 *
 * - `center`: normalizado 0-1 respecto a ancho/alto de la imagen.
 * - `direction`: vector unitario en espacio de PÍXELES, de muñeca hacia codo.
 * - `length`: largo sobre el eje del antebrazo, como fracción del ANCHO de la imagen.
 * - `thickness`: grosor perpendicular al eje, como fracción del ANCHO de la imagen.
 * - `angle`: atan2 de `direction` (píxeles), listo para ctx.rotate().
 *
 * Medir todo en unidades de ancho evita deformar el ROI en imágenes no
 * cuadradas (antes se mezclaban ejes normalizados x/y con distinta escala).
 */
export type ForearmRoi = {
  center: Point;
  direction: Point;
  length: number;
  thickness: number;
  angle: number;
  source: "pose" | "hand" | "tracking";
  /**
   * true si el largo viene de codo y muñeca detectados (pose); false si es una
   * estimación desde la mano. El tracking hereda este valor del ROI que sigue.
   */
  measured?: boolean;
};

const HAND_MCP = [5, 9, 13, 17];
const THICKNESS_RATIO = 0.4; // grosor = 40% del largo
const HAND_LENGTH_FACTOR = 2.5; // antebrazo ≈ 2.5 × largo de palma
const MIN_LENGTH = 0.12;
const MAX_LENGTH = 0.7;

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

function normalize(point: Point): Point {
  const length = Math.hypot(point.x, point.y);

  if (length === 0) return { x: 0, y: -1 };

  return {
    x: point.x / length,
    y: point.y / length
  };
}

function average(points: Point[]): Point {
  return {
    x: points.reduce((sum, point) => sum + point.x, 0) / points.length,
    y: points.reduce((sum, point) => sum + point.y, 0) / points.length
  };
}

/** Diferencia normalizada -> unidades de ancho de imagen. aspect = W / H */
function toWidthUnits(dx: number, dy: number, aspect: number): Point {
  return { x: dx, y: dy / aspect };
}

/** Unidades de ancho de imagen -> diferencia normalizada. */
function toNormalized(v: Point, aspect: number): Point {
  return { x: v.x, y: v.y * aspect };
}

export function createForearmRoiFromPose(
  wrist: Point,
  elbow: Point,
  aspect: number
): ForearmRoi {
  const vector = toWidthUnits(
    elbow.x - wrist.x,
    elbow.y - wrist.y,
    aspect
  );

  const length = Math.hypot(vector.x, vector.y);
  const direction = normalize(vector);

  return {
    center: {
      x: (wrist.x + elbow.x) / 2,
      y: (wrist.y + elbow.y) / 2
    },
    direction,
    length,
    thickness: length * THICKNESS_RATIO,
    angle: Math.atan2(direction.y, direction.x),
    source: "pose",
    measured: true
  };
}

export function createForearmRoiFromHand(
  wrist: Point,
  landmarks: Point[],
  aspect: number
): ForearmRoi | null {
  if (landmarks.length < 18) return null;

  const mcpCenter = average(HAND_MCP.map(i => landmarks[i]));

  const vector = toWidthUnits(
    wrist.x - mcpCenter.x,
    wrist.y - mcpCenter.y,
    aspect
  );

  const palm = Math.hypot(vector.x, vector.y);

  if (palm < 1e-4) return null;

  const direction = { x: vector.x / palm, y: vector.y / palm };
  const length = clamp(palm * HAND_LENGTH_FACTOR, MIN_LENGTH, MAX_LENGTH);

  const offset = toNormalized(
    { x: direction.x * length * 0.5, y: direction.y * length * 0.5 },
    aspect
  );

  return {
    center: {
      x: wrist.x + offset.x,
      y: wrist.y + offset.y
    },
    direction,
    length,
    thickness: length * THICKNESS_RATIO,
    angle: Math.atan2(direction.y, direction.x),
    source: "hand",
    measured: false
  };
}

/** Geometría del ROI en píxeles para una imagen de width × height. */
export function roiGeometry(
  roi: ForearmRoi,
  width: number,
  height: number
) {
  return {
    cx: roi.center.x * width,
    cy: roi.center.y * height,
    angle: roi.angle,
    length: roi.length * width,
    thickness: roi.thickness * width
  };
}