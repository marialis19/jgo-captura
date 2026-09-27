export type Point = {
  x: number;
  y: number;
};

export type ForearmRoi = {
  center: Point;
  direction: Point;
  width: number;
  height: number;
  angle: number;
  source: "pose" | "hand" | "tracking";
};

const HAND_MCP = [5, 9, 13, 17];

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

export function directionFromHand(
  wrist: Point,
  landmarks: Point[]
): Point | null {
  if (landmarks.length < 18) return null;

  const mcpCenter = average(HAND_MCP.map(i => landmarks[i]));

  return normalize({
    x: wrist.x - mcpCenter.x,
    y: wrist.y - mcpCenter.y
  });
}

export function createForearmRoi(
  wrist: Point,
  direction: Point,
  source: "hand" | "tracking" = "hand"
): ForearmRoi {
  const width = 0.16;
  const height = 0.32;

  return {
    center: {
      x: wrist.x + direction.x * height * 0.5,
      y: wrist.y + direction.y * height * 0.5
    },
    direction,
    width,
    height,
    angle: Math.atan2(direction.y, direction.x),
    source
  };
}

export function createForearmRoiFromPose(
  wrist: Point,
  elbow: Point
): ForearmRoi {
  const vector = {
    x: elbow.x - wrist.x,
    y: elbow.y - wrist.y
  };

  const height = Math.hypot(vector.x, vector.y);
  const direction = normalize(vector);

  return {
    center: {
      x: (wrist.x + elbow.x) / 2,
      y: (wrist.y + elbow.y) / 2
    },
    direction,
    width: 0.16,
    height,
    angle: Math.atan2(direction.y, direction.x),
    source: "pose"
  };
}