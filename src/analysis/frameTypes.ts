import type { ForearmRoi } from "./forearmRoi";
import type { MaskStats } from "./forearmMask";

export type Side = "Left" | "Right" | "none";

export type Landmark = {
  x: number;
  y: number;
  z: number;
};

export type FrameResult = {
  name: string;
  handDetected: boolean;
  handSide: Side;
  wristX: number;
  wristY: number;
  handLandmarks: Landmark[];
  imageUrl: string;
  poseDetected: boolean;
  elbowVisibility: number;
  wristVisibility: number;
  /** Largo codo-muñeca en unidades de ancho de imagen. */
  forearmLength: number;
  elbowX: number;
  elbowY: number;
  forearmRoi?: ForearmRoi;
  trackingDx?: number;
  trackingDy?: number;
  trackingError?: number;

  /** Tamaño real de la foto (ya con orientación EXIF aplicada). */
  imageWidth: number;
  imageHeight: number;
  /** true si el EXIF rota la foto 90°/270° (la máscara está en la orientación "derecha"). */
  orientationSwapped: boolean;

  /** Máscara de persona (0/1) a resolución de trabajo. null si no hay ROI. */
  person: Uint8Array | null;
  maskWidth: number;
  maskHeight: number;

  /** Miniatura para la hoja de contactos. */
  thumb: HTMLCanvasElement | null;
};

export type FrameQuality = {
  stats: MaskStats;
  previewUrl: string;
};