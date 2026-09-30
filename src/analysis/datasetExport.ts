import type JSZip from "jszip";
import { cleanForearmMask, effectiveRoiGeometry } from "./forearmMask";
import type { MaskParams } from "./forearmMask";
import {
  alphaToBinaryCanvas,
  canvasToBlob,
  upscaleMaskAlpha
} from "./maskRender";
import type { FrameQuality, FrameResult } from "./frameTypes";

export type ExportOptions = {
  params: MaskParams;
  includeCutouts: boolean;
  sourceName: string;
  segmenterLabels: string[];
};

const flatName = (name: string) => name.replace(/[\\/]+/g, "__");
const stemOf = (name: string) => name.replace(/\.[^.]+$/, "");
const tick = () => new Promise<void>(resolve => setTimeout(resolve, 0));

async function uprightJpeg(bytes: Uint8Array) {
  const bitmap = await createImageBitmap(new Blob([bytes as BlobPart]));
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  canvas.getContext("2d")?.drawImage(bitmap, 0, 0);
  bitmap.close();

  const blob = await canvasToBlob(canvas, "image/jpeg", 0.95);
  return new Uint8Array(await blob.arrayBuffer());
}

/**
 * Estructura del ZIP:
 *   images/<frame>.jpg          foto original (o re-codificada derecha si el EXIF la rotaba 90°)
 *   masks/<frame>.jpg.png       blanco = antebrazo, negro = ignorar (convención COLMAP: <imagen>.png)
 *   cutouts/<frame>.png         (opcional) foto con alpha = máscara
 *   manifest.json               parámetros, ROI, calidad y estado de revisión de TODOS los frames
 */
export async function exportDataset(
  zip: JSZip,
  frames: FrameResult[],
  isAccepted: (name: string) => boolean,
  quality: Record<string, FrameQuality>,
  options: ExportOptions,
  onProgress: (done: number, total: number) => void
): Promise<Blob> {
  const { default: JSZipCtor } = await import("jszip");
  const out = new JSZipCtor();
  const store = { compression: "STORE" as const };

  const toExport = frames.filter(
    f => isAccepted(f.name) && f.forearmRoi && f.person
  );

  const manifestFrames: Record<string, unknown>[] = [];
  let done = 0;

  for (const frame of frames) {
    const roi = frame.forearmRoi;
    const stats = quality[frame.name]?.stats;
    const accepted = isAccepted(frame.name) && !!roi && !!frame.person;

    const base: Record<string, unknown> = {
      source_name: frame.name,
      accepted,
      image: null,
      mask: null,
      cutout: null,
      image_width: frame.imageWidth,
      image_height: frame.imageHeight,
      hand_detected: frame.handDetected,
      pose_detected: frame.poseDetected,
      side: frame.handSide,
      roi: roi
        ? {
            source: roi.source,
            center_norm: [roi.center.x, roi.center.y],
            direction: [roi.direction.x, roi.direction.y],
            angle_rad: roi.angle,
            length_frac_width: roi.length,
            thickness_frac_width: roi.thickness,
            effective_half_length_frac_width:
              effectiveRoiGeometry(roi, 1, 1, options.params).halfL,
            effective_half_thickness_frac_width:
              effectiveRoiGeometry(roi, 1, 1, options.params).halfT
          }
        : null,
      tracking:
        frame.trackingError === undefined
          ? null
          : { dx: frame.trackingDx, dy: frame.trackingDy, error: frame.trackingError },
      mask_quality: stats
        ? {
            coverage: stats.coverage,
            fragments: stats.fragments,
            edge_contact: stats.edgeContact,
            flags: stats.flags
          }
        : null
    };

    if (!accepted || !roi || !frame.person) {
      base.reason = !roi
        ? "sin ROI"
        : "descartado en la revisión";
      manifestFrames.push(base);
      continue;
    }

    const entry = zip.file(frame.name);

    if (!entry) {
      base.accepted = false;
      base.reason = "no se encontró la foto en el ZIP de origen";
      manifestFrames.push(base);
      continue;
    }

    const flat = flatName(frame.name);
    const rawBytes = await entry.async("uint8array");
    const imageBytes = frame.orientationSwapped
      ? await uprightJpeg(rawBytes)
      : rawBytes;

    const cleaned = cleanForearmMask(
      frame.person,
      frame.maskWidth,
      frame.maskHeight,
      roi,
      options.params
    );

    const alpha = upscaleMaskAlpha(
      cleaned.mask,
      cleaned.width,
      cleaned.height,
      frame.imageWidth,
      frame.imageHeight
    );

    const maskBlob = await canvasToBlob(alphaToBinaryCanvas(alpha), "image/png");

    const imagePath = `images/${flat}`;
    const maskPath = `masks/${flat}.png`;

    out.file(imagePath, imageBytes, store);
    out.file(maskPath, await maskBlob.arrayBuffer(), store);

    base.image = imagePath;
    base.mask = maskPath;
    base.image_reencoded_upright = frame.orientationSwapped;

    if (options.includeCutouts) {
      const bitmap = await createImageBitmap(new Blob([imageBytes as BlobPart]));
      const cut = document.createElement("canvas");
      cut.width = frame.imageWidth;
      cut.height = frame.imageHeight;

      const context = cut.getContext("2d");

      if (context) {
        context.drawImage(bitmap, 0, 0, cut.width, cut.height);
        context.globalCompositeOperation = "destination-in";
        context.drawImage(alpha, 0, 0);

        const cutPath = `cutouts/${stemOf(flat)}.png`;
        out.file(cutPath, await (await canvasToBlob(cut, "image/png")).arrayBuffer(), store);
        base.cutout = cutPath;
      }

      bitmap.close();
    }

    manifestFrames.push(base);
    done++;
    onProgress(done, toExport.length);
    await tick();
  }

  const manifest = {
    format_version: 1,
    created_at: new Date().toISOString(),
    source_zip: options.sourceName,
    mask_convention:
      "PNG a resolución de la foto. 255 (blanco) = antebrazo, 0 (negro) = ignorar. Nombre: <imagen>.png (compatible con COLMAP --ImageReader.mask_path).",
    pipeline: {
      person_segmenter: "MediaPipe selfie_segmenter",
      person_segmenter_labels: options.segmenterLabels,
      forearm_mask: "persona ∩ ROI orientado → apertura/cierre → componente mayor → relleno de huecos",
      params: options.params
    },
    counts: {
      total_frames: frames.length,
      exported: done,
      without_roi: frames.filter(f => !f.forearmRoi).length
    },
    frames: manifestFrames
  };

  out.file("manifest.json", JSON.stringify(manifest, null, 2));

  return out.generateAsync({ type: "blob", compression: "STORE" });
}