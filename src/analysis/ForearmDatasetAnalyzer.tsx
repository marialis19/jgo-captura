import { useEffect, useMemo, useRef, useState } from "react";
import JSZip from "jszip";
import {
  FilesetResolver,
  HandLandmarker,
  ImageSegmenter,
  PoseLandmarker
} from "@mediapipe/tasks-vision";
import {
  createForearmRoiFromHand,
  createForearmRoiFromPose
} from "./forearmRoi";
import { trackForearmRoi } from "./forearmTracking";
import type { GrayFrame } from "./forearmTracking";
import type { ForearmRoi, Point } from "./forearmRoi";
import {
  DEFAULT_MASK_PARAMS,
  cleanForearmMask,
  effectiveRoiGeometry
} from "./forearmMask";
import type { MaskParams } from "./forearmMask";
import {
  drawRoiOutline,
  makeThumb,
  maskToCanvas,
  renderPreview
} from "./maskRender";
import { exportDataset } from "./datasetExport";
import type {
  FrameQuality,
  FrameResult,
  Landmark,
  Side
} from "./frameTypes";

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm";

const HAND_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const POSE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";

const SEGMENTER_MODEL =
  "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite";

const MAX_TRACKING_GAP = 3;
const GRAY_WIDTH = 320;
/** Lado mayor de la resolución de trabajo de la máscara de persona. */
const MASK_MAX_SIDE = 1024;
const MAX_VIEW_WIDTH = 1400;
/** Distancia máxima (en anchos de imagen) entre muñeca de pose y de mano para confiar en la pose. */
const POSE_HAND_MAX_DISTANCE = 0.1;

type ViewMode = "overlay" | "mask" | "cutout";

const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17]
];

/* ---------- Modelos: se crean una sola vez ---------- */

function once<T>(factory: () => Promise<T>) {
  let promise: Promise<T> | null = null;

  return () => {
    if (!promise) {
      promise = factory().catch(error => {
        promise = null;
        throw error;
      });
    }

    return promise;
  };
}

const getVision = once(() => FilesetResolver.forVisionTasks(WASM_URL));

const getPose = once(async () =>
  PoseLandmarker.createFromOptions(await getVision(), {
    baseOptions: { modelAssetPath: POSE_MODEL },
    runningMode: "IMAGE",
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5
  })
);

const getHand = once(async () =>
  HandLandmarker.createFromOptions(await getVision(), {
    baseOptions: { modelAssetPath: HAND_MODEL },
    runningMode: "IMAGE",
    numHands: 1,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5
  })
);

const getSegmenter = once(async () =>
  ImageSegmenter.createFromOptions(await getVision(), {
    baseOptions: { modelAssetPath: SEGMENTER_MODEL },
    runningMode: "IMAGE",
    outputCategoryMask: true
  })
);

/* ---------- Helpers ---------- */

const distance = (a: Point, b: Point) => Math.hypot(a.x - b.x, a.y - b.y);

let grayCanvas: HTMLCanvasElement | null = null;

function bitmapToGray(bitmap: ImageBitmap, width = GRAY_WIDTH): GrayFrame {
  const height = Math.round((bitmap.height / bitmap.width) * width);

  if (!grayCanvas) grayCanvas = document.createElement("canvas");

  grayCanvas.width = width;
  grayCanvas.height = height;

  const context = grayCanvas.getContext("2d", {
    willReadFrequently: true
  });

  if (!context) {
    throw new Error("No se pudo crear el contexto de tracking");
  }

  context.drawImage(bitmap, 0, 0, width, height);

  const image = context.getImageData(0, 0, width, height);
  const gray = new Uint8Array(width * height);

  for (let i = 0, p = 0; i < image.data.length; i += 4, p++) {
    gray[p] =
      0.299 * image.data[i] +
      0.587 * image.data[i + 1] +
      0.114 * image.data[i + 2];
  }

  return { data: gray, width, height };
}

function isFrameEntry(entry: JSZip.JSZipObject) {
  if (entry.dir) return false;
  if (!/\.jpe?g$/i.test(entry.name)) return false;
  if (entry.name.includes("__MACOSX/")) return false;

  const base = entry.name.split("/").pop() ?? "";

  return !base.startsWith(".");
}

/** Índice de la categoría "persona" según las etiquetas del modelo. */
function personCategoryIndex(segmenter: ImageSegmenter) {
  const index = segmenter
    .getLabels()
    .findIndex(label => label && !/background/i.test(label));

  return index >= 0 ? index : 1;
}

/** Máscara de persona (0/1) a resolución de trabajo. */
async function segmentPerson(
  segmenter: ImageSegmenter,
  bitmap: ImageBitmap
) {
  const scale = Math.min(
    1,
    MASK_MAX_SIDE / Math.max(bitmap.width, bitmap.height)
  );

  const work =
    scale < 1
      ? await createImageBitmap(bitmap, {
          resizeWidth: Math.round(bitmap.width * scale),
          resizeHeight: Math.round(bitmap.height * scale),
          resizeQuality: "medium"
        })
      : bitmap;

  try {
    const result = segmenter.segment(work);

    try {
      const mask = result.categoryMask;

      if (!mask) throw new Error("El segmenter no devolvió categoryMask");

      const personIndex = personCategoryIndex(segmenter);
      const data = mask.getAsUint8Array();
      const person = new Uint8Array(data.length);

      for (let i = 0; i < data.length; i++) {
        person[i] = data[i] === personIndex ? 1 : 0;
      }

      return { person, width: mask.width, height: mask.height };
    } finally {
      result.close();
    }
  } finally {
    if (work !== bitmap) work.close();
  }
}

/* ---------- Componente ---------- */

export default function ForearmDatasetAnalyzer() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const zipRef = useRef<JSZip | null>(null);
  const zipNameRef = useRef("dataset.zip");

  const [labels, setLabels] = useState<string[]>([]);
  const [results, setResults] = useState<FrameResult[]>([]);
  const [selectedFrame, setSelectedFrame] = useState("");
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [failedFrames, setFailedFrames] = useState(0);

  const [params, setParams] = useState<MaskParams>(DEFAULT_MASK_PARAMS);
  const [quality, setQuality] = useState<Record<string, FrameQuality>>({});
  const [computing, setComputing] = useState(false);
  const [overrides, setOverrides] = useState<Record<string, boolean>>({});

  const [viewMode, setViewMode] = useState<ViewMode>("overlay");
  const [showHand, setShowHand] = useState(true);
  const [onlyFlagged, setOnlyFlagged] = useState(false);
  const [includeCutouts, setIncludeCutouts] = useState(false);
  const [exporting, setExporting] = useState<string | null>(null);

  async function analyzeZip(zipFile: File) {
    setProcessing(true);
    setResults([]);
    setSelectedFrame("");
    setError(null);
    setFailedFrames(0);
    setQuality({});
    setOverrides({});
    setProgress("");
    zipRef.current = null;
    zipNameRef.current = zipFile.name;

    try {
      const zip = await JSZip.loadAsync(zipFile);
      zipRef.current = zip;

      const [pose, hand, segmenter] = await Promise.all([
        getPose(),
        getHand(),
        getSegmenter()
      ]);

      const entries = Object.values(zip.files)
        .filter(isFrameEntry)
        .sort((a, b) =>
          a.name.localeCompare(b.name, undefined, { numeric: true })
        );

      if (!entries.length) {
        setError("El ZIP no contiene imágenes JPG.");
        return;
      }

      const analyzed: FrameResult[] = [];
      let failed = 0;

      let lastRoi: ForearmRoi | undefined;
      let lastRoiGray: GrayFrame | undefined;
      let trackingGap = 0;

      for (let index = 0; index < entries.length; index++) {
        const entry = entries[index];
        let imageUrl: string | null = null;
        let bitmap: ImageBitmap | null = null;

        setProgress(`Frame ${index + 1}/${entries.length}`);

        try {
          const raw = await entry.async("blob");
          const blob = new Blob([raw], { type: "image/jpeg" });

          imageUrl = URL.createObjectURL(blob);
          bitmap = await createImageBitmap(blob);

          // ¿El EXIF rota la foto 90°? (para no desalinear máscara e imagen al exportar)
          const rawBitmap = await createImageBitmap(blob, {
            imageOrientation: "none"
          });
          const orientationSwapped = rawBitmap.width !== bitmap.width;
          rawBitmap.close();

          const aspect = bitmap.width / bitmap.height;

          const poseResult = pose.detect(bitmap);
          const handResult = hand.detect(bitmap);

          /* --- Mano --- */
          let handDetected = false;
          let handSide: Side = "none";
          let wristX = 0;
          let wristY = 0;
          let handLandmarks: Landmark[] = [];

          if (handResult.landmarks.length) {
            const landmarks = handResult.landmarks[0];

            handLandmarks = landmarks.map(({ x, y, z }) => ({ x, y, z }));
            wristX = landmarks[0].x;
            wristY = landmarks[0].y;
            handDetected = true;

            handSide =
              handResult.handedness[0]?.[0]?.categoryName === "Left"
                ? "Left"
                : "Right";
          }

          /* --- Pose --- */
          let poseDetected = false;
          let elbowVisibility = 0;
          let wristVisibility = 0;
          let forearmLength = 0;
          let elbowX = 0;
          let elbowY = 0;
          let forearmRoi: ForearmRoi | undefined;

          if (poseResult.landmarks.length) {
            const landmarks = poseResult.landmarks[0];

            const leftElbow = landmarks[13];
            const rightElbow = landmarks[14];
            const leftWrist = landmarks[15];
            const rightWrist = landmarks[16];

            const leftScore =
              (leftElbow.visibility ?? 0) + (leftWrist.visibility ?? 0);
            const rightScore =
              (rightElbow.visibility ?? 0) + (rightWrist.visibility ?? 0);

            let useLeft = leftScore >= rightScore;

            if (handDetected) {
              const handWrist = { x: wristX, y: wristY };
              useLeft =
                distance(leftWrist, handWrist) <=
                distance(rightWrist, handWrist);

              handSide = useLeft ? "Left" : "Right";
            }

            const elbow = useLeft ? leftElbow : rightElbow;
            const poseWrist = useLeft ? leftWrist : rightWrist;

            elbowVisibility = elbow.visibility ?? 0;
            wristVisibility = poseWrist.visibility ?? 0;
            elbowX = elbow.x;
            elbowY = elbow.y;

            poseDetected =
              elbowVisibility >= 0.5 || wristVisibility >= 0.5;

            forearmLength = Math.hypot(
              elbow.x - poseWrist.x,
              (elbow.y - poseWrist.y) / aspect
            );

            // Si la muñeca de pose está lejos de la muñeca de la mano, la pose
            // probablemente sigue otra cosa: se descarta y se usa la mano.
            const poseMismatch =
              handDetected &&
              Math.hypot(
                poseWrist.x - wristX,
                (poseWrist.y - wristY) / aspect
              ) > POSE_HAND_MAX_DISTANCE;

            if (
              elbowVisibility >= 0.7 &&
              wristVisibility >= 0.7 &&
              forearmLength >= 0.1 &&
              !poseMismatch
            ) {
              forearmRoi = createForearmRoiFromPose(
                { x: poseWrist.x, y: poseWrist.y },
                { x: elbow.x, y: elbow.y },
                aspect
              );
            }
          }

          if (!forearmRoi && handDetected) {
            forearmRoi =
              createForearmRoiFromHand(
                { x: wristX, y: wristY },
                handLandmarks.map(({ x, y }) => ({ x, y })),
                aspect
              ) ?? undefined;
          }

          /* --- Tracking --- */
          let trackingDx: number | undefined;
          let trackingDy: number | undefined;
          let trackingError: number | undefined;

          const currentGray = bitmapToGray(bitmap);

          if (forearmRoi) {
            lastRoi = forearmRoi;
            lastRoiGray = currentGray;
            trackingGap = 0;
          } else if (
            lastRoi &&
            lastRoiGray &&
            trackingGap < MAX_TRACKING_GAP
          ) {
            const tracked = trackForearmRoi(
              lastRoi,
              lastRoiGray,
              currentGray
            );

            if (tracked) {
              forearmRoi = tracked.roi;
              lastRoi = tracked.roi;
              lastRoiGray = currentGray;
              trackingDx = tracked.dx;
              trackingDy = tracked.dy;
              trackingError = tracked.error;
            }

            trackingGap++;
          } else {
            trackingGap++;
          }

          /* --- Máscara de persona (solo si hay ROI) --- */
          let person: Uint8Array | null = null;
          let maskWidth = 0;
          let maskHeight = 0;

          if (forearmRoi) {
            try {
              const segmented = await segmentPerson(segmenter, bitmap);
              person = segmented.person;
              maskWidth = segmented.width;
              maskHeight = segmented.height;
            } catch (segError) {
              console.warn(`Sin máscara (${entry.name}):`, segError);
            }
          }

          analyzed.push({
            name: entry.name,
            handDetected,
            handSide,
            wristX,
            wristY,
            handLandmarks,
            imageUrl,
            poseDetected,
            elbowVisibility,
            wristVisibility,
            forearmLength,
            elbowX,
            elbowY,
            forearmRoi,
            trackingDx,
            trackingDy,
            trackingError,
            imageWidth: bitmap.width,
            imageHeight: bitmap.height,
            orientationSwapped,
            person,
            maskWidth,
            maskHeight,
            thumb: makeThumb(bitmap)
          });

          imageUrl = null;
        } catch (frameError) {
          failed++;
          console.warn(`Frame omitido (${entry.name}):`, frameError);
        } finally {
          bitmap?.close();
          if (imageUrl) URL.revokeObjectURL(imageUrl);
        }
      }

      setFailedFrames(failed);
      setResults(analyzed);
      setProgress("");

      const first = analyzed.find(frame => frame.forearmRoi && frame.person);

      if (first) setSelectedFrame(first.name);
    } catch (zipError) {
      console.error("Error analizando ZIP:", zipError);
      setError(
        zipError instanceof Error
          ? `Error analizando ZIP: ${zipError.message}`
          : "Error analizando ZIP"
      );
    } finally {
      setProcessing(false);
    }
  }

  /* Etiquetas del segmenter (para verificar la polaridad de la máscara) */
  useEffect(() => {
    let alive = true;

    getSegmenter()
      .then(segmenter => {
        if (alive) setLabels(segmenter.getLabels());
      })
      .catch(segError => {
        console.error("No se pudo cargar el segmenter:", segError);
      });

    return () => {
      alive = false;
    };
  }, []);

  /* Calidad + miniaturas de todos los frames (se recalcula al mover los parámetros) */
  useEffect(() => {
    if (!results.length) {
      setQuality({});
      setComputing(false);
      return;
    }

    let cancelled = false;

    const timer = setTimeout(async () => {
      setComputing(true);

      const next: Record<string, FrameQuality> = {};

      for (const frame of results) {
        if (cancelled) return;

        if (frame.forearmRoi && frame.person) {
          const cleaned = cleanForearmMask(
            frame.person,
            frame.maskWidth,
            frame.maskHeight,
            frame.forearmRoi,
            params
          );

          next[frame.name] = {
            stats: cleaned.stats,
            previewUrl: renderPreview(frame, cleaned, params)
          };
        }

        await new Promise<void>(resolve => setTimeout(resolve, 0));
      }

      if (!cancelled) {
        setQuality(next);
        setComputing(false);
      }
    }, 300);

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [results, params]);

  const selected = results.find(frame => frame.name === selectedFrame);

  const cleaned = useMemo(() => {
    if (!selected?.forearmRoi || !selected.person) return null;

    return cleanForearmMask(
      selected.person,
      selected.maskWidth,
      selected.maskHeight,
      selected.forearmRoi,
      params
    );
  }, [selected, params]);

  /* Visor grande */
  useEffect(() => {
    const canvas = canvasRef.current;
    const frame = selected;
    const roi = frame?.forearmRoi;

    if (!canvas || !frame || !roi) return;

    let cancelled = false;

    const render = async () => {
      const image = new Image();
      image.src = frame.imageUrl;
      await image.decode();

      if (cancelled) return;

      const context = canvas.getContext("2d");
      if (!context) return;

      const scale = Math.min(1, MAX_VIEW_WIDTH / image.naturalWidth);
      const W = Math.round(image.naturalWidth * scale);
      const H = Math.round(image.naturalHeight * scale);

      canvas.width = W;
      canvas.height = H;
      context.imageSmoothingQuality = "high";

      const alpha = cleaned
        ? maskToCanvas(cleaned.mask, cleaned.width, cleaned.height, "alpha")
        : null;

      if (viewMode === "mask" && alpha) {
        context.fillStyle = "#000";
        context.fillRect(0, 0, W, H);
        context.drawImage(alpha, 0, 0, W, H);
        return;
      }

      if (viewMode === "cutout" && alpha) {
        const temp = document.createElement("canvas");
        temp.width = W;
        temp.height = H;

        const tempContext = temp.getContext("2d");
        if (!tempContext) return;

        tempContext.drawImage(image, 0, 0, W, H);
        tempContext.globalCompositeOperation = "destination-in";
        tempContext.drawImage(alpha, 0, 0, W, H);

        context.fillStyle = "#000";
        context.fillRect(0, 0, W, H);
        context.drawImage(temp, 0, 0);
        return;
      }

      context.drawImage(image, 0, 0, W, H);

      if (cleaned) {
        context.drawImage(
          maskToCanvas(cleaned.mask, cleaned.width, cleaned.height, "overlay"),
          0,
          0,
          W,
          H
        );
      }

      const unit = Math.max(1, W / 1000);

      if (showHand && frame.handDetected) {
        context.lineWidth = 3 * unit;
        context.strokeStyle = "#00ff66";

        for (const [start, end] of HAND_CONNECTIONS) {
          const a = frame.handLandmarks[start];
          const b = frame.handLandmarks[end];

          if (!a || !b) continue;

          context.beginPath();
          context.moveTo(a.x * W, a.y * H);
          context.lineTo(b.x * W, b.y * H);
          context.stroke();
        }
      }

      drawRoiOutline(
        context,
        effectiveRoiGeometry(roi, W, H, params),
        4 * unit,
        "#ff40ff"
      );

      context.font = `bold ${26 * unit}px Arial`;
      context.fillStyle = "#ff40ff";
      context.fillText(
        `ROI · ${roi.source.toUpperCase()}`,
        20 * unit,
        38 * unit
      );
    };

    render().catch(renderError => {
      console.error("Error dibujando frame:", renderError);
    });

    return () => {
      cancelled = true;
    };
  }, [selected, cleaned, viewMode, showHand, params]);

  /* Liberar object URLs */
  useEffect(() => {
    return () => {
      results.forEach(frame => {
        URL.revokeObjectURL(frame.imageUrl);
      });
    };
  }, [results]);

  const isAccepted = (name: string) =>
    overrides[name] ?? quality[name]?.stats.ok ?? false;

  async function handleExport() {
    const zip = zipRef.current;

    if (!zip) return;

    setExporting("Preparando...");
    setError(null);

    try {
      const blob = await exportDataset(
        zip,
        results,
        isAccepted,
        quality,
        {
          params,
          includeCutouts,
          sourceName: zipNameRef.current,
          segmenterLabels: labels
        },
        (done, total) => setExporting(`Exportando ${done}/${total}...`)
      );

      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");

      link.href = url;
      link.download = `${zipNameRef.current.replace(/\.zip$/i, "")}_forearm_dataset.zip`;
      link.click();

      setTimeout(() => URL.revokeObjectURL(url), 10000);
    } catch (exportError) {
      console.error("Error exportando dataset:", exportError);
      setError(
        exportError instanceof Error
          ? `Error exportando: ${exportError.message}`
          : "Error exportando dataset"
      );
    } finally {
      setExporting(null);
    }
  }

  const framesWithMask = results.filter(f => f.forearmRoi && f.person);
  const withoutRoi = results.filter(f => !f.forearmRoi);
  const acceptedCount = framesWithMask.filter(f => isAccepted(f.name)).length;
  const flaggedCount = framesWithMask.filter(
    f => (quality[f.name]?.stats.flags.length ?? 0) > 0
  ).length;

  const handCount = results.filter(f => f.handDetected).length;
  const poseCount = results.filter(f => f.poseDetected).length;
  const roiCount = results.filter(f => f.forearmRoi).length;
  const trackingCount = results.filter(
    f => f.forearmRoi?.source === "tracking"
  ).length;

  const gridFrames = framesWithMask.filter(
    f =>
      !onlyFlagged || (quality[f.name]?.stats.flags.length ?? 0) > 0
  );

  const selectedQuality = selected ? quality[selected.name] : undefined;

  const slider = (
    label: string,
    key: "thicknessScale" | "lengthScale" | "endTrim" | "morphRadius",
    min: number,
    max: number,
    step: number,
    format: (value: number) => string
  ) => (
    <label style={{ display: "block", marginBottom: 6 }}>
      {label}: <strong>{format(params[key])}</strong>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={params[key]}
        style={{ display: "block", width: 260 }}
        onChange={event =>
          setParams(previous => ({
            ...previous,
            [key]: Number(event.target.value)
          }))
        }
      />
    </label>
  );

  return (
    <section>
      <h2>C14 — Aislamiento del antebrazo</h2>

      <input
        type="file"
        accept=".zip"
        disabled={processing}
        onChange={event => {
          const file = event.target.files?.[0];

          if (file) {
            void analyzeZip(file);
          }

          event.target.value = "";
        }}
      />

      {processing && <p>Procesando... {progress}</p>}
      {error && <p role="alert">{error}</p>}
      {failedFrames > 0 && <p>Frames omitidos por error: {failedFrames}</p>}

      {results.length > 0 && (
        <>
          <p>
            Frames: {results.length} · ROI: {roiCount} (tracking:{" "}
            {trackingCount}) · Mano: {handCount} · Pose: {poseCount} · Sin
            ROI (excluidos): {withoutRoi.length}
          </p>

          <p>
            Etiquetas del segmenter:{" "}
            {labels.length ? labels.join(", ") : "(no disponibles)"}
          </p>

          <fieldset style={{ marginBottom: 12 }}>
            <legend>Parámetros de la máscara</legend>

            {slider("Grosor del ROI", "thicknessScale", 0.6, 2, 0.05, v => `×${v.toFixed(2)}`)}
            {slider("Largo del ROI (solo ROI estimados desde la mano)", "lengthScale", 0.8, 2.5, 0.05, v => `×${v.toFixed(2)}`)}
            {slider("Recorte de extremos", "endTrim", 0, 0.25, 0.01, v => `${(v * 100).toFixed(0)}%`)}
            {slider("Limpieza morfológica", "morphRadius", 0, 8, 1, v => `${v} px`)}

            <label>
              <input
                type="checkbox"
                checked={params.invertPerson}
                onChange={event =>
                  setParams(previous => ({
                    ...previous,
                    invertPerson: event.target.checked
                  }))
                }
              />{" "}
              Invertir máscara de persona (solo si pinta el fondo)
            </label>

            <p style={{ margin: "6px 0 0" }}>
              {computing ? "Recalculando máscaras..." : "Máscaras al día."}
            </p>
          </fieldset>

          <p>
            Aceptados para exportar:{" "}
            <strong>
              {acceptedCount}/{framesWithMask.length}
            </strong>{" "}
            · Con avisos: {flaggedCount}
          </p>

          <label>
            <input
              type="checkbox"
              checked={onlyFlagged}
              onChange={event => setOnlyFlagged(event.target.checked)}
            />{" "}
            Mostrar solo frames con avisos
          </label>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(170px, 1fr))",
              gap: 8,
              margin: "8px 0 16px"
            }}
          >
            {gridFrames.map(frame => {
              const q = quality[frame.name];
              const accepted = isAccepted(frame.name);
              const isSelected = frame.name === selectedFrame;
              const short = frame.name.split("/").pop() ?? frame.name;

              return (
                <figure
                  key={frame.name}
                  onClick={() => setSelectedFrame(frame.name)}
                  style={{
                    margin: 0,
                    cursor: "pointer",
                    borderRadius: 4,
                    border: `3px solid ${
                      isSelected ? "#0090ff" : accepted ? "#2e9e5b" : "#d33"
                    }`
                  }}
                >
                  {q?.previewUrl ? (
                    <img
                      src={q.previewUrl}
                      alt={short}
                      style={{ width: "100%", display: "block" }}
                    />
                  ) : (
                    <div style={{ padding: 24, textAlign: "center" }}>…</div>
                  )}

                  <figcaption style={{ fontSize: 12, padding: 4 }}>
                    <label onClick={event => event.stopPropagation()}>
                      <input
                        type="checkbox"
                        checked={accepted}
                        onChange={event =>
                          setOverrides(previous => ({
                            ...previous,
                            [frame.name]: event.target.checked
                          }))
                        }
                      />{" "}
                      {short}
                    </label>

                    {q && (
                      <div>
                        cob. {(q.stats.coverage * 100).toFixed(0)}% ·{" "}
                        {frame.forearmRoi?.source}
                      </div>
                    )}

                    {q && q.stats.flags.length > 0 && (
                      <div style={{ color: "#c60" }}>
                        {q.stats.flags.join(", ")}
                      </div>
                    )}
                  </figcaption>
                </figure>
              );
            })}
          </div>

          {selected && (
            <>
              <p>
                <strong>{selected.name}</strong> · Mano: {selected.handSide}
                {selectedQuality &&
                  ` · cobertura ${(selectedQuality.stats.coverage * 100).toFixed(0)}% · contacto con borde del ROI ${(selectedQuality.stats.edgeContact * 100).toFixed(0)}%`}
              </p>

              <div style={{ marginBottom: 6 }}>
                <select
                  value={viewMode}
                  onChange={event =>
                    setViewMode(event.target.value as ViewMode)
                  }
                >
                  <option value="overlay">Foto + máscara + ROI</option>
                  <option value="mask">Solo máscara</option>
                  <option value="cutout">Recorte sobre negro</option>
                </select>{" "}
                <label>
                  <input
                    type="checkbox"
                    checked={showHand}
                    onChange={event => setShowHand(event.target.checked)}
                  />{" "}
                  Esqueleto de mano
                </label>
              </div>

              <canvas
                ref={canvasRef}
                style={{
                  display: "block",
                  maxWidth: "100%",
                  height: "auto"
                }}
              />
            </>
          )}

          <h3>Exportar dataset</h3>

          <label>
            <input
              type="checkbox"
              checked={includeCutouts}
              onChange={event => setIncludeCutouts(event.target.checked)}
            />{" "}
            Incluir recortes PNG con transparencia (más pesado)
          </label>

          <p>
            <button
              disabled={!!exporting || computing || acceptedCount === 0}
              onClick={() => void handleExport()}
            >
              {exporting ?? `Descargar ZIP (${acceptedCount} frames)`}
            </button>
          </p>

          <details>
            <summary>Tabla de detección por frame</summary>

            <table>
              <thead>
                <tr>
                  <th>Frame</th>
                  <th>Mano</th>
                  <th>Lado</th>
                  <th>Pose</th>
                  <th>Codo</th>
                  <th>Muñeca</th>
                  <th>Longitud (× ancho)</th>
                  <th>ROI</th>
                  <th>ΔX (px@{GRAY_WIDTH})</th>
                  <th>ΔY (px@{GRAY_WIDTH})</th>
                  <th>Error</th>
                  <th>Cobertura</th>
                  <th>Avisos</th>
                  <th>Aceptado</th>
                </tr>
              </thead>

              <tbody>
                {results.map(frame => {
                  const q = quality[frame.name];

                  return (
                    <tr key={frame.name}>
                      <td>{frame.name}</td>
                      <td>{frame.handDetected ? "Sí" : "No"}</td>
                      <td>{frame.handSide}</td>
                      <td>{frame.poseDetected ? "Sí" : "No"}</td>
                      <td>{frame.elbowVisibility.toFixed(2)}</td>
                      <td>{frame.wristVisibility.toFixed(2)}</td>
                      <td>{frame.forearmLength.toFixed(3)}</td>
                      <td>{frame.forearmRoi?.source ?? "—"}</td>
                      <td>{frame.trackingDx?.toFixed(1) ?? "—"}</td>
                      <td>{frame.trackingDy?.toFixed(1) ?? "—"}</td>
                      <td>{frame.trackingError?.toFixed(1) ?? "—"}</td>
                      <td>{q ? `${(q.stats.coverage * 100).toFixed(0)}%` : "—"}</td>
                      <td>{q?.stats.flags.join(", ") || "—"}</td>
                      <td>{q ? (isAccepted(frame.name) ? "Sí" : "No") : "—"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </details>
        </>
      )}
    </section>
  );
}
