import {
  useEffect,
  useRef,
  useState
} from "react";
import JSZip from "jszip";
import {
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker
} from "@mediapipe/tasks-vision";
import {
  createForearmRoi,
  createForearmRoiFromPose,
  directionFromHand,
  directionFromPose
} from "./forearmRoi";
import type {
  ForearmRoi,
  Point
} from "./forearmRoi";

const HAND_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const POSE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";

type Landmark = {
  x: number;
  y: number;
  z: number;
};

type FrameResult = {
  name: string;
  handDetected: boolean;
  handSide: "Left" | "Right" | "none";
  wristX: number;
  wristY: number;
  handLandmarks: Landmark[];
  imageUrl: string;
  poseDetected: boolean;
  elbowVisibility: number;
  wristVisibility: number;
  forearmLength: number;
  elbowX: number;
  elbowY: number;
  forearmRoi?: ForearmRoi;
};

const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17]
];

export default function ForearmDatasetAnalyzer() {
  const poseLandmarkerRef =
    useRef<PoseLandmarker | null>(null);

  const handLandmarkerRef =
    useRef<HandLandmarker | null>(null);

  const canvasRef =
    useRef<HTMLCanvasElement | null>(null);

  const [results, setResults] =
    useState<FrameResult[]>([]);

  const [selectedFrame, setSelectedFrame] =
    useState("");

  const [processing, setProcessing] =
    useState(false);

  async function createPoseLandmarker() {
    if (poseLandmarkerRef.current) {
      return poseLandmarkerRef.current;
    }

    const vision =
      await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
      );

    const landmarker =
      await PoseLandmarker.createFromOptions(
        vision,
        {
          baseOptions: {
            modelAssetPath: POSE_MODEL
          },
          runningMode: "IMAGE",
          numPoses: 1,
          minPoseDetectionConfidence: 0.5,
          minPosePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5
        }
      );

    poseLandmarkerRef.current = landmarker;

    return landmarker;
  }

  async function createHandLandmarker() {
    if (handLandmarkerRef.current) {
      return handLandmarkerRef.current;
    }

    const vision =
      await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
      );

    const landmarker =
      await HandLandmarker.createFromOptions(
        vision,
        {
          baseOptions: {
            modelAssetPath: HAND_MODEL
          },
          runningMode: "IMAGE",
          numHands: 1,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5
        }
      );

    handLandmarkerRef.current = landmarker;

    return landmarker;
  }

  function distance(
    a: Point,
    b: Point
  ) {
    return Math.hypot(
      a.x - b.x,
      a.y - b.y
    );
  }

  async function analyzeZip(file: File) {
    setProcessing(true);
    setResults([]);
    setSelectedFrame("");

    const zip = await JSZip.loadAsync(file);

    const poseLandmarker =
      await createPoseLandmarker();

    const handLandmarker =
      await createHandLandmarker();

    const imageFiles = Object.values(zip.files)
      .filter((entry) =>
        /\.jpe?g$/i.test(entry.name)
      )
      .sort((a, b) =>
        a.name.localeCompare(b.name)
      );

    const analyzed: FrameResult[] = [];

    for (const entry of imageFiles) {
      const blob = await entry.async("blob");
      const imageUrl = URL.createObjectURL(blob);
      const imageBitmap =
        await createImageBitmap(blob);

      const poseResult =
        poseLandmarker.detect(imageBitmap);

      const handResult =
        handLandmarker.detect(imageBitmap);

      let handDetected = false;
      let handSide: "Left" | "Right" | "none" =
        "none";

      let wristX = 0;
      let wristY = 0;
      let handLandmarks: Landmark[] = [];

      if (handResult.landmarks.length > 0) {
        const hand = handResult.landmarks[0];

        handLandmarks = hand.map(
          ({ x, y, z }) => ({ x, y, z })
        );

        const wrist = hand[0];

        handDetected = true;
        wristX = wrist.x;
        wristY = wrist.y;

        if (
          handResult.handedness.length > 0 &&
          handResult.handedness[0].length > 0
        ) {
          const category =
            handResult.handedness[0][0];

          handSide =
            category.categoryName === "Left"
              ? "Left"
              : "Right";
        }
      }

      let poseDetected = false;
      let elbowVisibility = 0;
      let wristVisibility = 0;
      let forearmLength = 0;
      let elbowX = 0;
      let elbowY = 0;
      let forearmRoi: ForearmRoi | undefined;

      if (poseResult.landmarks.length > 0) {
        const landmarks =
          poseResult.landmarks[0];

        const leftElbow = landmarks[13];
        const rightElbow = landmarks[14];
        const leftWrist = landmarks[15];
        const rightWrist = landmarks[16];

        const leftElbowVisibility =
          leftElbow.visibility ?? 0;

        const rightElbowVisibility =
          rightElbow.visibility ?? 0;

        const leftWristVisibility =
          leftWrist.visibility ?? 0;

        const rightWristVisibility =
          rightWrist.visibility ?? 0;

        const leftScore =
          leftElbowVisibility +
          leftWristVisibility;

        const rightScore =
          rightElbowVisibility +
          rightWristVisibility;

        const useLeft =
          leftScore >= rightScore;

        const elbow =
          useLeft
            ? leftElbow
            : rightElbow;

        const poseWrist =
          useLeft
            ? leftWrist
            : rightWrist;

        elbowVisibility =
          useLeft
            ? leftElbowVisibility
            : rightElbowVisibility;

        wristVisibility =
          useLeft
            ? leftWristVisibility
            : rightWristVisibility;

        elbowX = elbow.x;
        elbowY = elbow.y;

        poseDetected =
          elbowVisibility >= 0.5 ||
          wristVisibility >= 0.5;

        if (poseDetected) {
          forearmLength = distance(
            elbow,
            poseWrist
          );
        }

        if (
          elbowVisibility >= 0.5 &&
          wristVisibility >= 0.5
        ) {
          const direction =
            directionFromPose(
              {
                x: elbow.x,
                y: elbow.y
              },
              {
                x: poseWrist.x,
                y: poseWrist.y
              }
            );

          forearmRoi =
            createForearmRoiFromPose(
              {
                x: poseWrist.x,
                y: poseWrist.y
              },
              { 
                x: elbow.x,
                y: elbow.y
              }
            );
        }
      }

      if (
        !forearmRoi &&
        handDetected &&
        handLandmarks.length >= 18
      ) {
        const landmarks: Point[] =
          handLandmarks.map(
            ({ x, y }) => ({ x, y })
          );

        const direction =
          directionFromHand(
            {
              x: wristX,
              y: wristY
            },
            landmarks
          );

        if (direction) {
          forearmRoi =
            createForearmRoi(
              {
                x: wristX,
                y: wristY
              },
              direction,
              "hand"
            );
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
        forearmRoi
      });

      imageBitmap.close();
    }

    setResults(analyzed);

    const firstDetected =
      analyzed.find(
        (frame) => frame.forearmRoi
      );

    if (firstDetected) {
      setSelectedFrame(
        firstDetected.name
      );
    }

    setProcessing(false);
  }

  useEffect(() => {
    const frame =
      results.find(
        (item) =>
          item.name === selectedFrame
      );

    if (
      !frame ||
      !frame.handDetected ||
      frame.handLandmarks.length === 0
    ) {
      return;
    }

    const canvas = canvasRef.current;

    if (!canvas) return;

    const image = new Image();

    image.onload = () => {
      const context =
        canvas.getContext("2d");

      if (!context) return;

      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;

      context.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
      );

      context.drawImage(
        image,
        0,
        0
      );

      context.lineWidth = 4;
      context.strokeStyle = "#00ff66";

      for (const [start, end] of HAND_CONNECTIONS) {
        const a =
          frame.handLandmarks[start];

        const b =
          frame.handLandmarks[end];

        if (!a || !b) continue;

        context.beginPath();

        context.moveTo(
          a.x * canvas.width,
          a.y * canvas.height
        );

        context.lineTo(
          b.x * canvas.width,
          b.y * canvas.height
        );

        context.stroke();
      }

      frame.handLandmarks.forEach(
        (landmark, index) => {
          const x =
            landmark.x * canvas.width;

          const y =
            landmark.y * canvas.height;

          context.beginPath();

          context.arc(
            x,
            y,
            index === 0 ? 10 : 6,
            0,
            Math.PI * 2
          );

          context.fillStyle =
            index === 0
              ? "#ff3b30"
              : "#00ff66";

          context.fill();
        }
      );

      const wrist =
        frame.handLandmarks[0];

      if (wrist) {
        context.font =
          "bold 28px Arial";

        context.fillStyle =
          "#ff3b30";

        context.fillText(
          "WRIST",
          wrist.x * canvas.width + 15,
          wrist.y * canvas.height - 15
        );
      }

      if (frame.forearmRoi) {
        const roi = frame.forearmRoi;

        const centerX =
          roi.center.x * canvas.width;

        const centerY =
          roi.center.y * canvas.height;

        const width =
          roi.width * canvas.width;

        const height =
          roi.height * canvas.height;

        context.save();

        context.translate(
          centerX,
          centerY
        );

        context.rotate(
          roi.angle
        );

        context.strokeStyle = "#00ffff";
        context.lineWidth = 6;
        context.setLineDash([16, 10]);

        context.strokeRect(
          -height / 2,
          -width / 2,
          height,
          width
        );

        context.setLineDash([]);

        context.beginPath();

        context.moveTo(
          -height / 2,
          0
        );

        context.lineTo(
          height / 2,
          0
        );

        context.stroke();

        context.restore();

        context.font =
          "bold 28px Arial";

        context.fillStyle =
          "#00ffff";

        context.fillText(
          `FOREARM · ${roi.source.toUpperCase()}`,
          20,
          40
        );
      }
    };

    image.src = frame.imageUrl;
  }, [results, selectedFrame]);

  useEffect(() => {
    return () => {
      results.forEach((frame) => {
        URL.revokeObjectURL(frame.imageUrl);
      });
    };
  }, [results]);

  const handDetectedCount =
    results.filter(
      (frame) => frame.handDetected
    ).length;

  const poseDetectedCount =
    results.filter(
      (frame) => frame.poseDetected
    ).length;

  const bothDetectedCount =
    results.filter(
      (frame) =>
        frame.handDetected &&
        frame.poseDetected
    ).length;

  const roiDetectedCount =
    results.filter(
      (frame) => frame.forearmRoi
    ).length;

  const selected =
    results.find(
      (frame) =>
        frame.name === selectedFrame
    );

  return (
    <section>
      <h2>
        C10 — Análisis del antebrazo
      </h2>

      <input
        type="file"
        accept=".zip"
        disabled={processing}
        onChange={(event) => {
          const file =
            event.target.files?.[0];

          if (file) {
            void analyzeZip(file);
          }
        }}
      />

      {processing && (
        <p>Procesando frames...</p>
      )}

      {results.length > 0 && (
        <>
          <div>
            <p>
              Frames analizados:{" "}
              {results.length}
            </p>

            <p>
              Mano detectada:{" "}
              {handDetectedCount}/
              {results.length}
            </p>

            <p>
              Pose detectada:{" "}
              {poseDetectedCount}/
              {results.length}
            </p>

            <p>
              Mano + pose:{" "}
              {bothDetectedCount}/
              {results.length}
            </p>

            <p>
              ROI antebrazo:{" "}
              {roiDetectedCount}/
              {results.length}
            </p>
          </div>

          <div>
            <label htmlFor="frame-select">
              Ver frame:
            </label>

            <select
              id="frame-select"
              value={selectedFrame}
              onChange={(event) =>
                setSelectedFrame(
                  event.target.value
                )
              }
            >
              {results
                .filter(
                  (frame) =>
                    frame.forearmRoi
                )
                .map((frame) => (
                  <option
                    key={frame.name}
                    value={frame.name}
                  >
                    {frame.name}
                  </option>
                ))}
            </select>
          </div>

          {selected && (
            <div>
              <p>
                {selected.name} · Mano:{" "}
                {selected.handSide}
              </p>

              <canvas
                ref={canvasRef}
                style={{
                  display: "block",
                  maxWidth: "100%",
                  height: "auto"
                }}
              />
            </div>
          )}

          <table>
            <thead>
              <tr>
                <th>Frame</th>
                <th>Mano</th>
                <th>Lado</th>
                <th>Muñeca X</th>
                <th>Muñeca Y</th>
                <th>Pose</th>
                <th>Codo</th>
                <th>Muñeca</th>
                <th>Longitud</th>
                <th>ROI</th>
              </tr>
            </thead>

            <tbody>
              {results.map((frame) => (
                <tr key={frame.name}>
                  <td>{frame.name}</td>

                  <td>
                    {frame.handDetected
                      ? "Sí"
                      : "No"}
                  </td>

                  <td>
                    {frame.handSide}
                  </td>

                  <td>
                    {frame.wristX.toFixed(3)}
                  </td>

                  <td>
                    {frame.wristY.toFixed(3)}
                  </td>

                  <td>
                    {frame.poseDetected
                      ? "Sí"
                      : "No"}
                  </td>

                  <td>
                    {frame.elbowVisibility.toFixed(
                      2
                    )}
                  </td>

                  <td>
                    {frame.wristVisibility.toFixed(
                      2
                    )}
                  </td>

                  <td>
                    {frame.forearmLength.toFixed(
                      3
                    )}
                  </td>

                  <td>
                    {frame.forearmRoi
                      ? frame.forearmRoi.source
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </section>
  );
}