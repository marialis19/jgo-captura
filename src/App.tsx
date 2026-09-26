import { useEffect, useRef, useState } from "react";
import {
  ImageSegmenter,
  FilesetResolver
} from "@mediapipe/tasks-vision";
import JSZip from "jszip";
import "./App.css";
import ForearmDatasetAnalyzer
  from "./analysis/ForearmDatasetAnalyzer";

type CapturedFrame = {
  image: Blob;
  timestamp: number;
};

function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const previousFrameRef =
    useRef<Uint8Array | null>(null);

  const capturePreviousFrameRef =
    useRef<Uint8Array | null>(null);

  const segmenterRef =
    useRef<ImageSegmenter | null>(null);

  const capturedFramesRef = useRef(0);
  const selectedFramesRef =
    useRef<CapturedFrame[]>([]);

  const captureBusyRef = useRef(false);

  const [segmentationStatus, setSegmentationStatus] =
    useState("");

  const [segmentationArea, setSegmentationArea] =
    useState("");

  const [cameraActive, setCameraActive] =
    useState(false);

  const [error, setError] = useState("");
  const [cameraInfo, setCameraInfo] = useState("");
  const [frameQuality, setFrameQuality] =
    useState("");

  const [composition, setComposition] =
    useState("");

  const [capturing, setCapturing] =
    useState(false);

  const [capturedFrames, setCapturedFrames] =
    useState(0);

  const [selectedFrames, setSelectedFrames] =
    useState(0);

  const [captureDuration, setCaptureDuration] =
    useState("");

  const captureStartTimeRef =
    useRef<number | null>(null);

  const analyzeFrame = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas || video.readyState < 2) {
      return;
    }

    const context = canvas.getContext("2d", {
      willReadFrequently: true
    });

    if (!context) return;

    const width = 320;
    const height = Math.round(
      (video.videoHeight / video.videoWidth) * width
    );

    canvas.width = width;
    canvas.height = height;

    context.drawImage(
      video,
      0,
      0,
      width,
      height
    );

    const imageData = context.getImageData(
      0,
      0,
      width,
      height
    );

    const pixels = imageData.data;
    const grayscale = new Uint8Array(
      width * height
    );

    for (
      let i = 0, pixel = 0;
      i < pixels.length;
      i += 4, pixel++
    ) {
      grayscale[pixel] =
        0.299 * pixels[i] +
        0.587 * pixels[i + 1] +
        0.114 * pixels[i + 2];
    }

    let temporalChange = 0;

    if (previousFrameRef.current) {
      for (let i = 0; i < grayscale.length; i++) {
        temporalChange += Math.abs(
          grayscale[i] -
          previousFrameRef.current[i]
        );
      }

      temporalChange /= grayscale.length;
    }

    previousFrameRef.current = grayscale;

    let brightness = 0;

    for (let i = 0; i < pixels.length; i += 4) {
      const gray =
        0.299 * pixels[i] +
        0.587 * pixels[i + 1] +
        0.114 * pixels[i + 2];

      brightness += gray;
    }

    brightness /= width * height;

    let averageDifference = 0;

    for (let i = 0; i < pixels.length; i += 4) {
      const gray =
        0.299 * pixels[i] +
        0.587 * pixels[i + 1] +
        0.114 * pixels[i + 2];

      averageDifference += Math.abs(
        gray - brightness
      );
    }

    averageDifference /= width * height;

    let sharpness = 0;

    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const index = (y * width + x) * 4;

        const center =
          0.299 * pixels[index] +
          0.587 * pixels[index + 1] +
          0.114 * pixels[index + 2];

        const leftIndex = index - 4;
        const rightIndex = index + 4;
        const topIndex = index - width * 4;
        const bottomIndex = index + width * 4;

        const left =
          0.299 * pixels[leftIndex] +
          0.587 * pixels[leftIndex + 1] +
          0.114 * pixels[leftIndex + 2];

        const right =
          0.299 * pixels[rightIndex] +
          0.587 * pixels[rightIndex + 1] +
          0.114 * pixels[rightIndex + 2];

        const top =
          0.299 * pixels[topIndex] +
          0.587 * pixels[topIndex + 1] +
          0.114 * pixels[topIndex + 2];

        const bottom =
          0.299 * pixels[bottomIndex] +
          0.587 * pixels[bottomIndex + 1] +
          0.114 * pixels[bottomIndex + 2];

        const laplacian =
          left +
          right +
          top +
          bottom -
          4 * center;

        sharpness += laplacian * laplacian;
      }
    }

    sharpness /=
      (width - 2) * (height - 2);

    const goodBrightness =
      brightness >= 45 && brightness <= 210;

    const goodSharpness =
      sharpness >= 100;

    const status =
      goodBrightness && goodSharpness
        ? "Bueno"
        : "Revisar";

    setFrameQuality(
      `Nitidez: ${sharpness.toFixed(0)} · ` +
      `Brillo: ${brightness.toFixed(0)} · ` +
      `Cambio: ${temporalChange.toFixed(1)} · ` +
      status
    );

    const goodComposition =
      averageDifference >= 20;

    setComposition(
      `Variación: ${averageDifference.toFixed(1)} · ` +
      (goodComposition
        ? "Escena útil"
        : "Escena poco definida")
    );
  };

  const captureFrame = async () => {
    if (captureBusyRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas || video.readyState < 2) {
      return;
    }

    captureBusyRef.current = true;

    try {
      const analysisWidth = 640;
      const analysisHeight = Math.round(
        (video.videoHeight / video.videoWidth) *
        analysisWidth
      );

      canvas.width = analysisWidth;
      canvas.height = analysisHeight;

      const context = canvas.getContext("2d", {
        willReadFrequently: true
      });

      if (!context) return;

      context.drawImage(
        video,
        0,
        0,
        analysisWidth,
        analysisHeight
      );

      const frame = context.getImageData(
        0,
        0,
        analysisWidth,
        analysisHeight
      );

      capturedFramesRef.current += 1;
      setCapturedFrames(
        capturedFramesRef.current
      );

      const grayscale = new Uint8Array(
        frame.width * frame.height
      );

      for (
        let i = 0, j = 0;
        i < frame.data.length;
        i += 4, j++
      ) {
        grayscale[j] =
          0.299 * frame.data[i] +
          0.587 * frame.data[i + 1] +
          0.114 * frame.data[i + 2];
      }

      const previous =
        capturePreviousFrameRef.current;

      capturePreviousFrameRef.current =
        grayscale;

      if (!previous) return;

      let difference = 0;

      for (let i = 0; i < grayscale.length; i++) {
        difference += Math.abs(
          grayscale[i] - previous[i]
        );
      }

      const averageDifference =
        difference / grayscale.length;

      if (averageDifference < 8) {
        return;
      }

      const captureWidth = 1280;
      const captureHeight = Math.round(
        (video.videoHeight / video.videoWidth) *
        captureWidth
      );

      canvas.width = captureWidth;
      canvas.height = captureHeight;

      context.drawImage(
        video,
        0,
        0,
        captureWidth,
        captureHeight
      );

      const blob = await new Promise<Blob | null>(
        (resolve) => {
          canvas.toBlob(
            resolve,
            "image/jpeg",
            0.9
          );
        }
      );

      if (!blob) return;

      selectedFramesRef.current.push({
        image: blob,
        timestamp: performance.now()
      });

      setSelectedFrames(
        selectedFramesRef.current.length
      );
    } finally {
      captureBusyRef.current = false;
    }
  };

  const startCapture = () => {
    capturedFramesRef.current = 0;
    selectedFramesRef.current = [];

    previousFrameRef.current = null;
    capturePreviousFrameRef.current = null;

    setCapturedFrames(0);
    setSelectedFrames(0);
    setCaptureDuration("");

    captureStartTimeRef.current =
      performance.now();

    setCapturing(true);
  };

  const stopCapture = () => {
    setCapturing(false);

    if (captureStartTimeRef.current !== null) {
      const duration =
        (performance.now() -
          captureStartTimeRef.current) /
        1000;

      setCaptureDuration(
        `${duration.toFixed(1)} segundos`
      );
    }
  };

  const exportSelectedFrames = async () => {
    if (selectedFramesRef.current.length === 0) {
      return;
    }

    const zip = new JSZip();
    const framesFolder = zip.folder("frames");

    if (!framesFolder) {
      setError(
        "No se pudo crear la carpeta de frames."
      );
      return;
    }

    for (
      let index = 0;
      index < selectedFramesRef.current.length;
      index++
    ) {
      const { image } =
        selectedFramesRef.current[index];

      framesFolder.file(
        `frame_${String(index + 1).padStart(4, "0")}.jpg`,
        image
      );
    }

    const zipBlob =
      await zip.generateAsync({
        type: "blob"
      });

    const url =
      URL.createObjectURL(zipBlob);

    const link =
      document.createElement("a");

    link.href = url;
    link.download =
      "jgo-captura-dataset.zip";

    document.body.appendChild(link);
    link.click();
    link.remove();

    URL.revokeObjectURL(url);
  };

  const initializeSegmenter = async () => {
    if (segmenterRef.current) return;

    const vision =
      await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/" +
        "@mediapipe/tasks-vision/wasm"
      );

    segmenterRef.current =
      await ImageSegmenter.createFromOptions(
        vision,
        {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/" +
              "mediapipe-models/image_segmenter/" +
              "selfie_segmenter/float16/latest/" +
              "selfie_segmenter.tflite"
          },
          runningMode: "VIDEO",
          outputCategoryMask: true
        }
      );
  };

  const analyzeSegmentation = () => {
    const video = videoRef.current;
    const segmenter = segmenterRef.current;

    if (
      !video ||
      !segmenter ||
      video.readyState < 2
    ) {
      return;
    }

    const result =
      segmenter.segmentForVideo(
        video,
        performance.now()
      );

    const mask = result.categoryMask;

    if (!mask) {
      setSegmentationArea(
        "Máscara no disponible"
      );
      return;
    }

    const maskData =
      mask.getAsUint8Array();

    let foregroundPixels = 0;

    for (let i = 0; i < maskData.length; i++) {
      if (maskData[i] > 0) {
        foregroundPixels++;
      }
    }

    const percentage =
      (foregroundPixels / maskData.length) *
      100;

    setSegmentationArea(
      `Persona detectada: ${percentage.toFixed(1)}%`
    );

    mask.close();
  };

  const startCamera = async () => {
    try {
      setError("");

      await initializeSegmenter();

      setSegmentationStatus(
        "Segmentador listo"
      );
    } catch (error) {
      console.error(
        "Error al inicializar MediaPipe:",
        error
      );

      setError(
        "No se pudo inicializar el segmentador."
      );

      return;
    }

    try {
      const stream =
        await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            facingMode: {
              ideal: "environment"
            },
            width: {
              ideal: 1920
            },
            height: {
              ideal: 1080
            }
          }
        });

      streamRef.current = stream;

      const track =
        stream.getVideoTracks()[0];

      const settings = track.getSettings();

      setCameraInfo(
        `${settings.width} × ${settings.height} · ` +
        `${settings.frameRate} FPS · ` +
        `${settings.facingMode}`
      );

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setCameraActive(true);
    } catch (error) {
      console.error(
        "Error al acceder a la cámara:",
        error
      );

      setError(
        "No se pudo acceder a la cámara."
      );
    }
  };

  const stopCamera = () => {
    setCapturing(false);

    streamRef.current
      ?.getTracks()
      .forEach((track) => track.stop());

    streamRef.current = null;

    previousFrameRef.current = null;
    capturePreviousFrameRef.current = null;

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setCameraActive(false);
  };

  useEffect(() => {
    return () => {
      streamRef.current
        ?.getTracks()
        .forEach((track) => track.stop());
    };
  }, []);

  useEffect(() => {
    if (!cameraActive) return;

    const interval =
      window.setInterval(
        analyzeFrame,
        500
      );

    return () => {
      window.clearInterval(interval);
    };
  }, [cameraActive]);

  useEffect(() => {
    if (!cameraActive) return;

    let animationFrameId: number;

    const processFrame = () => {
      analyzeSegmentation();

      animationFrameId =
        requestAnimationFrame(processFrame);
    };

    processFrame();

    return () => {
      cancelAnimationFrame(
        animationFrameId
      );
    };
  }, [cameraActive]);

  useEffect(() => {
    if (!capturing) return;

    const interval =
      window.setInterval(
        captureFrame,
        200
      );

    return () => {
      window.clearInterval(interval);
    };
  }, [capturing]);

  return (
    <main>
      <h1>JGO Capture</h1>

      <p>
        PoC-C1 · Captura de antebrazo
      </p>

      <div className="camera-container">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
        />

        <canvas
          ref={canvasRef}
          hidden
        />

        <div className="capture-guide">
          <span>
            Ubicá el antebrazo dentro de
            esta zona
          </span>
        </div>
      </div>

      {cameraInfo && (
        <p>{cameraInfo}</p>
      )}

      {segmentationStatus && (
        <p>{segmentationStatus}</p>
      )}

      {segmentationArea && (
        <p>{segmentationArea}</p>
      )}

      {frameQuality && (
        <p>{frameQuality}</p>
      )}

      {composition && (
        <p>{composition}</p>
      )}

      {!cameraActive ? (
        <button onClick={startCamera}>
          Activar cámara
        </button>
      ) : (
        <button onClick={stopCamera}>
          Detener cámara
        </button>
      )}

      {cameraActive && (
        <>
          {!capturing ? (
            <button onClick={startCapture}>
              Iniciar captura 360°
            </button>
          ) : (
            <button onClick={stopCapture}>
              Detener captura 360°
            </button>
          )}

          {capturedFrames > 0 &&
            !capturing && (
              <button
                onClick={exportSelectedFrames}
              >
                Exportar frames seleccionados
              </button>
            )}

          <p>
            Frames capturados: {capturedFrames}
          </p>

          <p>
            Frames seleccionados: {selectedFrames}
          </p>

          {captureDuration && (
            <p>
              Duración de la captura:{" "}
              {captureDuration}
            </p>
          )}
        </>
      )}

      {error && (
        <p className="error">
          {error}
        </p>
      )}

      <ForearmDatasetAnalyzer />
    </main>    
  );
}


export default App;