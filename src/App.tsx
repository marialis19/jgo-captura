import { useEffect, useRef, useState } from "react";
import "./App.css";

function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const previousFrameRef = useRef<Uint8Array | null>(null);

  const [cameraActive, setCameraActive] = useState(false);
  const [error, setError] = useState("");
  const [cameraInfo, setCameraInfo] = useState("");
  const [frameQuality, setFrameQuality] = useState("");

  const analyzeFrame = () => {
  const video = videoRef.current;
  const canvas = canvasRef.current;

  if (!video || !canvas || video.readyState < 2) return;

  const context = canvas.getContext("2d", { willReadFrequently: true });

  if (!context) return;

  const width = 320;
  const height = Math.round((video.videoHeight / video.videoWidth) * width);

  canvas.width = width;
  canvas.height = height;

  context.drawImage(video, 0, 0, width, height);

  const imageData = context.getImageData(0, 0, width, height);
  const pixels = imageData.data;

  const grayscale = new Uint8Array(width * height);

  for (let i = 0, pixel = 0; i < pixels.length; i += 4, pixel++) {
    grayscale[pixel] =
      0.299 * pixels[i] +
     0.587 * pixels[i + 1] +
      0.114 * pixels[i + 2];
  }

  let temporalChange = 0;

  if (previousFrameRef.current) {
    for (let i = 0; i < grayscale.length; i++) {
      temporalChange += Math.abs(
        grayscale[i] - previousFrameRef.current[i]
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

      const laplacian = left + right + top + bottom - 4 * center;

      sharpness += laplacian * laplacian;
    }
  }

  sharpness /= (width - 2) * (height - 2);

  const goodBrightness = brightness >= 45 && brightness <= 210;
  const goodSharpness = sharpness >= 100;

  const status = goodBrightness && goodSharpness ? "Bueno" : "Revisar";

  setFrameQuality(
    `Nitidez: ${sharpness.toFixed(0)} · Brillo: ${brightness.toFixed(0)} · Cambio: ${temporalChange.toFixed(1)} · ${status}`
  );
};

  const startCamera = async () => {
    try {
      setError("");

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 1920 },
          height: { ideal: 1080 }
        }
      });

      streamRef.current = stream;
      const track = stream.getVideoTracks()[0];
      const settings = track.getSettings();

      setCameraInfo(
        `${settings.width} × ${settings.height} · ${settings.frameRate} FPS · ${settings.facingMode}`
      );

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setCameraActive(true);
    } catch (error) {
      console.error(error);
      setError("No se pudo acceder a la cámara.");
    }
  };

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());

    streamRef.current = null;


    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setCameraActive(false);
  };

  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  useEffect(() => {
    if (!cameraActive) return;
    const interval = window.setInterval(analyzeFrame, 500);
    return () => { window.clearInterval(interval); };
  }, [cameraActive]);

  return (
    <main>
      <h1>JGO Capture</h1>

      <p>PoC-C1 · Captura de antebrazo</p>

      <div className="camera-container">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
        />
        <canvas ref={canvasRef} hidden />
      </div>

      {cameraInfo && <p>{cameraInfo}</p>}
      {frameQuality && <p>{frameQuality}</p>}
      {!cameraActive ? (
        <button onClick={startCamera}>
          Activar cámara
        </button>
      ) : (
        <button onClick={stopCamera}>
          Detener cámara
        </button>
      )}

      {error && <p className="error">{error}</p>}
    </main>
  );
}

export default App;