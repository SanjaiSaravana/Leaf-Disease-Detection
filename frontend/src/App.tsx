import { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";

type Box = { x1: number; y1: number; x2: number; y2: number };

type PredictResponse = {
  cnn: {
    class_name: string;
    confidence: number;
    probabilities: Record<string, number>;
  };
  detections: Array<{ class_name: string; confidence: number; box: Box }>;
  image_width: number;
  image_height: number;
};

function drawDetections(
  canvas: HTMLCanvasElement,
  img: HTMLImageElement,
  detections: PredictResponse["detections"]
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const rect = img.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  ctx.clearRect(0, 0, rect.width, rect.height);
  const nw = img.naturalWidth;
  const nh = img.naturalHeight;
  if (!nw || !nh) return;

  const scale = Math.min(rect.width / nw, rect.height / nh);
  const drawW = nw * scale;
  const drawH = nh * scale;
  const offX = (rect.width - drawW) / 2;
  const offY = (rect.height - drawH) / 2;

  const mapX = (x: number) => offX + (x / nw) * drawW;
  const mapY = (y: number) => offY + (y / nh) * drawH;

  ctx.strokeStyle = "rgba(94, 207, 138, 0.95)";
  ctx.lineWidth = 2;
  ctx.font = "500 12px var(--mono), monospace";

  detections.forEach((d) => {
    const { x1, y1, x2, y2 } = d.box;
    const mx1 = mapX(x1);
    const my1 = mapY(y1);
    const w = mapX(x2) - mx1;
    const h = mapY(y2) - my1;
    ctx.strokeRect(mx1, my1, w, h);
    const label = `${d.class_name} ${(d.confidence * 100).toFixed(0)}%`;
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = "rgba(20, 28, 25, 0.92)";
    ctx.fillRect(mx1, my1 - 18, tw + 8, 18);
    ctx.fillStyle = "rgba(180, 240, 200, 0.98)";
    ctx.fillText(label, mx1 + 4, my1 - 5);
  });
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [apiStatus, setApiStatus] = useState<"unknown" | "up" | "down">("unknown");
  const [modelsReady, setModelsReady] = useState<boolean | null>(null);

  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const checkHealth = useCallback(async () => {
    try {
      const r = await fetch("/api/health");
      if (!r.ok) {
        setApiStatus("down");
        setModelsReady(null);
        return;
      }
      const j = (await r.json()) as { models_loaded?: boolean };
      setApiStatus("up");
      setModelsReady(Boolean(j.models_loaded));
    } catch {
      setApiStatus("down");
      setModelsReady(null);
    }
  }, []);

  useEffect(() => {
    checkHealth();
    const t = setInterval(checkHealth, 15000);
    return () => clearInterval(t);
  }, [checkHealth]);

  useEffect(() => {
    if (!previewUrl || !result?.detections || !imgRef.current || !canvasRef.current) return;

    const img = imgRef.current;
    const canvas = canvasRef.current;

    const redraw = () => drawDetections(canvas, img, result.detections);

    if (img.complete) redraw();
    else img.onload = redraw;

    const ro = new ResizeObserver(redraw);
    ro.observe(img);
    return () => ro.disconnect();
  }, [previewUrl, result]);

  const onPick = (f: File | null) => {
    setError(null);
    setResult(null);
    setFile(f);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(f ? URL.createObjectURL(f) : null);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const f = e.dataTransfer.files?.[0];
    if (f?.type.startsWith("image/")) onPick(f);
  };

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const body = new FormData();
      body.append("file", file);
      const res = await fetch("/api/predict", { method: "POST", body });
      if (!res.ok) {
        const j = await res.json().catch(() => ({}));
        const raw = (j as { detail?: string | Array<{ msg?: string }> }).detail;
        const detail =
          typeof raw === "string"
            ? raw
            : Array.isArray(raw)
              ? raw.map((x) => x.msg).filter(Boolean).join("; ")
              : undefined;
        throw new Error(detail || res.statusText || "Request failed");
      }
      const data = (await res.json()) as PredictResponse;
      setResult(data);
      void checkHealth();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const probs = result?.cnn.probabilities
    ? Object.entries(result.cnn.probabilities).sort((a, b) => b[1] - a[1])
    : [];

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="brand-mark" aria-hidden />
          <div>
            <h1>Plant pathology lab</h1>
            <p className="tagline">AMFFT analysis on your leaf images</p>
          </div>
        </div>
        <div
          className={`pill ${
            apiStatus === "up" && modelsReady
              ? "pill-ok"
              : apiStatus === "up"
                ? "pill-warn"
                : apiStatus === "down"
                  ? "pill-bad"
                  : ""
          }`}
        >
          <span className="dot" />
          {apiStatus === "unknown"
            ? "Checking API…"
            : apiStatus === "down"
              ? "API offline — start backend (port 8000)"
              : modelsReady
                ? "API connected · models ready"
                : "API up · models load on first analysis"}
        </div>
      </header>

      <main className="grid">
        <section className="card panel">
          <h2>Image</h2>
          <p className="hint">JPEG, PNG, or WebP — drag a file or browse.</p>

          <label
            className="dropzone"
            onDrop={onDrop}
            onDragOver={onDragOver}
          >
            <input
              type="file"
              accept="image/jpeg,image/png,image/webp,image/bmp"
              onChange={(e) => onPick(e.target.files?.[0] ?? null)}
            />
            <div className="drop-inner">
              <span className="drop-title">Drop image here</span>
              <span className="drop-sub">or click to choose</span>
            </div>
          </label>

          {file && (
            <p className="file-name">
              Selected: <strong>{file.name}</strong>
            </p>
          )}

          <button type="button" className="btn primary" disabled={!file || loading} onClick={analyze}>
            {loading ? "Analyzing…" : "Run analysis"}
          </button>

          {error && <div className="alert error">{error}</div>}

          <div className="preview-wrap">
            {previewUrl ? (
              <div className="preview-stage">
                <img ref={imgRef} src={previewUrl} alt="Upload preview" className="preview-img" />
                <canvas ref={canvasRef} className="overlay" />
              </div>
            ) : (
              <div className="preview-placeholder">Preview appears after you select an image</div>
            )}
          </div>
        </section>

        <section className="card panel">
          <h2>Results</h2>

          {!result && !loading && (
            <p className="muted">Run analysis to see CNN classification and AMFFT detection.</p>
          )}

          {loading && <p className="muted pulse">Loading model inference…</p>}

          {result && (
            <>
              <div className="hero-result">
                <span className="label">CNN (image-level)</span>
                <p className="hero-class">{result.cnn.class_name}</p>
                <p className="hero-conf">{(result.cnn.confidence * 100).toFixed(1)}% confidence</p>
              </div>

              <h3 className="subhead">Class probabilities</h3>
              <ul className="prob-list">
                {probs.map(([name, p]) => (
                  <li key={name}>
                    <span className="prob-name">{name}</span>
                    <span className="prob-bar-wrap">
                      <span className="prob-bar" style={{ width: `${Math.min(100, p * 100)}%` }} />
                    </span>
                    <span className="prob-val">{(p * 100).toFixed(1)}%</span>
                  </li>
                ))}
              </ul>

              <h3 className="subhead">AMFFT DETECTION ({result.detections.length})</h3>
              {result.detections.length === 0 ? (
                <p className="muted">No boxes above the model confidence threshold.</p>
              ) : (
                <ul className="det-list">
                  {result.detections.map((d, i) => (
                    <li key={`${d.class_name}-${i}`}>
                      <span className="det-class">{d.class_name}</span>
                      <span className="det-conf">{(d.confidence * 100).toFixed(1)}%</span>
                      <span className="det-box mono">
                        {Math.round(d.box.x1)}, {Math.round(d.box.y1)} → {Math.round(d.box.x2)},{" "}
                        {Math.round(d.box.y2)}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </>
          )}
        </section>
      </main>

      <footer className="footer">
        <span>Frontend proxies <code>/api</code> to <code>http://127.0.0.1:8000</code></span>
      </footer>
    </div>
  );
}
