"""
REST API for hybrid CNN + YOLO plant disease inference.
Serves the trained assets at the project root (best_cnn_model.h5, YOLO weights).
"""
from __future__ import annotations

import io
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CNN_PATH = ROOT / "best_cnn_model.h5"
CLASS_NAMES = [
    "Corn Common Rust",
    "Peach_Bacterial_Spot",
    "Strawberry leaf scorch",
    "Tomato Late Blight",
]
IMG_SIZE = (224, 224)
YOLO_CONF = 0.25

_cnn_model: Any = None
_yolo_model: Any = None
_load_lock = threading.Lock()
_load_error: str | None = None


def _set_tf_memory_growth() -> None:
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def _find_yolo_weights() -> Path:
    candidates = list(ROOT.glob("runs/detect/*/weights/best.pt"))
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    fallback = ROOT / "yolov8s.pt"
    if not fallback.is_file():
        raise FileNotFoundError(f"No YOLO weights found; expected {fallback}")
    return fallback


def _apply_legacy_efficientnet_h5_compat() -> None:
    """Keras 3 rejects DepthwiseConv2D configs saved with `groups` from older Keras."""
    import tensorflow as tf

    cls = tf.keras.layers.DepthwiseConv2D
    orig = cls.from_config.__func__  # type: ignore[attr-defined]

    def from_config(cls2, config):  # noqa: ARG001
        cfg = dict(config)
        cfg.pop("groups", None)
        return orig(cls2, cfg)

    cls.from_config = classmethod(from_config)  # type: ignore[assignment]


def load_models() -> None:
    global _cnn_model, _yolo_model
    import tensorflow as tf
    from ultralytics import YOLO

    _set_tf_memory_growth()
    if not CNN_PATH.is_file():
        raise FileNotFoundError(f"CNN weights not found: {CNN_PATH}")
    _apply_legacy_efficientnet_h5_compat()
    try:
        cnn = tf.keras.models.load_model(str(CNN_PATH), compile=False, safe_mode=False)
    except TypeError:
        cnn = tf.keras.models.load_model(str(CNN_PATH), compile=False)
    wpath = _find_yolo_weights()
    yolo = YOLO(str(wpath))
    _cnn_model = cnn
    _yolo_model = yolo


def ensure_models() -> None:
    global _load_error
    if _cnn_model is not None and _yolo_model is not None:
        return
    with _load_lock:
        if _cnn_model is not None and _yolo_model is not None:
            return
        try:
            load_models()
            _load_error = None
        except Exception as exc:
            _load_error = str(exc)
            raise


def run_cnn(image_path: str) -> dict[str, Any]:
    import tensorflow as tf

    assert _cnn_model is not None
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = _cnn_model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    return {
        "class_name": CLASS_NAMES[idx],
        "confidence": float(preds[idx]),
        "probabilities": {
            CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))
        },
    }


def run_yolo(image_path: str) -> list[dict[str, Any]]:
    assert _yolo_model is not None
    results = _yolo_model.predict(image_path, conf=YOLO_CONF, verbose=False)
    out: list[dict[str, Any]] = []
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0])
            cls_idx = int(box.cls[0])
            label = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else str(cls_idx)
            out.append(
                {
                    "class_name": label,
                    "confidence": conf,
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                }
            )
    return out


app = FastAPI(title="Plant pathology hybrid API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    ready = _cnn_model is not None and _yolo_model is not None
    body: dict[str, Any] = {"status": "ok", "models_loaded": ready}
    if _load_error:
        body["last_load_error"] = _load_error
    return body


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected an image file")
    data = await file.read()
    if len(data) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 25MB)")
    try:
        Image.open(io.BytesIO(data)).verify()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image") from exc

    try:
        ensure_models()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Models failed to load (use the Python env with TensorFlow + Ultralytics): {exc}",
        ) from exc

    with Image.open(io.BytesIO(data)) as im:
        image_width, image_height = im.size

    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    if suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        suffix = ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        cnn = run_cnn(tmp_path)
        detections = run_yolo(tmp_path)
        return {
            "cnn": cnn,
            "detections": detections,
            "image_width": image_width,
            "image_height": image_height,
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
