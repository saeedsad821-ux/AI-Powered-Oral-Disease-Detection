from __future__ import annotations

from pathlib import Path
import json
import os
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"
CLASS_MAPPING = json.loads((ARTIFACTS_DIR / "class_mapping.json").read_text(encoding="utf-8"))


MODEL_CATALOG = [
    {"id": "custom_cnn", "name": "Custom CNN", "path": None, "available": False},
    {"id": "mobilenetv2", "name": "MobileNetV2", "path": None, "available": False},
    {"id": "efficientnetb3", "name": "EfficientNetB3", "path": None, "available": False},
    {"id": "densenet121", "name": "DenseNet121", "path": None, "available": False},
]


def list_available_models() -> list[dict[str, object]]:
    """Scan the models/ directory and expose the discovered artifacts."""
    discovered: list[dict[str, object]] = []
    model_files = sorted(list(MODELS_DIR.glob("*.keras")) + list(MODELS_DIR.glob("*.h5")))

    if model_files:
        for model_file in model_files:
            stem = model_file.stem.lower()
            if "custom" in stem or "cnn" in stem:
                model_id = "custom_cnn"
                model_name = "Custom CNN"
            elif "mobilenet" in stem:
                model_id = "mobilenetv2"
                model_name = "MobileNetV2"
            elif "efficientnet" in stem or "stage1" in stem or "effnet" in stem:
                model_id = "efficientnetb3"
                model_name = "EfficientNetB3"
            elif "densenet" in stem or "dense" in stem:
                model_id = "densenet121"
                model_name = "DenseNet121"
            else:
                model_id = "efficientnetb3"
                model_name = "EfficientNetB3"

            discovered.append({"id": model_id, "name": model_name, "path": str(model_file), "available": True})

    if not discovered:
        return [{**entry, "path": None, "available": False} for entry in MODEL_CATALOG]

    return discovered


def build_model_from_catalog(model_id: str) -> tuple[tf.keras.Model, str, bool]:
    """Load the selected model from disk when available or initialize a lightweight preview model otherwise."""
    # Import TensorFlow lazily to avoid requiring it for UI-only tasks
    import tensorflow as tf

    models = list_available_models()
    metadata = next((item for item in models if item["id"] == model_id), None)
    if metadata is None:
        metadata = models[0] if models else None
    if metadata is None:
        raise ValueError(f"Unknown model id: {model_id}")

    if metadata["available"]:
        path = metadata["path"]
        model = tf.keras.models.load_model(path, compile=False)
        return model, str(path), True

    # Fallback preview model for UI without a local artifact.
    base = tf.keras.applications.EfficientNetB0(include_top=False, weights=None, input_shape=(224, 224, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_MAPPING["classes"]), activation="softmax")(x)
    model = tf.keras.Model(inputs=base.input, outputs=outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model, "preview", False


def predict_with_model(model: tf.keras.Model, image_tensor: tf.Tensor, model_name: str) -> dict[str, object]:
    """Predict disease class and confidence from a preprocessed image tensor."""
    start = time.perf_counter()
    logits = model(image_tensor, training=False)
    probs = np.asarray(logits[0]).astype(float)
    top_indices = np.argsort(probs)[::-1][:3]
    top_labels = [CLASS_MAPPING["classes"][i] for i in top_indices]
    top_scores = [float(probs[i]) for i in top_indices]
    predicted_index = int(np.argmax(probs))
    predicted_label = CLASS_MAPPING["classes"][predicted_index]
    confidence = float(probs[predicted_index])
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    return {
        "predicted_label": predicted_label,
        "confidence": confidence,
        "top_predictions": list(zip(top_labels, top_scores)),
        "probabilities": [float(p) for p in probs],
        "model_name": model_name,
        "inference_ms": elapsed_ms,
        "class_names": CLASS_MAPPING["classes"],
    }
