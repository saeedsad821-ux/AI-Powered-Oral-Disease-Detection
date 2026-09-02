from __future__ import annotations

from pathlib import Path
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
IMG_SIZE = 224


def load_image_to_array(path_or_image: str | Path | Image.Image) -> np.ndarray:
    """Load an image from disk or accept a PIL image and return a numpy array."""
    if isinstance(path_or_image, Image.Image):
        image = path_or_image.convert("RGB")
        return np.array(image)

    image = Image.open(path_or_image).convert("RGB")
    return np.array(image)


def preprocess_for_model(image_array: np.ndarray, model_name: str) -> tf.Tensor:
    """Apply the same preprocessing contract used by the training pipeline.

    The project M2 pipeline keeps images in [0,255] float32 and lets the model
    apply its own preprocessing internally. This helper mirrors that contract
    while still supporting the backbone-specific normalization in inference.
    """
    # Import TensorFlow lazily so the UI can load without TensorFlow installed
    import tensorflow as tf

    image = tf.convert_to_tensor(image_array, dtype=tf.float32)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32)

    if model_name.lower().startswith("mobilenet"):
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    elif model_name.lower().startswith("efficientnet"):
        image = tf.keras.applications.efficientnet.preprocess_input(image)
    elif model_name.lower().startswith("densenet"):
        image = tf.keras.applications.densenet.preprocess_input(image)
    elif model_name.lower().startswith("custom"):
        image = tf.cast(image / 255.0, tf.float32)
    else:
        image = tf.cast(image / 255.0, tf.float32)

    return tf.expand_dims(image, axis=0)


def preprocess_and_time(image_array: np.ndarray, model_name: str) -> tuple[tf.Tensor, float]:
    """Measure preprocessing time for the dashboard."""
    start = time.perf_counter()
    tensor = preprocess_for_model(image_array, model_name)
    elapsed = time.perf_counter() - start
    return tensor, elapsed
