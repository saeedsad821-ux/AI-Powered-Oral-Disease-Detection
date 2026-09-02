from __future__ import annotations

from pathlib import Path
import base64
from io import BytesIO

from PIL import Image
import numpy as np
import streamlit as st


def image_to_bytes(image: Image.Image) -> bytes:
    """Convert a PIL image to bytes for easy display or caching."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def display_image_preview(image: Image.Image) -> None:
    """Render a centered image preview in the Streamlit UI."""
    st.image(image, caption="Uploaded oral image", width="stretch")


def resize_for_display(image: Image.Image, max_width: int = 700) -> Image.Image:
    """Resize the uploaded image for display while preserving aspect ratio."""
    width, height = image.size
    ratio = min(max_width / width, 1.0)
    new_size = (max(int(width * ratio), 1), max(int(height * ratio), 1))
    return image.resize(new_size, Image.Resampling.LANCZOS)
