from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_project_overview() -> dict[str, object]:
    """Load high-level project metadata used across the app."""
    analysis = json.loads((ROOT / "reports" / "analysis_summary.json").read_text(encoding="utf-8"))
    return {
        "summary": "This project builds an explainable AI pipeline for oral disease detection using transfer learning, medical-style preprocessing, and a modern deployed dashboard.",
        "highlights": [
            "Six-class oral disease classification from real-world oral photographs.",
            "Clinical-ready preprocessing pipeline with image resizing and normalization.",
            "Model comparison across Custom CNN, MobileNetV2, EfficientNetB3, and DenseNet121.",
            "Streamlit deployment for interactive inference and educational presentation.",
        ],
        "dataset_images": f"{analysis['total_images']:,}",
        "classes": analysis["classes"],
        "primary_model": "EfficientNetB3",
        "status": "Milestone 5 in progress",
    }


def load_dataset_summary() -> dict[str, object]:
    """Load dataset metrics from the EDA reports."""
    quality = json.loads((ROOT / "reports" / "quality_report.json").read_text(encoding="utf-8"))
    preprocessing = json.loads((ROOT / "reports" / "preprocessing_summary.json").read_text(encoding="utf-8"))
    analysis = json.loads((ROOT / "reports" / "analysis_summary.json").read_text(encoding="utf-8"))
    return {
        "corrupted_images": quality["corrupted_images"],
        "unsupported_files": quality["unsupported_files"],
        "class_counts": analysis["per_class"],
        "image_size": preprocessing["image_size"],
        "batch_size": preprocessing["batch_size"],
        "train_images": preprocessing["train_images"],
        "val_images": preprocessing["val_images"],
        "test_images": preprocessing["test_images"],
        "imbalance_ratio": analysis.get("imbalance_ratio", 0),
        "class_weights": preprocessing["class_weights"],
    }
