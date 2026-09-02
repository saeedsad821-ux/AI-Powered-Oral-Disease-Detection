from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]

OV_FONT = "DejaVu Sans"
OV_BG = "#0B1220"
OV_CARD = "#131E30"
OV_TEXT = "#E2E8F0"
OV_MUTED = "#94A3B8"
OV_GRID = "#24304A"
OV_PALETTE = ["#42A5F5", "#64B5F6", "#90CAF9", "#1E88E5", "#0D47A1", "#1565C0", "#2196F3"]


def _apply_theme(ax) -> None:
    ax.set_facecolor(OV_CARD)
    ax.tick_params(colors=OV_TEXT, labelsize=10)
    ax.xaxis.label.set_color(OV_MUTED)
    ax.yaxis.label.set_color(OV_MUTED)
    ax.title.set_color(OV_TEXT)
    for spine in ax.spines.values():
        spine.set_color(OV_GRID)
    ax.grid(axis="y", color=OV_GRID, linewidth=0.6, alpha=0.6)


def render_dataset_plots() -> None:
    """Render a few dataset insight charts from the generated reports."""
    import streamlit as st

    summary = pd.read_csv(ROOT / "reports" / "class_statistics.csv")
    summary = summary.rename(columns={"total_images": "count"})
    fig = plt.figure(figsize=(8, 4.6), facecolor=OV_BG)
    ax = fig.add_subplot(111)
    bars = sns.barplot(data=summary, x="class", y="count", palette=OV_PALETTE, ax=ax, width=0.65)
    for i, bar in enumerate(bars.patches):
        bar.set_edgecolor("none")
    ax.set_title("Class Distribution", fontweight="bold")
    ax.set_ylabel("Image Count")
    ax.set_xlabel("Class")
    ax.tick_params(axis="x", rotation=20)
    _apply_theme(ax)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)


def render_resolution_chart() -> None:
    """Render a resolution histogram from the EDA outputs."""
    import streamlit as st

    from PIL import Image

    image_path = ROOT / "reports" / "resolution_distribution.png"
    if image_path.exists():
        image = Image.open(image_path)
        st.image(image, caption="Resolution analysis", width="stretch")


def render_quality_chart() -> None:
    """Render the quality report image from the EDA outputs."""
    import streamlit as st

    from PIL import Image

    image_path = ROOT / "reports" / "quality_report.png"
    if image_path.exists():
        image = Image.open(image_path)
        st.image(image, caption="Data quality", width="stretch")
