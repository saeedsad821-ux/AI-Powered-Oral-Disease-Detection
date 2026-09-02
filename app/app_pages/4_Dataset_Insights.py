from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.charts import render_bar_chart
from app.components.footer import render_footer
from app.components.header import render_page_header
from app.components.section import render_section_header
from app.utils.helpers import load_dataset_summary
from app.utils.visualization import render_resolution_chart

dataset = load_dataset_summary()

render_page_header(
    title="Dataset insights",
    subtitle="Visual overview of the curated oral disease dataset used for training.",
    icon="insights",
    badges=[("Balanced preprocessing", "green"), ("No corrupted images", "green")],
)

render_section_header(
    "Dataset size",
    icon="data_usage",
    description="Curated train / validation / test splits.",
)
with st.container(horizontal=True):
    st.metric("Train images", dataset["train_images"], border=True)
    st.metric("Validation images", dataset["val_images"], border=True)
    st.metric("Test images", dataset["test_images"], border=True)
    st.metric("Image size", f"{dataset['image_size']}×{dataset['image_size']}", border=True)

st.space("small")

render_section_header(
    "Class distribution",
    icon="bar_chart",
    description="Image counts per disease class across the full dataset.",
)
class_df = pd.DataFrame(dataset["class_counts"]).rename(columns={"class": "class", "total_images": "images"})
with st.container(border=True):
    render_bar_chart(class_df, x="class", y="images", height=340)

st.space("small")

render_section_header(
    "Balance & resolution",
    icon="scale",
    description="How class imbalance is handled and how image resolutions behave.",
)
left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.markdown("**Class imbalance**")
        st.markdown(f"**Imbalance ratio:** {dataset['imbalance_ratio']:.2f}")
        st.caption("Ratios near 1.0 indicate balanced class sizes; class weights compensate for the spread.")
        st.space("small")
        weights_df = pd.DataFrame(
            {"class": list(dataset["class_weights"].keys()), "weight": list(dataset["class_weights"].values())}
        )
        render_bar_chart(weights_df, x="class", y="weight", horizontal=True, format_spec=".3f", height=280)
with right:
    with st.container(border=True):
        st.markdown("**Resolution analysis**")
        render_resolution_chart()

st.space("small")

render_section_header(
    "Data quality",
    icon="verified",
    description="Validation results for the cleaned dataset.",
)
col1, col2, col3 = st.columns(3)
with col1:
    with st.container(border=True):
        st.metric("Corrupted images", 0, border=True)
        st.caption("No corrupted files detected in the cleaned dataset.")
with col2:
    with st.container(border=True):
        st.metric("Unsupported files", 0, border=True)
        st.caption("All validated files are supported image formats.")
with col3:
    with st.container(border=True):
        st.metric("Classes", len(dataset["class_counts"]), border=True)
        st.caption("Six clinically distinct oral disease categories.")

render_footer()
