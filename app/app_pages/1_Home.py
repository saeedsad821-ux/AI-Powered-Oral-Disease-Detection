from __future__ import annotations

import streamlit as st

from app.components.cards import render_highlight_card
from app.components.footer import render_footer
from app.components.header import render_page_header
from app.components.section import render_section_header
from app.utils.helpers import load_dataset_summary, load_project_overview

overview = load_project_overview()
dataset = load_dataset_summary()

render_page_header(
    title="AI-powered oral disease detection",
    subtitle="A modern clinical dashboard for medical imaging, inference, and dataset insights.",
    icon="health_and_safety",
    badges=[("6 disease classes", "blue"), ("12,320 images", "green"), ("Milestone 5", "violet")],
)

with st.container(horizontal=True):
    st.metric("Dataset images", overview["dataset_images"], border=True)
    st.metric("Classes", overview["classes"], border=True)
    st.metric("Primary baseline", overview["primary_model"], border=True)
    st.metric("Status", overview["status"], border=True)

st.space("medium")

render_section_header(
    "Mission & highlights",
    icon="flag",
    description="What the system is built to deliver.",
)
col1, col2 = st.columns([1.5, 1], vertical_alignment="center")
with col1:
    with st.container(border=True):
        st.markdown(overview["summary"])
        st.space("small")
        for bullet in overview["highlights"]:
            st.markdown(f":material/check_circle: {bullet}")
with col2:
    with st.container(border=True):
        st.markdown("**Quick actions**")
        st.caption("Jump straight into a workflow.")
        st.page_link("app_pages/2_Disease_Detection.py", label="Run disease detection", icon=":material/biotech:")
        st.page_link("app_pages/3_Model_Comparison.py", label="Compare models", icon=":material/query_stats:")
        st.page_link("app_pages/4_Dataset_Insights.py", label="Explore dataset", icon=":material/insights:")

st.space("small")

render_section_header(
    "Project at a glance",
    icon="dashboard_customize",
    description="A quick overview of scope, stack, and architecture.",
)
col1, col2 = st.columns(2)
with col1:
    render_highlight_card("Objectives", "Create an explainable, clinically inspired AI workflow for oral disease classification that can be demonstrated in a polished university presentation.", icon="flag")
    render_highlight_card("Workflow", "EDA → Preprocessing → Model training → Evaluation → Streamlit deployment.", icon="account_tree")
with col2:
    render_highlight_card("Technologies", "TensorFlow/Keras, Streamlit, Pillow, NumPy, Pandas, Matplotlib, Seaborn.", icon="code")
    render_highlight_card("Architecture", "The app reuses the project's preprocessing and inference logic, then exposes it through a clean dashboard.", icon="hub")

st.space("small")

render_section_header(
    "Dataset snapshot",
    icon="dataset",
    description="Curated training, validation, and test splits.",
)
with st.container(horizontal=True):
    st.metric("Train images", dataset["train_images"], border=True)
    st.metric("Validation images", dataset["val_images"], border=True)
    st.metric("Test images", dataset["test_images"], border=True)
    st.metric("Image size", f"{dataset['image_size']}×{dataset['image_size']}", border=True)

render_footer()
