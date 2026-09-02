from __future__ import annotations

import streamlit as st

from app.components.cards import render_info_card
from app.components.footer import render_footer
from app.components.header import render_page_header
from app.components.section import render_section_header

render_page_header(
    title="About the project",
    subtitle="A clear explanation of the methodology behind the AI-powered oral disease detection system.",
    icon="info",
)

render_section_header(
    "Pipeline",
    icon="timeline",
    description="From raw oral images to a deployed Streamlit dashboard.",
)
with st.container(border=True):
    st.mermaid_chart("""
flowchart LR
    A[Oral images] --> B[Preprocessing]
    B --> C[Augmentation]
    C --> D[Transfer learning]
    D --> E[Evaluation]
    E --> F[Streamlit app]
""")

st.space("small")

render_section_header(
    "Methodology",
    icon="psychology",
    description="The design principles behind the system.",
)
col1, col2 = st.columns(2)
with col1:
    render_info_card("Objective", "Design and deploy an explainable deep-learning system that classifies six oral disease categories from images while remaining suitable for educational and presentation purposes.", icon="flag")
    render_info_card("Methodology", "Transfer learning is used to adapt established CNN backbones to oral disease images, while the app reuses the project's trained workflow for inference.", icon="psychology")
with col2:
    render_info_card("Workflow", "The project follows an end-to-end pipeline: EDA, preprocessing, model training, evaluation, and now deployment in Streamlit.", icon="account_tree")
    render_info_card("AI pipeline", "Images are resized, converted to tensors, normalized according to the model contract, and passed through the selected model to generate predictions and confidence scores.", icon="memory")

st.space("small")

render_section_header(
    "Future work",
    icon="rocket_launch",
    description="Planned extensions beyond the current milestone.",
)
render_info_card("Future work", "Future extensions include Grad-CAM explainability, model retraining on larger datasets, and deployment in a cloud environment.", icon="rocket_launch")

render_footer()
