from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.charts import render_bar_chart
from app.components.footer import render_footer
from app.components.header import render_page_header
from app.components.section import render_section_header

render_page_header(
    title="Model comparison",
    subtitle="A clinical benchmark of the four architectures evaluated in this project.",
    icon="query_stats",
)

comparison = [
    {"rank": 1, "model": "EfficientNetB3", "accuracy": 93.02, "params": "Heavy", "suitability": "Best fit for this task"},
    {"rank": 2, "model": "DenseNet121", "accuracy": 92.57, "params": "Heavy", "suitability": "Accuracy-first alternative"},
    {"rank": 3, "model": "MobileNetV2", "accuracy": 89.57, "params": "Light", "suitability": "Edge-friendly deployment"},
    {"rank": 4, "model": "Custom CNN", "accuracy": 83.97, "params": "Light", "suitability": "Lightweight baseline"},
]

render_section_header(
    "Leaderboard",
    icon="leaderboard",
    description="Ranking by top-1 accuracy across the four evaluated architectures.",
)
with st.container(border=True):
    leaderboard = pd.DataFrame(comparison)
    leaderboard["rank"] = leaderboard["rank"].map({1: ":material/military_tech:", 2: ":material/emoji_events:", 3: ":material/workspace_premium:", 4: ":material/tag:"})
    st.dataframe(
        leaderboard[["rank", "model", "accuracy", "params", "suitability"]],
        hide_index=True,
        column_config={
            "accuracy": st.column_config.NumberColumn("Accuracy", format="%.2f%%"),
        },
        width="stretch",
    )

st.space("small")

render_section_header(
    "Accuracy comparison",
    icon="bar_chart",
    description="Top-1 accuracy of each architecture, ordered from lowest to highest.",
)
acc_df = pd.DataFrame(comparison)[["model", "accuracy"]].sort_values("accuracy")
with st.container(border=True):
    render_bar_chart(acc_df, x="model", y="accuracy", format_spec=".2f", height=340)

st.space("small")

render_section_header(
    "Architecture notes",
    icon="note_alt",
    description="Design rationale and recommended use for each model.",
)
notes = [
    ("EfficientNetB3", "High representational power and strong medical-image transfer performance. Slightly heavier than lighter backbones.", "Best fit for the current task"),
    ("DenseNet121", "Feature reuse through dense connections. Higher parameter count and heavier inference.", "Strong where accuracy is prioritized"),
    ("MobileNetV2", "Efficient mobile backbone with strong transfer-learning behavior. Needs careful fine-tuning to avoid overfitting.", "Very suitable for edge-friendly deployment"),
    ("Custom CNN", "Compact baseline. Requires more data and can be less stable.", "Good as a lightweight baseline"),
]
for index, (name, detail, suitability) in enumerate(notes):
    with st.container(border=True):
        st.markdown(f"**{name}**")
        st.write(detail)
        st.markdown(f":blue-badge[{suitability}]")

render_footer()
