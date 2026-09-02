from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = ROOT / "app"
for path in (str(ROOT), str(APP_ROOT)):
    if path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, str(ROOT))
sys.path.append(str(APP_ROOT))

try:
    from app.utils.prediction import list_available_models
except ModuleNotFoundError:
    from utils.prediction import list_available_models


def render_sidebar() -> None:
    """Render the app sidebar: brand, navigation, model selector, and status footer."""
    logo_path = APP_ROOT / "assets" / "logo.png"
    if logo_path.exists() and logo_path.stat().st_size > 0:
        try:
            st.logo(str(logo_path), size="medium")
        except Exception:
            pass

    with st.sidebar:
        st.markdown("### :material/medical_services: Oral Vision")
        st.caption("AI-powered oral disease detection")

        st.space("small")

        with st.container(border=True):
            st.markdown("**:material/neurology: Active model**")
            models = list_available_models()
            if not models:
                st.warning("No local model artifacts were detected.")
                return

            if "selected_model_id" not in st.session_state:
                st.session_state["selected_model_id"] = models[0]["id"]

            selected = st.pills(
                "Model",
                options=[model["id"] for model in models],
                format_func=lambda model_id: next(item["name"] for item in models if item["id"] == model_id),
                key="selected_model_id",
                label_visibility="collapsed",
            )

            selected_metadata = next(item for item in models if item["id"] == selected)
            status_text, status_icon = (
                ("Artifact loaded", ":material/verified:") if selected_metadata["available"] else ("Preview model", ":material/science:")
            )
            st.markdown(f"{status_icon} {status_text}")
            st.caption(f"Source: {selected_metadata['path'] or 'pretrained initialization'}")

        st.space("medium")

        st.markdown("**:material/monitor_heart: Project status**")
        st.markdown(":green-badge[Milestone 5] :gray-badge[In progress]")
        st.caption("Six-class oral disease classification · 12,320 images")
