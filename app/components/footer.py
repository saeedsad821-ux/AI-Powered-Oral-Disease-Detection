from __future__ import annotations

import streamlit as st


def render_footer() -> None:
    """Render the bottom footer for the app."""
    st.space("medium")
    with st.container(border=True, horizontal=True, horizontal_alignment="distribute"):
        st.markdown(":material/medical_services: **Oral Vision**")
        st.markdown(":material/dataset: 12,320 images · 6 classes")
        st.markdown(":material/school: AI Tools course · Milestone 5")
    st.space("small")
    st.caption("Built for the AI Tools course project · Medical AI dashboard")
