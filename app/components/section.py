from __future__ import annotations

import re

import streamlit as st


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def render_section_header(title: str, icon: str | None = None, description: str | None = None, key_suffix: str | None = None) -> None:
    """Render a structured section header with icon, accent bar, and optional description."""
    key = f"ov_sec_{_slugify(key_suffix or title)}"
    with st.container(key=key):
        label = f":material/{icon}: {title}" if icon else title
        st.markdown(f"### {label}", anchors=False)
        if description:
            st.caption(description)
