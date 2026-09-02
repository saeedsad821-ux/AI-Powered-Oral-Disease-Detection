from __future__ import annotations

import streamlit as st


def render_page_header(
    title: str,
    subtitle: str,
    icon: str,
    badges: list[tuple[str, str]] | None = None,
) -> None:
    """Render a consistent page hero with title, subtitle, icon, and inline badges.

    badges is a list of (label, color) pairs rendered as inline colored badges.
    """
    with st.container(key="page_hero"):
        st.markdown(f"### :material/{icon}: {title}", anchors=False)
        st.caption(subtitle)
        if badges:
            badge_text = " ".join(f":{color}-badge[{label}]" for label, color in badges)
            st.markdown(badge_text)
    st.space("small")
