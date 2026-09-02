from __future__ import annotations

import streamlit as st


def render_info_card(title: str, body: str, icon: str | None = None) -> None:
    """Render a native bordered information card with an optional Material icon."""
    with st.container(border=True):
        label = f":material/{icon}: {title}" if icon else title
        st.markdown(f"**{label}**")
        st.write(body)


def render_highlight_card(title: str, body: str, icon: str, badge: str | None = None, badge_color: str = "blue") -> None:
    """Render a highlight card with an icon and an optional inline status badge."""
    with st.container(border=True):
        header = f":material/{icon}: {title}"
        if badge:
            header += f" :{badge_color}-badge[{badge}]"
        st.markdown(f"**{header}**")
        st.write(body)


def render_team_card(name: str, role: str, description: str, icon: str = "person") -> None:
    """Render a native team profile card with an avatar icon."""
    with st.container(border=True):
        st.markdown(f":material/{icon}: **{name}**")
        st.caption(role)
        st.write(description)
