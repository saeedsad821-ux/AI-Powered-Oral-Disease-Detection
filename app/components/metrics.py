from __future__ import annotations

import streamlit as st


def render_metric_row(values: list[tuple[str, str | int]]) -> None:
    """Render a responsive row of native metric tiles with icons and borders."""
    with st.container(horizontal=True):
        for label, value in values:
            st.metric(label, value, border=True)


def render_kpi(label: str, value: str | int, delta: str | None = None, icon: str | None = None) -> None:
    """Render a single KPI metric tile, optionally with a delta and an icon."""
    rendered_label = f":material/{icon}: {label}" if icon else label
    st.metric(rendered_label, value, delta=delta, border=True)
