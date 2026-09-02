from __future__ import annotations

import streamlit as st

OV_FONT = "'Inter', 'Segoe UI', sans-serif"
OV_PALETTE = ["#42A5F5", "#64B5F6", "#90CAF9", "#1E88E5", "#0D47A1", "#1565C0", "#2196F3"]

_MUTED = "#94A3B8"
_GRID = "#24304A"


def _axis_config() -> dict:
    return {
        "font": OV_FONT,
        "axis": {
            "labelFont": OV_FONT,
            "titleFont": OV_FONT,
            "labelColor": _MUTED,
            "titleColor": _MUTED,
            "gridColor": _GRID,
            "domainColor": _GRID,
            "tickColor": _GRID,
            "labelFontSize": 12,
            "titleFontSize": 13,
        },
        "view": {"stroke": "transparent"},
        "background": "transparent",
    }


def render_bar_chart(
    data,
    x: str,
    y: str,
    horizontal: bool = False,
    height: int = 360,
    use_palette: bool = True,
    format_spec: str | None = None,
) -> None:
    """Render a consistent, theme-styled bar chart (Vega-Lite)."""
    if horizontal:
        x_enc = {"field": y, "type": "quantitative", "title": y}
        y_enc = {"field": x, "type": "nominal", "title": x, "sort": None}
        mark = {"type": "bar", "cornerRadiusEnd": 6}
    else:
        x_enc = {"field": x, "type": "nominal", "title": x, "sort": None, "axis": {"labelAngle": -20}}
        y_enc = {"field": y, "type": "quantitative", "title": y}
        mark = {"type": "bar", "cornerRadiusEnd": 6}

    color_enc = (
        {"field": x, "type": "nominal", "scale": {"range": OV_PALETTE}, "legend": None}
        if use_palette
        else {"value": OV_PALETTE[0]}
    )

    tooltip = [
        {"field": x, "type": "nominal", "title": x},
        {"field": y, "type": "quantitative", "title": y, "format": format_spec or ",.2f"},
    ]

    spec = {
        "mark": mark,
        "width": "container",
        "height": height,
        "encoding": {"x": x_enc, "y": y_enc, "color": color_enc, "tooltip": tooltip},
        "config": _axis_config(),
    }

    st.vega_lite_chart(data, spec, use_container_width=True)
