"""Turn a chart spec produced by the SQL agent into a Plotly figure.

Keeping this separate from SQL generation means the model never writes plotting
code directly - it only picks from a fixed menu of chart types and field names,
which we validate against the actual result columns before rendering.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px

SUPPORTED_TYPES = {"bar", "line", "pie", "scatter", "histogram"}


def render_chart(df: pd.DataFrame, chart_spec: dict | None):
    if not chart_spec or df is None or df.empty:
        return None

    chart_type = str(chart_spec.get("type") or "").lower()
    if chart_type not in SUPPORTED_TYPES:
        return None

    x = chart_spec.get("x")
    y = chart_spec.get("y")
    color = chart_spec.get("group_by")
    title = chart_spec.get("title") or ""

    if x not in df.columns:
        return None
    if color not in df.columns:
        color = None
    if y is not None and y not in df.columns:
        y = None

    try:
        if chart_type == "bar":
            return px.bar(df, x=x, y=y, color=color, title=title)
        if chart_type == "line":
            return px.line(df, x=x, y=y, color=color, title=title)
        if chart_type == "pie":
            if y is None:
                return None
            return px.pie(df, names=x, values=y, title=title)
        if chart_type == "scatter":
            return px.scatter(df, x=x, y=y, color=color, title=title)
        if chart_type == "histogram":
            return px.histogram(df, x=x, color=color, title=title)
    except Exception:
        return None
    return None
