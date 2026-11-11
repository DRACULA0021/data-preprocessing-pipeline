"""Downloadable processing summaries and dataset exports."""

from __future__ import annotations

from html import escape
from io import BytesIO
import json
from typing import Any

import pandas as pd


def report_as_dataframe(report: dict[str, Any]) -> pd.DataFrame:
    """Flatten a processing summary into a key/value report table."""
    return pd.DataFrame(
        [
            {"item": key, "value": json.dumps(value, default=str)}
            for key, value in report.items()
        ]
    )


def export_excel(
    frame: pd.DataFrame,
    report: dict[str, Any],
    *,
    stats: pd.DataFrame | None = None,
) -> bytes:
    """Export a dataset and report sheets as an in-memory Excel workbook."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="processed_data", index=False)
        report_as_dataframe(report).to_excel(writer, sheet_name="processing_report", index=False)
        if stats is not None and not stats.empty:
            stats.to_excel(writer, sheet_name="statistics", index=False)
    return output.getvalue()


def export_html(
    frame: pd.DataFrame,
    report: dict[str, Any],
    *,
    stats: pd.DataFrame | None = None,
) -> bytes:
    """Create a self-contained HTML report with a small dataset preview."""
    report_rows = "".join(
        f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>"
        for key, value in report.items()
    )
    stats_html = (
        stats.to_html(index=False, border=0, classes="data")
        if stats is not None and not stats.empty
        else "<p>No numeric statistics available.</p>"
    )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Preprocessing report</title>
<style>body{{font:15px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{border:1px solid #ddd;
padding:.5rem;text-align:left}}th{{background:#f3f5f7}}h1,h2{{color:#20334a}}</style>
</head><body><h1>Data preprocessing report</h1><h2>Processing summary</h2>
<table>{report_rows}</table><h2>Statistics</h2>{stats_html}
<h2>Dataset preview (first 100 rows)</h2>{frame.head(100).to_html(index=False, border=0, classes="data")}
</body></html>"""
    return html.encode("utf-8")
