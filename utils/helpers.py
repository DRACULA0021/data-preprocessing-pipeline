"""Small helpers shared by the dashboard and report generation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd


def history_entry(action: str, details: Any) -> dict[str, str]:
    """Create a timestamped, display-ready processing history entry."""
    return {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "action": action,
        "details": str(details),
    }


def dataframe_for_display(frame: pd.DataFrame) -> pd.DataFrame:
    """Make mixed Python-object columns Arrow-safe without changing pipeline data."""
    display_frame = frame.copy()
    for column in display_frame.select_dtypes(include=["object"]).columns:
        display_frame[column] = display_frame[column].astype("string")
    return display_frame
