"""Read supported tabular formats without writing or changing uploaded files."""

from __future__ import annotations

from collections.abc import Iterable
from io import BytesIO
from pathlib import Path

import pandas as pd


def make_unique_columns(columns: Iterable[object]) -> list[str]:
    """Return trimmed, non-empty, unique string column names."""
    used: set[str] = set()
    result: list[str] = []
    for position, raw_name in enumerate(columns, start=1):
        base = str(raw_name).strip() or f"column_{position}"
        name = base
        suffix = 1
        while name in used:
            name = f"{base}_{suffix}"
            suffix += 1
        used.add(name)
        result.append(name)
    return result


def load_data_file(filename: str, content: bytes) -> pd.DataFrame:
    """Load one CSV, Excel, or JSON file from bytes."""
    extension = Path(filename).suffix.lower()
    buffer = BytesIO(content)
    try:
        if extension == ".csv":
            frame = pd.read_csv(buffer, sep=None, engine="python")
        elif extension in {".xlsx", ".xls"}:
            frame = pd.read_excel(buffer)
        elif extension == ".json":
            frame = pd.read_json(buffer)
        else:
            raise ValueError(f"Unsupported file type: {extension or '(no extension)'}")
    except (ImportError, ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read {filename!r}: {exc}") from exc

    frame.columns = make_unique_columns(frame.columns)
    return frame


def load_uploaded_files(
    files: Iterable[tuple[str, bytes]],
) -> dict[str, pd.DataFrame]:
    """Load uploaded (filename, bytes) pairs, reporting file-specific errors."""
    loaded: dict[str, pd.DataFrame] = {}
    for filename, content in files:
        if filename in loaded:
            raise ValueError(f"Two uploaded files have the same name: {filename!r}")
        loaded[filename] = load_data_file(filename, content)
    if not loaded:
        raise ValueError("Upload at least one supported data file.")
    return loaded
