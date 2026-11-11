"""Dataset shape, type, and quality summaries."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)


def inspect_dataframe(frame: pd.DataFrame) -> dict[str, object]:
    """Summarize dimensions, types, nulls, and duplicate rows."""
    dtype_names = frame.dtypes.astype(str)
    constant_columns: list[str] = []
    near_constant_columns: list[str] = []
    for column in frame.columns:
        observed = frame[column].dropna()
        if observed.nunique() <= 1:
            constant_columns.append(column)
        elif not observed.empty and observed.value_counts(normalize=True).iloc[0] >= 0.98:
            near_constant_columns.append(column)
    return {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "shape": tuple(frame.shape),
        "dtypes": dtype_names.to_dict(),
        "missing": frame.isna().sum().astype(int).to_dict(),
        "missing_total": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "numeric_columns": [
            column
            for column in frame.columns
            if is_numeric_dtype(frame[column].dtype)
            and not is_bool_dtype(frame[column].dtype)
        ],
        "categorical_columns": [
            column
            for column in frame.columns
            if not (
                is_numeric_dtype(frame[column].dtype)
                or is_bool_dtype(frame[column].dtype)
                or is_datetime64_any_dtype(frame[column].dtype)
            )
        ],
        "boolean_columns": [
            column for column in frame.columns if is_bool_dtype(frame[column].dtype)
        ],
        "datetime_columns": [
            column
            for column in frame.columns
            if is_datetime64_any_dtype(frame[column].dtype)
        ],
        "constant_columns": constant_columns,
        "near_constant_columns": near_constant_columns,
    }


def common_columns(frames: dict[str, pd.DataFrame]) -> list[str]:
    """Return column names present in every input frame."""
    iter_frames = iter(frames.values())
    first = next(iter_frames, None)
    if first is None:
        return []
    shared = set(first.columns)
    for frame in iter_frames:
        shared.intersection_update(frame.columns)
    return [column for column in first.columns if column in shared]
