"""Selectable, non-destructive data cleaning operations."""

from __future__ import annotations

from collections.abc import Sequence
import re

import pandas as pd
from pandas.api.types import is_numeric_dtype

from modules.file_handler import make_unique_columns


MISSING_STRATEGIES = (
    "none",
    "auto",
    "mean",
    "median",
    "mode",
    "forward fill",
    "backward fill",
    "drop rows",
    "drop columns",
)


def standardize_name(column: object) -> str:
    """Normalize a column name to lowercase snake_case."""
    name = re.sub(r"\s+", "_", str(column).strip().lower())
    name = re.sub(r"[^\w]+", "_", name, flags=re.UNICODE).strip("_")
    return name or "column"


def _fill_missing(
    frame: pd.DataFrame, strategy: str, columns: Sequence[str]
) -> pd.DataFrame:
    if strategy in {"none", "auto"}:
        for column in columns:
            if strategy == "auto":
                if is_numeric_dtype(frame[column].dtype):
                    frame[column] = frame[column].fillna(frame[column].median())
                else:
                    mode = frame[column].mode(dropna=True)
                    if not mode.empty:
                        frame[column] = frame[column].fillna(mode.iloc[0])
        return frame

    selected = [column for column in columns if column in frame.columns]
    if not selected:
        return frame
    if strategy == "mean":
        for column in selected:
            if not frame[column].isna().any():
                continue
            if not is_numeric_dtype(frame[column].dtype):
                raise ValueError(f"Mean imputation requires a numeric column: {column}")
            frame[column] = frame[column].fillna(frame[column].mean())
    elif strategy == "median":
        for column in selected:
            if not frame[column].isna().any():
                continue
            if not is_numeric_dtype(frame[column].dtype):
                raise ValueError(f"Median imputation requires a numeric column: {column}")
            frame[column] = frame[column].fillna(frame[column].median())
    elif strategy == "mode":
        for column in selected:
            mode = frame[column].mode(dropna=True)
            if not mode.empty:
                frame[column] = frame[column].fillna(mode.iloc[0])
    elif strategy == "forward fill":
        frame[selected] = frame[selected].ffill()
    elif strategy == "backward fill":
        frame[selected] = frame[selected].bfill()
    elif strategy == "drop rows":
        frame = frame.dropna(subset=selected)
    elif strategy == "drop columns":
        columns_with_missing = [
            column for column in selected if frame[column].isna().any()
        ]
        frame = frame.drop(columns=columns_with_missing)
    else:
        raise ValueError(f"Unknown missing-value strategy: {strategy}")
    return frame


def clean_dataframe(
    source: pd.DataFrame,
    *,
    missing_strategy: str = "none",
    missing_columns: Sequence[str] | None = None,
    numeric_columns: Sequence[str] | None = None,
    columns_to_remove: Sequence[str] | None = None,
    remove_duplicates: bool = True,
    normalize_names: bool = True,
    strip_whitespace: bool = True,
    normalize_categories: bool = False,
    remove_constant_columns: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return a cleaned copy and an auditable summary; never mutate source."""
    if missing_strategy not in MISSING_STRATEGIES:
        raise ValueError(f"Unsupported missing-value strategy: {missing_strategy}")
    frame = source.copy(deep=True)
    before_shape = frame.shape
    missing_before = int(frame.isna().sum().sum())
    original_columns = list(frame.columns)
    modifications: list[str] = []

    if normalize_names:
        normalized = make_unique_columns(standardize_name(name) for name in frame.columns)
        renamed_columns = dict(zip(original_columns, normalized))
        if normalized != list(frame.columns):
            modifications.append("standardized column names")
        frame.columns = normalized
    else:
        renamed_columns = {column: column for column in original_columns}
    if strip_whitespace:
        for column in frame.select_dtypes(include=["object", "string"]).columns:
            stripped = frame[column].map(
                lambda value: value.strip() if isinstance(value, str) else value
            )
            if not stripped.equals(frame[column]):
                modifications.append(f"trimmed whitespace: {column}")
            frame[column] = stripped

    converted_columns: dict[str, int] = {}
    for selected_column in numeric_columns or []:
        column = renamed_columns.get(selected_column, selected_column)
        if column not in frame.columns:
            continue
        was_missing = frame[column].isna()
        converted = pd.to_numeric(frame[column], errors="coerce")
        converted_columns[column] = int((converted.isna() & ~was_missing).sum())
        frame[column] = converted
        modifications.append(
            f"converted to numeric: {column} "
            f"(invalid values coerced to missing: {converted_columns[column]})"
        )

    if normalize_categories:
        for column in frame.select_dtypes(include=["object", "string"]).columns:
            normalized = frame[column].map(
                lambda value: value.casefold() if isinstance(value, str) else value
            )
            if not normalized.equals(frame[column]):
                modifications.append(f"normalized category casing: {column}")
            frame[column] = normalized

    duplicates_removed = int(frame.duplicated().sum()) if remove_duplicates else 0
    if remove_duplicates:
        frame = frame.drop_duplicates()
        if duplicates_removed:
            modifications.append("removed duplicate rows")

    chosen_missing_columns = (
        [renamed_columns.get(column, column) for column in missing_columns]
        if missing_columns is not None
        else list(frame.columns)
    )
    chosen_missing_columns = [
        column for column in chosen_missing_columns if column in frame.columns
    ]
    removed_columns = [
        renamed_columns.get(column, column)
        for column in (columns_to_remove or [])
        if renamed_columns.get(column, column) in frame.columns
    ]
    if removed_columns:
        frame = frame.drop(columns=removed_columns)
        modifications.append("removed selected columns")
    chosen_missing_columns = [
        column for column in chosen_missing_columns if column in frame.columns
    ]
    if missing_strategy != "none":
        frame = _fill_missing(frame, missing_strategy, chosen_missing_columns)
        modifications.append(f"missing values: {missing_strategy}")

    removed_constants: list[str] = []
    if remove_constant_columns:
        removed_constants = [
            column for column in frame.columns if frame[column].nunique(dropna=False) <= 1
        ]
        frame = frame.drop(columns=removed_constants)
        if removed_constants:
            modifications.append("removed constant columns")

    return frame, {
        "original_shape": before_shape,
        "final_shape": frame.shape,
        "duplicates_removed": duplicates_removed,
        "missing_before": missing_before,
        "missing_after": int(frame.isna().sum().sum()),
        "modifications": modifications,
        "numeric_columns_converted": converted_columns,
        "columns_removed": removed_columns,
        "constant_columns_removed": removed_constants,
    }
