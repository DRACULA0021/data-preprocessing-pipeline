"""Explicit concatenation and keyed-merge operations."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from modules.file_handler import make_unique_columns


def combine_dataframes(
    frames: dict[str, pd.DataFrame],
    *,
    method: str,
    join_type: str = "outer",
    keys: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Combine frames only using the integration method and keys chosen by the user."""
    if not frames:
        raise ValueError("There are no datasets to combine.")
    names = list(frames)
    if len(names) == 1:
        result = frames[names[0]].copy()
        result.columns = make_unique_columns(result.columns)
        return result, {
            "method": "single file",
            "files": names,
            "rows": len(result),
            "columns": len(result.columns),
        }

    if method == "concatenate":
        result = pd.concat(
            [frames[name] for name in names],
            ignore_index=True,
            sort=False,
        )
        report: dict[str, object] = {
            "method": "concatenate rows",
            "files": names,
            "input_rows": {name: len(frames[name]) for name in names},
            "columns_per_file": {
                name: list(frames[name].columns) for name in names
            },
            "rows": len(result),
            "columns": len(result.columns),
        }
        return result, report

    if method != "merge":
        raise ValueError("Method must be 'concatenate' or 'merge'.")
    if join_type not in {"inner", "left", "right", "outer"}:
        raise ValueError("Join type must be inner, left, right, or outer.")
    selected_keys = list(keys or [])
    if not selected_keys:
        raise ValueError("Select at least one key column for a merge.")
    for name, frame in frames.items():
        missing_keys = [key for key in selected_keys if key not in frame.columns]
        if missing_keys:
            raise ValueError(f"{name!r} is missing merge columns: {missing_keys}")

    result = frames[names[0]].copy()
    for name in names[1:]:
        right = frames[name].copy()
        collisions = [
            column
            for column in right.columns
            if column in result.columns and column not in selected_keys
        ]
        if collisions:
            right = right.rename(columns={column: f"{name}_{column}" for column in collisions})
        result = result.merge(
            right,
            how=join_type,
            on=selected_keys,
            suffixes=("", f"_{name}"),
        )
    return result, {
        "method": f"{join_type} merge",
        "files": names,
        "keys": selected_keys,
        "rows": len(result),
        "columns": len(result.columns),
    }
