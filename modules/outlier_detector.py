"""Review-oriented IQR and Z-score outlier summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_outliers(frame: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
    """Count potential IQR outliers and extreme absolute Z-scores per numeric column."""
    if z_threshold <= 0:
        raise ValueError("Z-score threshold must be greater than zero.")
    records: list[dict[str, object]] = []
    for column in frame.select_dtypes(include="number").columns:
        values = frame[column].dropna().astype(float)
        if values.empty:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            iqr_outliers = int(((values < q1) | (values > q3)).sum())
        else:
            iqr_outliers = int(
                ((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum()
            )
        std = values.std(ddof=0)
        z_count = (
            int((np.abs((values - values.mean()) / std) > z_threshold).sum())
            if std > 0
            else 0
        )
        records.append(
            {
                "column": column,
                "iqr_outliers": iqr_outliers,
                "z_score_extremes": z_count,
                "z_threshold": z_threshold,
            }
        )
    return pd.DataFrame(
        records,
        columns=["column", "iqr_outliers", "z_score_extremes", "z_threshold"],
    )
