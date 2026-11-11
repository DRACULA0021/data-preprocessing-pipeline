"""Distribution skewness classification and suggested transformations."""

from __future__ import annotations

import pandas as pd
from scipy.stats import skew


def analyze_skewness(frame: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Report skewness for numeric columns and suggest review-only actions."""
    results: list[dict[str, object]] = []
    for column in frame.select_dtypes(include="number").columns:
        values = frame[column].dropna()
        if len(values) < 3 or values.nunique() < 2:
            value = float("nan")
            label = "insufficient variation"
            recommendation = "No transformation suggested"
        else:
            value = float(skew(values, bias=False))
            if value > threshold:
                label = "positively skewed"
                recommendation = (
                    "Review log1p or Yeo-Johnson; log1p requires non-negative values"
                )
            elif value < -threshold:
                label = "negatively skewed"
                recommendation = "Review Yeo-Johnson or investigate the distribution"
            else:
                label = "approximately symmetric"
                recommendation = "No transformation suggested"
        results.append(
            {
                "column": column,
                "skewness": value,
                "distribution": label,
                "recommendation": recommendation,
            }
        )
    return pd.DataFrame(
        results,
        columns=["column", "skewness", "distribution", "recommendation"],
    )
