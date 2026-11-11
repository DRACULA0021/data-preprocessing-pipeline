"""Reusable statistical summaries and Plotly chart builders."""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from modules.data_inspector import inspect_dataframe


def statistical_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Describe numeric columns and report modes for numeric and categorical data."""
    numeric = frame.select_dtypes(include="number")
    if numeric.empty:
        summary = pd.DataFrame(columns=["column"])
    else:
        summary = numeric.describe().T
        summary["median"] = numeric.median()
        summary["variance"] = numeric.var()
        summary["skewness"] = numeric.skew()
        summary["kurtosis"] = numeric.kurtosis()
        summary = summary.reset_index(names="column")
    numeric_modes = numeric.mode(dropna=True)
    if not summary.empty and not numeric_modes.empty:
        numeric_mode_values = numeric_modes.iloc[0].to_dict()
        summary["mode"] = summary["column"].map(numeric_mode_values)

    categorical = frame.select_dtypes(exclude="number")
    category_rows: list[dict[str, object]] = []
    for column in categorical.columns:
        values = frame[column].dropna().mode()
        if not values.empty:
            category_rows.append(
                {
                    "column": column,
                    "mode": values.iloc[0],
                    "mode_count": int((frame[column] == values.iloc[0]).sum()),
                }
            )
    if category_rows:
        return pd.concat(
            [summary, pd.DataFrame(category_rows)],
            ignore_index=True,
            sort=False,
        )
    return summary


def categorical_frequencies(
    frame: pd.DataFrame, column: str, limit: int = 30
) -> pd.DataFrame:
    """Return the most frequent categories, including missing values."""
    if column not in frame.columns:
        raise ValueError(f"Unknown column: {column}")
    return (
        frame[column]
        .astype("string")
        .fillna("(missing)")
        .value_counts(dropna=False)
        .head(limit)
        .rename_axis("category")
        .reset_index(name="count")
    )


def missing_values_figure(frame: pd.DataFrame) -> go.Figure:
    """Plot missingness per column; avoids a costly row-by-column heatmap."""
    missing = frame.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        figure = go.Figure()
        figure.add_annotation(text="No missing values", showarrow=False)
        return figure
    return px.bar(
        x=missing.index,
        y=missing.values,
        labels={"x": "Column", "y": "Missing values"},
        title="Missing values by column",
    )


def missing_values_heatmap(frame: pd.DataFrame) -> go.Figure:
    """Show a bounded sample of row/column missingness to keep rendering practical."""
    columns = list(frame.columns[:30])
    if not columns or not frame.isna().any().any():
        figure = go.Figure()
        figure.add_annotation(text="No missing values", showarrow=False)
        return figure
    sample = frame[columns]
    if len(sample) > 500:
        sample = sample.sample(500, random_state=42)
    mask = sample.isna().astype(int)
    return px.imshow(
        mask,
        aspect="auto",
        color_continuous_scale=["#f1f5f9", "#ef4444"],
        labels={"x": "Column", "y": "Sample row", "color": "Missing"},
        title="Missing values heatmap (up to 500 rows and 30 columns)",
    )


def correlation_heatmap_matplotlib(frame: pd.DataFrame) -> plt.Figure | None:
    """Create a Seaborn correlation heatmap for up to 20 numeric columns."""
    numeric = frame.select_dtypes(include="number").iloc[:, :20]
    if numeric.shape[1] < 2:
        return None
    figure, axis = plt.subplots(figsize=(max(6, numeric.shape[1] * 0.55), 5))
    sns.heatmap(
        numeric.corr(),
        ax=axis,
        cmap="vlag",
        center=0,
        annot=numeric.shape[1] <= 10,
        fmt=".2f",
    )
    axis.set_title("Correlation heatmap")
    figure.tight_layout()
    return figure


def distribution_figure(frame: pd.DataFrame, column: str) -> go.Figure:
    if column not in frame.columns:
        raise ValueError(f"Unknown column: {column}")
    if pd.api.types.is_numeric_dtype(frame[column].dtype):
        return px.histogram(frame, x=column, marginal="box", title=f"Distribution: {column}")
    counts = categorical_frequencies(frame, column)
    return px.bar(counts, x="category", y="count", title=f"Top categories: {column}")


def correlation_figure(frame: pd.DataFrame) -> go.Figure:
    numeric = frame.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        figure = go.Figure()
        figure.add_annotation(text="At least two numeric columns are needed", showarrow=False)
        return figure
    correlation = numeric.corr()
    return px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Numeric feature correlations",
    )


def pair_figure(frame: pd.DataFrame, columns: list[str]) -> go.Figure:
    selected = [
        column
        for column in columns
        if column in frame.select_dtypes(include="number").columns
    ][:5]
    if len(selected) < 2:
        figure = go.Figure()
        figure.add_annotation(text="Select at least two numeric features", showarrow=False)
        return figure
    sample = frame[selected].dropna()
    if len(sample) > 500:
        sample = sample.sample(500, random_state=42)
    return px.scatter_matrix(sample, dimensions=selected, title="Selected feature relationships")


def overview(frame: pd.DataFrame) -> dict[str, object]:
    """Expose the compact overview used by the dashboard and reports."""
    return inspect_dataframe(frame)
