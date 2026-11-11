"""Reviewable sklearn-based preprocessing with optional leakage-safe splitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)


def _categorical_values_as_strings(values: np.ndarray) -> np.ndarray:
    return values.astype(str)


@dataclass
class PreprocessingResult:
    """Transformed data and fitted state needed for reuse or train/test workflows."""

    transformed: pd.DataFrame | None
    train_features: pd.DataFrame | None
    test_features: pd.DataFrame | None
    train_target: pd.Series | None
    test_target: pd.Series | None
    transformer: ColumnTransformer
    config: dict[str, Any]


def detect_column_types(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Classify columns, treating booleans as categorical features."""
    numeric: list[str] = []
    categorical: list[str] = []
    datetime: list[str] = []
    for column in frame.columns:
        dtype = frame[column].dtype
        if is_datetime64_any_dtype(dtype):
            datetime.append(column)
        elif is_bool_dtype(dtype):
            categorical.append(column)
        elif is_numeric_dtype(dtype):
            numeric.append(column)
        else:
            categorical.append(column)
    return {"numeric": numeric, "categorical": categorical, "datetime": datetime}


def extract_datetime_features(
    source: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Replace selected date columns with year, month, day, and weekday features."""
    frame = source.copy(deep=True)
    added: list[str] = []
    for column in columns:
        if column not in frame.columns:
            raise ValueError(f"Unknown date column: {column}")
        dates = pd.to_datetime(frame[column], errors="coerce")
        for suffix, values in (
            ("year", dates.dt.year),
            ("month", dates.dt.month),
            ("day", dates.dt.day),
            ("dayofweek", dates.dt.day_of_week),
        ):
            feature = f"{column}_{suffix}"
            if feature in frame.columns and feature != column:
                raise ValueError(f"Date feature name already exists: {feature}")
            frame[feature] = values
            added.append(feature)
        frame = frame.drop(columns=[column])
    return frame, added


def _scaler(name: str) -> Any:
    scalers = {
        "none": None,
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }
    if name not in scalers:
        raise ValueError(f"Unknown scaler: {name}")
    return scalers[name]


def _build_transformer(
    frame: pd.DataFrame,
    scaler: str,
    encoder: str,
    log_columns: list[str],
) -> ColumnTransformer:
    types = detect_column_types(frame)
    numeric = types["numeric"]
    categorical = types["categorical"]
    if types["datetime"]:
        raise ValueError(
            "Extract date features or remove datetime columns before preprocessing: "
            f"{types['datetime']}"
        )
    invalid_log_columns = [column for column in log_columns if column not in numeric]
    if invalid_log_columns:
        raise ValueError(f"Log transform requires numeric columns: {invalid_log_columns}")
    for column in log_columns:
        observed = frame[column].dropna()
        if not observed.empty and observed.min() < 0:
            raise ValueError(f"log1p requires non-negative values: {column}")
    if encoder not in {"one-hot", "ordinal"}:
        raise ValueError("Encoder must be 'one-hot' or 'ordinal'.")

    scale = _scaler(scaler)
    remainder_numeric = [column for column in numeric if column not in log_columns]
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if remainder_numeric:
        steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
        ]
        if scale is not None:
            steps.append(("scaler", scale))
        transformers.append(("numeric", Pipeline(steps), remainder_numeric))
    if log_columns:
        log_steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "log1p",
                FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
            ),
        ]
        if scale is not None:
            log_steps.append(("scaler", _scaler(scaler)))
        transformers.append(("log_numeric", Pipeline(log_steps), log_columns))
    if categorical:
        category_encoder: Any
        if encoder == "one-hot":
            category_encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )
        else:
            category_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-1,
            )
        category_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="__missing__",
                        keep_empty_features=True,
                    ),
                ),
                (
                    "stringify",
                    FunctionTransformer(
                        _categorical_values_as_strings,
                        feature_names_out="one-to-one",
                    ),
                ),
                ("encoder", category_encoder),
            ]
        )
        transformers.append(("categorical", category_pipeline, categorical))
    if not transformers:
        raise ValueError("No supported feature columns were found.")
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def preprocess_dataframe(
    source: pd.DataFrame,
    *,
    scaler: str = "none",
    encoder: str = "one-hot",
    log_columns: list[str] | None = None,
    feature_columns: list[str] | None = None,
    split: bool = False,
    target_column: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> PreprocessingResult:
    """Fit/transform features, fitting on training data only when split is enabled."""
    if source.empty:
        raise ValueError("Cannot preprocess an empty dataset.")
    selected_log_columns = list(log_columns or [])
    if feature_columns is not None:
        missing_features = [column for column in feature_columns if column not in source]
        if missing_features:
            raise ValueError(f"Unknown feature columns: {missing_features}")
    if split:
        if target_column not in source.columns:
            raise ValueError("Choose a target column before enabling the train/test split.")
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between zero and one.")
        available_features = source.drop(columns=[target_column])
        features = (
            available_features[feature_columns]
            if feature_columns is not None
            else available_features
        )
        target = source[target_column]
        train_indices, test_indices = train_test_split(
            np.arange(len(source)),
            test_size=test_size,
            random_state=random_state,
        )
        x_train = features.iloc[train_indices]
        x_test = features.iloc[test_indices]
        transformer = _build_transformer(
            x_train, scaler, encoder, selected_log_columns
        )
        transformer.set_output(transform="pandas")
        train_result = transformer.fit_transform(x_train)
        test_result = transformer.transform(x_test)
        return PreprocessingResult(
            transformed=None,
            train_features=train_result,
            test_features=test_result,
            train_target=target.iloc[train_indices].copy(),
            test_target=target.iloc[test_indices].copy(),
            transformer=transformer,
            config={
                "scaler": scaler,
                "encoder": encoder,
                "log_columns": selected_log_columns,
                "feature_columns": list(features.columns),
                "split": True,
                "target_column": target_column,
                "test_size": test_size,
                "random_state": random_state,
                "fit_scope": "training features only",
            },
        )

    selected_features = source[feature_columns] if feature_columns is not None else source
    transformer = _build_transformer(selected_features, scaler, encoder, selected_log_columns)
    transformer.set_output(transform="pandas")
    transformed = transformer.fit_transform(selected_features)
    return PreprocessingResult(
        transformed=transformed,
        train_features=None,
        test_features=None,
        train_target=None,
        test_target=None,
        transformer=transformer,
        config={
            "scaler": scaler,
            "encoder": encoder,
            "log_columns": selected_log_columns,
            "feature_columns": list(selected_features.columns),
            "split": False,
            "fit_scope": "entire dataset (exploration/export only)",
        },
    )
