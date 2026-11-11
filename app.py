"""Streamlit dashboard for the reusable preprocessing modules."""

from __future__ import annotations

import hashlib
import json
import time

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.data_cleaner import MISSING_STRATEGIES, clean_dataframe
from modules.data_inspector import common_columns, inspect_dataframe
from modules.data_merger import combine_dataframes
from modules.data_preprocessor import (
    detect_column_types,
    extract_datetime_features,
    preprocess_dataframe,
)
from modules.eda import (
    correlation_figure,
    correlation_heatmap_matplotlib,
    distribution_figure,
    missing_values_figure,
    missing_values_heatmap,
    pair_figure,
    statistical_summary,
)
from modules.file_handler import load_uploaded_files
from modules.outlier_detector import detect_outliers
from modules.report_generator import export_excel, export_html, report_as_dataframe
from modules.skewness_analyzer import analyze_skewness
from utils.helpers import dataframe_for_display, history_entry


st.set_page_config(
    page_title="Automated Data Preprocessing & EDA",
    page_icon="📊",
    layout="wide",
)


def _init_state() -> None:
    defaults = {
        "raw_frames": {},
        "working_df": None,
        "integration_report": {},
        "cleaning_report": {},
        "preprocessing_config": {},
        "preprocessing_result": None,
        "preprocessing_source": None,
        "history": [],
        "processing_seconds": 0.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _processing_report(frame: pd.DataFrame) -> dict[str, object]:
    original_shape = st.session_state.get("original_shape")
    inspection = inspect_dataframe(frame)
    skew_report = analyze_skewness(frame)
    outlier_report = detect_outliers(frame)
    return {
        "files_processed": len(st.session_state.raw_frames),
        "original_dimensions": original_shape or frame.shape,
        "final_dimensions": frame.shape,
        "duplicate_records_removed": st.session_state.cleaning_report.get(
            "duplicates_removed", 0
        ),
        "missing_values_before_cleaning": st.session_state.cleaning_report.get(
            "missing_before", inspection["missing_total"]
        ),
        "missing_values_after_cleaning": inspection["missing_total"],
        "columns_modified": st.session_state.cleaning_report.get("modifications", []),
        "integration": st.session_state.integration_report,
        "preprocessing": st.session_state.preprocessing_config,
        "outliers_detected": int(outlier_report["iqr_outliers"].sum())
        if not outlier_report.empty
        else 0,
        "skewness_analysis": skew_report.to_dict(orient="records"),
        "processing_time_seconds": round(st.session_state.processing_seconds, 3),
    }


def main() -> None:
    _init_state()
    st.title("Automated Data Preprocessing & EDA")
    st.caption(
        "Upload data, review integration and cleaning choices, explore distributions, "
        "then export your processed dataset and an audit-friendly report."
    )
    tabs = st.tabs(
        ["Home & Upload", "Integration", "Data Cleaning", "Preprocessing", "EDA", "Reports"]
    )

    with tabs[0]:
        uploads = st.file_uploader(
            "Upload one or more CSV, Excel, or JSON files",
            type=["csv", "xlsx", "xls", "json"],
            accept_multiple_files=True,
        )
        if uploads:
            payloads = [(uploaded.name, uploaded.getvalue()) for uploaded in uploads]
            digest = hashlib.sha256()
            for name, content in payloads:
                digest.update(name.encode("utf-8"))
                digest.update(content)
            signature = digest.hexdigest()
            if signature != st.session_state.get("upload_signature"):
                try:
                    frames = load_uploaded_files(payloads)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.session_state.raw_frames = frames
                    st.session_state.working_df = next(iter(frames.values())).copy(deep=True)
                    st.session_state.upload_signature = signature
                    st.session_state.original_shape = st.session_state.working_df.shape
                    st.session_state.integration_report = (
                        {"method": "single file", "files": list(frames)}
                        if len(frames) == 1
                        else {}
                    )
                    st.session_state.cleaning_report = {}
                    st.session_state.preprocessing_config = {}
                    st.session_state.preprocessing_result = None
                    st.session_state.preprocessing_source = None
                    st.session_state.processing_seconds = 0.0
                    st.session_state.history = [
                        history_entry("Uploaded files", ", ".join(frames))
                    ]
                    st.success(f"Loaded {len(frames)} file(s). Original uploads remain unchanged.")
        if st.session_state.raw_frames:
            st.subheader("Uploaded file inspection")
            for name, frame in st.session_state.raw_frames.items():
                with st.expander(f"{name} — {frame.shape[0]:,} rows × {frame.shape[1]} columns"):
                    st.json(inspect_dataframe(frame))
                    st.dataframe(dataframe_for_display(frame.head(20)), width="stretch")
            if len(st.session_state.raw_frames) > 1:
                st.info(
                    "Choose concatenate or keyed merge in Integration. Until then, "
                    "the first file is shown as a preview; it is not combined implicitly."
                )
        else:
            st.info("Upload files to begin. CSV, .xlsx, .xls, and JSON are supported.")

    with tabs[1]:
        st.subheader("Data integration")
        frames = st.session_state.raw_frames
        if not frames:
            st.info("Upload files on the Home & Upload tab first.")
        elif len(frames) == 1:
            st.success("One file is ready; no integration step is needed.")
        else:
            method = st.radio(
                "How should these datasets be combined?",
                ["concatenate", "merge"],
                horizontal=True,
                help="Concatenate adds rows. Merge joins on keys you choose.",
            )
            join_type = "outer"
            keys: list[str] = []
            if method == "merge":
                shared = common_columns(frames)
                if not shared:
                    st.warning("No common columns were found. Concatenation is available instead.")
                keys = st.multiselect("Merge key column(s)", shared)
                join_type = st.selectbox("Join type", ["inner", "left", "right", "outer"])
            if st.button(
                "Combine files",
                type="primary",
                disabled=(method == "merge" and (not keys or not common_columns(frames))),
            ):
                try:
                    combined, report = combine_dataframes(
                        frames, method=method, join_type=join_type, keys=keys
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.session_state.working_df = combined
                    st.session_state.integration_report = report
                    st.session_state.original_shape = combined.shape
                    st.session_state.history.append(history_entry("Integrated datasets", report))
                    st.success(f"Combined dataset: {combined.shape[0]:,} rows × {combined.shape[1]} columns")
            if st.session_state.integration_report:
                st.json(st.session_state.integration_report)

    with tabs[2]:
        st.subheader("Data cleaning")
        frame = st.session_state.working_df
        if frame is None:
            st.info("Load a file or combine datasets first.")
        else:
            st.caption("Operations are applied to a working copy; uploaded data is never overwritten.")
            text_columns = frame.select_dtypes(include=["object", "string"]).columns.tolist()
            with st.form("cleaning_form"):
                missing_strategy = st.selectbox(
                    "Missing-value strategy",
                    MISSING_STRATEGIES,
                    help="Auto uses median for numeric columns and mode for other columns.",
                )
                missing_columns = st.multiselect(
                    "Columns for the selected missing-value strategy",
                    list(frame.columns),
                    default=list(frame.columns),
                )
                numeric_columns = st.multiselect(
                    "Text columns to convert to numeric",
                    text_columns,
                    help="Unparseable values become missing and are included in the cleaning summary.",
                )
                columns_to_remove = st.multiselect(
                    "Remove irrelevant columns", list(frame.columns)
                )
                remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
                normalize_names = st.checkbox("Standardize column names", value=True)
                strip_whitespace = st.checkbox("Trim text whitespace", value=True)
                normalize_categories = st.checkbox(
                    "Normalize text categories to lowercase", value=False
                )
                remove_constants = st.checkbox("Remove constant columns", value=False)
                clean_clicked = st.form_submit_button("Apply selected cleaning")
            if clean_clicked:
                started = time.perf_counter()
                try:
                    cleaned, report = clean_dataframe(
                        frame,
                        missing_strategy=missing_strategy,
                        missing_columns=missing_columns,
                        numeric_columns=numeric_columns,
                        columns_to_remove=columns_to_remove,
                        remove_duplicates=remove_duplicates,
                        normalize_names=normalize_names,
                        strip_whitespace=strip_whitespace,
                        normalize_categories=normalize_categories,
                        remove_constant_columns=remove_constants,
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.session_state.working_df = cleaned
                    st.session_state.cleaning_report = report
                    st.session_state.processing_seconds += time.perf_counter() - started
                    st.session_state.history.append(history_entry("Cleaned data", report))
                    st.success(
                        f"Cleaning complete: {report['duplicates_removed']} duplicate rows removed."
                    )
            if st.session_state.cleaning_report:
                st.json(st.session_state.cleaning_report)
            st.dataframe(
                dataframe_for_display(st.session_state.working_df.head(50)),
                width="stretch",
            )

    with tabs[3]:
        st.subheader("Preprocessing")
        frame = st.session_state.working_df
        if frame is None:
            st.info("Load a file or combine datasets first.")
        else:
            types = detect_column_types(frame)
            st.write(
                {
                    "numeric": types["numeric"],
                    "categorical (including boolean)": types["categorical"],
                    "datetime": types["datetime"],
                }
            )
            with st.form("preprocessing_form"):
                split = st.checkbox("Create a train/test split (fit transformations on training data)")
                target = (
                    st.selectbox("Target column", list(frame.columns))
                    if split
                    else None
                )
                feature_options = [
                    column for column in frame.columns if column != target
                ]
                feature_columns = st.multiselect(
                    "Features to include",
                    feature_options,
                    default=feature_options,
                    help="Leave all selected to preprocess every feature. The target is excluded from fitting.",
                )
                dates = st.multiselect(
                    "Extract date parts and replace selected date columns",
                    feature_columns,
                )
                scaler = st.selectbox(
                    "Numerical scaling", ["none", "standard", "minmax", "robust"]
                )
                encoder = st.selectbox(
                    "Categorical encoding", ["one-hot", "ordinal (label-style)"]
                )
                encoder = "ordinal" if encoder.startswith("ordinal") else encoder
                selected_numeric = [
                    column for column in feature_columns if column in types["numeric"]
                ]
                log_columns = st.multiselect(
                    "Optional log1p columns (non-negative numeric columns only)",
                    selected_numeric,
                )
                test_size = st.slider(
                    "Test set fraction", 0.1, 0.5, 0.2, 0.05, disabled=not split
                )
                preprocess_clicked = st.form_submit_button("Preview and apply preprocessing")
            if preprocess_clicked:
                started = time.perf_counter()
                try:
                    features, added_dates = extract_datetime_features(frame, dates) if dates else (frame.copy(), [])
                    selected_logs = [column for column in log_columns if column in features.columns]
                    preprocessing_columns = [
                        column for column in feature_columns if column not in dates
                    ] + added_dates
                    result = preprocess_dataframe(
                        features,
                        scaler=scaler,
                        encoder=encoder,
                        log_columns=selected_logs,
                        feature_columns=preprocessing_columns,
                        split=split,
                        target_column=target,
                        test_size=test_size,
                    )
                except (ValueError, TypeError) as exc:
                    st.error(str(exc))
                else:
                    st.session_state.preprocessing_result = result
                    st.session_state.preprocessing_source = frame.copy(deep=True)
                    st.session_state.preprocessing_config = {
                        **result.config,
                        "date_features_added": added_dates,
                    }
                    st.session_state.processing_seconds += time.perf_counter() - started
                    st.session_state.history.append(
                        history_entry("Preprocessed data", st.session_state.preprocessing_config)
                    )
                    if result.transformed is not None:
                        st.session_state.working_df = result.transformed
                    st.success("Preprocessing complete. Preview the transformed data below.")
            result = st.session_state.preprocessing_result
            if result is not None:
                if result.transformed is not None:
                    st.dataframe(
                        dataframe_for_display(result.transformed.head(50)),
                        width="stretch",
                    )
                    st.caption(f"Output shape: {result.transformed.shape}")
                    source = st.session_state.preprocessing_source
                    comparable = (
                        [
                            column
                            for column in result.transformed.select_dtypes(
                                include="number"
                            ).columns
                            if source is not None
                            and column in source.select_dtypes(include="number").columns
                        ]
                        if source is not None
                        else []
                    )
                    if comparable:
                        comparison_column = st.selectbox(
                            "Compare numeric distribution before and after",
                            comparable,
                        )
                        before_chart, after_chart = st.columns(2)
                        before_chart.plotly_chart(
                            px.histogram(
                                source,
                                x=comparison_column,
                                title=f"Before: {comparison_column}",
                            ),
                            width="stretch",
                        )
                        after_chart.plotly_chart(
                            px.histogram(
                                result.transformed,
                                x=comparison_column,
                                title=f"After: {comparison_column}",
                            ),
                            width="stretch",
                        )
                else:
                    train_tab, test_tab = st.tabs(["Training set", "Test set"])
                    with train_tab:
                        st.dataframe(
                            dataframe_for_display(result.train_features.head(50)),
                            width="stretch",
                        )
                        st.caption(f"Training features: {result.train_features.shape}")
                    with test_tab:
                        st.dataframe(
                            dataframe_for_display(result.test_features.head(50)),
                            width="stretch",
                        )
                        st.caption(f"Test features: {result.test_features.shape}")
                    st.info("Training and test exports are available on the Reports tab.")

    with tabs[4]:
        st.subheader("Exploratory data analysis")
        frame = st.session_state.working_df
        if frame is None:
            st.info("Load a file first.")
        else:
            overview = inspect_dataframe(frame)
            metric_columns = st.columns(4)
            metric_columns[0].metric("Rows", f"{overview['rows']:,}")
            metric_columns[1].metric("Columns", overview["columns"])
            metric_columns[2].metric("Missing values", overview["missing_total"])
            metric_columns[3].metric("Duplicate rows", overview["duplicate_rows"])
            st.write("**Missing values by column**")
            st.plotly_chart(missing_values_figure(frame), width="stretch")
            st.plotly_chart(missing_values_heatmap(frame), width="stretch")
            numeric = frame.select_dtypes(include="number").columns.tolist()
            categorical = overview["categorical_columns"]
            if numeric:
                st.write("**Numerical statistics**")
                st.dataframe(
                    dataframe_for_display(statistical_summary(frame)),
                    width="stretch",
                )
            columns = list(frame.columns)
            if columns:
                column = st.selectbox("Distribution column", columns)
                st.plotly_chart(distribution_figure(frame, column), width="stretch")
            if numeric:
                st.write("**Outlier review**")
                chosen_numeric = st.selectbox("Numeric column for box/violin plots", numeric)
                box_col, violin_col = st.columns(2)
                box_col.plotly_chart(px.box(frame, y=chosen_numeric), width="stretch")
                violin_col.plotly_chart(
                    px.violin(frame, y=chosen_numeric, box=True), width="stretch"
                )
                st.write("**Skewness and potential outliers (not removed automatically)**")
                skew_report = analyze_skewness(frame)
                outlier_report = detect_outliers(frame)
                left, right = st.columns(2)
                left.dataframe(skew_report, width="stretch")
                right.dataframe(outlier_report, width="stretch")
                st.plotly_chart(correlation_figure(frame), width="stretch")
                heatmap = correlation_heatmap_matplotlib(frame)
                if heatmap is not None:
                    st.pyplot(heatmap, clear_figure=True)
                if len(numeric) > 1:
                    scatter_x, scatter_y = st.columns(2)
                    x_column = scatter_x.selectbox("Scatter X feature", numeric, key="scatter_x")
                    y_options = [column for column in numeric if column != x_column]
                    y_column = scatter_y.selectbox(
                        "Scatter Y feature",
                        y_options,
                        key="scatter_y",
                    )
                    st.plotly_chart(
                        px.scatter(frame, x=x_column, y=y_column),
                        width="stretch",
                    )
                    selected = st.multiselect(
                        "Pair-plot features (up to five)",
                        numeric,
                        default=numeric[: min(3, len(numeric))],
                    )
                    st.plotly_chart(pair_figure(frame, selected), width="stretch")
            if categorical:
                cat_column = st.selectbox("Category count plot", categorical)
                st.plotly_chart(
                    px.bar(
                        frame[cat_column].astype("string").fillna("(missing)")
                        .value_counts().head(30).rename_axis("category").reset_index(name="count"),
                        x="category",
                        y="count",
                    ),
                    width="stretch",
                )

    with tabs[5]:
        st.subheader("Downloads and processing history")
        frame = st.session_state.working_df
        if frame is None:
            st.info("Load data to generate reports.")
        else:
            report = _processing_report(frame)
            stats = statistical_summary(frame)
            st.json(report)
            report_csv = report_as_dataframe(report).to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download processed data (CSV)",
                frame.to_csv(index=False).encode("utf-8"),
                file_name="processed_data.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download dataset and report (Excel)",
                export_excel(frame, report, stats=stats),
                file_name="preprocessing_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            st.download_button(
                "Download HTML report",
                export_html(frame, report, stats=stats),
                file_name="preprocessing_report.html",
                mime="text/html",
            )
            st.download_button(
                "Download processing summary (CSV)",
                report_csv,
                file_name="processing_summary.csv",
                mime="text/csv",
            )
            config = st.session_state.preprocessing_config
            if config:
                st.download_button(
                    "Download preprocessing configuration (JSON)",
                    json.dumps(config, indent=2, default=str).encode("utf-8"),
                    file_name="preprocessing_config.json",
                    mime="application/json",
                )
            result = st.session_state.preprocessing_result
            if result is not None and result.config.get("split"):
                for label, data in (
                    ("train_features", result.train_features),
                    ("test_features", result.test_features),
                ):
                    if data is not None:
                        target_series = (
                            result.train_target
                            if label == "train_features"
                            else result.test_target
                        )
                        download_frame = data.copy()
                        if target_series is not None:
                            download_frame[result.config["target_column"]] = target_series
                        st.download_button(
                            f"Download {label.replace('_', ' ')}",
                            download_frame.to_csv(index=False).encode("utf-8"),
                            file_name=f"{label}.csv",
                            mime="text/csv",
                        )
            st.markdown("#### Processing history (this browser session)")
            if st.session_state.history:
                st.dataframe(
                    pd.DataFrame(st.session_state.history), width="stretch"
                )
            else:
                st.caption("No processing actions recorded yet.")


if __name__ == "__main__":
    main()
