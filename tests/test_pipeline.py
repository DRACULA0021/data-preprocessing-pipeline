"""Focused tests for file loading, combination, cleaning, and preprocessing."""

from __future__ import annotations

from io import BytesIO
import unittest

import numpy as np
import pandas as pd

from modules.data_cleaner import clean_dataframe
from modules.data_merger import combine_dataframes
from modules.data_preprocessor import preprocess_dataframe
from modules.data_preprocessor import extract_datetime_features
from modules.file_handler import load_data_file, load_uploaded_files, make_unique_columns
from modules.outlier_detector import detect_outliers
from modules.report_generator import export_excel, export_html
from modules.skewness_analyzer import analyze_skewness
from utils.helpers import dataframe_for_display


class FileHandlerTests(unittest.TestCase):
    def test_duplicate_column_names_are_made_unique(self) -> None:
        self.assertEqual(
            make_unique_columns([" A ", "A", "", "A_1"]),
            ["A", "A_1", "column_3", "A_1_1"],
        )

    def test_csv_load_does_not_change_input_bytes(self) -> None:
        content = b"name,age\n Ada ,37\n"
        original = content[:]
        frame = load_data_file("people.csv", content)
        self.assertEqual(content, original)
        self.assertEqual(frame.columns.tolist(), ["name", "age"])

    def test_excel_and_json_inputs_load(self) -> None:
        excel_buffer = BytesIO()
        pd.DataFrame({"StockCode": ["85123A", 10002, None]}).to_excel(
            excel_buffer, index=False
        )
        excel = load_data_file("values.xlsx", excel_buffer.getvalue())
        json_frame = load_data_file("values.json", b'[{"n":1},{"n":2}]')
        self.assertEqual(excel["StockCode"].iloc[0], "85123A")
        self.assertEqual(excel["StockCode"].iloc[1], 10002)
        display_frame = dataframe_for_display(excel)
        import pyarrow as pa

        pa.Table.from_pandas(display_frame, preserve_index=False)
        self.assertEqual(json_frame["n"].tolist(), [1, 2])

    def test_mixed_object_columns_are_safe_for_arrow_display(self) -> None:
        import pyarrow as pa

        source = pd.DataFrame({"StockCode": ["85123A", 10002, None]})
        display_frame = dataframe_for_display(source)
        pa.Table.from_pandas(display_frame, preserve_index=False)
        self.assertEqual(source.loc[1, "StockCode"], 10002)
        self.assertEqual(display_frame.loc[1, "StockCode"], "10002")


class IntegrationAndCleaningTests(unittest.TestCase):
    def test_concat_and_merge_report_decisions(self) -> None:
        first = pd.DataFrame({"id": [1, 2], "name": ["a", "b"]})
        second = pd.DataFrame({"id": [2, 3], "value": [10, 20]})
        concatenated, concat_report = combine_dataframes(
            {"a.csv": first, "b.csv": second}, method="concatenate"
        )
        merged, merge_report = combine_dataframes(
            {"a.csv": first, "b.csv": second},
            method="merge",
            join_type="outer",
            keys=["id"],
        )
        self.assertEqual(len(concatenated), 4)
        self.assertEqual(concat_report["method"], "concatenate rows")
        self.assertEqual(len(merged), 3)
        self.assertEqual(merge_report["keys"], ["id"])

    def test_cleaning_returns_copy_and_applies_selected_operations(self) -> None:
        source = pd.DataFrame(
            {" First Name ": [" Ada ", " Ada "], "Score": [2.0, 2.0]}
        )
        cleaned, report = clean_dataframe(
            source,
            missing_strategy="none",
            remove_duplicates=True,
            remove_constant_columns=False,
        )
        self.assertEqual(source.shape, (2, 2))
        self.assertEqual(cleaned.shape, (1, 2))
        self.assertEqual(cleaned.columns.tolist(), ["first_name", "score"])
        self.assertEqual(cleaned.loc[0, "first_name"], "Ada")
        self.assertEqual(report["duplicates_removed"], 1)

    def test_mean_imputation_rejects_categorical_columns(self) -> None:
        frame = pd.DataFrame({"category": ["a", None]})
        with self.assertRaisesRegex(ValueError, "Mean imputation requires"):
            clean_dataframe(frame, missing_strategy="mean")

    def test_drop_columns_only_drops_selected_columns_with_missing_values(self) -> None:
        frame = pd.DataFrame({"keep": [1, 2], "drop": [None, 3]})
        cleaned, _ = clean_dataframe(
            frame,
            missing_strategy="drop columns",
            remove_duplicates=False,
        )
        self.assertEqual(cleaned.columns.tolist(), ["keep"])

    def test_selected_text_column_can_be_converted_to_numeric(self) -> None:
        frame = pd.DataFrame({"value": ["1.5", "bad"]})
        cleaned, report = clean_dataframe(
            frame,
            numeric_columns=["value"],
            remove_duplicates=False,
            normalize_names=False,
        )
        self.assertTrue(pd.isna(cleaned.loc[1, "value"]))
        self.assertEqual(report["numeric_columns_converted"]["value"], 1)


class AnalysisAndPreprocessingTests(unittest.TestCase):
    def test_end_to_end_uploaded_files_to_transformed_report(self) -> None:
        uploaded = load_uploaded_files(
            [
                ("amounts.csv", b"id,amount\n1,2.0\n2,\n"),
                ("groups.json", b'[{"id":1,"group":"a"},{"id":2,"group":"b"}]'),
            ]
        )
        combined, _ = combine_dataframes(
            uploaded, method="merge", join_type="outer", keys=["id"]
        )
        cleaned, _ = clean_dataframe(
            combined, missing_strategy="auto", remove_duplicates=False
        )
        transformed = preprocess_dataframe(
            cleaned, scaler="standard", encoder="one-hot"
        ).transformed
        self.assertEqual(transformed.shape[0], 2)
        self.assertTrue(np.isfinite(transformed.to_numpy()).all())

    def test_reports_detect_outlier_and_skew(self) -> None:
        frame = pd.DataFrame({"value": [1, 2, 3, 4, 100]})
        outliers = detect_outliers(frame)
        skew = analyze_skewness(frame)
        self.assertGreater(outliers.loc[0, "iqr_outliers"], 0)
        self.assertEqual(skew.loc[0, "distribution"], "positively skewed")

    def test_train_test_preprocessing_fits_on_training_only(self) -> None:
        frame = pd.DataFrame(
            {
                "amount": np.arange(20, dtype=float),
                "kind": ["a", "b"] * 10,
                "target": [0, 1] * 10,
            }
        )
        result = preprocess_dataframe(
            frame,
            scaler="standard",
            encoder="one-hot",
            split=True,
            target_column="target",
            test_size=0.25,
        )
        self.assertIsNone(result.transformed)
        self.assertEqual(len(result.train_features), 15)
        self.assertEqual(len(result.test_features), 5)
        self.assertEqual(result.config["fit_scope"], "training features only")
        self.assertTrue(np.isfinite(result.train_features.to_numpy()).all())

    def test_preprocessing_handles_mixed_categorical_values(self) -> None:
        frame = pd.DataFrame({"category": pd.Series([1, "one", None], dtype=object)})
        result = preprocess_dataframe(frame, encoder="one-hot")
        self.assertEqual(len(result.transformed), 3)

    def test_preprocessing_can_select_features(self) -> None:
        frame = pd.DataFrame({"use": [1, 2, 3], "ignore": [10, 20, 30]})
        result = preprocess_dataframe(
            frame,
            feature_columns=["use"],
            scaler="none",
        )
        self.assertEqual(result.transformed.columns.tolist(), ["use"])

    def test_log_transform_rejects_negative_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "log1p requires non-negative"):
            preprocess_dataframe(
                pd.DataFrame({"amount": [-1.0, 2.0, 3.0]}),
                log_columns=["amount"],
            )

    def test_datetime_features_replace_original_column(self) -> None:
        frame = pd.DataFrame({"created": ["2024-01-02", "not a date"]})
        updated, added = extract_datetime_features(frame, ["created"])
        self.assertNotIn("created", updated.columns)
        self.assertEqual(len(added), 4)
        self.assertEqual(updated.loc[0, "created_year"], 2024)

    def test_excel_and_html_reports_are_generated(self) -> None:
        frame = pd.DataFrame({"value": [1, 2]})
        report = {"files_processed": 1}
        self.assertTrue(export_excel(frame, report).startswith(b"PK"))
        self.assertIn(b"Preprocessing report", export_html(frame, report))


if __name__ == "__main__":
    unittest.main()
