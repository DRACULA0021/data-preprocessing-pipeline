# Running the Application: Step-by-Step Guide

This guide explains how to set up, launch, and use the Automated Data Preprocessing and EDA application on Windows.

## 1. Prerequisites

- Windows 10 or later
- Python 3.10 or later
- PowerShell
- Input files in CSV, Excel (`.xlsx` or `.xls`), or JSON format

Check that Python is installed:

```powershell
py --version
```

## 2. Open the project folder

In PowerShell, change to the project directory:

```powershell
cd C:\Users\admin\OneDrive\Desktop\data_preprocessing
```

If you placed the project somewhere else, replace this path with that location.

## 3. Create and activate a virtual environment

Create an isolated Python environment for this project:

```powershell
py -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell prevents script activation, you can use the environment's Python executable directly in the commands below instead.

## 4. Install the project dependencies

Upgrade `pip` and install the packages listed for this project:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 5. Start the dashboard

From the project directory, run:

```powershell
python -m streamlit run app.py
```

Streamlit prints a local address, usually `http://localhost:8501`. Open that address in your browser. To stop the application, press `Ctrl+C` in the PowerShell window.

If the virtual environment is not activated, use its executable explicitly:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## 6. Use the application

1. **Home & Upload:** Upload one or more CSV, Excel, or JSON files. Review each file's dimensions, columns, types, missing values, and preview.
2. **Integration:** For multiple files, choose whether to concatenate rows or merge on selected common keys. Select a join type for merges and review the combination report.
3. **Data Cleaning:** Select which operations to apply, such as duplicate removal, whitespace trimming, column-name standardization, type conversion, missing-value handling, category normalization, or column removal.
4. **Preprocessing:** Select features, optional date-part extraction, a numerical scaler, categorical encoding, and optional `log1p` columns. A log transform is allowed only for non-negative numeric data.
5. **Train/test split (optional):** Choose a target column and enable the split to fit preprocessing transformations on the training features only. Without a split, the transformed dataset is suitable for exploration and export, but not a leakage-safe training workflow.
6. **EDA:** Review statistics, distributions, category counts, missing values, correlations, feature relationships, skewness, and potential outliers. Outliers are reported, not automatically removed.
7. **Reports:** Download the processed dataset and processing summary in CSV, Excel, or HTML formats. Download the preprocessing configuration as JSON when one is available. If you enabled a split, download train and test feature sets separately.

Uploaded source files are read in memory and are not overwritten. Cleaning and preprocessing actions affect the current working dataset in the browser session.

## Technology Used

| Technology | Purpose |
|---|---|
| Python | Application language and pipeline implementation |
| Streamlit | Interactive web dashboard |
| pandas | Tabular data loading, cleaning, integration, and reporting |
| NumPy | Numerical operations and data validation |
| scikit-learn | Imputation, categorical encoding, scaling, and train/test splitting |
| SciPy | Skewness analysis |
| Matplotlib and Seaborn | Statistical plotting, including correlation heatmaps |
| Plotly | Interactive charts and EDA visualizations |
| openpyxl | Excel workbook export |
| xlrd | Reading legacy `.xls` Excel files |

Dependency versions are specified in [requirements.txt](./requirements.txt). Implementation details for the individual modules and the broader workflow are in [README.md](./README.md).

## 7. Run the tests (optional)

The project includes standard-library unit and end-to-end tests:

```powershell
python -m unittest discover -s tests -v
```

When using the virtual environment without activating it:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Troubleshooting

- **`py` is not recognized:** Install Python and enable the Python launcher, or substitute `python` for `py` in the environment setup commands.
- **`streamlit` is not recognized:** Activate `.venv` and run `python -m streamlit run app.py`, or use the explicit `.venv\Scripts\python.exe` command above.
- **A dependency is missing:** Confirm PowerShell is using the project's `.venv\Scripts\python.exe`, then reinstall with `python -m pip install -r requirements.txt`.
- **An upload cannot be read:** Check that the file extension is supported and that the file is a valid CSV, Excel, or JSON dataset.
- **A merge produces more rows than expected:** Check whether the selected merge keys contain duplicates in either input. Many-to-many key matches can multiply rows.
- **A log transformation is rejected:** `log1p` requires non-negative values; review the data or choose another transformation.

## Code and file-by-file explanation

This section describes what each project source file and support file does, and how its main code pieces fit into the application. The descriptions follow the current implementation.

### Application entry point

#### [`app.py`](./app.py)

This file is the Streamlit user interface and connects the reusable modules. Running `python -m streamlit run app.py` executes this file.

- **Imports:** Loads Streamlit, pandas, Plotly, Python utilities, and the project's module functions. `hashlib` identifies whether uploaded files changed; `json` serializes reusable settings; `time` measures processing duration.
- **`st.set_page_config(...)`:** Sets the browser tab title, icon, and wide dashboard layout.
- **`_init_state()`:** Creates missing Streamlit session-state values. These hold uploaded DataFrames, the current working DataFrame, operation summaries, preprocessing results/configuration, history, and elapsed processing time. Session state keeps the selected data available as Streamlit reruns the script after UI interactions.
- **`_processing_report(frame)`:** Collects file count, original/final dimensions, cleaning changes, integration and preprocessing choices, missing counts, detected outliers, skewness results, and elapsed time into one report dictionary for display and downloads.
- **`main()`:** Builds six dashboard tabs and handles the application workflow:
  1. **Home & Upload:** Reads uploaded bytes, computes a SHA-256 signature to avoid reloading unchanged uploads on each rerun, calls the file loader, saves raw and working DataFrames separately, and shows each file's inspection and preview. `dataframe_for_display()` is applied only to display copies so mixed-type values are safe for Streamlit's Arrow renderer without changing the working data.
  2. **Integration:** Offers row concatenation or an explicit keyed merge. It validates merge choices, displays the combination report, and updates the working DataFrame only after the user clicks **Combine files**.
  3. **Data Cleaning:** Presents checkboxes and selectors, measures the operation time, calls `clean_dataframe()`, and displays its change summary and preview.
  4. **Preprocessing:** Detects column types, lets the user choose features, date extraction, scaler, encoder, log columns, and optional train/test split, then calls `preprocess_dataframe()`. It retains the fitted transformer and selected configuration. When splitting is enabled, training features fit the transforms and test features are only transformed.
  5. **EDA:** Shows shape and quality metrics, missingness charts, statistics, distributions, categorical counts, box/violin charts, skewness/outlier tables, correlations, scatter plots, and selected-feature pair plots.
  6. **Reports:** Creates CSV, Excel, HTML, and JSON downloads; when a split exists it also offers separate train/test feature files with the target column restored.
- **`if __name__ == "__main__":`:** Calls `main()` when Streamlit executes this file as the application entry point.

### Data loading, inspection, integration, and cleaning

#### [`modules/file_handler.py`](./modules/file_handler.py)

- **`make_unique_columns(columns)`:** Trims column labels, replaces blank names with `column_N`, and appends suffixes to repeated names so every DataFrame column label is unique.
- **`load_data_file(filename, content)`:** Reads bytes into memory, picks a pandas reader from the extension (CSV, Excel, or JSON), normalizes the column labels, and raises a filename-specific `ValueError` if reading fails. It never writes to the original uploaded file.
- **`load_uploaded_files(files)`:** Loads a collection of `(filename, bytes)` pairs into a filename-to-DataFrame mapping. It rejects duplicate filenames and empty uploads.

#### [`modules/data_inspector.py`](./modules/data_inspector.py)

- **`inspect_dataframe(frame)`:** Reports dimensions, pandas data types, missing counts, duplicate rows, and numeric/categorical/boolean/datetime column groups. It also labels columns as constant or near-constant; the current near-constant threshold is a top value occupying at least 98% of non-missing values.
- **`common_columns(frames)`:** Returns column names present in every input DataFrame, preserving the first frame's column order. The integration screen uses this list to offer merge keys.

#### [`modules/data_merger.py`](./modules/data_merger.py)

- **`combine_dataframes(frames, method, join_type, keys)`:** Returns both a combined copy/result and a report. `concatenate` stacks rows by matching column names. `merge` joins frames in sequence using the selected key columns and one of `inner`, `left`, `right`, or `outer`. It checks that keys exist and that the requested method/join is valid. Overlapping non-key columns are renamed with the input filename to reduce naming conflicts. Merging does not automatically check key uniqueness; duplicate keys can yield many-to-many row multiplication.

#### [`modules/data_cleaner.py`](./modules/data_cleaner.py)

- **`MISSING_STRATEGIES`:** The supported missing-value options shown in the UI: none, automatic, mean, median, mode, forward fill, backward fill, drop rows, or drop columns.
- **`standardize_name(column)`:** Converts a column label to lowercase snake_case and replaces punctuation/whitespace with underscores.
- **`_fill_missing(frame, strategy, columns)`:** Internal helper that applies the selected imputation or dropping behavior to selected columns. Mean/median require numeric columns when missing values need filling; `auto` uses numeric median and non-numeric mode.
- **`clean_dataframe(source, ...)`:** Deep-copies its input so the source remains unchanged. It can normalize column labels, trim text, convert user-selected columns to numbers (unparseable values become missing), normalize category casing, remove duplicates, apply a missing-value strategy, remove chosen columns, and optionally drop constant columns. It returns the cleaned DataFrame plus an audit summary with before/after dimensions, counts, and modifications.

### Preprocessing and analysis

#### [`modules/data_preprocessor.py`](./modules/data_preprocessor.py)

- **`_categorical_values_as_strings(values)`:** Normalizes imputed categorical values to strings before sklearn encoding; this supports columns that contain a mixture of numbers and text.
- **`PreprocessingResult`:** Dataclass that packages the transformed data (or split training/test data), target series, fitted `ColumnTransformer`, and the configuration needed to describe the operation.
- **`detect_column_types(frame)`:** Separates numeric, categorical, and datetime columns; boolean values are treated as categorical.
- **`extract_datetime_features(source, columns)`:** Works on a deep copy, parses selected columns as dates, adds year/month/day/day-of-week fields, and removes each original date column. Invalid dates become missing date parts.
- **`_scaler(name)`:** Maps `none`, `standard`, `minmax`, and `robust` to the corresponding sklearn scaler, or reports an unsupported choice.
- **`_build_transformer(frame, scaler, encoder, log_columns)`:** Builds sklearn pipelines. Numeric values are median-imputed and optionally scaled; selected non-negative numeric columns are median-imputed, transformed with `log1p`, and optionally scaled. Categorical columns are filled with a missing-category marker, string-normalized, and one-hot or ordinal encoded. It rejects unhandled datetime columns, invalid log columns, negative values for `log1p`, and unsupported encoders.
- **`preprocess_dataframe(source, ...)`:** Validates the input and selected feature list. With `split=True`, it separates the target, splits row indices, fits the transformer on training features only, and transforms test features with that already-fitted transformer. With `split=False`, it fits and transforms the selected features together; that mode is intended for exploration/export rather than leakage-safe model evaluation. Both modes return a `PreprocessingResult` and a configuration dictionary.

#### [`modules/eda.py`](./modules/eda.py)

- **`statistical_summary(frame)`:** Builds descriptive statistics for numeric columns, including median, variance, skewness, kurtosis, and mode; it also adds the mode for non-numeric columns when one exists.
- **`categorical_frequencies(frame, column, limit)`:** Counts category values, including missing values, and returns at most `limit` rows.
- **`missing_values_figure(frame)`:** Makes a Plotly bar chart of missing values per column, or a no-missing-values annotation.
- **`missing_values_heatmap(frame)`:** Builds a row/column missingness heatmap bounded to the first 30 columns and at most 500 sampled rows.
- **`correlation_heatmap_matplotlib(frame)`:** Uses Seaborn and Matplotlib to render correlations for up to 20 numeric columns; returns `None` if fewer than two are available.
- **`distribution_figure(frame, column)`:** Creates a histogram with a box marginal for numeric columns, or a bar chart for the most frequent categories.
- **`correlation_figure(frame)`:** Creates an interactive Plotly correlation heatmap when at least two numeric features exist.
- **`pair_figure(frame, columns)`:** Creates a Plotly scatter-matrix for up to five selected numeric features and samples at most 500 complete rows.
- **`overview(frame)`:** Exposes the inspector's overview summary for callers that need it.

#### [`modules/skewness_analyzer.py`](./modules/skewness_analyzer.py)

- **`analyze_skewness(frame, threshold=0.5)`:** Computes SciPy skewness for numeric columns with enough non-constant values, labels distributions as positive, negative, or approximately symmetric, and provides a suggested transformation for review. It reports recommendations only; it does not transform data.

#### [`modules/outlier_detector.py`](./modules/outlier_detector.py)

- **`detect_outliers(frame, z_threshold=3.0)`:** For each numeric column, counts values outside the 1.5×IQR fences and values with absolute Z-score above the threshold. It validates the threshold and reports counts only; it does not remove outliers.

#### [`modules/report_generator.py`](./modules/report_generator.py)

- **`report_as_dataframe(report)`:** Converts a report dictionary to a two-column table; nested values are serialized as JSON strings.
- **`export_excel(frame, report, stats=None)`:** Builds an in-memory `.xlsx` workbook with processed data, processing report, and (when provided) statistics sheets.
- **`export_html(frame, report, stats=None)`:** Builds a self-contained HTML report with escaped report values, optional statistics, and a preview of up to 100 data rows.

### Helpers, tests, and project configuration

#### [`utils/helpers.py`](./utils/helpers.py)

- **`history_entry(action, details)`:** Creates a UTC timestamped record for the current browser session's processing history.
- **`dataframe_for_display(frame)`:** Makes a display-only copy of a DataFrame and converts `object` columns to pandas string values. This avoids Arrow serialization failures for mixed values such as text identifiers and numbers. It does not mutate or replace pipeline data.

#### [`tests/test_pipeline.py`](./tests/test_pipeline.py)

Contains standard-library `unittest` cases grouped by component:

- **`FileHandlerTests`:** Checks unique column naming, unchanged upload bytes, CSV/Excel/JSON loading, and Arrow-compatible display copies.
- **`IntegrationAndCleaningTests`:** Checks concatenate and merge results, copy-based cleaning, invalid mean-imputation requests, dropping columns with missing values, and selected text-to-number conversion.
- **`AnalysisAndPreprocessingTests`:** Exercises an uploaded CSV/JSON → merge → clean → preprocess flow; skew and outlier results; train-only fitting on a split; mixed category values; feature selection; negative log rejection; datetime extraction; and Excel/HTML report creation.

Run these tests with:

```powershell
python -m unittest discover -s tests -v
```

#### [`requirements.txt`](./requirements.txt)

Lists the installable Python packages and minimum versions:

- `pandas` and `numpy` provide DataFrame and numerical operations.
- `scikit-learn` provides imputers, encoders, scalers, pipelines, and splitting.
- `scipy` computes skewness.
- `matplotlib` and `seaborn` support statistical plotting.
- `plotly` provides interactive charts.
- `streamlit` provides the dashboard.
- `openpyxl` writes `.xlsx` workbooks; `xlrd` reads legacy `.xls` workbooks.

#### [`.gitignore`](./.gitignore)

Prevents the virtual environment, Python bytecode/cache, Streamlit secrets, local raw/processed data, and generated reports from being included in Git. The `!*.gitkeep` rules retain the otherwise empty data and report folders.

#### `modules/__init__.py` and `utils/__init__.py`

These package marker files contain only a module docstring. They make the project directories importable Python packages; they do not run processing logic.

#### `data/raw/.gitkeep`, `data/processed/.gitkeep`, and `reports/.gitkeep`

These are empty placeholders that keep the folders present in a checkout. The current app processes uploads in memory and downloads results to the user's browser; it does not automatically save uploaded or processed datasets to these folders.

#### [`README.md`](./README.md) and [`RUNNING_GUIDE.md`](./RUNNING_GUIDE.md)

`README.md` gives the project overview, architecture, setup, workflow, and limits. This `RUNNING_GUIDE.md` gives the commands to run the application, explains the dashboard steps and technologies, and documents the role of each source and support file.
