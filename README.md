# Automated Data Preprocessing and EDA

A local Streamlit application for reviewing, cleaning, preprocessing, and exploring CSV, Excel, and JSON datasets. Uploaded source files are read in memory and never overwritten. The pipeline does not guess whether separate files should be joined: you choose row concatenation or a keyed merge and inspect a report of that choice.

## Architecture and workflow

```text
app.py (Streamlit UI and session state)
  ├── file_handler       read uploads; normalize duplicate column labels
  ├── data_inspector     shapes, types, missing values, common fields
  ├── data_merger        explicit concatenate or keyed joins
  ├── data_cleaner       selectable, copy-based cleaning operations
  ├── data_preprocessor sklearn transforms and optional train/test split
  ├── eda                statistics and interactive Plotly figures
  ├── skewness_analyzer  skew classification and suggestions
  ├── outlier_detector   IQR and Z-score counts; no automatic removal
  └── report_generator   CSV, Excel, and HTML exports
```

The normal workflow is **upload → inspect → integrate (if needed) → clean → preview preprocessing → explore → export**. Each stage operates on a working DataFrame in the current browser session. Nothing is written to the `data/raw` or `data/processed` folders automatically; downloads are initiated by the user.

Preprocessing is opt-in. Scaling defaults to `none`; log transforms require non-negative numeric data; categorical encoding can use one-hot or ordinal encoding. If a train/test split is selected, the sklearn imputer, encoder, and scaler are fitted on training features only. Without a split, the transformed result is for exploration/export—not a leakage-safe model training dataset.

## Setup (Windows PowerShell)

Python 3.10 or later is recommended.

```powershell
cd C:\Users\admin\OneDrive\Desktop\data_preprocessing
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. If PowerShell blocks activation, run the venv interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Legacy `.xls` files use `xlrd`; `.xlsx` exports use `openpyxl`.

## Tests

Run the standard-library tests without installing a test framework:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests cover file immutability, duplicate column labels, explicit integration, copy-based cleaning, type validation, outlier/skew reporting, datetime feature extraction, and train-only preprocessing fit.

## Notes and limits

- Merging can multiply rows when a selected key is duplicated on both sides; inspect key uniqueness and the resulting dimensions before continuing.
- Concatenation aligns columns by name and leaves absent cells missing.
- One-hot encoding can increase the feature count for high-cardinality columns. Select appropriate categorical columns / encoding for the dataset size.
- Outlier counts and skew recommendations are diagnostic only. They do not remove or alter extreme values.
- Processing history is held in Streamlit session state and is not a permanent database.
- Excel output and large one-hot results are limited by available memory.
