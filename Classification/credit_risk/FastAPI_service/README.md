# Credit Risk API

A credit default risk model trained on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset and served through a FastAPI endpoint.

## Pipeline overview

```
data/*.csv  →  feature_engineering.py  →  train.py  →  models/credit_risk_artifact.joblib  →  credit_risk_api.py  →  score_applicants.py
```

1. **`feature_engineering.py`** — feature logic shared by training and serving.
2. **`train.py`** — builds the training table, fits the preprocessing pipeline and model, saves the artifact.
3. **`credit_risk_api.py`** — loads the artifact and exposes it as a `POST /predict` endpoint.
4. **`score_applicants.py`** — example client that reads a CSV/XLSX of applicants and calls the running API for each row.

## Features

### `add_derived_personal_features(df)`
Converts raw applicant fields into cleaner personal features:
- `DAYS_BIRTH` / `DAYS_EMPLOYED` → `AGE_YEARS`, `EMPLOYED_YEARS`
- Clips implausible values (`CNT_CHILDREN` > 10, `EMPLOYED_YEARS` > 100) to `NaN`

### `feature_on_current_data(df)`
Financial and credit-score ratios computed from the single application:
- `DEBT_BURDEN`, `PAYMENT_BURDEN`, `CREDIT_GOODS_RATIO`, `CREDIT_TO_AGE_RATIO`
- `EXT_SOURCE_MEAN` / `MIN` / `STD` / `PROD` / `AGE` / `EMPLOYED` (aggregates of the three external credit-bureau scores)
- `EMPLOYED_TO_AGE_RATIO`

These two functions are the only feature logic run **at prediction time** — see [Input data for `/predict`](#input-data-for-predict) below.

### `feature_on_bureau(bureau_df, bureau_balance_df)` — training only
Per-client summary of external bureau credit history: `DEBT_TO_CREDIT_RATIO`, `MAX_OVERDUE_DAYS`, `WORST_STATUS`.

### `feature_on_prev_home_credit(previous_df, pos_df, cc_df, inst_df)` — training only
Per-client summary of prior Home Credit loans: previous-approval rate, POS/cash balance behavior, credit card utilization, and installment payment lateness.

## Training (`train.py`)

- Reads `application_train.csv` plus the five auxiliary tables (`bureau`, `bureau_balance`, `previous_application`, `POS_CASH_balance`, `credit_card_balance`, `installments_payments`) with narrowed `dtype`/`usecols` to control memory use on the multi-hundred-MB files.
- Merges all engineered features onto the application table by `SK_ID_CURR`.
- Preprocessing: median imputation for numeric columns, most-frequent imputation + one-hot encoding for categoricals (`sklearn.ColumnTransformer`).
- Model: `XGBClassifier`, hyperparameters taken from an Optuna search (see `credit_risk_with_API.ipynb`), with `scale_pos_weight` set for class imbalance.
- Saves `models/credit_risk_artifact.joblib`, a dict of `{pipeline, model, raw_input_columns}`.

## Serving (`credit_risk_api.py`)

- Loads the artifact once at startup.
- `POST /predict` accepts a single applicant:
  ```json
  { "features": { "AMT_INCOME_TOTAL": 150000, "DAYS_BIRTH": -12000, "...": "..." } }
  ```
  and returns:
  ```json
  { "probability": 0.1732 }
  ```
- Request handling: runs `add_derived_personal_features` and `feature_on_current_data` on the input, reindexes to the training columns (missing columns become `NaN` and get imputed), then calls `pipeline.transform` + `model.predict_proba`.

### Input data for `/predict`

**By design, the API only uses the applicant's own application data** — the same fields found in `application_train.csv` / `application_test.csv` (income, credit amount, `EXT_SOURCE_*`, personal/housing info, etc.). It does **not** recompute the bureau- or previous-application-derived features (`DEBT_TO_CREDIT_RATIO`, `PREV_APPROVAL_RATE`, `CC_UTILITY_MEAN`, etc.); those are left for the imputer to fill with their training-set median/mode for every request. This is a deliberate simplification, not a bug — extending the API to accept bureau/previous-loan data is a possible future enhancement, not implemented here.

## How to run

From inside `credit_risk/`:

1. **Install dependencies** (no `requirements.txt` yet — install as needed):
   ```bash
   pip install pandas numpy scikit-learn xgboost joblib fastapi "uvicorn[standard]" pydantic requests openpyxl
   ```
2. **Train the model** (only needed once, or after changing features/hyperparameters):
   ```bash
   python train.py
   ```
   Reads everything from `data/`, prints validation ROC-AUC, writes `models/credit_risk_artifact.joblib`.
3. **Start the API server:**
   ```bash
   uvicorn credit_risk_api:app --reload
   ```
   Interactive docs available at `http://127.0.0.1:8000/docs`.
4. **Score a batch of applicants from a file** (with the server running):
   - Edit `INPUT_FILE` in `score_applicants.py` to point at your `.csv`/`.xlsx`.
   - Run:
     ```bash
     python score_applicants.py
     ```
   - Output written to `data/new_applicants_scored.csv` with a `PROBABILITY` column added.

## Directory structure

```
credit_risk/
├── data/                       # raw Home Credit CSVs (not all needed at inference)
├── models/
│   └── credit_risk_artifact.joblib
├── feature_engineering.py      # shared feature functions
├── train.py                    # training pipeline
├── credit_risk_api.py          # FastAPI serving app
├── score_applicants.py         # batch-scoring client script
└── README.md
```
