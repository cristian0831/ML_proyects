import gc
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from feature_engineering import (
    add_derived_personal_features,
    feature_on_bureau,
    feature_on_current_data,
    feature_on_prev_home_credit,
)

DATA_DIR = "data"
MODEL_DIR = "models"

# best hyperparameters found by the notebook's Optuna search (study.best_params)
BEST_XGB_PARAMS = {
    "n_estimators": 499,
    "max_depth": 4,
    "learning_rate": 0.045718654302699654,
    "subsample": 0.9404390394415453,
    "colsample_bytree": 0.8914700079542485,
}

# Only the columns each aggregation actually needs, with compact dtypes.
# bureau_balance/POS_CASH/credit_card/installments are 4M-27M rows each,
# so loading full width at default (object/int64/float64) dtypes is what
# was blowing past available RAM.
BUREAU_USECOLS = ["SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "AMT_CREDIT_SUM",
                   "AMT_CREDIT_SUM_DEBT", "CREDIT_DAY_OVERDUE"]
BUREAU_DTYPES = {"SK_ID_CURR": "int32", "SK_ID_BUREAU": "int32", "CREDIT_ACTIVE": "category",
                  "AMT_CREDIT_SUM": "float32", "AMT_CREDIT_SUM_DEBT": "float32",
                  "CREDIT_DAY_OVERDUE": "int32"}

BUREAU_BALANCE_USECOLS = ["SK_ID_BUREAU", "STATUS"]
BUREAU_BALANCE_DTYPES = {"SK_ID_BUREAU": "int32", "STATUS": "category"}

PREVIOUS_USECOLS = ["SK_ID_CURR", "SK_ID_PREV", "NAME_CONTRACT_STATUS"]
PREVIOUS_DTYPES = {"SK_ID_CURR": "int32", "SK_ID_PREV": "int32", "NAME_CONTRACT_STATUS": "category"}

POS_USECOLS = ["SK_ID_CURR", "MONTHS_BALANCE", "SK_DPD"]
POS_DTYPES = {"SK_ID_CURR": "int32", "MONTHS_BALANCE": "int32", "SK_DPD": "int32"}

CC_USECOLS = ["SK_ID_CURR", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"]
CC_DTYPES = {"SK_ID_CURR": "int32", "AMT_BALANCE": "float32", "AMT_CREDIT_LIMIT_ACTUAL": "float32"}

INSTALLMENTS_USECOLS = ["SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT",
                         "AMT_INSTALMENT", "AMT_PAYMENT"]
INSTALLMENTS_DTYPES = {"SK_ID_CURR": "int32", "DAYS_INSTALMENT": "float32",
                        "DAYS_ENTRY_PAYMENT": "float32", "AMT_INSTALMENT": "float32",
                        "AMT_PAYMENT": "float32"}


def read_csv(name, usecols, dtypes):
    return pd.read_csv(os.path.join(DATA_DIR, name), usecols=usecols, dtype=dtypes)


def compute_bureau_features():
    bureau_df = read_csv("bureau.csv", BUREAU_USECOLS, BUREAU_DTYPES)
    bureau_balance_df = read_csv("bureau_balance.csv", BUREAU_BALANCE_USECOLS, BUREAU_BALANCE_DTYPES)
    features = feature_on_bureau(bureau_df, bureau_balance_df)
    del bureau_df, bureau_balance_df
    gc.collect()
    return features


def compute_prev_features():
    previous_df = read_csv("previous_application.csv", PREVIOUS_USECOLS, PREVIOUS_DTYPES)
    pos_df = read_csv("POS_CASH_balance.csv", POS_USECOLS, POS_DTYPES)
    cc_df = read_csv("credit_card_balance.csv", CC_USECOLS, CC_DTYPES)
    inst_df = read_csv("installments_payments.csv", INSTALLMENTS_USECOLS, INSTALLMENTS_DTYPES)
    features = feature_on_prev_home_credit(previous_df, pos_df, cc_df, inst_df)
    del previous_df, pos_df, cc_df, inst_df
    gc.collect()
    return features


def build_train_df():
    application_train_df = pd.read_csv(os.path.join(DATA_DIR, "application_train.csv"))
    application_train_df = add_derived_personal_features(application_train_df)
    train_df = feature_on_current_data(application_train_df)
    del application_train_df

    bureau_features = compute_bureau_features()
    train_df = train_df.merge(bureau_features, on="SK_ID_CURR", how="left")
    del bureau_features

    prev_features = compute_prev_features()
    train_df = train_df.merge(prev_features, on="SK_ID_CURR", how="left")
    del prev_features
    gc.collect()

    return train_df


def build_pipeline(X_train):
    num_attribs = X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_attribs = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("num", num_pipeline, num_attribs),
            ("cat", cat_pipeline, cat_attribs),
        ]
    )


def main():
    train_df = build_train_df()

    X = train_df.drop(columns=["TARGET"]).replace([np.inf, -np.inf], np.nan)
    y = train_df["TARGET"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline(X_train)
    train_prepared = pipeline.fit_transform(X_train)
    val_prepared = pipeline.transform(X_val)

    model = XGBClassifier(
        **BEST_XGB_PARAMS,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=42,
        eval_metric="auc",
        n_jobs=-1,
    )
    model.fit(train_prepared, y_train)

    val_auc = roc_auc_score(y_val, model.predict_proba(val_prepared)[:, 1])
    print(f"Validation ROC-AUC: {val_auc:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    artifact_path = os.path.join(MODEL_DIR, "credit_risk_artifact.joblib")
    joblib.dump(
        {
            "pipeline": pipeline,
            "model": model,
            "raw_input_columns": X_train.columns.tolist(),
        },
        artifact_path,
    )
    print(f"Saved artifact to {artifact_path}")


if __name__ == "__main__":
    main()
