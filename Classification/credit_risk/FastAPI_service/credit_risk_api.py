from typing import Dict, Union

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from feature_engineering import add_derived_personal_features, feature_on_current_data

## ------------ load the trained artifact --------------------- #

artifact = joblib.load("models/credit_risk_artifact.joblib")
pipeline = artifact["pipeline"]
model = artifact["model"]
raw_input_columns = artifact["raw_input_columns"]

## ------------ initialize the API --------------------------- #

app = FastAPI()

## -------------- validate the input data --------------------- #

class PredictRequest(BaseModel):
    features: Dict[str, Union[float, int, str, None]]

## ---------------- adding a POST endpoint -------------------- #

@app.post("/predict")
def predict(request: PredictRequest):
    if not request.features:
        raise HTTPException(status_code=400, detail="features must not be empty")

    df = pd.DataFrame([request.features])

    try:
        df = add_derived_personal_features(df)
        df = feature_on_current_data(df)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"missing required feature: {exc}")

    df = df.reindex(columns=raw_input_columns)
    df = df.replace([np.inf, -np.inf], np.nan)

    X = pipeline.transform(df)
    probability = float(model.predict_proba(X)[0, 1])

    return {"probability": probability}
