import json

import pandas as pd
import requests

API_URL = "http://127.0.0.1:8000/predict"
INPUT_FILE = "data/new_applicants.xlsx"  # change to your .csv/.xlsx path
OUTPUT_FILE = "data/new_applicants_scored.csv"


def load_applicants(path):
    if path.endswith(".xlsx"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def score_applicants(df):
    records = json.loads(df.to_json(orient="records"))
    probabilities = []
    for record in records:
        response = requests.post(API_URL, json={"features": record})
        response.raise_for_status()
        probabilities.append(response.json()["probability"])
    return probabilities


def main():
    df = load_applicants(INPUT_FILE)
    df["PROBABILITY"] = score_applicants(df)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved scored applicants to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
