"""One-time preprocessing: raw TMDB 5000 CSVs -> processed artifacts for the API.

Usage: python scripts/build_dataset.py
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import (
    DIRECTOR_WEIGHT,
    FEATURE_MATRIX_PATH,
    MOVIES_PARQUET,
    PROCESSED_DIR,
    RAW_CREDITS_CSV,
    RAW_MOVIES_CSV,
    TOP_CAST_COUNT,
    VECTORIZER_PATH,
)
from app.features import build_soup, extract_director, extract_top_cast, parse_names


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not RAW_MOVIES_CSV.exists() or not RAW_CREDITS_CSV.exists():
        raise FileNotFoundError(
            "Missing raw TMDB CSVs. Download tmdb_5000_movies.csv and "
            f"tmdb_5000_credits.csv into {RAW_MOVIES_CSV.parent} "
            "(https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)."
        )
    return pd.read_csv(RAW_MOVIES_CSV), pd.read_csv(RAW_CREDITS_CSV)


def merge_datasets(movies: pd.DataFrame, credits: pd.DataFrame) -> pd.DataFrame:
    credits = credits.rename(columns={"movie_id": "id"})[["id", "cast", "crew"]]
    return movies.merge(credits, on="id", how="inner")


def build_processed_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["genres"] = df["genres"].apply(parse_names)
    df["keywords"] = df["keywords"].apply(parse_names)
    df["cast"] = df["cast"].apply(lambda s: extract_top_cast(s, TOP_CAST_COUNT))
    df["director"] = df["crew"].apply(extract_director)
    df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year

    df["soup"] = df.apply(
        lambda row: build_soup(
            row["genres"], row["cast"], row["director"], row["keywords"], DIRECTOR_WEIGHT
        ),
        axis=1,
    )

    return df[
        [
            "id",
            "title",
            "genres",
            "cast",
            "director",
            "vote_average",
            "vote_count",
            "release_year",
            "soup",
        ]
    ].reset_index(drop=True)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    movies, credits = load_raw()
    merged = merge_datasets(movies, credits)
    processed = build_processed_table(merged)

    vectorizer = CountVectorizer(stop_words="english")
    feature_matrix = vectorizer.fit_transform(processed["soup"])

    processed.drop(columns=["soup"]).to_parquet(MOVIES_PARQUET, index=False)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    sparse.save_npz(FEATURE_MATRIX_PATH, feature_matrix)

    print(f"Processed {len(processed)} movies.")
    print(f"Feature vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Saved: {MOVIES_PARQUET}\n       {VECTORIZER_PATH}\n       {FEATURE_MATRIX_PATH}")


if __name__ == "__main__":
    main()
