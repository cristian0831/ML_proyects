from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

RAW_MOVIES_CSV = RAW_DIR / "tmdb_5000_movies.csv"
RAW_CREDITS_CSV = RAW_DIR / "tmdb_5000_credits.csv"

MOVIES_PARQUET = PROCESSED_DIR / "movies.parquet"
VECTORIZER_PATH = PROCESSED_DIR / "vectorizer.joblib"
FEATURE_MATRIX_PATH = PROCESSED_DIR / "feature_matrix.npz"

TOP_CAST_COUNT = 5
DIRECTOR_WEIGHT = 3
