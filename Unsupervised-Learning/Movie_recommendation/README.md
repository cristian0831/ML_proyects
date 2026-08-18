# Movie Recommendation API

A content-based movie recommendation API built with FastAPI. Users submit
preferred genres, actors, and/or a director; the API returns ranked movie
recommendations using cosine similarity over a vectorized feature corpus
built from the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).

## How it works

- `scripts/build_dataset.py` is a one-time offline step: it merges the raw
  TMDB movies/credits CSVs, extracts genres/cast/director/keywords, builds a
  per-movie "feature soup", and fits a `CountVectorizer` over the corpus.
  The processed movie table, fitted vectorizer, and feature matrix are saved
  to `data/processed/`.
- The FastAPI app loads those pre-built artifacts once at startup. Each
  request builds a query vector from the user's genres/actors/director using
  the *same* fitted vectorizer, scores it against the corpus with cosine
  similarity, and returns the top matches. If no preferences are given, it
  falls back to an IMDB-style weighted rating.

## Setup

### 1. Get the dataset

Download the two TMDB 5000 CSVs from Kaggle (free account required):

https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

Place both files here:

```
data/raw/tmdb_5000_movies.csv
data/raw/tmdb_5000_credits.csv
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Build the processed dataset

```bash
python scripts/build_dataset.py
```

This writes `movies.parquet`, `vectorizer.joblib`, and `feature_matrix.npz`
into `data/processed/`.

### 4. Run the API

```bash
uvicorn app.main:app --reload
```

The API is now available at http://127.0.0.1:8000 (interactive docs at
`/docs`).

## Example request

```bash
curl -X POST http://127.0.0.1:8000/recommendations \
  -H "Content-Type: application/json" \
  -d '{"genres": ["Action", "Science Fiction"], "actors": ["Tom Hardy"], "top_n": 5}'
```

```json
{
  "results": [
    {
      "title": "...",
      "genres": ["Action", "Science Fiction"],
      "director": "...",
      "cast": ["...", "..."],
      "vote_average": 7.8,
      "release_year": 2015,
      "similarity_score": 0.63
    }
  ],
  "count": 5
}
```

## Tests

The test suite mocks the recommender via FastAPI dependency overrides, so it
runs without the dataset present:

```bash
pytest
```
