# Movie Recommendation API

A content-based movie recommendation API built with FastAPI. Users submit
preferred genres, actors, and/or a director as JSON; the API returns ranked
movie recommendations using cosine similarity over a vectorized feature
corpus built from the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset Description](#dataset-description)
- [Goal of the Project](#goal-of-the-project)
- [Project Structure](#project-structure)
- [Features — Model Training Methodology](#features--model-training-methodology)
- [How to Run](#how-to-run)
- [Tests](#tests)

## Problem Statement

Choosing what to watch next is hard when the options are framed only as "users
who liked X also liked Y" — that requires a large history of other users'
ratings, which a small or cold-start project doesn't have. This project takes
a different angle: a user describes what they *want* directly (genres, actors,
a director) and the API finds movies whose own content — genre, cast,
director, plot keywords — matches that description most closely. No user
accounts, no rating history, no collaborative signal required; recommendations
are derived purely from the attributes of the movies themselves.

## Dataset Description

The [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
(Kaggle) ships as two CSV files, joined on movie ID:

**`tmdb_5000_movies.csv`** — one row per movie, 20 columns. The ones this
project uses: `id`, `title`, `genres` (JSON-string list of `{id, name}`),
`keywords` (JSON-string list of `{id, name}`), `vote_average`, `vote_count`,
`release_date`.

**`tmdb_5000_credits.csv`** — `movie_id`, `title`, `cast` (JSON-string list of
cast members with `name`, ordered by billing), `crew` (JSON-string list of
crew members with `name` and `job`, used to find the entry where
`job == "Director"`).

Neither file is checked into the repo (`.gitignore` excludes
`data/raw/*.csv`) — download them manually and place them in `data/raw/`, as
described in [How to Run](#how-to-run).

## Goal of the Project

Build a working, end-to-end content-based recommender exposed as a typed HTTP
API:

- Preprocess raw, messy CSV data (JSON-encoded columns) into a clean feature
  representation for each movie.
- Train a vectorization model (`CountVectorizer`) over those features once,
  offline, and persist it.
- Serve recommendations at request time by projecting a user's free-form
  query into the same feature space and ranking by similarity — with a
  sensible fallback (weighted rating) when the user gives no preferences.
- Validate all input/output through Pydantic models so the API is
  self-documenting (`/docs`) and rejects malformed requests automatically.

## Project Structure

```
Movie_recommendation/
├── data/
│   ├── raw/                     # user-downloaded tmdb_5000_movies.csv, tmdb_5000_credits.csv
│   └── processed/                # generated artifacts (git-ignored)
│       ├── movies.parquet        # cleaned per-movie table
│       ├── vectorizer.joblib     # fitted CountVectorizer
│       └── feature_matrix.npz    # sparse bag-of-words matrix, one row per movie
├── app/
│   ├── main.py                   # FastAPI app, lifespan startup, endpoints
│   ├── schemas.py                # Pydantic request/response models
│   ├── recommender.py            # loads artifacts, ranks movies against a query
│   ├── features.py               # shared feature extraction/normalization (used by both
│   │                              #   build_dataset.py and recommender.py, to keep training-time
│   │                              #   and query-time vectors consistent)
│   └── config.py                 # file paths / constants
├── scripts/
│   └── build_dataset.py          # one-time: raw CSVs -> processed artifacts
├── tests/
│   └── test_api.py               # FastAPI TestClient tests (dataset-free, via dependency override)
├── requirements.txt
├── .gitignore
└── README.md
```

## Features — Model Training Methodology

This is content-based filtering, not collaborative filtering: there's no
user-item ratings matrix, only movie attributes.

**1. Extraction** (`app/features.py`, applied in `scripts/build_dataset.py`)

For each movie, `genres`/`keywords`/`cast`/`crew` are parsed out of their raw
JSON-string form (`ast.literal_eval`), and reduced to:
- all genre names
- the top 5 billed cast members (`TOP_CAST_COUNT` in `app/config.py`)
- the director (the crew member whose `job` is `"Director"`)

**2. Normalization**

Every token (genre, actor name, director name) has its internal spaces
stripped and is lowercased — e.g. `"Tom Hardy"` → `"tomhardy"`. This prevents
`CountVectorizer`'s default whitespace tokenizer from splitting multi-word
names into separate words, which would otherwise cause false matches (e.g.
"Tom" from "Tom Hardy" colliding with "Tom" from "Tom Hanks").

**3. The "soup"**

All of a movie's normalized tokens are concatenated into one space-joined
string — genres + cast + director (repeated `DIRECTOR_WEIGHT = 3` times to
weight it more heavily than a single cast member) + keywords. This single
string is what gets vectorized, so one `CountVectorizer` encodes every
feature type at once.

**4. Vectorization**

`CountVectorizer(stop_words="english")` is `fit_transform`-ed once over every
movie's soup, producing a sparse bag-of-words matrix (rows = movies, columns
= vocabulary terms, values = token counts). Both the fitted vectorizer and
the resulting matrix are persisted (`vectorizer.joblib`, `feature_matrix.npz`)
so this training step never has to re-run at request time.

**5. Query-time scoring** (`app/recommender.py`)

A user's `genres`/`actors`/`director` are normalized and built into a soup
the same way, then passed through the *already-fitted* vectorizer's
`.transform()` (never `.fit_transform()` — the vocabulary must match
training). `cosine_similarity` between that query vector and every row of
the feature matrix gives a similarity score per movie; the top `top_n` are
returned. If the query is empty, movies are ranked instead by an IMDB-style
weighted rating over `vote_average`/`vote_count`.

## How to Run

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
into `data/processed/`. Re-run this only if the raw CSVs or the feature
extraction logic in `app/features.py` changes.

### 4. Run the API

```bash
uvicorn app.main:app --reload
```

The API is now available at http://127.0.0.1:8000 (interactive docs at
`/docs`).

### Example request

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
