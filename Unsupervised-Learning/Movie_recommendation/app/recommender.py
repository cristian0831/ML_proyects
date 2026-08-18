import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from app.config import DIRECTOR_WEIGHT, FEATURE_MATRIX_PATH, MOVIES_PARQUET, VECTORIZER_PATH
from app.features import build_soup


class Recommender:
    """Content-based recommender: matches a free-form genre/actor/director
    query against a pre-vectorized movie corpus via cosine similarity."""

    def __init__(self) -> None:
        self.movies = pd.read_parquet(MOVIES_PARQUET)
        self.vectorizer = joblib.load(VECTORIZER_PATH)
        self.feature_matrix = sparse.load_npz(FEATURE_MATRIX_PATH)
        self._weighted_rating = self._compute_weighted_rating()

    def _compute_weighted_rating(self) -> pd.Series:
        """IMDB-style weighted rating, used as the ranking fallback when the
        caller supplies no genres/actors/director to match against."""
        v = self.movies["vote_count"]
        r = self.movies["vote_average"]
        m = v.quantile(0.60)
        c = r.mean()
        return (v / (v + m)) * r + (m / (v + m)) * c

    def recommend(
        self,
        genres: list[str],
        actors: list[str],
        director: str | None,
        top_n: int,
    ) -> list[dict]:
        has_query = bool(genres or actors or director)

        if has_query:
            soup = build_soup(genres, actors, director, director_weight=DIRECTOR_WEIGHT)
            query_vector = self.vectorizer.transform([soup])
            scores = cosine_similarity(query_vector, self.feature_matrix).ravel()
            order = np.argsort(-scores)[:top_n]
        else:
            scores = np.zeros(len(self.movies))
            order = np.argsort(-self._weighted_rating.to_numpy())[:top_n]

        results = []
        for idx in order:
            row = self.movies.iloc[idx]
            release_year = row["release_year"]
            results.append(
                {
                    "title": row["title"],
                    "genres": list(row["genres"]),
                    "director": row["director"],
                    "cast": list(row["cast"]),
                    "vote_average": float(row["vote_average"]),
                    "release_year": None if pd.isna(release_year) else int(release_year),
                    "similarity_score": float(scores[idx]),
                }
            )
        return results
