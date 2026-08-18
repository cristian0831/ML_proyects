from fastapi.testclient import TestClient

from app.main import app, get_recommender

FAKE_MOVIES = [
    {
        "title": "Fake Action Movie",
        "genres": ["Action", "Adventure"],
        "director": "Jane Doe",
        "cast": ["Actor One", "Actor Two"],
        "vote_average": 7.5,
        "release_year": 2010,
        "similarity_score": 0.85,
    },
    {
        "title": "Fake Drama Movie",
        "genres": ["Drama"],
        "director": "John Smith",
        "cast": ["Actor Three"],
        "vote_average": 6.2,
        "release_year": 2015,
        "similarity_score": 0.40,
    },
]


class FakeRecommender:
    def recommend(self, genres, actors, director, top_n):
        return FAKE_MOVIES[:top_n]


# Not entering TestClient as a context manager, so the app's lifespan (which
# loads the real dataset artifacts) never runs; the endpoint only ever sees
# the overridden fake recommender below.
app.dependency_overrides[get_recommender] = lambda: FakeRecommender()
client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommendations_valid_request():
    response = client.post("/recommendations", json={"genres": ["Action"], "top_n": 1})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["results"][0]["title"] == "Fake Action Movie"


def test_recommendations_defaults_to_empty_query():
    response = client.post("/recommendations", json={})
    assert response.status_code == 200
    assert response.json()["count"] == len(FAKE_MOVIES)


def test_recommendations_rejects_top_n_too_low():
    response = client.post("/recommendations", json={"top_n": 0})
    assert response.status_code == 422


def test_recommendations_rejects_top_n_too_high():
    response = client.post("/recommendations", json={"top_n": 100})
    assert response.status_code == 422
