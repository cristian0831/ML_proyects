from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request

from app.recommender import Recommender
from app.schemas import RecommendationRequest, RecommendationResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.recommender = Recommender()
    yield


app = FastAPI(title="Movie Recommendation API", lifespan=lifespan)


def get_recommender(request: Request) -> Recommender:
    return request.app.state.recommender


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/recommendations", response_model=RecommendationResponse)
def recommendations(
    payload: RecommendationRequest,
    recommender: Recommender = Depends(get_recommender),
) -> RecommendationResponse:
    results = recommender.recommend(
        genres=payload.genres,
        actors=payload.actors,
        director=payload.director,
        top_n=payload.top_n,
    )
    return RecommendationResponse(results=results, count=len(results))
