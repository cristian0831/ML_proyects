from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    genres: list[str] = Field(default_factory=list)
    actors: list[str] = Field(default_factory=list)
    director: str | None = None
    top_n: int = Field(default=10, ge=1, le=50)


class MovieRecommendation(BaseModel):
    title: str
    genres: list[str]
    director: str | None
    cast: list[str]
    vote_average: float
    release_year: int | None
    similarity_score: float


class RecommendationResponse(BaseModel):
    results: list[MovieRecommendation]
    count: int
