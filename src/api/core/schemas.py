"""
Pydantic schemas for CineBandit API request and response validation.
All I/O is strictly typed — FastAPI validates incoming JSON automatically.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from enum import Enum


# ── Enums ─────────────────────────────────────────────────────────────────────

class Reaction(str, Enum):
    like    = "like"
    dislike = "dislike"
    skip    = "skip"


# ── Reward mapping ────────────────────────────────────────────────────────────

REACTION_TO_REWARD = {
    Reaction.like:    1.0,
    Reaction.dislike: 0.0,
    Reaction.skip:    0.0,
}


# ── Request schemas ───────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    user_id: int = Field(..., gt=0, description="User ID (1-943 for ML-100K)")
    n: int = Field(5, ge=1, le=20, description="Number of recommendations")
    exclude: Optional[List[int]] = Field(
        default=None,
        description="Movie IDs to exclude (already seen)"
    )

    model_config = {"json_schema_extra": {
        "example": {"user_id": 1, "n": 5, "exclude": []}
    }}


class FeedbackRequest(BaseModel):
    user_id: int = Field(..., gt=0, description="User ID")
    movie_id: int = Field(..., gt=0, description="Movie ID that was recommended")
    reaction: Reaction = Field(..., description="User reaction: like/dislike/skip")

    model_config = {"json_schema_extra": {
        "example": {"user_id": 1, "movie_id": 50, "reaction": "like"}
    }}


class RetrainRequest(BaseModel):
    reason: str = Field(
        "manual",
        description="Reason for triggering retraining"
    )


# ── Response schemas ──────────────────────────────────────────────────────────

class MovieRecommendation(BaseModel):
    movie_id: int
    title: str
    genres: List[str]
    year: int
    ucb_score: float = Field(description="UCB score (higher = more confident)")


class RecommendResponse(BaseModel):
    user_id: int
    recommendations: List[MovieRecommendation]
    model_timestep: int = Field(description="Model version at time of request")


class FeedbackResponse(BaseModel):
    status: str
    user_id: int
    movie_id: int
    reaction: str
    reward: float
    rolling_ctr: float = Field(description="Rolling CTR over last 100 events")
    model_timestep: int


class HealthResponse(BaseModel):
    status: str   # "ok"


class ReadyResponse(BaseModel):
    status: str       # "ok" or "not_ready"
    model_loaded: bool
    model_timestep: Optional[int] = None


class ModelInfoResponse(BaseModel):
    algorithm: str
    alpha: float
    timesteps: int
    arms_seen: int
    total_movies: int
    mean_ctr: Optional[float] = None
    model_path: Optional[str] = None


class DriftStatusResponse(BaseModel):
    drift_score: float
    threshold: float
    alert: bool
    status: str   # "ok" or "drift_detected"


class RetrainResponse(BaseModel):
    status: str
    reason: str
    message: str


class MetricsSnapshotResponse(BaseModel):
    rolling_ctr: float
    model_timestep: int
    arms_seen: int
    drift_score: float
