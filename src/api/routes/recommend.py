"""
/recommend endpoint — returns top-N movie recommendations for a user.
"""

import time
import logging
from fastapi import APIRouter, HTTPException

from src.api.core.schemas import RecommendRequest, RecommendResponse
from src.api.core.model_store import store
from src.api.core.metrics import (
    RECOMMENDATION_REQUESTS,
    RECOMMENDATION_LATENCY,
    API_ERRORS
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/recommend", response_model=RecommendResponse, tags=["Inference"])
def recommend(request: RecommendRequest):
    """
    Get top-N movie recommendations for a user.

    The model scores all candidate movies using LinUCB's UCB formula
    and returns the N highest-scoring ones the user hasn't seen.
    """
    if not store.is_ready:
        RECOMMENDATION_REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    t0 = time.perf_counter()

    try:
        recommendations = store.recommend(
            user_id=request.user_id,
            n=request.n,
            exclude=request.exclude or []
        )
    except ValueError as e:
        API_ERRORS.labels(endpoint="recommend", error_type="validation").inc()
        RECOMMENDATION_REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"[recommend] Unexpected error: {e}")
        API_ERRORS.labels(endpoint="recommend", error_type="internal").inc()
        RECOMMENDATION_REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=500, detail="Internal server error")

    latency = time.perf_counter() - t0
    RECOMMENDATION_LATENCY.observe(latency)
    RECOMMENDATION_REQUESTS.labels(status="success").inc()

    return RecommendResponse(
        user_id=request.user_id,
        recommendations=recommendations,
        model_timestep=store.info["timesteps"]
    )
