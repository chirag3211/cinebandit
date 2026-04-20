"""
/feedback endpoint — receives user reaction and incrementally updates
the LinUCB model.
"""

import logging
from fastapi import APIRouter, HTTPException

from src.api.core.schemas import FeedbackRequest, FeedbackResponse, REACTION_TO_REWARD
from src.api.core.model_store import store
from src.api.core.metrics import (
    FEEDBACK_RECEIVED,
    CUMULATIVE_REWARD,
    API_ERRORS,
    ctr_tracker
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse, tags=["Inference"])
def feedback(request: FeedbackRequest):
    """
    Submit user feedback for a recommendation.

    This triggers an incremental LinUCB parameter update — the model
    learns from every interaction in real time. This is the core of
    the continuous learning loop.

    Rewards:
      - like    → 1.0
      - dislike → 0.0
      - skip    → 0.0
    """
    if not store.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    reward = REACTION_TO_REWARD[request.reaction]

    try:
        store.update(
            user_id=request.user_id,
            movie_id=request.movie_id,
            reward=reward
        )
    except ValueError as e:
        API_ERRORS.labels(endpoint="feedback", error_type="validation").inc()
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"[feedback] Unexpected error: {e}")
        API_ERRORS.labels(endpoint="feedback", error_type="internal").inc()
        raise HTTPException(status_code=500, detail="Internal server error")

    # Update Prometheus metrics
    FEEDBACK_RECEIVED.labels(reaction=request.reaction.value).inc()
    if reward > 0:
        CUMULATIVE_REWARD.inc(reward)

    rolling_ctr = ctr_tracker.record(reward)

    logger.info(
        f"[feedback] user={request.user_id} movie={request.movie_id} "
        f"reaction={request.reaction.value} reward={reward} "
        f"rolling_ctr={rolling_ctr:.3f}"
    )

    return FeedbackResponse(
        status="ok",
        user_id=request.user_id,
        movie_id=request.movie_id,
        reaction=request.reaction.value,
        reward=reward,
        rolling_ctr=round(rolling_ctr, 4),
        model_timestep=store.info["timesteps"]
    )
