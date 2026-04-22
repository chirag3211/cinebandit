"""
System routes:
  GET  /health          — liveness probe (always 200 if server is up)
  GET  /ready           — readiness probe (200 only if model is loaded)
  GET  /model/info      — current model metadata
  GET  /drift/status    — current drift score and alert state
  POST /retrain/trigger — manually trigger retraining
  GET  /snapshot        — quick metrics snapshot for Streamlit dashboard
"""

import logging
from fastapi import APIRouter, HTTPException

from src.api.core.schemas import (
    HealthResponse, ReadyResponse, ModelInfoResponse,
    DriftStatusResponse, RetrainRequest, RetrainResponse,
    MetricsSnapshotResponse
)
from src.api.core.model_store import store
from src.api.core.metrics import (
    RETRAIN_RUNS, DRIFT_SCORE, ctr_tracker
)

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory drift score — updated by the drift detection DAG via /drift/update
_drift_score: float = 0.0
_drift_threshold: float = 0.3


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Liveness probe — returns 200 if the server process is running."""
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse, tags=["System"])
def ready():
    """
    Readiness probe — returns 200 only if a model is loaded and ready
    to serve recommendations. Used by Docker health checks and Airflow.
    """
    if not store.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    return ReadyResponse(
        status="ok",
        model_loaded=True,
        model_timestep=store.info.get("timesteps")
    )


@router.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    """Return metadata about the currently loaded model."""
    if not store.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")

    info = store.info
    return ModelInfoResponse(
        algorithm=info.get("algorithm", "LinUCB"),
        alpha=info.get("alpha", 0.0),
        timesteps=info.get("timesteps", 0),
        arms_seen=info.get("arms_seen", 0),
        total_movies=info.get("total_movies", 0),
        mean_ctr=info.get("mean_ctr"),
        model_path=info.get("model_path")
    )


@router.get("/drift/status", response_model=DriftStatusResponse, tags=["Monitoring"])
def drift_status():
    """
    Return the current data drift score and whether the alert threshold
    has been exceeded. Updated periodically by the Airflow drift DAG.
    """
    alert = _drift_score > _drift_threshold
    return DriftStatusResponse(
        drift_score=round(_drift_score, 4),
        threshold=_drift_threshold,
        alert=alert,
        status="drift_detected" if alert else "ok"
    )


@router.post("/drift/update", tags=["Monitoring"])
def drift_update(score: float):
    """
    Internal endpoint — called by the Airflow drift detection DAG
    to push the latest KL divergence score into the API.
    """
    global _drift_score
    _drift_score = score
    DRIFT_SCORE.set(score)
    logger.info(f"[drift] Score updated: {score:.4f} "
                f"(alert={'YES' if score > _drift_threshold else 'NO'})")
    return {"status": "ok", "drift_score": score}


@router.post("/retrain/trigger", response_model=RetrainResponse, tags=["Model"])
def retrain_trigger(request: RetrainRequest):
    """
    Manually trigger a retraining pipeline run.
    In production this calls the Airflow REST API to trigger the
    retraining DAG. For now it logs the request and returns a response.
    """
    RETRAIN_RUNS.labels(status="triggered").inc()
    logger.info(f"[retrain] Trigger requested. Reason: {request.reason}")

    # TODO (Week 3): Replace with Airflow REST API call:
    # requests.post("http://airflow:8080/api/v1/dags/retrain/dagRuns",
    #               json={"conf": {"reason": request.reason}})

    return RetrainResponse(
        status="triggered",
        reason=request.reason,
        message="Retraining pipeline triggered. "
                "Check Airflow UI for progress."
    )


@router.post("/model/reload", tags=["Model"])
def model_reload():
    """
    Hot-reload the model from disk without restarting the server.
    Called by the Airflow retraining DAG after a new model is saved.
    """
    success = store.reload()
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Model reload failed — check logs"
        )
    # Update Prometheus gauges after reload
    from src.api.core.metrics import MODEL_VERSION, MODEL_ALPHA, ARMS_SEEN
    info = store.info
    MODEL_VERSION.set(info.get("timesteps", 0))
    MODEL_ALPHA.set(info.get("alpha", 0))
    ARMS_SEEN.set(info.get("arms_seen", 0))

    return {"status": "ok", "message": "Model reloaded successfully",
            "timesteps": store.info.get("timesteps")}


@router.get("/snapshot", response_model=MetricsSnapshotResponse, tags=["Monitoring"])
def snapshot():
    """
    Quick metrics snapshot for the Streamlit dashboard.
    Returns the most important live metrics in a single call.
    """
    info = store.info if store.is_ready else {}
    return MetricsSnapshotResponse(
        rolling_ctr=round(ctr_tracker.current_ctr, 4),
        model_timestep=info.get("timesteps", 0),
        arms_seen=info.get("arms_seen", 0),
        drift_score=round(_drift_score, 4)
    )