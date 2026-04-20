"""
CineBandit FastAPI application.

Entry point for the model serving layer. Wires together:
  - Lifespan (startup model load)
  - Route registration
  - Prometheus /metrics endpoint
  - CORS middleware
  - Structured logging
  - Global exception handler
"""

import os
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from src.api.core.model_store import store
from src.api.core.metrics import MODEL_LOAD_TIME, MODEL_VERSION, MODEL_ALPHA
from src.api.routes import recommend, feedback, system

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ── Model path from environment (overridden in Docker) ────────────────────────

MODEL_PATH = Path(
    os.environ.get("MODEL_PATH", "outputs/linucb_alpha0.1.pkl")
)


# ── Lifespan (replaces deprecated @app.on_event) ─────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    logger.info("=" * 50)
    logger.info("  CineBandit API starting up")
    logger.info("=" * 50)

    t0 = time.perf_counter()
    store.load(MODEL_PATH)
    load_time = time.perf_counter() - t0

    MODEL_LOAD_TIME.set(load_time)
    if store.is_ready:
        info = store.info
        MODEL_VERSION.set(info.get("timesteps", 0))
        MODEL_ALPHA.set(info.get("alpha", 0))
        logger.info(f"  Startup complete in {load_time:.2f}s")
    else:
        logger.warning("  Started without a loaded model — /ready will return 503")

    yield

    logger.info("CineBandit API shutting down.")


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="CineBandit API",
    description=(
        "Adaptive movie recommendation system powered by LinUCB. "
        "Serves recommendations, accepts feedback, and exposes monitoring endpoints."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Allow Streamlit frontend (any origin in dev, tighten in prod)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Prometheus metrics endpoint ───────────────────────────────────────────────

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(recommend.router)
app.include_router(feedback.router)
app.include_router(system.router)

# ── Global exception handler ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# ── Root ──────────────────────────────────────────────────────────────────────

@app.get("/", tags=["System"])
def root():
    return {
        "service": "CineBandit API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
        "metrics": "/metrics"
    }
