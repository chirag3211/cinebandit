"""
DVC Stage 2: Model Training
Loads processed data, trains LinUCB, saves model and metrics.
"""

import yaml
import json
import pickle
import logging
import time
from pathlib import Path

import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.bandit.simulator import run_simulation
from src.evaluation.metrics import cumulative_ctr, recommendation_diversity, arm_coverage

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Load params ───────────────────────────────────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

ALPHA      = params["train"]["alpha"]
POOL_SIZE  = params["train"]["pool_size"]
SEED       = params["train"]["seed"]
MAX_STEPS  = params["train"]["max_steps"]
PROCESSED  = Path("data/processed")
OUTPUTS    = Path("outputs")
METRICS    = Path("metrics")

OUTPUTS.mkdir(exist_ok=True)
METRICS.mkdir(exist_ok=True)


def main():
    logger.info("=== Stage 2: Training ===")
    logger.info(f"  alpha={ALPHA} pool_size={POOL_SIZE} seed={SEED}")

    # ── Load processed data ───────────────────────────────────────────────────
    logger.info("Loading processed data...")
    train_ratings = pd.read_parquet(PROCESSED / "train_ratings.parquet")
    movies        = pd.read_parquet(PROCESSED / "movies.parquet")
    users         = pd.read_parquet(PROCESSED / "users.parquet")
    profiles      = pd.read_parquet(PROCESSED / "profiles.parquet")

    # ── Run simulation ────────────────────────────────────────────────────────
    logger.info("Running training simulation...")
    t0 = time.time()
    model, train_results = run_simulation(
        ratings=train_ratings, movies=movies,
        users=users, profiles=profiles,
        alpha=ALPHA, pool_size=POOL_SIZE,
        max_steps=MAX_STEPS, seed=SEED, verbose=True
    )
    train_time = time.time() - t0

    # ── Metrics ───────────────────────────────────────────────────────────────
    train_ctr  = float(train_results["reward"].mean()) if not train_results.empty else 0.0
    diversity  = recommendation_diversity(train_results, movies)
    coverage   = arm_coverage(model, len(movies))
    match_rate = len(train_results) / len(train_ratings)

    metrics = {
        "train_ctr":       round(train_ctr, 4),
        "genre_diversity": round(diversity, 4),
        "arm_coverage":    round(coverage, 4),
        "match_rate":      round(match_rate, 4),
        "train_steps":     len(train_results),
        "model_timesteps": model.t,
        "train_time_s":    round(train_time, 2),
        "alpha":           ALPHA,
    }

    with open(METRICS / "train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"  train_ctr={train_ctr:.4f} diversity={diversity:.3f} "
                f"coverage={coverage:.1%} time={train_time:.1f}s")

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = OUTPUTS / f"linucb_alpha{ALPHA}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"  Model saved to {model_path}")

    logger.info("=== Stage 2 complete ✅ ===")


if __name__ == "__main__":
    main()