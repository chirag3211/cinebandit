"""
DVC Stage 3: Evaluation
Loads trained model and test data, computes evaluation metrics and plots.
"""

import yaml
import json
import pickle
import logging
from pathlib import Path

import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.bandit.simulator import run_simulation
from src.evaluation.metrics import (
    print_summary, plot_results,
    recommendation_diversity, arm_coverage, cumulative_ctr
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Load params ───────────────────────────────────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

ALPHA     = params["train"]["alpha"]
POOL_SIZE = params["train"]["pool_size"]
SEED      = params["train"]["seed"]
PROCESSED = Path("data/processed")
OUTPUTS   = Path("outputs")
METRICS   = Path("metrics")
PLOTS     = Path("plots")

METRICS.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)


def main():
    logger.info("=== Stage 3: Evaluation ===")

    # ── Load model ────────────────────────────────────────────────────────────
    model_path = OUTPUTS / f"linucb_alpha{ALPHA}.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"  Loaded model from {model_path} (t={model.t:,})")

    # ── Load test data ────────────────────────────────────────────────────────
    test_ratings = pd.read_parquet(PROCESSED / "test_ratings.parquet")
    movies       = pd.read_parquet(PROCESSED / "movies.parquet")
    users        = pd.read_parquet(PROCESSED / "users.parquet")
    profiles     = pd.read_parquet(PROCESSED / "profiles.parquet")

    # ── Run test simulation ───────────────────────────────────────────────────
    logger.info("Running test simulation...")
    _, test_results = run_simulation(
        ratings=test_ratings, movies=movies,
        users=users, profiles=profiles,
        alpha=ALPHA, pool_size=POOL_SIZE,
        seed=SEED + 1, verbose=True
    )

    # ── Compute metrics ───────────────────────────────────────────────────────
    test_ctr   = float(test_results["reward"].mean()) if not test_results.empty else 0.0
    diversity  = recommendation_diversity(test_results, movies)
    coverage   = arm_coverage(model, len(movies))

    eval_metrics = {
        "test_ctr":        round(test_ctr, 4),
        "genre_diversity": round(diversity, 4),
        "arm_coverage":    round(coverage, 4),
        "test_steps":      len(test_results),
        "total_reward":    int(test_results["reward"].sum()) if not test_results.empty else 0,
    }

    with open(METRICS / "eval_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)

    logger.info(f"  test_ctr={test_ctr:.4f} diversity={diversity:.3f} "
                f"coverage={coverage:.1%}")

    # ── Generate plots ────────────────────────────────────────────────────────
    logger.info("Generating evaluation plots...")

    # Load train results for train plot
    train_ratings = pd.read_parquet(PROCESSED / "train_ratings.parquet")
    _, train_results = run_simulation(
        ratings=train_ratings, movies=movies,
        users=users, profiles=profiles,
        alpha=ALPHA, pool_size=POOL_SIZE,
        seed=SEED, verbose=False
    )

    plot_results(train_results, model, movies,
                 save_path=PLOTS / "train_evaluation.png")
    plot_results(test_results, model, movies,
                 save_path=PLOTS / "test_evaluation.png")

    logger.info(f"  Plots saved to {PLOTS}/")

    # Print full summary
    print_summary(model, test_results, movies)

    logger.info("=== Stage 3 complete ✅ ===")


if __name__ == "__main__":
    main()