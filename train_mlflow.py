"""
train_mlflow.py — MLflow-instrumented training script (UPDATED)
"""

import os
import time
import pickle
import argparse
import tempfile
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import mlflow
from mlflow.models import infer_signature

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.loader import prepare_dataset
from src.bandit.simulator import run_simulation
from src.evaluation.metrics import (
    print_summary,
    plot_results,
    recommendation_diversity,
    arm_coverage,
    cumulative_ctr,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

EXPERIMENT_NAME = "CineBandit-LinUCB"
MODEL_NAME      = "CineBandit"


# ── Experiment helpers ───────────────────────────────────────────────────────

def get_or_create_experiment(name: str) -> str:
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
        logger.info(f"[mlflow] Created experiment '{name}' (id={exp_id})")
    else:
        exp_id = exp.experiment_id
        logger.info(f"[mlflow] Using experiment '{name}' (id={exp_id})")
    return exp_id


def get_champion_ctr() -> float:
    client = mlflow.tracking.MlflowClient()
    try:
        mv = client.get_model_version_by_alias(
            name=MODEL_NAME,
            alias="production"
        )
        run = client.get_run(mv.run_id)
        return float(run.data.metrics.get("test_ctr", 0.0))
    except Exception:
        return 0.0


def promote_if_better(run_id: str, test_ctr: float) -> str:
    client = mlflow.tracking.MlflowClient()

    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, MODEL_NAME)
    version = mv.version

    logger.info(f"[mlflow] Registered model v{version} from run {run_id}")

    champion_ctr = get_champion_ctr()
    logger.info(
        f"[mlflow] Challenger CTR={test_ctr:.4f} vs Champion CTR={champion_ctr:.4f}"
    )

    if test_ctr > champion_ctr:
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="production",
            version=version
        )
        logger.info(
            f"[mlflow] ✅ Model v{version} set as 'production' "
            f"(CTR {champion_ctr:.4f} → {test_ctr:.4f})"
        )
    else:
        logger.info("[mlflow] ❌ Challenger did not beat production model")

    return version


# ── PyFunc Wrapper ───────────────────────────────────────────────────────────

class LinUCBWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame
    ) -> pd.DataFrame:
        return pd.DataFrame({
            "message": ["Use the FastAPI /recommend endpoint for inference"]
        })


# ── Training ─────────────────────────────────────────────────────────────────

def train_and_log(
    alpha: float = 0.1,
    pool_size: int = 20,
    max_steps: int = None,
    seed: int = 42,
    auto_promote: bool = True,
) -> dict:

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD"] = "true"
    logger.info(f"[mlflow] Tracking URI: {tracking_uri}")

    exp_id = get_or_create_experiment(EXPERIMENT_NAME)

    logger.info("[train] Loading data...")
    ratings, movies, users, profiles = prepare_dataset()

    split = int(len(ratings) * 0.8)
    train_ratings = ratings.iloc[:split].reset_index(drop=True)
    test_ratings  = ratings.iloc[split:].reset_index(drop=True)

    with mlflow.start_run(experiment_id=exp_id) as run:
        run_id = run.info.run_id
        logger.info(f"[mlflow] Run ID: {run_id}")

        mlflow.log_params({
            "algorithm":   "LinUCB",
            "alpha":       alpha,
            "pool_size":   pool_size,
            "context_dim": 60,
            "seed":        seed,
            "n_train":     len(train_ratings),
            "n_test":      len(test_ratings),
        })

        # ── Training ──────────────────────────────────────────────────────────
        logger.info(f"[train] Training with alpha={alpha} ...")
        t0 = time.time()
        model, train_results = run_simulation(
            ratings=train_ratings, movies=movies,
            users=users, profiles=profiles,
            alpha=alpha, pool_size=pool_size,
            max_steps=max_steps, seed=seed, verbose=True
        )
        train_time = time.time() - t0

        # ── Testing ───────────────────────────────────────────────────────────
        logger.info("[train] Running test simulation...")
        _, test_results = run_simulation(
            ratings=test_ratings, movies=movies,
            users=users, profiles=profiles,
            alpha=alpha, pool_size=pool_size,
            max_steps=max_steps, seed=seed + 1, verbose=True
        )

        # ── Metrics ───────────────────────────────────────────────────────────
        train_ctr    = float(train_results["reward"].mean()) if not train_results.empty else 0.0
        test_ctr     = float(test_results["reward"].mean())  if not test_results.empty  else 0.0
        diversity    = recommendation_diversity(train_results, movies)
        coverage     = arm_coverage(model, len(movies))
        match_rate   = len(train_results) / len(train_ratings) if len(train_ratings) > 0 else 0.0

        mlflow.log_metrics({
            "train_ctr":       round(train_ctr, 4),
            "test_ctr":        round(test_ctr, 4),
            "train_reward":    float(train_results["reward"].sum()) if not train_results.empty else 0.0,
            "test_reward":     float(test_results["reward"].sum())  if not test_results.empty  else 0.0,
            "genre_diversity": round(diversity, 4),
            "arm_coverage":    round(coverage, 4),
            "match_rate":      round(match_rate, 4),
            "train_steps":     len(train_results),
            "test_steps":      len(test_results),
            "train_time_s":    round(train_time, 2),
            "model_timesteps": model.t,
        })

        # Log rolling CTR curve step-by-step
        if not train_results.empty:
            cum_ctr = cumulative_ctr(train_results).values
            for step, ctr in enumerate(cum_ctr[::50]):
                mlflow.log_metric("cumulative_ctr", ctr, step=step * 50)

        # ── Save model pickle locally ─────────────────────────────────────────
        model_path = OUTPUT_DIR / f"linucb_alpha{alpha}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # ── Log artifacts to MLflow (graceful fallback if storage unavailable) ─
        try:
            mlflow.log_artifact(str(model_path), artifact_path="model_pickle")
            logger.info("[mlflow] Model pickle logged as artifact")
        except Exception as e:
            logger.warning(f"[mlflow] Could not log pickle artifact: {e}")

        # ── Log model to registry ─────────────────────────────────────────────
        try:
            sample_input  = pd.DataFrame({"user_id": [1.0], "n": [5.0]})
            sample_output = pd.DataFrame({"message": ["example"]})
            signature = infer_signature(sample_input, sample_output)
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=LinUCBWrapper(model),
                input_example=sample_input,
                signature=signature,
            )
            logger.info("[mlflow] PyFunc model logged")
        except Exception as e:
            logger.warning(f"[mlflow] Could not log pyfunc model: {e}")

        # ── Log evaluation plots ───────────────────────────────────────────────
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                if not train_results.empty:
                    plot_path = tmpdir / "train_evaluation.png"
                    plot_results(train_results, model, movies, save_path=plot_path)
                    mlflow.log_artifact(str(plot_path), artifact_path="plots")
                if not test_results.empty:
                    plot_path = tmpdir / "test_evaluation.png"
                    plot_results(test_results, model, movies, save_path=plot_path)
                    mlflow.log_artifact(str(plot_path), artifact_path="plots")
            logger.info("[mlflow] Evaluation plots logged")
        except Exception as e:
            logger.warning(f"[mlflow] Could not log plots: {e}")

        # ── Tags ──────────────────────────────────────────────────────────────
        mlflow.set_tags({
            "model_type": "contextual_bandit",
            "dataset":    "movielens_100k",
            "status":     "completed",
        })

        logger.info(f"[mlflow] Run complete. train_ctr={train_ctr:.4f} test_ctr={test_ctr:.4f}")
        print_summary(model, test_results, movies)

    # ── Promotion ─────────────────────────────────────────────────────────────
    if auto_promote:
        version = promote_if_better(run_id, test_ctr)
    else:
        version = None

    return {
        "run_id":     run_id,
        "alpha":      alpha,
        "train_ctr":  train_ctr,
        "test_ctr":   test_ctr,
        "version":    version,
        "model_path": str(model_path),
    }


# ── Alpha sweep ───────────────────────────────────────────────────────────────

def tune_alpha(
    alphas: list = [0.05, 0.1, 0.5, 1.0, 1.5],
    pool_size: int = 20,
    max_steps: int = 30000,
    seed: int = 42,
):
    """Run multiple alpha values, each as a separate MLflow run."""
    results = []
    for alpha in alphas:
        logger.info(f"\n{'─'*50}\n  Training alpha={alpha}\n{'─'*50}")
        result = train_and_log(
            alpha=alpha, pool_size=pool_size,
            max_steps=max_steps, seed=seed, auto_promote=True,
        )
        results.append(result)

    print("\n=== Alpha Sweep Summary ===")
    for r in sorted(results, key=lambda x: x["test_ctr"], reverse=True):
        print(f"  alpha={r['alpha']:.2f} | test_ctr={r['test_ctr']:.4f} | "
              f"run={r['run_id'][:8]}... | version={r['version']}")
    best = max(results, key=lambda x: x["test_ctr"])
    print(f"\n  Best: alpha={best['alpha']} (test_ctr={best['test_ctr']:.4f})")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CineBandit — MLflow Training")
    parser.add_argument("--alpha",      type=float, default=0.1)
    parser.add_argument("--pool-size",  type=int,   default=20)
    parser.add_argument("--steps",      type=int,   default=None)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--tune-alpha", action="store_true")
    parser.add_argument("--no-promote", action="store_true")
    args = parser.parse_args()

    if args.tune_alpha:
        tune_alpha(
            pool_size=args.pool_size,
            max_steps=args.steps or 30000,
            seed=args.seed,
        )
    else:
        result = train_and_log(
            alpha=args.alpha,
            pool_size=args.pool_size,
            max_steps=args.steps,
            seed=args.seed,
            auto_promote=not args.no_promote,
        )
        print(f"\nRun ID   : {result['run_id']}")
        print(f"Alpha    : {result['alpha']}")
        print(f"Train CTR: {result['train_ctr']:.4f}")
        print(f"Test CTR : {result['test_ctr']:.4f}")
        print(f"Version  : {result['version']}")
        print(f"Model    : {result['model_path']}")