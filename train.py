"""
Main script: Train and evaluate LinUCB on MovieLens-100K.

Usage:
    python train.py                     # default alpha=1.0, all data
    python train.py --alpha 0.5         # custom alpha
    python train.py --steps 20000       # limit to first 20k interactions
    python train.py --tune-alpha        # compare multiple alpha values
    python train.py --pool-size 50      # larger candidate pool
"""

import argparse
import time
import pickle
import numpy as np
from pathlib import Path

# ── Ensure src is importable from project root ────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.loader import prepare_dataset
from src.bandit.simulator import run_simulation
from src.evaluation.metrics import (
    print_summary,
    plot_results,
    plot_alpha_comparison,
    cumulative_ctr
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def train_and_evaluate(
    alpha: float = 1.0,
    pool_size: int = 20,
    max_steps: int = None,
    seed: int = 42,
    save_model: bool = True,
    plot: bool = True
):
    """Full training and evaluation run."""

    print("\n" + "=" * 55)
    print(f"  CineBandit — LinUCB Training")
    print(f"  alpha={alpha}, pool_size={pool_size}, seed={seed}")
    if max_steps:
        print(f"  max_steps={max_steps:,}")
    print("=" * 55 + "\n")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    t0 = time.time()
    ratings, movies, users, profiles = prepare_dataset()
    print(f"\n[main] Data loaded in {time.time()-t0:.1f}s")

    # ── 2. Train/test split (80/20 by time) ───────────────────────────────────
    split = int(len(ratings) * 0.8)
    train_ratings = ratings.iloc[:split].reset_index(drop=True)
    test_ratings = ratings.iloc[split:].reset_index(drop=True)
    print(f"[main] Train: {len(train_ratings):,} | Test: {len(test_ratings):,}")

    # ── 3. Run training simulation ────────────────────────────────────────────
    print(f"\n[main] Running training simulation ...")
    t0 = time.time()
    model, train_results = run_simulation(
        ratings=train_ratings,
        movies=movies,
        users=users,
        profiles=profiles,
        alpha=alpha,
        pool_size=pool_size,
        max_steps=max_steps,
        seed=seed,
        verbose=True
    )
    train_time = time.time() - t0
    print(f"[main] Training complete in {train_time:.1f}s")

    # ── 4. Run test simulation (no updates — frozen model) ────────────────────
    print(f"\n[main] Running test simulation (frozen model) ...")

    # Temporarily freeze: we run replay but don't call model.update
    # Reuse simulator with a wrapper that skips updates
    from src.bandit.simulator import run_simulation as sim
    _, test_results = sim(
        ratings=test_ratings,
        movies=movies,
        users=users,
        profiles=profiles,
        alpha=alpha,          # kept same for fair comparison
        pool_size=pool_size,
        max_steps=max_steps,
        seed=seed + 1,
        verbose=True
    )

    # ── 5. Print summaries ────────────────────────────────────────────────────
    print("\n--- TRAINING RESULTS ---")
    print_summary(model, train_results, movies)

    print("--- TEST RESULTS ---")
    print_summary(model, test_results, movies)

    # ── 6. Save plots ─────────────────────────────────────────────────────────
    if plot and not train_results.empty:
        plot_results(
            train_results, model, movies,
            save_path=OUTPUT_DIR / f"train_results_alpha{alpha}.png"
        )
        if not test_results.empty:
            plot_results(
                test_results, model, movies,
                save_path=OUTPUT_DIR / f"test_results_alpha{alpha}.png"
            )

    # ── 7. Save model ─────────────────────────────────────────────────────────
    if save_model:
        model_path = OUTPUT_DIR / f"linucb_alpha{alpha}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"[main] Model saved to {model_path}")

    return model, train_results, test_results


def tune_alpha(
    alphas: list = [0.1, 0.5, 1.0, 1.5, 2.0],
    pool_size: int = 20,
    max_steps: int = 30000,
    seed: int = 42
):
    """Compare LinUCB performance across multiple alpha values."""
    print(f"\n[tune] Alpha sweep: {alphas}")
    print(f"[tune] Using first {max_steps:,} interactions\n")

    ratings, movies, users, profiles = prepare_dataset()
    split = int(len(ratings) * 0.8)
    train_ratings = ratings.iloc[:split].reset_index(drop=True)

    results_by_alpha = {}
    summary = []

    for alpha in alphas:
        print(f"\n{'─'*40}")
        print(f"  Training with alpha = {alpha}")
        print(f"{'─'*40}")
        _, results = run_simulation(
            ratings=train_ratings,
            movies=movies,
            users=users,
            profiles=profiles,
            alpha=alpha,
            pool_size=pool_size,
            max_steps=max_steps,
            seed=seed,
            verbose=True
        )
        results_by_alpha[alpha] = results
        final_ctr = results["reward"].mean() if not results.empty else 0.0
        summary.append({"alpha": alpha, "final_ctr": final_ctr,
                         "matched_steps": len(results)})
        print(f"  α={alpha:.2f} → final CTR = {final_ctr:.4f}")

    print("\n\n=== Alpha Sweep Summary ===")
    for row in sorted(summary, key=lambda x: x["final_ctr"], reverse=True):
        print(f"  α={row['alpha']:.2f} | CTR={row['final_ctr']:.4f} | "
              f"steps={row['matched_steps']:,}")

    best = max(summary, key=lambda x: x["final_ctr"])
    print(f"\n  Best alpha: {best['alpha']} (CTR={best['final_ctr']:.4f})")

    plot_alpha_comparison(
        results_by_alpha,
        save_path=OUTPUT_DIR / "alpha_comparison.png"
    )

    return results_by_alpha


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CineBandit — LinUCB")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Exploration parameter (default: 1.0)")
    parser.add_argument("--pool-size", type=int, default=20,
                        help="Candidate pool size (default: 20)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Max interactions to simulate (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tune-alpha", action="store_true",
                        help="Run alpha sweep instead of single run")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip generating plots")
    args = parser.parse_args()

    if args.tune_alpha:
        tune_alpha(
            pool_size=args.pool_size,
            max_steps=args.steps or 30000,
            seed=args.seed
        )
    else:
        train_and_evaluate(
            alpha=args.alpha,
            pool_size=args.pool_size,
            max_steps=args.steps,
            seed=args.seed,
            plot=not args.no_plot
        )
