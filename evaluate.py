"""
Standalone evaluation script.

Usage:
    python evaluate.py                          # evaluates default model
    python evaluate.py --model outputs/linucb_alpha1.0.pkl
    python evaluate.py --alpha 0.5 --steps 20000
    python evaluate.py --tune-alpha             # compare all saved models
"""

import argparse
import pickle
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.loader import prepare_dataset
from src.bandit.simulator import run_simulation
from src.evaluation.metrics import (
    print_summary,
    plot_results,
    plot_alpha_comparison,
    recommendation_diversity,
    arm_coverage,
    cumulative_ctr,
    rolling_ctr
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def evaluate_model(model_path: str = None, alpha: float = 1.0,
                   max_steps: int = None, seed: int = 42):
    """Load a saved model and evaluate it on test data."""

    # ── Load data ─────────────────────────────────────────────────────────────
    ratings, movies, users, profiles = prepare_dataset()
    split = int(len(ratings) * 0.8)
    test_ratings = ratings.iloc[split:].reset_index(drop=True)
    print(f"[eval] Test set: {len(test_ratings):,} interactions")

    # ── Load or retrain model ─────────────────────────────────────────────────
    if model_path and Path(model_path).exists():
        print(f"[eval] Loading model from {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"[eval] Model loaded: alpha={model.alpha}, "
              f"arms={len(model.arms)}, timesteps={model.t}")
    else:
        print(f"[eval] No saved model found — training fresh with alpha={alpha}")
        train_ratings = ratings.iloc[:split].reset_index(drop=True)
        from src.bandit.simulator import run_simulation as sim
        model, _ = sim(
            ratings=train_ratings, movies=movies, users=users,
            profiles=profiles, alpha=alpha, max_steps=max_steps,
            seed=seed, verbose=True
        )

    # ── Run test simulation ───────────────────────────────────────────────────
    print("\n[eval] Running test simulation ...")
    _, test_results = run_simulation(
        ratings=test_ratings, movies=movies, users=users,
        profiles=profiles, alpha=model.alpha,
        max_steps=max_steps, seed=seed + 1, verbose=True
    )

    # ── Print full summary ────────────────────────────────────────────────────
    print_summary(model, test_results, movies)

    # ── Detailed breakdown ────────────────────────────────────────────────────
    if not test_results.empty:
        _print_detailed_breakdown(model, test_results, movies)

    # ── Save plot ─────────────────────────────────────────────────────────────
    plot_results(
        test_results, model, movies,
        save_path=OUTPUT_DIR / f"eval_results_alpha{model.alpha}.png"
    )

    return model, test_results


def _print_detailed_breakdown(model, results, movies):
    """Print additional evaluation details."""
    import numpy as np

    print("\n" + "─" * 55)
    print("  Detailed Breakdown")
    print("─" * 55)

    # CTR over time quartiles
    n = len(results)
    q = n // 4
    if q > 0:
        print("\n  CTR by time quartile (learning curve):")
        for i in range(4):
            chunk = results.iloc[i*q:(i+1)*q]
            print(f"    Q{i+1} (steps {i*q+1:,}–{(i+1)*q:,}): "
                  f"CTR = {chunk['reward'].mean():.4f}")

    # Top 10 most recommended movies by CTR (min 3 pulls)
    top = (
        results.groupby("movie_id")["reward"]
        .agg(["count", "mean", "sum"])
        .rename(columns={"count": "pulls", "mean": "ctr", "sum": "total_reward"})
        .query("pulls >= 3")
        .sort_values("ctr", ascending=False)
        .head(10)
    )
    movie_titles = movies.set_index("movie_id")["title"]
    print("\n  Top-10 movies by CTR (min 3 pulls):")
    print(f"  {'Title':<30} {'Pulls':>6} {'CTR':>6} {'Rewards':>8}")
    print(f"  {'─'*30} {'─'*6} {'─'*6} {'─'*8}")
    for mid, row in top.iterrows():
        title = str(movie_titles.get(mid, f"Movie {mid}"))[:28]
        print(f"  {title:<30} {int(row['pulls']):>6} "
              f"{row['ctr']:>6.3f} {int(row['total_reward']):>8}")

    # Genre distribution of recommendations
    from src.data.loader import GENRES
    recommended_movies = results["movie_id"].values
    movie_genres = movies.set_index("movie_id")[GENRES]
    genre_counts = {}
    for mid in recommended_movies:
        if mid in movie_genres.index:
            for g, v in zip(GENRES, movie_genres.loc[mid].values):
                if v == 1:
                    genre_counts[g] = genre_counts.get(g, 0) + 1

    total = sum(genre_counts.values())
    if total > 0:
        print("\n  Genre distribution of recommendations:")
        sorted_genres = sorted(genre_counts.items(),
                               key=lambda x: x[1], reverse=True)
        for genre, count in sorted_genres[:8]:
            bar = "█" * int(count / total * 30)
            print(f"  {genre:<15} {bar:<30} {count/total*100:5.1f}%")

    # Reward rate over time (early vs late)
    early_ctr = results.iloc[:n//2]["reward"].mean() if n > 1 else 0
    late_ctr = results.iloc[n//2:]["reward"].mean() if n > 1 else 0
    improvement = (late_ctr - early_ctr) / max(early_ctr, 1e-9) * 100
    print(f"\n  Learning improvement:")
    print(f"    Early CTR (first half) : {early_ctr:.4f}")
    print(f"    Late CTR  (second half): {late_ctr:.4f}")
    print(f"    Improvement            : {improvement:+.1f}%")
    print("─" * 55 + "\n")


def compare_all_saved_models():
    """Compare all saved models in the outputs directory."""
    model_files = list(OUTPUT_DIR.glob("linucb_alpha*.pkl"))

    if not model_files:
        print("[eval] No saved models found in outputs/. "
              "Run train.py first.")
        return

    print(f"[eval] Found {len(model_files)} saved models: "
          f"{[f.name for f in model_files]}")

    ratings, movies, users, profiles = prepare_dataset()
    split = int(len(ratings) * 0.8)
    test_ratings = ratings.iloc[split:].reset_index(drop=True)

    results_by_alpha = {}
    summary = []

    for model_file in sorted(model_files):
        with open(model_file, "rb") as f:
            model = pickle.load(f)

        print(f"\n[eval] Testing alpha={model.alpha} ...")
        _, test_results = run_simulation(
            ratings=test_ratings, movies=movies, users=users,
            profiles=profiles, alpha=model.alpha,
            seed=43, verbose=False
        )

        final_ctr = test_results["reward"].mean() if not test_results.empty else 0
        results_by_alpha[model.alpha] = test_results
        summary.append({
            "alpha": model.alpha,
            "test_ctr": final_ctr,
            "matched_steps": len(test_results)
        })
        print(f"  alpha={model.alpha:.2f} → test CTR={final_ctr:.4f} "
              f"({len(test_results):,} matched steps)")

    print("\n\n=== Model Comparison Summary ===")
    for row in sorted(summary, key=lambda x: x["test_ctr"], reverse=True):
        marker = " ← BEST" if row == max(
            summary, key=lambda x: x["test_ctr"]
        ) else ""
        print(f"  alpha={row['alpha']:.2f} | "
              f"test CTR={row['test_ctr']:.4f}{marker}")

    plot_alpha_comparison(
        results_by_alpha,
        save_path=OUTPUT_DIR / "model_comparison.png"
    )
    print(f"\n[eval] Comparison plot saved to "
          f"{OUTPUT_DIR / 'model_comparison.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CineBandit — Evaluate LinUCB")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to saved .pkl model file")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Alpha to use if no model file provided")
    parser.add_argument("--steps", type=int, default=None,
                        help="Max test steps (default: all)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all saved models in outputs/")
    args = parser.parse_args()

    if args.compare:
        compare_all_saved_models()
    else:
        model_path = args.model or f"outputs/linucb_alpha{args.alpha}.pkl"
        evaluate_model(
            model_path=model_path,
            alpha=args.alpha,
            max_steps=args.steps
        )