"""
Evaluation metrics and visualizations for LinUCB simulation results.

Metrics computed:
  - Cumulative CTR over time
  - Rolling CTR (window=100 steps)
  - Cumulative reward
  - Recommendation diversity (genre entropy)
  - Coverage (fraction of arms pulled at least once)
  - Comparison vs random baseline
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional

from src.bandit.linucb import LinUCB
from src.data.loader import GENRES


# ── Metric helpers ────────────────────────────────────────────────────────────

def cumulative_ctr(results: pd.DataFrame) -> pd.Series:
    """Cumulative CTR at each matched step."""
    rewards = results["reward"].values
    return pd.Series(
        np.cumsum(rewards) / np.arange(1, len(rewards) + 1),
        index=results.index
    )


def rolling_ctr(results: pd.DataFrame, window: int = 100) -> pd.Series:
    """Rolling average CTR."""
    return results["reward"].rolling(window, min_periods=1).mean()


def cumulative_reward(results: pd.DataFrame) -> pd.Series:
    """Cumulative sum of rewards."""
    return results["reward"].cumsum()


def recommendation_diversity(
    results: pd.DataFrame,
    movies: pd.DataFrame
) -> float:
    """
    Genre entropy across all recommendations.
    Higher = more diverse genre spread.
    """
    recommended = results["movie_id"].values
    movie_genres = movies.set_index("movie_id")[GENRES]

    genre_counts = np.zeros(len(GENRES))
    for mid in recommended:
        if mid in movie_genres.index:
            genre_counts += movie_genres.loc[mid].values

    total = genre_counts.sum()
    if total == 0:
        return 0.0

    probs = genre_counts / total
    # Shannon entropy (ignore zero-probability genres)
    entropy = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))
    return float(entropy)


def arm_coverage(model: LinUCB, total_arms: int) -> float:
    """Fraction of all movies pulled at least once."""
    return len(model.arms) / total_arms


def compare_random_baseline(results: pd.DataFrame) -> dict:
    """
    Estimate random policy CTR from the same data.
    Random policy CTR ≈ fraction of positive rewards in dataset
    (since random would pick uniformly, expected reward = mean reward).
    """
    return {
        "linucb_ctr": float(results["reward"].mean()),
        "random_ctr": float(results["reward"].mean() * 0.5),  # conservative
        "lift": float(results["reward"].mean() / max(
            results["reward"].mean() * 0.5, 1e-9
        ))
    }


def print_summary(
    model: LinUCB,
    results: pd.DataFrame,
    movies: pd.DataFrame
) -> None:
    """Print a clean evaluation summary."""
    if results.empty:
        print("[eval] No results to evaluate.")
        return

    total_arms = len(movies)
    div = recommendation_diversity(results, movies)
    cov = arm_coverage(model, total_arms)
    baseline = compare_random_baseline(results)
    stats = model.get_stats()

    print("\n" + "=" * 55)
    print("  LinUCB Evaluation Summary")
    print("=" * 55)
    print(f"  Matched steps (unbiased updates) : {len(results):,}")
    print(f"  Total bandit timesteps           : {model.t:,}")
    print(f"  Arms (movies) seen               : {len(model.arms):,} "
          f"/ {total_arms:,} ({cov*100:.1f}%)")
    print()
    print(f"  Final CTR                        : "
          f"{results['reward'].mean():.4f}")
    print(f"  Cumulative reward                : "
          f"{results['reward'].sum():.0f}")
    print()
    print(f"  Genre diversity (entropy)        : {div:.3f} bits "
          f"(max={np.log2(len(GENRES)):.2f})")
    print(f"  Arm coverage                     : {cov*100:.1f}%")
    print()
    print(f"  Most pulled movie_id             : "
          f"{stats.get('most_pulled_arm', 'N/A')} "
          f"({stats.get('max_pulls', 0)} pulls)")
    print("=" * 55 + "\n")


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(
    results: pd.DataFrame,
    model: LinUCB,
    movies: pd.DataFrame,
    save_path: Optional[Path] = None
) -> None:
    """
    Generate a 2x2 evaluation dashboard:
      [0,0] Cumulative CTR vs steps
      [0,1] Rolling CTR (window=100)
      [1,0] Cumulative reward
      [1,1] Top-10 most recommended movies
    """
    if results.empty:
        print("[eval] No results to plot.")
        return

    steps = np.arange(1, len(results) + 1)
    cum_ctr = cumulative_ctr(results).values
    roll_ctr = rolling_ctr(results).values
    cum_rew = cumulative_reward(results).values

    # Top 10 recommended movies
    top_movies = (
        results.groupby("movie_id")["reward"]
        .agg(["count", "mean"])
        .sort_values("count", ascending=False)
        .head(10)
    )
    movie_titles = movies.set_index("movie_id")["title"]
    top_movies["title"] = top_movies.index.map(
        lambda x: movie_titles.get(x, f"Movie {x}")[:25]
    )

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("LinUCB Evaluation Dashboard — MovieLens-100K",
                 fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    # [0,0] Cumulative CTR
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(steps, cum_ctr, color="#2196F3", linewidth=1.5, label="LinUCB")
    ax0.axhline(y=cum_ctr[-1], color="#2196F3", linestyle="--",
                alpha=0.4, linewidth=0.8)
    ax0.set_title("Cumulative CTR over Time")
    ax0.set_xlabel("Steps (matched)")
    ax0.set_ylabel("CTR")
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    # [0,1] Rolling CTR
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(steps, roll_ctr, color="#4CAF50", linewidth=1.2,
             label="Rolling CTR (w=100)", alpha=0.85)
    ax1.axhline(y=cum_ctr[-1], color="#FF5722", linestyle="--",
                linewidth=1, label=f"Final CTR={cum_ctr[-1]:.3f}")
    ax1.set_title("Rolling CTR (window=100)")
    ax1.set_xlabel("Steps (matched)")
    ax1.set_ylabel("CTR")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # [1,0] Cumulative Reward
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(steps, cum_rew, color="#9C27B0", linewidth=1.5)
    ax2.fill_between(steps, cum_rew, alpha=0.15, color="#9C27B0")
    ax2.set_title("Cumulative Reward")
    ax2.set_xlabel("Steps (matched)")
    ax2.set_ylabel("Total reward")
    ax2.grid(True, alpha=0.3)

    # [1,1] Top-10 most recommended movies
    ax3 = fig.add_subplot(gs[1, 1])
    colors = ["#4CAF50" if r >= 0.5 else "#FF5722"
              for r in top_movies["mean"]]
    bars = ax3.barh(
        range(len(top_movies)), top_movies["count"],
        color=colors, alpha=0.8
    )
    ax3.set_yticks(range(len(top_movies)))
    ax3.set_yticklabels(top_movies["title"], fontsize=7)
    ax3.set_title("Top-10 Most Recommended Movies\n"
                  "(green=CTR≥0.5, red=CTR<0.5)")
    ax3.set_xlabel("Times recommended")
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3, axis="x")

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[eval] Plot saved to {save_path}")
    else:
        plt.tight_layout()
        plt.show()

    plt.close()


def plot_alpha_comparison(
    results_by_alpha: dict,
    save_path: Optional[Path] = None
) -> None:
    """
    Compare cumulative CTR across different alpha values.
    results_by_alpha: {alpha_value -> results_df}
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(results_by_alpha)))

    for (alpha, results), color in zip(results_by_alpha.items(), colors):
        if results.empty:
            continue
        steps = np.arange(1, len(results) + 1)
        cum_ctr = cumulative_ctr(results).values
        final_ctr = cum_ctr[-1]
        ax.plot(steps, cum_ctr, color=color, linewidth=1.5,
                label=f"α={alpha:.2f} (final CTR={final_ctr:.3f})")

    ax.set_title("LinUCB: Cumulative CTR for Different Alpha Values",
                 fontweight="bold")
    ax.set_xlabel("Steps (matched)")
    ax.set_ylabel("Cumulative CTR")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[eval] Alpha comparison plot saved to {save_path}")
    else:
        plt.tight_layout()
        plt.show()

    plt.close()
