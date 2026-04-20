"""
Drift detection for CineBandit.

Computes KL divergence between the baseline genre distribution
(computed during initial data ingestion) and the current interaction
distribution. A high KL divergence indicates user preference shift,
triggering model retraining.

Also computes feature-level statistics for monitoring.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)

BASELINE_PATH = Path("data/baselines/genre_baseline.json")
DRIFT_THRESHOLD = 0.3   # KL divergence threshold — alert above this
GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western"
]


# ── Baseline computation ──────────────────────────────────────────────────────

def compute_genre_distribution(
    ratings: pd.DataFrame,
    movies: pd.DataFrame
) -> np.ndarray:
    """
    Compute the genre distribution of liked movies (reward=1).
    Returns a normalised probability vector over 19 genres.
    """
    liked = ratings[ratings["reward"] == 1].merge(
        movies[["movie_id"] + GENRES], on="movie_id", how="inner"
    )

    if liked.empty:
        return np.ones(len(GENRES)) / len(GENRES)

    genre_counts = liked[GENRES].sum(axis=0).values.astype(float)
    total = genre_counts.sum()
    if total == 0:
        return np.ones(len(GENRES)) / len(GENRES)

    return genre_counts / total


def compute_baseline_statistics(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    save_path: Path = BASELINE_PATH
) -> dict:
    """
    Compute and save baseline statistics from the full dataset.
    Called once during initial data ingestion.

    Returns a dict with:
      - genre_distribution : 19-dim probability vector
      - rating_mean        : mean rating
      - rating_std         : std of ratings
      - like_rate          : fraction of ratings >= 4
      - n_ratings          : total number of ratings
    """
    genre_dist = compute_genre_distribution(ratings, movies)

    stats = {
        "genre_distribution": genre_dist.tolist(),
        "genre_names":        GENRES,
        "rating_mean":        float(ratings["rating"].mean()),
        "rating_std":         float(ratings["rating"].std()),
        "like_rate":          float((ratings["rating"] >= 4).mean()),
        "n_ratings":          int(len(ratings)),
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"[drift] Baseline statistics saved to {save_path}")
    logger.info(f"[drift] like_rate={stats['like_rate']:.3f} "
                f"n_ratings={stats['n_ratings']:,}")
    return stats


def load_baseline(path: Path = BASELINE_PATH) -> Optional[dict]:
    """Load saved baseline statistics. Returns None if not found."""
    if not path.exists():
        logger.warning(f"[drift] Baseline not found at {path}")
        return None
    with open(path) as f:
        return json.load(f)


# ── KL Divergence ─────────────────────────────────────────────────────────────

def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute KL divergence KL(p || q).

    p = current distribution
    q = baseline distribution

    Adds epsilon smoothing to avoid log(0).
    Returns a scalar >= 0. Zero means identical distributions.
    """
    p = np.array(p, dtype=float) + eps
    q = np.array(q, dtype=float) + eps

    # Normalise
    p = p / p.sum()
    q = q / q.sum()

    return float(np.sum(p * np.log(p / q)))


# ── Drift detection ───────────────────────────────────────────────────────────

def detect_drift(
    current_ratings: pd.DataFrame,
    movies: pd.DataFrame,
    baseline_path: Path = BASELINE_PATH,
    threshold: float = DRIFT_THRESHOLD,
    min_samples: int = 100,
) -> Tuple[float, bool, dict]:
    """
    Compare current interaction distribution against baseline.

    Args:
        current_ratings : recent interactions (last N days)
        movies          : movie metadata
        baseline_path   : path to saved baseline JSON
        threshold       : KL divergence threshold for alert
        min_samples     : minimum interactions needed for reliable estimate

    Returns:
        drift_score : KL divergence (float)
        alert       : True if drift_score > threshold
        report      : detailed dict with all stats
    """
    baseline = load_baseline(baseline_path)
    if baseline is None:
        logger.warning("[drift] No baseline found — cannot detect drift")
        return 0.0, False, {"error": "no_baseline"}

    if len(current_ratings) < min_samples:
        logger.warning(
            f"[drift] Only {len(current_ratings)} samples — "
            f"need {min_samples} for reliable drift detection"
        )
        return 0.0, False, {"error": "insufficient_samples",
                            "n_samples": len(current_ratings)}

    # Compute current distribution
    current_dist = compute_genre_distribution(current_ratings, movies)
    baseline_dist = np.array(baseline["genre_distribution"])

    # KL divergence
    drift_score = kl_divergence(current_dist, baseline_dist)
    alert = drift_score > threshold

    # Per-genre shift
    genre_shifts = {
        genre: float(current_dist[i] - baseline_dist[i])
        for i, genre in enumerate(GENRES)
    }
    top_shifts = sorted(
        genre_shifts.items(), key=lambda x: abs(x[1]), reverse=True
    )[:5]

    report = {
        "drift_score":        round(drift_score, 4),
        "threshold":          threshold,
        "alert":              alert,
        "n_current_samples":  len(current_ratings),
        "n_baseline_samples": baseline["n_ratings"],
        "current_like_rate":  float((current_ratings["rating"] >= 4).mean())
                              if "rating" in current_ratings.columns else None,
        "baseline_like_rate": baseline["like_rate"],
        "top_genre_shifts":   dict(top_shifts),
        "current_distribution": {
            g: round(float(current_dist[i]), 4)
            for i, g in enumerate(GENRES)
        },
        "baseline_distribution": {
            g: round(float(baseline_dist[i]), 4)
            for i, g in enumerate(GENRES)
        },
    }

    level = "🚨 ALERT" if alert else "✅ OK"
    logger.info(
        f"[drift] {level} | KL={drift_score:.4f} | "
        f"threshold={threshold} | samples={len(current_ratings):,}"
    )
    if alert:
        logger.info(f"[drift] Top genre shifts: {dict(top_shifts)}")

    return drift_score, alert, report


# ── Simulate drift (for testing) ─────────────────────────────────────────────

def simulate_drift(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    drift_phase: int = 1,
    n_samples: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate user preference drift by reweighting genre preferences.

    drift_phase 0 = original distribution (no drift)
    drift_phase 1 = mild drift  (Action/Adventure users shift to Drama/Romance)
    drift_phase 2 = strong drift (broad genre shift)

    Returns a modified ratings DataFrame representing drifted interactions.
    """
    rng = np.random.RandomState(seed + drift_phase)

    # Genre weight multipliers per drift phase
    drift_weights = {
        0: {g: 1.0 for g in GENRES},
        1: {
            "Action": 0.5, "Adventure": 0.5, "Thriller": 0.6,
            "Drama": 2.0, "Romance": 2.0, "Comedy": 1.5,
            **{g: 1.0 for g in GENRES
               if g not in ["Action", "Adventure", "Thriller",
                             "Drama", "Romance", "Comedy"]}
        },
        2: {
            "Action": 0.3, "Sci-Fi": 0.4, "Horror": 0.3,
            "Drama": 3.0, "Romance": 2.5, "Musical": 2.0,
            "Documentary": 2.0,
            **{g: 1.0 for g in GENRES
               if g not in ["Action", "Sci-Fi", "Horror",
                             "Drama", "Romance", "Musical", "Documentary"]}
        },
    }

    weights = drift_weights.get(drift_phase, drift_weights[0])

    # Weight each movie by its genre preference under current drift
    movie_genres = movies.set_index("movie_id")[GENRES]
    movie_weights = pd.Series(index=movies["movie_id"], dtype=float)

    for mid in movies["movie_id"]:
        if mid in movie_genres.index:
            genre_vec = movie_genres.loc[mid].values
            weight = sum(
                weights.get(g, 1.0) * genre_vec[i]
                for i, g in enumerate(GENRES)
            )
            movie_weights[mid] = max(weight, 0.1)
        else:
            movie_weights[mid] = 1.0

    # Sample ratings with drift-weighted movie probabilities
    movie_probs = movie_weights / movie_weights.sum()
    sampled_movies = rng.choice(
        movie_weights.index,
        size=min(n_samples, len(ratings)),
        p=movie_probs.values,
        replace=True
    )

    sampled_users = rng.choice(
        ratings["user_id"].unique(),
        size=len(sampled_movies),
        replace=True
    )

    drifted = pd.DataFrame({
        "user_id":  sampled_users,
        "movie_id": sampled_movies,
        "rating":   rng.choice([1, 2, 3, 4, 5], size=len(sampled_movies),
                               p=[0.05, 0.1, 0.2, 0.35, 0.3]),
        "timestamp": ratings["timestamp"].max() + np.arange(len(sampled_movies)) * 600,
    })
    drifted["reward"] = (drifted["rating"] >= 4).astype(int)

    return drifted