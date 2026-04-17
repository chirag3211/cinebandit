"""
DVC Stage 1: Data Preparation
Loads raw MovieLens data, splits into train/test,
computes user profiles and baseline statistics.
Outputs versioned parquet files tracked by DVC.
"""

import yaml
import logging
from pathlib import Path

import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import (
    load_ratings, load_movies, load_users,
    build_user_genre_profiles, DATA_DIR
)
from src.drift.detector import compute_baseline_statistics

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Load params ───────────────────────────────────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

TRAIN_SPLIT  = params["data"]["train_split"]
MIN_RATINGS  = params["data"]["min_ratings"]
PROCESSED    = Path("data/processed")
BASELINES    = Path("data/baselines")

PROCESSED.mkdir(parents=True, exist_ok=True)
BASELINES.mkdir(parents=True, exist_ok=True)


def main():
    logger.info("=== Stage 1: Data Preparation ===")

    # ── Load raw data ─────────────────────────────────────────────────────────
    logger.info("Loading raw MovieLens data...")
    ratings = load_ratings(DATA_DIR)
    movies  = load_movies(DATA_DIR)
    users   = load_users(DATA_DIR)

    logger.info(f"  Ratings : {len(ratings):,}")
    logger.info(f"  Movies  : {len(movies):,}")
    logger.info(f"  Users   : {len(users):,}")

    # ── Train / test split (temporal) ─────────────────────────────────────────
    split = int(len(ratings) * TRAIN_SPLIT)
    train = ratings.iloc[:split].reset_index(drop=True)
    test  = ratings.iloc[split:].reset_index(drop=True)

    logger.info(f"  Train   : {len(train):,} | Test: {len(test):,}")

    # ── User genre profiles ───────────────────────────────────────────────────
    logger.info("Building user genre profiles...")
    profiles = build_user_genre_profiles(ratings, movies, min_ratings=MIN_RATINGS)
    logger.info(f"  Profiles: {len(profiles):,} users")

    # ── Save processed data ───────────────────────────────────────────────────
    logger.info("Saving processed data...")
    train.to_parquet(PROCESSED / "train_ratings.parquet", index=False)
    test.to_parquet(PROCESSED  / "test_ratings.parquet",  index=False)
    movies.to_parquet(PROCESSED / "movies.parquet",        index=False)
    users.to_parquet(PROCESSED  / "users.parquet",         index=False)
    profiles.to_parquet(PROCESSED / "profiles.parquet",    index=False)
    logger.info("  Saved train/test/movies/users/profiles to data/processed/")

    # ── Compute baseline statistics ───────────────────────────────────────────
    logger.info("Computing baseline statistics...")
    stats = compute_baseline_statistics(
        ratings, movies,
        save_path=BASELINES / "genre_baseline.json"
    )
    logger.info(f"  like_rate={stats['like_rate']:.3f} "
                f"rating_mean={stats['rating_mean']:.2f}")

    logger.info("=== Stage 1 complete ✅ ===")


if __name__ == "__main__":
    main()