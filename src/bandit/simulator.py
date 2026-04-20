"""
Simulation engine for offline LinUCB evaluation.

Uses the "replay" method (Li et al. 2011) for unbiased offline evaluation:
  - Iterate through historical interactions in timestamp order
  - For each (user, movie, reward) tuple:
      * Build candidate pool of K random movies + the true movie
      * Ask LinUCB to recommend from the pool
      * If LinUCB chose the true movie → update with true reward (unbiased)
      * Otherwise → skip (do not update — avoids selection bias)

This gives an unbiased estimate of the online policy's performance
using only offline logged data.
"""

import random
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from tqdm import tqdm

from src.bandit.linucb import LinUCB
from src.data.loader import (
    GENRES, build_context_vector, build_user_genre_profiles
)


def build_candidate_pool(
    true_movie_id: int,
    all_movie_ids: List[int],
    pool_size: int = 20,
    rng: Optional[random.Random] = None
) -> List[int]:
    """
    Build a candidate pool containing the true movie + K-1 random others.
    This simulates the "20 candidates shown to the bandit" setting.
    """
    rng = rng or random
    others = [m for m in all_movie_ids if m != true_movie_id]
    sampled = rng.sample(others, min(pool_size - 1, len(others)))
    pool = sampled + [true_movie_id]
    rng.shuffle(pool)
    return pool


def run_simulation(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    users: pd.DataFrame,
    profiles: pd.DataFrame,
    alpha: float = 1.0,
    pool_size: int = 20,
    max_steps: Optional[int] = None,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[LinUCB, pd.DataFrame]:
    """
    Run offline replay simulation.

    Args:
        ratings    : sorted-by-timestamp ratings DataFrame
        movies     : movie metadata DataFrame
        users      : user demographics DataFrame
        profiles   : user genre preference vectors
        alpha      : LinUCB exploration parameter
        pool_size  : candidate pool size per step
        max_steps  : limit simulation to first N steps (None = all)
        seed       : random seed for reproducibility
        verbose    : show progress bar

    Returns:
        model      : trained LinUCB instance
        results_df : per-step results DataFrame
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    # Index lookups for O(1) access
    movie_lookup = movies.set_index("movie_id")
    user_lookup = users.set_index("user_id")
    profile_lookup = profiles.set_index("user_id")

    all_movie_ids = movies["movie_id"].tolist()
    context_dim = 60  # 19 + 19 + 19 + 1 + 1 + 1

    model = LinUCB(dim=context_dim, alpha=alpha)

    data = ratings if max_steps is None else ratings.iloc[:max_steps]
    n = len(data)

    results = []
    matched = 0  # replay matches (unbiased updates)

    iterator = tqdm(data.itertuples(), total=n, desc="Simulating") \
        if verbose else data.itertuples()

    for row in iterator:
        user_id = row.user_id
        true_movie_id = row.movie_id
        true_reward = float(row.reward)

        # Skip users/movies not in metadata
        if user_id not in user_lookup.index:
            continue
        if true_movie_id not in movie_lookup.index:
            continue

        user_row = user_lookup.loc[user_id]
        true_movie_row = movie_lookup.loc[true_movie_id]

        # Get user genre profile (fallback to zeros if not enough history)
        if user_id in profile_lookup.index:
            profile_row = profile_lookup.loc[user_id]
            user_profile = profile_row[
                [f"pref_{g}" for g in GENRES]
            ].values.astype(np.float64)
        else:
            user_profile = np.zeros(len(GENRES))

        # Build candidate pool
        pool = build_candidate_pool(
            true_movie_id, all_movie_ids, pool_size, rng
        )

        # Build context vectors for all candidates
        context_map = {}
        for mid in pool:
            if mid not in movie_lookup.index:
                continue
            mrow = movie_lookup.loc[mid]
            ctx = build_context_vector(user_row, mrow, user_profile)
            context_map[mid] = ctx

        if not context_map:
            continue

        # LinUCB selects best arm
        chosen_id, chosen_ucb = model.recommend(context_map)

        # Replay: only update if LinUCB chose the true historical movie
        if chosen_id == true_movie_id:
            matched += 1
            model.update(true_movie_id, context_map[true_movie_id], true_reward)

            results.append({
                "t": model.t,
                "step": len(results) + 1,
                "user_id": user_id,
                "movie_id": true_movie_id,
                "reward": true_reward,
                "ucb_score": chosen_ucb,
                "matched": True
            })

        if verbose and len(results) % 500 == 0 and len(results) > 0:
            ctr = sum(r["reward"] for r in results) / len(results)
            match_rate = matched / (len(results) + 1e-9)
            iterator.set_postfix(
                matched=matched,
                CTR=f"{ctr:.3f}",
                match_rate=f"{match_rate:.3f}"
            )

    results_df = pd.DataFrame(results)
    if verbose:
        print(f"\n[sim] Total steps processed : {n:,}")
        print(f"[sim] Replay matches        : {matched:,} "
              f"({matched/n*100:.1f}%)")
        if len(results_df) > 0:
            print(f"[sim] Mean CTR              : "
                  f"{results_df['reward'].mean():.4f}")

    return model, results_df
