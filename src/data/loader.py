"""
MovieLens-100K data loader and feature engineering.
Produces context vectors for LinUCB.
"""

import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from pathlib import Path

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "ml-100k"

# All 19 MovieLens genres in canonical order
GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western"
]


def download_if_needed(data_dir: Path = DATA_DIR) -> None:
    """Download and extract MovieLens-100K if not already present."""
    if data_dir.exists() and (data_dir / "u.data").exists():
        print(f"[loader] Dataset already present at {data_dir}")
        return

    zip_path = data_dir.parent / "ml-100k.zip"
    data_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"[loader] Downloading MovieLens-100K from {DATA_URL} ...")
    try:
        urllib.request.urlretrieve(DATA_URL, zip_path)
    except Exception as e:
        raise RuntimeError(
            f"Download failed: {e}\n"
            "Please manually download ml-100k.zip from:\n"
            "  https://files.grouplens.org/datasets/movielens/ml-100k.zip\n"
            f"and extract it to: {data_dir.parent}"
        )

    print(f"[loader] Extracting to {data_dir.parent} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_dir.parent)
    print("[loader] Done.")


def load_ratings(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load u.data — the main ratings file."""
    path = data_dir / "u.data"
    df = pd.read_csv(
        path, sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )
    # Convert to binary reward: like=1 (rating>=4), dislike=0
    df["reward"] = (df["rating"] >= 4).astype(int)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_movies(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load u.item — movie metadata including genre one-hot vectors."""
    path = data_dir / "u.item"
    cols = ["movie_id", "title", "release_date", "video_release_date",
            "imdb_url"] + GENRES
    df = pd.read_csv(path, sep="|", names=cols, encoding="latin-1")
    df = df[["movie_id", "title", "release_date"] + GENRES].copy()

    # Extract release year (default 1995 if missing)
    df["year"] = pd.to_datetime(
        df["release_date"], errors="coerce"
    ).dt.year.fillna(1995).astype(int)

    # Normalize year to [0, 1] range
    yr_min, yr_max = df["year"].min(), df["year"].max()
    df["year_norm"] = (df["year"] - yr_min) / max(yr_max - yr_min, 1)

    # Genre vector as numpy array (19-dim)
    df["genre_vec"] = df[GENRES].values.tolist()

    return df[["movie_id", "title", "year", "year_norm", "genre_vec"] + GENRES]


def load_users(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load u.user — user demographics."""
    path = data_dir / "u.user"
    df = pd.read_csv(
        path, sep="|",
        names=["user_id", "age", "gender", "occupation", "zip"]
    )
    # Normalize age to [0, 1]
    df["age_norm"] = (df["age"] - df["age"].min()) / (
        df["age"].max() - df["age"].min()
    )
    # Binary gender encoding
    df["gender_bin"] = (df["gender"] == "M").astype(float)
    return df[["user_id", "age", "age_norm", "gender", "gender_bin", "occupation"]]


def build_user_genre_profiles(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    min_ratings: int = 5
) -> pd.DataFrame:
    """
    Build a per-user genre preference vector by averaging the genre
    vectors of movies the user liked (reward=1).

    Returns a DataFrame indexed by user_id with a 19-dim genre_profile.
    """
    liked = ratings[ratings["reward"] == 1].merge(
        movies[["movie_id"] + GENRES], on="movie_id"
    )

    profiles = {}
    genre_arr = liked[GENRES].values  # shape: (N, 19)

    for uid, grp in liked.groupby("user_id"):
        if len(grp) < min_ratings:
            profiles[uid] = np.zeros(len(GENRES))
        else:
            idx = liked["user_id"] == uid
            vecs = liked.loc[idx, GENRES].values
            profile = vecs.mean(axis=0)
            # L2 normalize so dot products are cosine similarities
            norm = np.linalg.norm(profile)
            profiles[uid] = profile / norm if norm > 0 else profile

    profile_df = pd.DataFrame.from_dict(
        profiles, orient="index",
        columns=[f"pref_{g}" for g in GENRES]
    )
    profile_df.index.name = "user_id"
    return profile_df.reset_index()


def build_context_vector(
    user_row: pd.Series,
    movie_row: pd.Series,
    user_profile: np.ndarray
) -> np.ndarray:
    """
    Build a context vector for a (user, movie) pair.

    Components:
      - user_profile     : 19-dim genre preference vector (normalized)
      - movie_genre_vec  : 19-dim genre one-hot
      - interaction      : element-wise product of profile × genre (19-dim)
      - year_norm        : 1-dim scalar
      - age_norm         : 1-dim scalar
      - gender_bin       : 1-dim scalar
    Total: 60 dimensions
    """
    movie_genre = np.array(movie_row["genre_vec"], dtype=float)

    # Interaction term: captures alignment between user taste and movie genre
    interaction = user_profile * movie_genre

    ctx = np.concatenate([
        user_profile,                        # 19
        movie_genre,                         # 19
        interaction,                         # 19
        [movie_row["year_norm"]],            # 1
        [user_row["age_norm"]],              # 1
        [user_row["gender_bin"]],            # 1
    ])
    return ctx.astype(np.float64)


def prepare_dataset(data_dir: Path = DATA_DIR):
    """
    Full data preparation pipeline.

    Returns:
        ratings   : pd.DataFrame with columns [user_id, movie_id, rating, reward, timestamp]
        movies    : pd.DataFrame with movie metadata
        users     : pd.DataFrame with user demographics
        profiles  : pd.DataFrame with user genre preference vectors
    """
    download_if_needed(data_dir)
    print("[loader] Loading ratings ...")
    ratings = load_ratings(data_dir)
    print(f"         {len(ratings):,} ratings loaded")

    print("[loader] Loading movies ...")
    movies = load_movies(data_dir)
    print(f"         {len(movies):,} movies loaded")

    print("[loader] Loading users ...")
    users = load_users(data_dir)
    print(f"         {len(users):,} users loaded")

    print("[loader] Building user genre profiles ...")
    profiles = build_user_genre_profiles(ratings, movies)
    print(f"         {len(profiles):,} user profiles built")

    return ratings, movies, users, profiles


if __name__ == "__main__":
    ratings, movies, users, profiles = prepare_dataset()
    print("\nSample ratings:")
    print(ratings.head())
    print("\nSample movies:")
    print(movies[["movie_id", "title", "year"] + GENRES[:5]].head())
    print("\nSample user profiles:")
    print(profiles.head())
    print(f"\nContext vector dimension: 60")
