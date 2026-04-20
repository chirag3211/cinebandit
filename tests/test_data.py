"""
Tests for data loading and feature engineering.
Covers: MovieLens loading, context vector construction,
        user profile building, and data validation.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import (
    GENRES, build_context_vector, build_user_genre_profiles,
    load_ratings, load_movies, load_users
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_ratings():
    """Small synthetic ratings DataFrame."""
    return pd.DataFrame({
        "user_id":   [1, 1, 2, 2, 3],
        "movie_id":  [10, 20, 10, 30, 20],
        "rating":    [5, 3, 4, 2, 5],
        "timestamp": [100, 200, 300, 400, 500],
        "reward":    [1, 0, 1, 0, 1],
    })


@pytest.fixture
def sample_movies():
    """Small synthetic movies DataFrame with genre vectors."""
    genre_row_action = [0] * 19
    genre_row_action[1] = 1  # Action

    genre_row_drama = [0] * 19
    genre_row_drama[8] = 1   # Drama

    genre_row_comedy = [0] * 19
    genre_row_comedy[5] = 1  # Comedy

    rows = [
        [10, "Movie A", 1995, 0.5, genre_row_action] + genre_row_action,
        [20, "Movie B", 2000, 0.7, genre_row_drama]  + genre_row_drama,
        [30, "Movie C", 1990, 0.3, genre_row_comedy] + genre_row_comedy,
    ]
    cols = ["movie_id", "title", "year", "year_norm", "genre_vec"] + GENRES
    return pd.DataFrame(rows, columns=cols)


@pytest.fixture
def sample_users():
    """Small synthetic users DataFrame."""
    return pd.DataFrame({
        "user_id":    [1, 2, 3],
        "age":        [25, 35, 45],
        "age_norm":   [0.0, 0.5, 1.0],
        "gender":     ["M", "F", "M"],
        "gender_bin": [1.0, 0.0, 1.0],
        "occupation": ["student", "engineer", "doctor"],
    })


# ── GENRES constant tests ─────────────────────────────────────────────────────

class TestGenres:

    def test_genres_length(self):
        """GENRES list has exactly 19 entries."""
        assert len(GENRES) == 19

    def test_genres_no_duplicates(self):
        """No duplicate genres."""
        assert len(GENRES) == len(set(GENRES))

    def test_known_genres_present(self):
        """Key genres are present."""
        for genre in ["Action", "Drama", "Comedy", "Thriller", "Romance"]:
            assert genre in GENRES


# ── Context vector tests ──────────────────────────────────────────────────────

class TestContextVector:

    def test_context_vector_dimension(self, sample_users, sample_movies):
        """Context vector has exactly 60 dimensions."""
        user_row    = sample_users.set_index("user_id").loc[1]
        movie_row   = sample_movies.set_index("movie_id").loc[10]
        user_profile = np.zeros(19)
        ctx = build_context_vector(user_row, movie_row, user_profile)
        assert ctx.shape == (60,)

    def test_context_vector_dtype(self, sample_users, sample_movies):
        """Context vector is float64."""
        user_row    = sample_users.set_index("user_id").loc[1]
        movie_row   = sample_movies.set_index("movie_id").loc[10]
        user_profile = np.zeros(19)
        ctx = build_context_vector(user_row, movie_row, user_profile)
        assert ctx.dtype == np.float64

    def test_context_vector_no_nan(self, sample_users, sample_movies):
        """Context vector contains no NaN values."""
        user_row    = sample_users.set_index("user_id").loc[1]
        movie_row   = sample_movies.set_index("movie_id").loc[10]
        user_profile = np.random.randn(19)
        ctx = build_context_vector(user_row, movie_row, user_profile)
        assert not np.any(np.isnan(ctx))

    def test_context_vector_changes_with_movie(self, sample_users, sample_movies):
        """Different movies produce different context vectors."""
        user_row     = sample_users.set_index("user_id").loc[1]
        movie_lookup = sample_movies.set_index("movie_id")
        user_profile = np.zeros(19)
        ctx_a = build_context_vector(user_row, movie_lookup.loc[10], user_profile)
        ctx_b = build_context_vector(user_row, movie_lookup.loc[20], user_profile)
        assert not np.allclose(ctx_a, ctx_b)

    def test_context_vector_changes_with_user_profile(self, sample_users, sample_movies):
        """Different user profiles produce different context vectors."""
        user_row  = sample_users.set_index("user_id").loc[1]
        movie_row = sample_movies.set_index("movie_id").loc[10]
        profile_a = np.zeros(19)
        profile_b = np.ones(19) / 19
        ctx_a = build_context_vector(user_row, movie_row, profile_a)
        ctx_b = build_context_vector(user_row, movie_row, profile_b)
        assert not np.allclose(ctx_a, ctx_b)

    def test_interaction_term_is_elementwise_product(self, sample_users, sample_movies):
        """The interaction term (positions 38-56) is profile * genre."""
        user_row     = sample_users.set_index("user_id").loc[1]
        movie_row    = sample_movies.set_index("movie_id").loc[10]
        user_profile = np.random.RandomState(0).rand(19)
        ctx = build_context_vector(user_row, movie_row, user_profile)
        genre_vec   = np.array(movie_row["genre_vec"], dtype=float)
        interaction = user_profile * genre_vec
        np.testing.assert_array_almost_equal(ctx[38:57], interaction)


# ── User profile tests ────────────────────────────────────────────────────────

class TestUserProfiles:

    def test_profiles_shape(self, sample_ratings, sample_movies):
        """Profile DataFrame has correct shape."""
        profiles = build_user_genre_profiles(sample_ratings, sample_movies)
        assert "user_id" in profiles.columns
        assert len(profiles.columns) == 1 + 19  # user_id + 19 genres

    def test_profiles_normalised(self, sample_ratings, sample_movies):
        """Each user profile vector has L2 norm <= 1."""
        profiles = build_user_genre_profiles(sample_ratings, sample_movies)
        genre_cols = [f"pref_{g}" for g in GENRES]
        for _, row in profiles.iterrows():
            vec  = row[genre_cols].values.astype(float)
            norm = np.linalg.norm(vec)
            assert norm <= 1.0 + 1e-6, f"Norm {norm} > 1 for user {row['user_id']}"

    def test_profiles_no_nan(self, sample_ratings, sample_movies):
        """No NaN values in profiles."""
        profiles = build_user_genre_profiles(sample_ratings, sample_movies)
        assert not profiles.isnull().any().any()

    def test_profile_covers_all_users(self, sample_ratings, sample_movies):
        """Profile is computed for every user that has liked at least one movie."""
        liked_users = set(
            sample_ratings[sample_ratings["reward"] == 1]["user_id"].unique()
        )
        profiles = build_user_genre_profiles(sample_ratings, sample_movies,
                                              min_ratings=1)
        profile_users = set(profiles["user_id"].unique())
        assert liked_users.issubset(profile_users)


# ── Data loading tests (uses real data if available) ─────────────────────────

class TestDataLoading:

    DATA_DIR = Path("data/raw/ml-100k")

    @pytest.mark.skipif(
        not Path("data/raw/ml-100k/u.data").exists(),
        reason="MovieLens-100K not downloaded"
    )
    def test_load_ratings_shape(self):
        """Ratings DataFrame has correct columns."""
        ratings = load_ratings(self.DATA_DIR)
        assert set(["user_id", "movie_id", "rating", "timestamp", "reward"]) \
               .issubset(set(ratings.columns))

    @pytest.mark.skipif(
        not Path("data/raw/ml-100k/u.data").exists(),
        reason="MovieLens-100K not downloaded"
    )
    def test_load_ratings_count(self):
        """MovieLens-100K has exactly 100,000 ratings."""
        ratings = load_ratings(self.DATA_DIR)
        assert len(ratings) == 100_000

    @pytest.mark.skipif(
        not Path("data/raw/ml-100k/u.data").exists(),
        reason="MovieLens-100K not downloaded"
    )
    def test_ratings_value_range(self):
        """All ratings are in [1, 5]."""
        ratings = load_ratings(self.DATA_DIR)
        assert ratings["rating"].min() >= 1
        assert ratings["rating"].max() <= 5

    @pytest.mark.skipif(
        not Path("data/raw/ml-100k/u.data").exists(),
        reason="MovieLens-100K not downloaded"
    )
    def test_reward_is_binary(self):
        """Reward column is binary (0 or 1)."""
        ratings = load_ratings(self.DATA_DIR)
        assert set(ratings["reward"].unique()).issubset({0, 1})

    @pytest.mark.skipif(
        not Path("data/raw/ml-100k/u.item").exists(),
        reason="MovieLens-100K not downloaded"
    )
    def test_load_movies_genre_vectors(self):
        """Each movie has a valid genre vector of length 19."""
        movies = load_movies(self.DATA_DIR)
        assert "genre_vec" in movies.columns
        for vec in movies["genre_vec"]:
            assert len(vec) == 19
            assert all(v in [0, 1] for v in vec)

    @pytest.mark.skipif(
        not Path("data/raw/ml-100k/u.user").exists(),
        reason="MovieLens-100K not downloaded"
    )
    def test_load_users_normalised_age(self):
        """age_norm is in [0, 1]."""
        users = load_users(self.DATA_DIR)
        assert users["age_norm"].min() >= 0.0
        assert users["age_norm"].max() <= 1.0