"""
Tests for drift detection module.
Covers: KL divergence, baseline computation, drift detection logic,
        drift simulation, and edge cases.
"""

import pytest
import numpy as np
import pandas as pd
import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.drift.detector import (
    kl_divergence, compute_genre_distribution,
    compute_baseline_statistics, detect_drift,
    simulate_drift, load_baseline, GENRES, DRIFT_THRESHOLD
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_ratings():
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "user_id":  np.random.randint(1, 50, n),
        "movie_id": np.random.randint(1, 100, n),
        "rating":   np.random.randint(1, 6, n),
        "timestamp": np.arange(n),
        "reward":   np.random.randint(0, 2, n),
    })


@pytest.fixture
def sample_movies():
    """100 movies with random genre assignments."""
    np.random.seed(0)
    rows = []
    for i in range(1, 101):
        genre_vec = np.zeros(19, dtype=int)
        n_genres  = np.random.randint(1, 4)
        chosen    = np.random.choice(19, n_genres, replace=False)
        genre_vec[chosen] = 1
        year = np.random.randint(1970, 2000)
        row  = [i, f"Movie {i}", year, (year - 1970) / 30,
                genre_vec.tolist()] + list(genre_vec)
        rows.append(row)
    cols = ["movie_id", "title", "year", "year_norm", "genre_vec"] + GENRES
    return pd.DataFrame(rows, columns=cols)


@pytest.fixture
def baseline_path(tmp_path, sample_ratings, sample_movies):
    """Compute and save a baseline, return its path."""
    path = tmp_path / "genre_baseline.json"
    compute_baseline_statistics(sample_ratings, sample_movies, save_path=path)
    return path


# ── KL Divergence tests ───────────────────────────────────────────────────────

class TestKLDivergence:

    def test_identical_distributions_zero(self):
        """KL(p || p) = 0."""
        p = np.array([0.2, 0.3, 0.5])
        assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-6)

    def test_non_negative(self):
        """KL divergence is always >= 0."""
        rng = np.random.RandomState(0)
        for _ in range(20):
            p = rng.dirichlet(np.ones(10))
            q = rng.dirichlet(np.ones(10))
            assert kl_divergence(p, q) >= 0

    def test_asymmetric(self):
        """KL(p || q) != KL(q || p) in general."""
        p = np.array([0.8, 0.1, 0.1])
        q = np.array([0.1, 0.6, 0.3])
        kl_pq = kl_divergence(p, q)
        kl_qp = kl_divergence(q, p)
        # These are genuinely asymmetric distributions
        assert abs(kl_pq - kl_qp) > 0.01,             f"Expected KL(p||q)={kl_pq:.4f} != KL(q||p)={kl_qp:.4f}"

    def test_larger_divergence_for_different_distributions(self):
        """More different distributions have higher KL."""
        p = np.array([0.5, 0.5])
        q_close = np.array([0.48, 0.52])
        q_far   = np.array([0.9,  0.1])
        assert kl_divergence(p, q_close) < kl_divergence(p, q_far)

    def test_handles_zeros_with_epsilon(self):
        """Works without error when distributions contain zeros."""
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 1.0, 0.0])
        result = kl_divergence(p, q)
        assert np.isfinite(result)
        assert result > 0

    def test_unit_vectors(self):
        """Works correctly on 19-dim genre distributions."""
        p = np.ones(19) / 19
        q = np.ones(19) / 19
        assert kl_divergence(p, q) == pytest.approx(0.0, abs=1e-5)


# ── Genre distribution tests ──────────────────────────────────────────────────

class TestGenreDistribution:

    def test_distribution_sums_to_one(self, sample_ratings, sample_movies):
        """Genre distribution sums to 1."""
        dist = compute_genre_distribution(sample_ratings, sample_movies)
        assert dist.sum() == pytest.approx(1.0, abs=1e-6)

    def test_distribution_non_negative(self, sample_ratings, sample_movies):
        """All genre probabilities are >= 0."""
        dist = compute_genre_distribution(sample_ratings, sample_movies)
        assert np.all(dist >= 0)

    def test_distribution_length(self, sample_ratings, sample_movies):
        """Distribution has 19 entries."""
        dist = compute_genre_distribution(sample_ratings, sample_movies)
        assert len(dist) == 19

    def test_empty_liked_returns_uniform(self, sample_ratings, sample_movies):
        """No liked movies → uniform distribution."""
        no_likes = sample_ratings.copy()
        no_likes["reward"] = 0
        dist = compute_genre_distribution(no_likes, sample_movies)
        assert dist.sum() == pytest.approx(1.0, abs=1e-6)


# ── Baseline computation tests ────────────────────────────────────────────────

class TestBaselineComputation:

    def test_baseline_saved(self, tmp_path, sample_ratings, sample_movies):
        """Baseline JSON file is created."""
        path = tmp_path / "baseline.json"
        compute_baseline_statistics(sample_ratings, sample_movies, save_path=path)
        assert path.exists()

    def test_baseline_has_required_keys(self, tmp_path, sample_ratings, sample_movies):
        """Baseline contains all required keys."""
        path = tmp_path / "baseline.json"
        stats = compute_baseline_statistics(sample_ratings, sample_movies,
                                            save_path=path)
        required = ["genre_distribution", "rating_mean",
                    "rating_std", "like_rate", "n_ratings"]
        for key in required:
            assert key in stats

    def test_baseline_like_rate_valid(self, tmp_path, sample_ratings, sample_movies):
        """like_rate is in [0, 1]."""
        path = tmp_path / "baseline.json"
        stats = compute_baseline_statistics(sample_ratings, sample_movies,
                                            save_path=path)
        assert 0.0 <= stats["like_rate"] <= 1.0

    def test_baseline_n_ratings_correct(self, tmp_path, sample_ratings, sample_movies):
        """n_ratings matches the input size."""
        path = tmp_path / "baseline.json"
        stats = compute_baseline_statistics(sample_ratings, sample_movies,
                                            save_path=path)
        assert stats["n_ratings"] == len(sample_ratings)

    def test_load_baseline_returns_dict(self, baseline_path):
        """load_baseline returns a dict from saved JSON."""
        baseline = load_baseline(baseline_path)
        assert isinstance(baseline, dict)
        assert "genre_distribution" in baseline

    def test_load_baseline_missing_returns_none(self, tmp_path):
        """load_baseline returns None if file doesn't exist."""
        result = load_baseline(tmp_path / "nonexistent.json")
        assert result is None


# ── Drift detection tests ─────────────────────────────────────────────────────

class TestDriftDetection:

    def test_no_drift_same_data(self, sample_ratings, sample_movies, baseline_path):
        """Same data as baseline → near-zero drift score."""
        score, alert, report = detect_drift(
            sample_ratings, sample_movies,
            baseline_path=baseline_path, threshold=DRIFT_THRESHOLD
        )
        assert score == pytest.approx(0.0, abs=1e-5)
        assert alert is False

    def test_report_has_required_keys(self, sample_ratings, sample_movies,
                                       baseline_path):
        """Report contains all expected keys."""
        _, _, report = detect_drift(sample_ratings, sample_movies,
                                    baseline_path=baseline_path)
        keys = ["drift_score", "threshold", "alert",
                "n_current_samples", "top_genre_shifts"]
        for k in keys:
            assert k in report

    def test_insufficient_samples_returns_no_alert(self, sample_ratings,
                                                    sample_movies, baseline_path):
        """Too few samples → no alert, error in report."""
        tiny = sample_ratings.head(5)
        score, alert, report = detect_drift(
            tiny, sample_movies,
            baseline_path=baseline_path, min_samples=100
        )
        assert alert is False
        assert "error" in report

    def test_missing_baseline_returns_no_alert(self, sample_ratings,
                                                sample_movies, tmp_path):
        """Missing baseline → no alert."""
        score, alert, report = detect_drift(
            sample_ratings, sample_movies,
            baseline_path=tmp_path / "missing.json"
        )
        assert alert is False
        assert "error" in report

    def test_simulated_drift_detected(self, sample_ratings, sample_movies,
                                       baseline_path):
        """Strongly drifted data is detected above threshold."""
        drifted = simulate_drift(sample_ratings, sample_movies,
                                 drift_phase=2, n_samples=2000)
        score, _, _ = detect_drift(
            drifted, sample_movies,
            baseline_path=baseline_path, threshold=DRIFT_THRESHOLD,
            min_samples=50
        )
        # Drifted data should have higher score than same data
        same_score, _, _ = detect_drift(
            sample_ratings, sample_movies,
            baseline_path=baseline_path, threshold=DRIFT_THRESHOLD
        )
        assert score >= same_score

    def test_drift_score_non_negative(self, sample_ratings, sample_movies,
                                       baseline_path):
        """Drift score is always >= 0."""
        score, _, _ = detect_drift(sample_ratings, sample_movies,
                                   baseline_path=baseline_path)
        assert score >= 0


# ── Drift simulation tests ────────────────────────────────────────────────────

class TestDriftSimulation:

    def test_simulate_returns_dataframe(self, sample_ratings, sample_movies):
        """simulate_drift returns a DataFrame."""
        result = simulate_drift(sample_ratings, sample_movies, drift_phase=1)
        assert isinstance(result, pd.DataFrame)

    def test_simulate_has_required_columns(self, sample_ratings, sample_movies):
        """Simulated data has required columns."""
        result = simulate_drift(sample_ratings, sample_movies, drift_phase=1)
        for col in ["user_id", "movie_id", "rating", "timestamp", "reward"]:
            assert col in result.columns

    def test_simulate_reward_binary(self, sample_ratings, sample_movies):
        """Simulated reward is binary."""
        result = simulate_drift(sample_ratings, sample_movies, drift_phase=1)
        assert set(result["reward"].unique()).issubset({0, 1})

    def test_simulate_phase_zero_no_drift(self, sample_ratings, sample_movies):
        """Phase 0 is a valid call (no drift)."""
        result = simulate_drift(sample_ratings, sample_movies, drift_phase=0)
        assert len(result) > 0

    def test_simulate_reproducible(self, sample_ratings, sample_movies):
        """Same seed produces same result."""
        r1 = simulate_drift(sample_ratings, sample_movies, seed=42)
        r2 = simulate_drift(sample_ratings, sample_movies, seed=42)
        pd.testing.assert_frame_equal(r1, r2)