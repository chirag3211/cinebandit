"""
Tests for LinUCB bandit implementation.
Covers: LinUCBArm, LinUCB class, Sherman-Morrison update,
        UCB computation, arm creation, and recommendation logic.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bandit.linucb import LinUCBArm, LinUCB


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def dim():
    return 10

@pytest.fixture
def alpha():
    return 1.0

@pytest.fixture
def arm(dim, alpha):
    return LinUCBArm(dim=dim, alpha=alpha)

@pytest.fixture
def model(dim, alpha):
    return LinUCB(dim=dim, alpha=alpha)

@pytest.fixture
def context(dim):
    rng = np.random.RandomState(42)
    x = rng.randn(dim)
    return x / np.linalg.norm(x)


# ── LinUCBArm tests ───────────────────────────────────────────────────────────

class TestLinUCBArm:

    def test_initialisation_shape(self, arm, dim):
        """A and b are initialised to correct shapes."""
        assert arm.A.shape == (dim, dim)
        assert arm.b.shape == (dim,)
        assert arm.A_inv.shape == (dim, dim)

    def test_initialisation_identity(self, arm, dim):
        """A starts as identity matrix."""
        np.testing.assert_array_almost_equal(arm.A, np.eye(dim))

    def test_initialisation_zero_b(self, arm, dim):
        """b starts as zero vector."""
        np.testing.assert_array_almost_equal(arm.b, np.zeros(dim))

    def test_ucb_returns_scalar(self, arm, context):
        """compute_ucb returns a scalar UCB score."""
        ucb, theta_x = arm.compute_ucb(context)
        assert isinstance(ucb, float)
        assert isinstance(theta_x, float)

    def test_ucb_non_negative_on_fresh_arm(self, arm, context):
        """Fresh arm UCB >= 0 since variance term is always positive."""
        ucb, _ = arm.compute_ucb(context)
        assert ucb >= 0

    def test_update_increments_pulls(self, arm, context):
        """Update increments n_pulls by 1."""
        assert arm.n_pulls == 0
        arm.update(context, reward=1.0)
        assert arm.n_pulls == 1
        arm.update(context, reward=0.0)
        assert arm.n_pulls == 2

    def test_update_accumulates_reward(self, arm, context):
        """Total reward accumulates correctly."""
        arm.update(context, reward=1.0)
        arm.update(context, reward=1.0)
        arm.update(context, reward=0.0)
        assert arm.total_reward == 2.0

    def test_empirical_ctr_zero_pulls(self, arm):
        """CTR is 0.0 when no pulls have been made."""
        assert arm.empirical_ctr == 0.0

    def test_empirical_ctr_correct(self, arm, context):
        """CTR = total_reward / n_pulls."""
        arm.update(context, reward=1.0)
        arm.update(context, reward=0.0)
        assert arm.empirical_ctr == pytest.approx(0.5)

    def test_sherman_morrison_update(self, arm, context):
        """A_inv after update equals inverse of A after update."""
        arm.update(context, reward=1.0)
        expected_inv = np.linalg.inv(arm.A)
        np.testing.assert_array_almost_equal(arm.A_inv, expected_inv, decimal=6)

    def test_multiple_updates_consistency(self, arm, dim):
        """A_inv remains consistent with A after multiple updates."""
        rng = np.random.RandomState(0)
        for _ in range(20):
            x = rng.randn(dim)
            arm.update(x, reward=float(rng.randint(0, 2)))
        expected_inv = np.linalg.inv(arm.A)
        np.testing.assert_array_almost_equal(arm.A_inv, expected_inv, decimal=5)

    def test_higher_alpha_higher_ucb(self, dim, context):
        """Higher alpha → higher UCB score (more exploration)."""
        arm_low  = LinUCBArm(dim=dim, alpha=0.1)
        arm_high = LinUCBArm(dim=dim, alpha=2.0)
        ucb_low,  _ = arm_low.compute_ucb(context)
        ucb_high, _ = arm_high.compute_ucb(context)
        assert ucb_high > ucb_low


# ── LinUCB model tests ────────────────────────────────────────────────────────

class TestLinUCB:

    def test_initialisation(self, model):
        """Model starts with no arms and timestep 0."""
        assert len(model.arms) == 0
        assert model.t == 0

    def test_recommend_creates_arm(self, model, dim):
        """Recommend creates arms lazily on first call."""
        rng = np.random.RandomState(1)
        ctx = {1: rng.randn(dim), 2: rng.randn(dim)}
        model.recommend(ctx)
        assert len(model.arms) == 2

    def test_recommend_returns_valid_movie(self, model, dim):
        """Recommend returns a movie_id from the candidate pool."""
        rng = np.random.RandomState(2)
        pool = {10: rng.randn(dim), 20: rng.randn(dim), 30: rng.randn(dim)}
        chosen, ucb = model.recommend(pool)
        assert chosen in pool
        assert isinstance(ucb, float)

    def test_recommend_excludes_seen(self, model, dim):
        """Exclude parameter prevents recommending seen movies."""
        rng = np.random.RandomState(3)
        pool = {1: rng.randn(dim), 2: rng.randn(dim), 3: rng.randn(dim)}
        chosen, _ = model.recommend(pool, exclude=[1, 2])
        assert chosen == 3

    def test_update_increments_timestep(self, model, dim):
        """Update increments global timestep."""
        x = np.ones(dim) / np.sqrt(dim)
        model.update(1, x, reward=1.0)
        assert model.t == 1
        model.update(2, x, reward=0.0)
        assert model.t == 2

    def test_update_creates_arm(self, model, dim):
        """Update creates arm if it doesn't exist."""
        x = np.ones(dim) / np.sqrt(dim)
        assert 99 not in model.arms
        model.update(99, x, reward=1.0)
        assert 99 in model.arms

    def test_history_recorded(self, model, dim):
        """Each update is recorded in history."""
        x = np.ones(dim) / np.sqrt(dim)
        model.update(1, x, 1.0)
        model.update(2, x, 0.0)
        assert len(model.history) == 2
        assert model.history[0]["movie_id"] == 1
        assert model.history[1]["movie_id"] == 2

    def test_exploitation_after_training(self, dim):
        """After training, model prefers arm with higher reward history."""
        model = LinUCB(dim=dim, alpha=0.01)  # low alpha = exploit
        rng = np.random.RandomState(42)

        # Train arm 1 with mostly likes
        x1 = np.zeros(dim); x1[0] = 1.0
        for _ in range(50):
            model.update(1, x1, reward=1.0)

        # Train arm 2 with mostly dislikes
        x2 = np.zeros(dim); x2[1] = 1.0
        for _ in range(50):
            model.update(2, x2, reward=0.0)

        # Model should prefer arm 1
        chosen, _ = model.recommend({1: x1, 2: x2})
        assert chosen == 1

    def test_set_alpha_updates_all_arms(self, model, dim):
        """set_alpha updates alpha on all existing arms."""
        x = np.ones(dim) / np.sqrt(dim)
        model.update(1, x, 1.0)
        model.update(2, x, 1.0)
        model.set_alpha(0.5)
        assert model.alpha == 0.5
        for arm in model.arms.values():
            assert arm.alpha == 0.5

    def test_get_stats_returns_dict(self, model, dim):
        """get_stats returns a dict with expected keys."""
        x = np.ones(dim) / np.sqrt(dim)
        model.update(1, x, 1.0)
        stats = model.get_stats()
        assert "total_timesteps" in stats
        assert "n_arms" in stats
        assert "mean_ctr" in stats

    def test_recommend_single_candidate(self, model, dim):
        """Works correctly with a single candidate."""
        x = np.ones(dim) / np.sqrt(dim)
        chosen, _ = model.recommend({42: x})
        assert chosen == 42

    def test_recommend_empty_after_exclude(self, model, dim):
        """Returns None when all candidates are excluded."""
        x = np.ones(dim) / np.sqrt(dim)
        chosen, ucb = model.recommend({1: x}, exclude=[1])
        assert chosen is None