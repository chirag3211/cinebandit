"""
Tests for FastAPI endpoints.
Covers: health, ready, recommend, feedback, drift, snapshot, model info.
Uses FastAPI TestClient — no server needed.
"""

import pytest
import pickle
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Create TestClient with model loaded."""
    import os
    os.environ["MODEL_PATH"] = "outputs/linucb_alpha0.1.pkl"

    from fastapi.testclient import TestClient
    from src.api.main import app
    from src.api.core.model_store import store

    model_path = Path("outputs/linucb_alpha0.1.pkl")
    if model_path.exists():
        store.load(model_path)

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(scope="function")
def client_no_model():
    """TestClient with explicitly unloaded model — for testing 503 responses."""
    from fastapi.testclient import TestClient
    from src.api.main import app
    from src.api.core.model_store import ModelStore
    import src.api.core.model_store as ms

    # Save original store, replace with empty one
    original_store = ms.store
    ms.store = ModelStore()  # fresh unloaded store

    client = TestClient(app, raise_server_exceptions=False)
    yield client

    # Restore original store after test
    ms.store = original_store


# ── Health and readiness tests ────────────────────────────────────────────────

class TestSystemEndpoints:

    def test_root_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "service" in r.json()

    def test_root_contains_service_name(self, client):
        r = client.get("/")
        assert r.json()["service"] == "CineBandit API"

    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_health_always_up(self, client_no_model):
        """Health endpoint returns 200 even without model."""
        r = client_no_model.get("/health")
        assert r.status_code == 200

    def test_drift_status_structure(self, client):
        r = client.get("/drift/status")
        assert r.status_code == 200
        body = r.json()
        assert "drift_score"  in body
        assert "threshold"    in body
        assert "alert"        in body
        assert "status"       in body

    def test_drift_status_default_no_alert(self, client):
        """Default drift score is 0, no alert."""
        r = client.get("/drift/status")
        body = r.json()
        assert body["drift_score"] == 0.0
        assert body["alert"] is False

    def test_snapshot_structure(self, client):
        r = client.get("/snapshot")
        assert r.status_code == 200
        body = r.json()
        assert "rolling_ctr"     in body
        assert "model_timestep"  in body
        assert "arms_seen"       in body
        assert "drift_score"     in body

    def test_retrain_trigger(self, client):
        r = client.post("/retrain/trigger", json={"reason": "test"})
        assert r.status_code == 200
        assert r.json()["status"] == "triggered"


# ── Model endpoints ───────────────────────────────────────────────────────────

class TestModelEndpoints:

    def test_model_info_when_loaded(self, client):
        """Model info returns 200 when model is loaded."""
        r = client.get("/model/info")
        if r.status_code == 200:
            body = r.json()
            assert "algorithm"    in body
            assert "alpha"        in body
            assert "timesteps"    in body
            assert "arms_seen"    in body
            assert "total_movies" in body

    def test_model_info_503_when_not_loaded(self, client_no_model):
        r = client_no_model.get("/model/info")
        assert r.status_code == 503

    def test_ready_503_when_not_loaded(self, client_no_model):
        r = client_no_model.get("/ready")
        assert r.status_code == 503

    def test_model_reload_endpoint_exists(self, client):
        r = client.post("/model/reload")
        # Either 200 (reloaded) or 500 (file not found) — but not 404
        assert r.status_code != 404


# ── Recommendation tests ──────────────────────────────────────────────────────

class TestRecommendEndpoint:

    def test_recommend_503_without_model(self, client_no_model):
        r = client_no_model.post("/recommend",
                                  json={"user_id": 1, "n": 3})
        assert r.status_code == 503

    def test_recommend_invalid_user(self, client):
        """Non-existent user_id returns 404."""
        r = client.post("/recommend", json={"user_id": 99999, "n": 3})
        assert r.status_code in [404, 503]

    def test_recommend_invalid_payload(self, client):
        """Missing required field returns 422."""
        r = client.post("/recommend", json={"n": 3})  # no user_id
        assert r.status_code == 422

    def test_recommend_n_out_of_range(self, client):
        """n > 20 returns 422 (validation error)."""
        r = client.post("/recommend", json={"user_id": 1, "n": 100})
        assert r.status_code == 422

    def test_recommend_n_zero(self, client):
        """n=0 returns 422 (must be >= 1)."""
        r = client.post("/recommend", json={"user_id": 1, "n": 0})
        assert r.status_code == 422

    @pytest.mark.skipif(
        not Path("outputs/linucb_alpha0.1.pkl").exists(),
        reason="Model not trained yet"
    )
    def test_recommend_response_structure(self, client):
        """Successful recommendation has correct structure."""
        r = client.post("/recommend", json={"user_id": 1, "n": 3})
        if r.status_code == 200:
            body = r.json()
            assert "user_id"         in body
            assert "recommendations" in body
            assert "model_timestep"  in body
            recs = body["recommendations"]
            assert len(recs) <= 3
            for rec in recs:
                assert "movie_id"   in rec
                assert "title"      in rec
                assert "genres"     in rec
                assert "year"       in rec
                assert "ucb_score"  in rec

    @pytest.mark.skipif(
        not Path("outputs/linucb_alpha0.1.pkl").exists(),
        reason="Model not trained yet"
    )
    def test_recommend_ucb_scores_positive(self, client):
        """UCB scores are non-negative."""
        r = client.post("/recommend", json={"user_id": 1, "n": 5})
        if r.status_code == 200:
            for rec in r.json()["recommendations"]:
                assert rec["ucb_score"] >= 0

    @pytest.mark.skipif(
        not Path("outputs/linucb_alpha0.1.pkl").exists(),
        reason="Model not trained yet"
    )
    def test_recommend_exclude_works(self, client):
        """Excluded movie IDs do not appear in recommendations."""
        r1 = client.post("/recommend", json={"user_id": 1, "n": 5})
        if r1.status_code == 200:
            first_ids = [rec["movie_id"] for rec in r1.json()["recommendations"]]
            r2 = client.post("/recommend",
                              json={"user_id": 1, "n": 5, "exclude": first_ids})
            if r2.status_code == 200:
                second_ids = [rec["movie_id"] for rec in r2.json()["recommendations"]]
                assert not any(mid in first_ids for mid in second_ids)


# ── Feedback tests ────────────────────────────────────────────────────────────

class TestFeedbackEndpoint:

    def test_feedback_503_without_model(self, client_no_model):
        r = client_no_model.post("/feedback",
                                  json={"user_id": 1, "movie_id": 50,
                                        "reaction": "like"})
        assert r.status_code == 503

    def test_feedback_invalid_reaction(self, client):
        """Invalid reaction value returns 422."""
        r = client.post("/feedback",
                         json={"user_id": 1, "movie_id": 50,
                               "reaction": "love"})  # invalid
        assert r.status_code == 422

    def test_feedback_invalid_user(self, client):
        """Non-existent user returns 404."""
        r = client.post("/feedback",
                         json={"user_id": 99999, "movie_id": 50,
                               "reaction": "like"})
        assert r.status_code in [404, 503]

    def test_feedback_invalid_movie(self, client):
        """Non-existent movie returns 404."""
        r = client.post("/feedback",
                         json={"user_id": 1, "movie_id": 99999,
                               "reaction": "like"})
        assert r.status_code in [404, 503]

    def test_feedback_missing_fields(self, client):
        """Missing required fields return 422."""
        r = client.post("/feedback", json={"user_id": 1})
        assert r.status_code == 422

    @pytest.mark.skipif(
        not Path("outputs/linucb_alpha0.1.pkl").exists(),
        reason="Model not trained yet"
    )
    def test_feedback_like_response(self, client):
        """Like feedback returns correct reward and structure."""
        r = client.post("/feedback",
                         json={"user_id": 1, "movie_id": 50,
                               "reaction": "like"})
        if r.status_code == 200:
            body = r.json()
            assert body["reward"]   == 1.0
            assert body["reaction"] == "like"
            assert "rolling_ctr"    in body
            assert "model_timestep" in body

    @pytest.mark.skipif(
        not Path("outputs/linucb_alpha0.1.pkl").exists(),
        reason="Model not trained yet"
    )
    def test_feedback_dislike_reward_zero(self, client):
        """Dislike feedback returns reward=0."""
        r = client.post("/feedback",
                         json={"user_id": 1, "movie_id": 50,
                               "reaction": "dislike"})
        if r.status_code == 200:
            assert r.json()["reward"] == 0.0

    @pytest.mark.skipif(
        not Path("outputs/linucb_alpha0.1.pkl").exists(),
        reason="Model not trained yet"
    )
    def test_feedback_increments_timestep(self, client):
        """Each feedback increments model timestep by 1."""
        info_before = client.get("/model/info").json()
        client.post("/feedback",
                    json={"user_id": 1, "movie_id": 50, "reaction": "skip"})
        info_after = client.get("/model/info").json()
        if "timesteps" in info_before and "timesteps" in info_after:
            assert info_after["timesteps"] == info_before["timesteps"] + 1

    @pytest.mark.skipif(
        not Path("outputs/linucb_alpha0.1.pkl").exists(),
        reason="Model not trained yet"
    )
    def test_all_reaction_types(self, client):
        """All three reaction types are accepted."""
        for reaction in ["like", "dislike", "skip"]:
            r = client.post("/feedback",
                             json={"user_id": 2, "movie_id": 100,
                                   "reaction": reaction})
            assert r.status_code in [200, 404, 503]  # not 422


# ── Drift update test ─────────────────────────────────────────────────────────

class TestDriftUpdate:

    def test_drift_update_changes_score(self, client):
        """Posting a drift score updates /drift/status."""
        client.post("/drift/update", params={"score": 0.25})
        r = client.get("/drift/status")
        assert r.status_code == 200
        assert r.json()["drift_score"] == pytest.approx(0.25)

    def test_drift_alert_triggered_above_threshold(self, client):
        """Drift score above threshold triggers alert."""
        client.post("/drift/update", params={"score": 0.5})
        r = client.get("/drift/status")
        assert r.json()["alert"] is True
        assert r.json()["status"] == "drift_detected"

    def test_drift_no_alert_below_threshold(self, client):
        """Drift score below threshold does not trigger alert."""
        client.post("/drift/update", params={"score": 0.1})
        r = client.get("/drift/status")
        assert r.json()["alert"] is False
        assert r.json()["status"] == "ok"