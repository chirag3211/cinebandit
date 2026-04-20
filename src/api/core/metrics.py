"""
Prometheus metrics for CineBandit API.

All metrics are defined here as module-level singletons and imported
by route handlers. This avoids duplicate metric registration errors
when FastAPI reloads modules during development.
"""

from prometheus_client import Counter, Histogram, Gauge, Summary

# ── Request metrics ───────────────────────────────────────────────────────────

RECOMMENDATION_REQUESTS = Counter(
    "cinebandit_recommendation_requests_total",
    "Total number of recommendation requests",
    ["status"]   # labels: success, error
)

RECOMMENDATION_LATENCY = Histogram(
    "cinebandit_recommendation_latency_seconds",
    "Recommendation request latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
)

FEEDBACK_RECEIVED = Counter(
    "cinebandit_feedback_total",
    "Total feedback events received",
    ["reaction"]  # labels: like, dislike, skip
)

# ── Model metrics ─────────────────────────────────────────────────────────────

MODEL_VERSION = Gauge(
    "cinebandit_model_timesteps",
    "Current model timestep (proxy for version)"
)

MODEL_ALPHA = Gauge(
    "cinebandit_model_alpha",
    "Current model exploration parameter alpha"
)

ARMS_SEEN = Gauge(
    "cinebandit_arms_seen_total",
    "Number of unique movies the model has updated on"
)

# ── CTR metrics ───────────────────────────────────────────────────────────────

CUMULATIVE_REWARD = Counter(
    "cinebandit_cumulative_reward_total",
    "Cumulative reward (likes) received since startup"
)

ROLLING_CTR = Gauge(
    "cinebandit_rolling_ctr",
    "Rolling CTR over last 100 feedback events"
)

# ── Retraining metrics ────────────────────────────────────────────────────────

RETRAIN_RUNS = Counter(
    "cinebandit_retrain_runs_total",
    "Total retraining pipeline runs triggered",
    ["status"]   # labels: triggered, success, failed
)

DRIFT_SCORE = Gauge(
    "cinebandit_drift_score",
    "Current KL divergence drift score (0 = no drift)"
)

# ── Infrastructure ────────────────────────────────────────────────────────────

MODEL_LOAD_TIME = Gauge(
    "cinebandit_model_load_time_seconds",
    "Time taken to load the model at startup"
)

API_ERRORS = Counter(
    "cinebandit_api_errors_total",
    "Total API errors",
    ["endpoint", "error_type"]
)


# ── Rolling CTR tracker ───────────────────────────────────────────────────────

class RollingCTRTracker:
    """
    Tracks CTR over a sliding window of the last N feedback events.
    Thread-safe via a simple list + lock.
    """

    def __init__(self, window: int = 100) -> None:
        import threading
        self.window = window
        self._rewards: list = []
        self._lock = threading.Lock()

    def record(self, reward: float) -> float:
        """Record a reward and return current rolling CTR."""
        with self._lock:
            self._rewards.append(reward)
            if len(self._rewards) > self.window:
                self._rewards.pop(0)
            ctr = sum(self._rewards) / len(self._rewards)
        ROLLING_CTR.set(ctr)
        return ctr

    @property
    def current_ctr(self) -> float:
        with self._lock:
            if not self._rewards:
                return 0.0
            return sum(self._rewards) / len(self._rewards)


# Module-level tracker instance
ctr_tracker = RollingCTRTracker(window=100)
