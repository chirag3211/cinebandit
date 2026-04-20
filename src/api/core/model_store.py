"""
ModelStore — singleton that holds the live LinUCB model in memory.

Responsibilities:
  - Load model from disk on startup
  - Expose recommend() and update() to route handlers
  - Track readiness state (is a model loaded?)
  - Support hot-reload (swap model without restarting the server)

Design: a single module-level instance is imported by all routes.
This avoids passing the model through FastAPI's dependency injection
for every request, keeping route handlers clean.
"""

import pickle
import logging
import threading
from pathlib import Path
from typing import Optional, Dict
import numpy as np

from src.bandit.linucb import LinUCB
from src.data.loader import (
    load_movies, load_users, build_user_genre_profiles,
    build_context_vector, GENRES, DATA_DIR
)

logger = logging.getLogger(__name__)

# Default model path — overridden by environment variable in Docker
DEFAULT_MODEL_PATH = Path("outputs/linucb_alpha0.1.pkl")


class ModelStore:
    """
    Thread-safe singleton model store.

    All public methods acquire a read/write lock so the API can serve
    recommendations while a background retraining job swaps in a new model.
    """

    def __init__(self) -> None:
        self._model: Optional[LinUCB] = None
        self._model_path: Optional[Path] = None
        self._lock = threading.RLock()

        # Data lookups — loaded once at startup
        self._movies = None
        self._users = None
        self._profiles = None
        self._movie_lookup = None
        self._user_lookup = None
        self._profile_lookup = None
        self._all_movie_ids = None

        self._ready = False

    # ── Startup ───────────────────────────────────────────────────────────────

    def load(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        """
        Load model + data lookups from disk.
        Called once at FastAPI startup.
        """
        logger.info(f"[store] Loading model from {model_path} ...")

        if not model_path.exists():
            logger.warning(
                f"[store] Model file not found at {model_path}. "
                "API will start but /ready will return False until a model is loaded."
            )
            self._ready = False
            return

        with self._lock:
            with open(model_path, "rb") as f:
                self._model = pickle.load(f)
            self._model_path = model_path
            logger.info(
                f"[store] Model loaded: alpha={self._model.alpha}, "
                f"arms={len(self._model.arms)}, t={self._model.t}"
            )

        self._load_data()
        self._ready = True
        logger.info("[store] ModelStore ready.")

    def _load_data(self) -> None:
        """Load and index movie/user/profile data for fast lookup."""
        logger.info("[store] Loading data lookups ...")
        self._movies = load_movies(DATA_DIR)
        self._users = load_users(DATA_DIR)
        self._profiles = build_user_genre_profiles(
            # load ratings for profile building
            __import__(
                "src.data.loader", fromlist=["load_ratings"]
            ).load_ratings(DATA_DIR),
            self._movies
        )
        self._movie_lookup = self._movies.set_index("movie_id")
        self._user_lookup = self._users.set_index("user_id")
        self._profile_lookup = self._profiles.set_index("user_id")
        self._all_movie_ids = self._movies["movie_id"].tolist()
        logger.info(
            f"[store] Data loaded: {len(self._movies)} movies, "
            f"{len(self._users)} users"
        )

    # ── Hot reload ────────────────────────────────────────────────────────────

    def reload(self, model_path: Optional[Path] = None) -> bool:
        """
        Swap in a new model without restarting the server.
        Also loads data lookups if not yet initialised.
        Returns True if reload succeeded.
        """
        path = model_path or self._model_path or DEFAULT_MODEL_PATH
        try:
            with open(path, "rb") as f:
                new_model = pickle.load(f)
            with self._lock:
                self._model = new_model
                self._model_path = path
            # Load data lookups if not already loaded
            if self._movie_lookup is None:
                self._load_data()
            with self._lock:
                self._ready = True
            logger.info(f"[store] Model hot-reloaded from {path}")
            return True
        except Exception as e:
            logger.error(f"[store] Reload failed: {e}")
            return False

    # ── Inference ─────────────────────────────────────────────────────────────

    def recommend(
        self,
        user_id: int,
        n: int = 5,
        exclude: Optional[list] = None
    ) -> list:
        """
        Recommend n movies for a given user.

        Returns a list of dicts:
          { movie_id, title, genres, year, ucb_score }
        """
        if not self._ready:
            raise RuntimeError("Model not loaded")

        exclude = set(exclude or [])

        # Get user data
        if user_id not in self._user_lookup.index:
            raise ValueError(f"Unknown user_id: {user_id}")

        user_row = self._user_lookup.loc[user_id]

        # Get user genre profile
        if user_id in self._profile_lookup.index:
            profile_row = self._profile_lookup.loc[user_id]
            user_profile = profile_row[
                [f"pref_{g}" for g in GENRES]
            ].values.astype(np.float64)
        else:
            user_profile = np.zeros(len(GENRES))

        # Build context map for all candidate movies
        import random
        rng = random.Random()
        candidates = [
            m for m in self._all_movie_ids if m not in exclude
        ]
        # Sample a pool of 100 candidates for efficiency
        pool = rng.sample(candidates, min(100, len(candidates)))

        context_map: Dict[int, np.ndarray] = {}
        for mid in pool:
            if mid not in self._movie_lookup.index:
                continue
            mrow = self._movie_lookup.loc[mid]
            ctx = build_context_vector(user_row, mrow, user_profile)
            context_map[mid] = ctx

        # Get top-n recommendations greedily
        recommendations = []
        chosen = set()

        with self._lock:
            for _ in range(min(n, len(context_map))):
                remaining = {
                    k: v for k, v in context_map.items()
                    if k not in chosen
                }
                if not remaining:
                    break
                best_id, best_ucb = self._model.recommend(remaining)
                chosen.add(best_id)

                mrow = self._movie_lookup.loc[best_id]
                genres = [g for g in GENRES if mrow[g] == 1]
                recommendations.append({
                    "movie_id": int(best_id),
                    "title": str(mrow["title"]),
                    "genres": genres,
                    "year": int(mrow["year"]),
                    "ucb_score": round(float(best_ucb), 4)
                })

        return recommendations

    # ── Feedback / update ─────────────────────────────────────────────────────

    def update(
        self,
        user_id: int,
        movie_id: int,
        reward: float
    ) -> None:
        """
        Incrementally update the bandit with observed feedback.
        Called after every like/dislike/skip.
        """
        if not self._ready:
            raise RuntimeError("Model not loaded")

        if user_id not in self._user_lookup.index:
            raise ValueError(f"Unknown user_id: {user_id}")
        if movie_id not in self._movie_lookup.index:
            raise ValueError(f"Unknown movie_id: {movie_id}")

        user_row = self._user_lookup.loc[user_id]
        movie_row = self._movie_lookup.loc[movie_id]

        if user_id in self._profile_lookup.index:
            profile_row = self._profile_lookup.loc[user_id]
            user_profile = profile_row[
                [f"pref_{g}" for g in GENRES]
            ].values.astype(np.float64)
        else:
            user_profile = np.zeros(len(GENRES))

        ctx = build_context_vector(user_row, movie_row, user_profile)

        with self._lock:
            self._model.update(movie_id, ctx, reward)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def info(self) -> dict:
        if not self._ready or self._model is None:
            return {"status": "not_loaded"}
        with self._lock:
            stats = self._model.get_stats()
        return {
            "model_path": str(self._model_path),
            "algorithm": "LinUCB",
            "alpha": self._model.alpha,
            "timesteps": self._model.t,
            "arms_seen": len(self._model.arms),
            "total_movies": len(self._all_movie_ids) if self._all_movie_ids else 0,
            **stats
        }


# Module-level singleton — imported by all route handlers
store = ModelStore()