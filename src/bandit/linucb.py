"""
LinUCB: Linear Upper Confidence Bound Bandit for Recommendation.

Reference: Li et al. (2010) "A Contextual-Bandit Approach to
Personalized News Article Recommendation", WWW 2010.

Algorithm (Disjoint LinUCB):
  For each arm (movie) a, maintain:
    A_a : (d x d) matrix = I + sum of x*x^T for chosen interactions
    b_a : (d,)   vector  = sum of r*x for chosen interactions

  At each step:
    theta_a = A_a^{-1} b_a          (ridge regression estimate)
    UCB_a   = theta_a^T x + alpha * sqrt(x^T A_a^{-1} x)

  Choose arm with highest UCB, observe reward, update A and b.

  Incremental update avoids recomputing inverse from scratch each step
  using the Sherman-Morrison formula.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class LinUCBArm:
    """
    State for a single arm (movie) in Disjoint LinUCB.

    Uses Sherman-Morrison for O(d^2) incremental inverse updates
    instead of O(d^3) full inversions.
    """

    def __init__(self, dim: int, alpha: float) -> None:
        self.dim = dim
        self.alpha = alpha

        # A = I_d (identity), b = 0_d
        self.A = np.eye(dim, dtype=np.float64)
        self.A_inv = np.eye(dim, dtype=np.float64)   # cached inverse
        self.b = np.zeros(dim, dtype=np.float64)

        self.n_pulls = 0
        self.total_reward = 0.0

    def compute_ucb(self, x: np.ndarray) -> Tuple[float, float]:
        """
        Compute UCB score for context vector x.

        Returns:
            ucb   : scalar UCB score
            theta_x : exploitation component (theta^T x)
        """
        theta = self.A_inv @ self.b
        theta_x = float(theta @ x)
        variance = float(x @ self.A_inv @ x)
        ucb = theta_x + self.alpha * np.sqrt(max(variance, 0.0))
        return ucb, theta_x

    def update(self, x: np.ndarray, reward: float) -> None:
        """
        Update arm parameters given observed context x and reward.
        Uses Sherman-Morrison to update A_inv in O(d^2).
        """
        # Sherman-Morrison: (A + x x^T)^{-1} =
        #   A^{-1} - (A^{-1} x x^T A^{-1}) / (1 + x^T A^{-1} x)
        Ax = self.A_inv @ x
        denom = 1.0 + float(x @ Ax)
        self.A_inv -= np.outer(Ax, Ax) / denom
        self.A += np.outer(x, x)
        self.b += reward * x

        self.n_pulls += 1
        self.total_reward += reward

    @property
    def empirical_ctr(self) -> float:
        """Click-through rate: fraction of pulls that returned reward=1."""
        return self.total_reward / self.n_pulls if self.n_pulls > 0 else 0.0


class LinUCB:
    """
    Disjoint LinUCB bandit for movie recommendation.

    Each movie (arm) maintains independent A and b matrices.
    Arms are created lazily on first encounter.

    Args:
        dim   : dimension of context vectors
        alpha : exploration parameter (higher = more exploration)
                Typical range: 0.1 – 2.0
    """

    def __init__(self, dim: int, alpha: float = 1.0) -> None:
        self.dim = dim
        self.alpha = alpha
        self.arms: Dict[int, LinUCBArm] = {}

        # Tracking
        self.t = 0                          # global timestep
        self.history: List[dict] = []       # for evaluation

    def _get_or_create_arm(self, movie_id: int) -> LinUCBArm:
        if movie_id not in self.arms:
            self.arms[movie_id] = LinUCBArm(self.dim, self.alpha)
        return self.arms[movie_id]

    def recommend(
        self,
        context_map: Dict[int, np.ndarray],
        exclude: Optional[List[int]] = None
    ) -> Tuple[int, float]:
        """
        Select the best movie from a candidate set.

        Args:
            context_map : {movie_id -> context_vector} for candidates
            exclude     : movie_ids to exclude (already seen)

        Returns:
            best_movie_id : chosen arm
            best_ucb      : UCB score of chosen arm
        """
        exclude = set(exclude or [])
        best_id, best_ucb = None, -np.inf

        for movie_id, x in context_map.items():
            if movie_id in exclude:
                continue
            arm = self._get_or_create_arm(movie_id)
            ucb, _ = arm.compute_ucb(x)
            if ucb > best_ucb:
                best_ucb = ucb
                best_id = movie_id

        return best_id, best_ucb

    def update(
        self,
        movie_id: int,
        context: np.ndarray,
        reward: float
    ) -> None:
        """
        Update the chosen arm with observed reward.

        Args:
            movie_id : arm that was chosen
            context  : context vector used for recommendation
            reward   : observed reward (1=like, 0=dislike/skip)
        """
        arm = self._get_or_create_arm(movie_id)
        arm.update(context, reward)
        self.t += 1

        self.history.append({
            "t": self.t,
            "movie_id": movie_id,
            "reward": reward,
            "n_arms_seen": len(self.arms)
        })

    def get_stats(self) -> dict:
        """Return summary statistics across all arms."""
        if not self.arms:
            return {}
        ctrs = [arm.empirical_ctr for arm in self.arms.values()]
        pulls = [arm.n_pulls for arm in self.arms.values()]
        return {
            "total_timesteps": self.t,
            "n_arms": len(self.arms),
            "mean_ctr": float(np.mean(ctrs)),
            "max_ctr": float(np.max(ctrs)),
            "total_reward": sum(
                arm.total_reward for arm in self.arms.values()
            ),
            "most_pulled_arm": max(self.arms, key=lambda k: self.arms[k].n_pulls),
            "max_pulls": int(np.max(pulls)),
        }

    def set_alpha(self, alpha: float) -> None:
        """Update exploration parameter for all arms."""
        self.alpha = alpha
        for arm in self.arms.values():
            arm.alpha = alpha
