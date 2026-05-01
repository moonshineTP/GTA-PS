"""
Tabular model learners for Dyna-style planning.

Both classes share the same three-method interface:
    update(s, a, s_next, r)
    get_model() -> (P_hat: (|A|,|S|,|S|), r_hat: (|S|,))
    reset()

CountModelLearner  — distributional model.
    P_hat[a,s,s'] = N(a,s,s') / Σ_{s''} N(a,s,s'').
    Smoothed by a Dirichlet pseudocount (prior).
    Memory: O(|A||S|²).

SampleModel        — sample model (Sutton & Barto §8.3).
    Stores only the most-recently observed (s', r) per (s,a).
    P_hat[a,s,:] is a point mass on last_next[a,s].
    Unvisited (s,a) pairs keep a uniform prior row.
    Memory: O(|A||S|).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class EmpiricalModel(ABC):
    """
    Abstract base for tabular empirical model learners.

    Defines the interface consumed by DynaHost and any environment wrapper
    that builds P_hat / r_hat from rollout data, mirroring the OracleModel
    boundary for the distributional / sample sides respectively.

    Concrete implementations
    ------------------------
    CountModel  — distributional (frequency counts), O(|A||S|²) memory.
    SampleModel — last-transition sample model, O(|A||S|) memory.
    """

    @property
    @abstractmethod
    def n_states(self) -> int:
        """Number of states |S|."""

    @property
    @abstractmethod
    def n_actions(self) -> int:
        """Number of actions |A|."""

    @abstractmethod
    def update(self, s: int, a: int, s_next: int, r: float) -> None:
        """Record a single (s, a, s’, r) transition."""

    @abstractmethod
    def update_batch(
        self,
        states: "np.ndarray",
        actions: "np.ndarray",
        next_states: "np.ndarray",
        rewards: "np.ndarray",
    ) -> None:
        """Vectorised batch update."""

    @abstractmethod
    def get_model(self) -> tuple["np.ndarray", "np.ndarray"]:
        """
        Returns
        -------
        P_hat : (|A|, |S|, |S|) row-stochastic ndarray
        r_hat : (|S|,) ndarray
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset all accumulated data to the initial state."""


class CountModel(EmpiricalModel):
    """
    Builds an empirical tabular model via visitation counts.

    Parameters
    ----------
    n_states  : int
    n_actions : int
    prior     : float
        Dirichlet pseudocount added to each successor cell (prevents
        zero-probability transitions before all cells are visited).
    """

    def __init__(self, n_states: int, n_actions: int, prior: float = 1e-3):
        self._n_states = n_states
        self._n_actions = n_actions
        self.prior = prior
        self._counts = np.full((n_actions, n_states, n_states), prior)
        self._reward_sum = np.zeros(n_states)
        self._reward_cnt = np.zeros(n_states)

    @property
    def n_states(self) -> int:
        return self._n_states

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def update(self, s: int, a: int, s_next: int, r: float) -> None:
        """Record a single (s, a, s', r) transition."""
        self._counts[a, s, s_next] += 1.0
        self._reward_sum[s] += r
        self._reward_cnt[s] += 1.0

    def update_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        rewards: np.ndarray,
    ) -> None:
        """Vectorised batch update."""
        for s, a, sp, r in zip(states, actions, next_states, rewards):
            self.update(int(s), int(a), int(sp), float(r))

    def get_model(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        P_hat : (|A|, |S|, |S|) row-stochastic
        r_hat : (|S|,)
        """
        row_sums = self._counts.sum(axis=2, keepdims=True)
        P_hat = self._counts / row_sums

        visited = self._reward_cnt > 0
        r_hat = np.where(visited, self._reward_sum / np.maximum(self._reward_cnt, 1), 0.0)
        return P_hat, r_hat

    def reset(self) -> None:
        self._counts[:] = self.prior
        self._reward_sum[:] = 0.0
        self._reward_cnt[:] = 0.0


class SampleModel(EmpiricalModel):
    """
    Dyna sample model (Sutton & Barto §8.3).

    Stores only the most-recently observed (s', r) for each (s, a).
    On get_model(), emits a deterministic transition kernel:

        P_hat[a, s, last_next[a,s]] = 1.0,  all other entries 0.

    Unvisited (s, a) pairs retain a uniform row (flat prior).

    Compared to CountModel:
      - Memory: O(|A||S|) vs O(|A||S|²).
      - get_model() cost: O(|A||S|) vs O(|A||S|²).
      - Planning uses the exact last seen transition; stochastic dynamics
        are not captured — use CountModel for stochastic envs.

    Parameters
    ----------
    n_states  : int
    n_actions : int
    """

    def __init__(self, n_states: int, n_actions: int):
        self._n_states = n_states
        self._n_actions = n_actions
        # -1 sentinel for unvisited (a, s) pairs
        self._last_next   = np.full((n_actions, n_states), -1, dtype=np.intp)
        self._last_reward = np.zeros((n_actions, n_states), dtype=np.float64)
        self._visited     = np.zeros((n_actions, n_states), dtype=bool)

    @property
    def n_states(self) -> int:
        return self._n_states

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def update(self, s: int, a: int, s_next: int, r: float) -> None:
        """Record the most-recent (s, a) → (s', r) transition."""
        self._last_next[a, s]   = int(s_next)
        self._last_reward[a, s] = float(r)
        self._visited[a, s]     = True

    def update_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        rewards: np.ndarray,
    ) -> None:
        """Vectorised batch update (processes in order; last transition wins)."""
        for s, a, sp, r in zip(states, actions, next_states, rewards):
            self.update(int(s), int(a), int(sp), float(r))

    def get_model(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        P_hat : (|A|, |S|, |S|) — point-mass rows for visited (a,s),
                uniform rows for unvisited.
        r_hat : (|S|,) — mean of last_reward over visited actions per state.
        """
        A, S = self.n_actions, self.n_states

        # Start with uniform prior; overwrite visited rows.
        P_hat = np.full((A, S, S), 1.0 / S, dtype=np.float64)
        a_idx, s_idx = np.where(self._visited)
        if len(a_idx):
            P_hat[a_idx, s_idx, :] = 0.0
            P_hat[a_idx, s_idx, self._last_next[a_idx, s_idx]] = 1.0

        # r_hat: mean over visited actions for each state s.
        r_sum = (self._last_reward * self._visited).sum(axis=0)   # (S,)
        r_cnt = self._visited.sum(axis=0).astype(np.float64)      # (S,)
        r_hat = np.where(r_cnt > 0, r_sum / np.maximum(r_cnt, 1.0), 0.0)
        return P_hat, r_hat

    def reset(self) -> None:
        self._last_next[:]   = -1
        self._last_reward[:] = 0.0
        self._visited[:]     = False
