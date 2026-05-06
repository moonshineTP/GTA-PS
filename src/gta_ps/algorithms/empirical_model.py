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
        Uniform prior probability assigned to unvisited (s, a) rows.
    """

    def __init__(self, n_states: int, n_actions: int, prior: float = 1e-3):
        self._n_states = n_states
        self._n_actions = n_actions
        self.prior = prior
        # Storage is sparse (init with 0, not prior) to avoid O(AS^2) density bottlenecks.
        self._counts = np.zeros((n_actions, n_states, n_states), dtype=np.float64)
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
        """Vectorised batch update using np.add.at for efficiency."""
        # Convert to intp for indexing
        s_idx = np.asarray(states, dtype=np.intp)
        a_idx = np.asarray(actions, dtype=np.intp)
        sn_idx = np.asarray(next_states, dtype=np.intp)
        
        np.add.at(self._counts, (a_idx, s_idx, sn_idx), 1.0)
        np.add.at(self._reward_sum, s_idx, rewards)
        np.add.at(self._reward_cnt, s_idx, 1.0)

    def get_row(self, s: int, a: int) -> np.ndarray:
        """Return the row-stochastic transition vector for (s, a). O(S)."""
        row_count = self._counts[a, s]
        total = row_count.sum()
        if total == 0:
            return np.full(self._n_states, 1.0 / self._n_states)
        return row_count / total

    def get_model(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns full model. O(AS^2) but vectorized (NumPy).
        """
        row_sums = self._counts.sum(axis=2, keepdims=True)
        # Avoid div by zero; unvisited rows will be handled via mask
        P_hat = self._counts / np.maximum(row_sums, 1e-6)
        
        unvisited = (row_sums.squeeze(2) == 0)
        if np.any(unvisited):
            P_hat[unvisited] = 1.0 / self._n_states

        visited = self._reward_cnt > 0
        r_hat = np.where(visited, self._reward_sum / np.maximum(self._reward_cnt, 1), 0.0)
        return P_hat, r_hat

    def reset(self) -> None:
        self._counts.fill(0.0)
        self._reward_sum.fill(0.0)
        self._reward_cnt.fill(0.0)


class SampleModel(EmpiricalModel):
    """
    Dyna sample model (Sutton & Barto §8.3).
    """

    def __init__(self, n_states: int, n_actions: int):
        self._n_states = n_states
        self._n_actions = n_actions
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
        for s, a, sp, r in zip(states, actions, next_states, rewards):
            self.update(int(s), int(a), int(sp), float(r))

    def get_row(self, s: int, a: int) -> np.ndarray:
        """O(S)."""
        row = np.zeros(self._n_states)
        if not self._visited[a, s]:
            row.fill(1.0 / self._n_states)
        else:
            row[self._last_next[a, s]] = 1.0
        return row

    def get_model(self) -> tuple[np.ndarray, np.ndarray]:
        A, S = self.n_actions, self.n_states
        P_hat = np.full((A, S, S), 1.0 / S, dtype=np.float64)
        a_idx, s_idx = np.where(self._visited)
        if len(a_idx):
            P_hat[a_idx, s_idx, :] = 0.0
            P_hat[a_idx, s_idx, self._last_next[a_idx, s_idx]] = 1.0

        r_sum = (self._last_reward * self._visited).sum(axis=0)
        r_cnt = self._visited.sum(axis=0).astype(np.float64)
        r_hat = np.where(r_cnt > 0, r_sum / np.maximum(r_cnt, 1.0), 0.0)
        return P_hat, r_hat

    def reset(self) -> None:
        self._last_next.fill(-1)
        self._last_reward.fill(0.0)
        self._visited.fill(False)
