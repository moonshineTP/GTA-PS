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

SampleModel        — sparse frequency model for small backups.
    Stores only observed successor counts and materialises one row on demand.
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
    SampleModel — sparse successor counts for small backups.
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
    Sparse empirical model used by small-backup Dyna planning.

    Counts are stored only for observed transitions. This provides
    N(s,a,s') / N(s,a) without materialising a dense transition tensor.
    """

    def __init__(self, n_states: int, n_actions: int):
        self._n_states = n_states
        self._n_actions = n_actions
        self._counts: dict[tuple[int, int], dict[int, int]] = {}
        self._totals = np.zeros((n_actions, n_states), dtype=np.int64)
        self._predecessor_counts: list[dict[tuple[int, int], int]] = [
            {} for _ in range(n_states)
        ]
        self._reward_sum = np.zeros(n_states, dtype=np.float64)
        self._reward_count = np.zeros(n_states, dtype=np.int64)

    @property
    def n_states(self) -> int:
        return self._n_states

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def update(self, s: int, a: int, s_next: int, r: float) -> None:
        key = (int(s), int(a))
        successors = self._counts.setdefault(key, {})
        successors[s_next] = successors.get(s_next, 0) + 1
        self._totals[a, s] += 1
        predecessors = self._predecessor_counts[s_next]
        predecessors[key] = predecessors.get(key, 0) + 1
        self._reward_sum[s] += r
        self._reward_count[s] += 1

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
        total = int(self._totals[a, s])
        if total == 0:
            row.fill(1.0 / self._n_states)
        else:
            for successor, count in self._counts[(s, a)].items():
                row[successor] = count / total
        return row

    def visit_count(self, s: int, a: int) -> int:
        return int(self._totals[a, s])

    def reward_mean(self, s: int) -> float:
        count = int(self._reward_count[s])
        return float(self._reward_sum[s] / count) if count else 0.0

    def predecessors(self, successor: int) -> list[tuple[int, int, float]]:
        result = []
        for (state, action), count in self._predecessor_counts[successor].items():
            probability = count / int(self._totals[action, state])
            result.append((state, action, probability))
        return sorted(result)

    def get_model(self) -> tuple[np.ndarray, np.ndarray]:
        A, S = self.n_actions, self.n_states
        P_hat = np.empty((A, S, S), dtype=np.float64)
        for action in range(A):
            for state in range(S):
                P_hat[action, state] = self.get_row(state, action)
        r_hat = np.divide(
            self._reward_sum,
            self._reward_count,
            out=np.zeros(S, dtype=np.float64),
            where=self._reward_count > 0,
        )
        return P_hat, r_hat

    def reset(self) -> None:
        self._counts.clear()
        self._totals.fill(0)
        self._predecessor_counts = [{} for _ in range(self._n_states)]
        self._reward_sum.fill(0.0)
        self._reward_count.fill(0)
