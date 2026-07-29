"""VaST-style value iteration over a linear state abstraction.

This module is a thin tabular proxy for VaST, not its variational neural
encoder. A standardized linear projection preserves the feature geometry, and
deterministic farthest-point tabulation builds ``a`` compact states. Planning
then applies frozen batches of Bellman backups to the induced compact MDP.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .batching import BatchedPriorityModule
from .dp_host import DpHost
from .gta_ps import GtaPS


class LinearStateProjection:
    """Project features linearly, then tabulate them into ``a`` local regions."""

    def __init__(self, a: int) -> None:
        if a < 1:
            raise ValueError(f"a must be positive, got {a!r}")
        self.a = int(a)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Return a deterministic abstract-state label for each feature row."""
        features = np.asarray(features, dtype=float)
        if features.ndim != 2 or features.shape[0] < 1 or features.shape[1] < 1:
            raise ValueError("features must have shape (n_states, n_features)")
        if not np.all(np.isfinite(features)):
            raise ValueError("features must contain only finite values")

        centered = features - features.mean(axis=0, keepdims=True)
        scale = centered.std(axis=0)
        scale[scale <= np.finfo(float).eps] = 1.0
        standardized = centered / scale

        if np.all(standardized == 0.0):
            return np.zeros(features.shape[0], dtype=np.int64)

        _, _, right_vectors = np.linalg.svd(standardized, full_matrices=False)
        projected = standardized @ right_vectors.T
        unique_points, inverse = np.unique(projected, axis=0, return_inverse=True)
        n_regions = min(self.a, unique_points.shape[0])

        first = 0
        centers = [first]
        min_distance = np.sum(
            (unique_points - unique_points[first]) ** 2,
            axis=1,
        )
        for _ in range(1, n_regions):
            next_center = int(np.argmax(min_distance))
            centers.append(next_center)
            distance = np.sum(
                (unique_points - unique_points[next_center]) ** 2,
                axis=1,
            )
            min_distance = np.minimum(min_distance, distance)

        center_points = unique_points[np.asarray(centers)]
        distances = np.sum(
            (unique_points[:, None, :] - center_points[None, :, :]) ** 2,
            axis=2,
        )
        unique_labels = np.argmin(distances, axis=1)
        return unique_labels[inverse].astype(np.int64, copy=False)


class LinearTabularModel:
    """Aggregate a tabular MDP through a fitted linear state projection."""

    def __init__(self, P: np.ndarray, features: np.ndarray, *, a: int) -> None:
        P = np.asarray(P, dtype=float)
        features = np.asarray(features, dtype=float)
        if P.ndim != 3 or P.shape[1] != P.shape[2]:
            raise ValueError("P must have shape (n_actions, n_states, n_states)")
        if features.ndim != 2 or features.shape[0] != P.shape[1]:
            raise ValueError("features must have one row per state")
        if not np.all(np.isfinite(P)):
            raise ValueError("P must contain only finite values")
        if np.any(P < 0.0) or not np.allclose(P.sum(axis=2), 1.0):
            raise ValueError("P must be row-stochastic")

        self.labels = LinearStateProjection(a).fit_transform(features)
        self.n_states = int(self.labels.max()) + 1
        self._concrete_states = P.shape[1]
        self._membership = np.eye(self.n_states)[self.labels]
        self._counts = self._membership.sum(axis=0)
        self.P = self._aggregate_transitions(P)

    def _aggregate_transitions(self, P: np.ndarray) -> np.ndarray:
        compact = np.empty((P.shape[0], self.n_states, self.n_states), dtype=float)
        for action in range(P.shape[0]):
            compact[action] = (
                self._membership.T @ P[action] @ self._membership
            ) / self._counts[:, None]
        return compact

    def aggregate(self, values: np.ndarray) -> np.ndarray:
        """Average a concrete-state vector within each abstract state."""
        values = np.asarray(values, dtype=float)
        if values.shape != (self._concrete_states,):
            raise ValueError(f"values must have shape ({self._concrete_states},)")
        if not np.all(np.isfinite(values)):
            raise ValueError("values must contain only finite values")
        return (self._membership.T @ values) / self._counts

    def lift(self, values: np.ndarray) -> np.ndarray:
        """Broadcast an abstract-state vector back to concrete states."""
        values = np.asarray(values, dtype=float)
        if values.shape != (self.n_states,):
            raise ValueError(f"values must have shape ({self.n_states},)")
        return values[self.labels]


class VaST:
    """Plan on a compact MDP induced by a linear state projection.

    ``K`` counts compact-state Bellman backups. A batch freezes the next
    ``batch_size`` states under the current Bellman-residual order.
    """

    def __init__(
        self,
        P: np.ndarray,
        r: np.ndarray,
        features: np.ndarray,
        *,
        gamma: float,
        a: int,
        batch_size: int,
    ) -> None:
        P = np.asarray(P, dtype=float)
        r = np.asarray(r, dtype=float)
        features = np.asarray(features, dtype=float)
        if P.ndim != 3 or P.shape[1] != P.shape[2]:
            raise ValueError("P must have shape (n_actions, n_states, n_states)")
        if r.shape != (P.shape[1],):
            raise ValueError("r must have shape (n_states,)")
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"gamma must lie in [0, 1), got {gamma!r}")
        if not np.all(np.isfinite(r)):
            raise ValueError("r must contain only finite values")
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size!r}")

        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.r = r.copy()
        self.abstraction = LinearTabularModel(P, features, a=a)
        self.n_abstract_states = self.abstraction.n_states
        self._r = self.abstraction.aggregate(self.r)
        self._V = np.zeros(self.n_abstract_states, dtype=float)
        self.V = np.zeros(P.shape[1], dtype=float)
        self._batch: deque[int] = deque()

    def _refill_batch(self) -> None:
        expected = np.einsum("asj,j->as", self.abstraction.P, self._V)
        residual = np.abs(self._r + self.gamma * expected.max(axis=0) - self._V)
        ranked = np.argsort(-residual, kind="stable")
        self._batch.extend(
            int(state) for state in ranked[: self.batch_size]
        )

    def warm_start(self, value: np.ndarray) -> None:
        """Initialize compact values from a concrete-state value vector."""
        value = np.asarray(value, dtype=float)
        if value.shape != self.V.shape:
            raise ValueError(f"value must have shape {self.V.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError("value must contain only finite values")
        self._V = self.abstraction.aggregate(value)
        self.V = self.abstraction.lift(self._V)
        self._batch.clear()

    def plan(self, K: int, r: np.ndarray | None = None) -> np.ndarray:
        """Apply ``K`` cyclic compact-state VI backups and lift the result."""
        if K < 0:
            raise ValueError(f"K must be nonnegative, got {K!r}")
        if r is not None:
            r = np.asarray(r, dtype=float)
            if r.shape != self.r.shape:
                raise ValueError(f"r must have shape {self.r.shape}")
            if not np.all(np.isfinite(r)):
                raise ValueError("r must contain only finite values")
            self.r = r.copy()
            self._r = self.abstraction.aggregate(self.r)
            self._batch.clear()

        for _ in range(K):
            if not self._batch:
                self._refill_batch()
            state = self._batch.popleft()
            expected = self.abstraction.P[:, state, :] @ self._V
            self._V[state] = self._r[state] + self.gamma * expected.max()

        self.V = self.abstraction.lift(self._V)
        return self.V.copy()

    def reset(self) -> None:
        """Reset values while retaining the learned abstraction and model."""
        self._V.fill(0.0)
        self.V.fill(0.0)
        self._batch.clear()


class GtaPsFa:
    """Run GTA-PS on the same compact table used by :class:`VaST`."""

    def __init__(
        self,
        P: np.ndarray,
        r: np.ndarray,
        features: np.ndarray,
        *,
        gamma: float,
        a: int,
        batch_size: int,
        q: int | None = None,
        gta_b: float = 1.0,
        **gta_kwargs,
    ) -> None:
        r = np.asarray(r, dtype=float)
        self.abstraction = LinearTabularModel(P, features, a=a)
        self.n_abstract_states = self.abstraction.n_states
        self.batch_size = int(batch_size)
        self.r = r.copy()
        compact_r = self.abstraction.aggregate(self.r)
        module = GtaPS(
            self.abstraction.P,
            compact_r,
            gamma=gamma,
            q=q,
            b=gta_b,
            **gta_kwargs,
        )
        module = BatchedPriorityModule(module, batch_size=batch_size)
        self._agent = DpHost.from_arrays(
            self.abstraction.P,
            compact_r,
            gamma,
            module,
        )
        self.V = np.zeros_like(self.r)

    def warm_start(self, value: np.ndarray) -> None:
        """Initialize host and scheduler values from concrete-state values."""
        compact_value = self.abstraction.aggregate(value)
        self._agent.V = compact_value.copy()
        self._agent.module.V = compact_value.copy()
        self.V = self.abstraction.lift(compact_value)

    def plan(self, K: int, r: np.ndarray | None = None) -> np.ndarray:
        """Apply GTA-PS backups to the compact MDP and lift the result."""
        if K < 0:
            raise ValueError(f"K must be nonnegative, got {K!r}")
        compact_r = None
        if r is not None:
            r = np.asarray(r, dtype=float)
            if r.shape != self.r.shape:
                raise ValueError(f"r must have shape {self.r.shape}")
            if not np.all(np.isfinite(r)):
                raise ValueError("r must contain only finite values")
            self.r = r.copy()
            compact_r = self.abstraction.aggregate(self.r)
        compact_value = self._agent.plan(K=K, r=compact_r)
        self.V = self.abstraction.lift(compact_value)
        return self.V.copy()

    def reset(self) -> None:
        """Reset compact GTA-PS values and lift the zero table."""
        self._agent.reset()
        self.V.fill(0.0)
