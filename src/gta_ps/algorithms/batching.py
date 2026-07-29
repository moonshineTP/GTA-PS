"""Frozen top-m batching for prioritized-sweeping modules."""
from __future__ import annotations

from collections import deque

import numpy as np

from .base import PrioritizedSweepingModule


class BatchedPriorityModule(PrioritizedSweepingModule):
    """Freeze the next ``batch_size`` choices from an existing priority module.

    The wrapped module still receives every backup and predecessor update.  It
    is queried again only after the frozen batch is exhausted, isolating the
    cost of delayed rekeying without changing the module's scoring rule.
    """

    def __init__(self, module: PrioritizedSweepingModule, batch_size: int) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size!r}")
        self.module = module
        self.batch_size = int(batch_size)
        self._batch: deque[int] = deque()

    @property
    def V(self) -> np.ndarray:
        """Expose the wrapped module's value state for DP warm starts."""
        return self.module.V

    @V.setter
    def V(self, value: np.ndarray) -> None:
        self.module.V = value

    def reward_update(self, delta_r: np.ndarray) -> None:
        """Apply the shift and discard a batch ranked under the old reward."""
        self._batch.clear()
        self.module.reward_update(delta_r)

    def pop_and_prioritize(self) -> int | None:
        """Return the next frozen state, refilling from the wrapped priority."""
        if not self._batch:
            self._batch.extend(self.pop_batch(self.batch_size))
        return self._batch.popleft() if self._batch else None

    def pop_batch(self, limit: int) -> list[int]:
        """Select top-m states before any member of the batch is updated."""
        selected: list[int] = []
        for _ in range(min(self.batch_size, max(0, limit))):
            state = self.module.pop_and_prioritize()
            if state is None:
                break
            selected.append(state)
        return selected

    def push_predecessors(self, s: int, xi_s: float) -> None:
        self.module.push_predecessors(s, xi_s)

    def notify_backup(self, s: int, adv_PS_row: np.ndarray, xi_s: float) -> None:
        self.module.notify_backup(s, adv_PS_row, xi_s)

    def update_model(self, P: np.ndarray, r: np.ndarray) -> None:
        self.module.update_model(P, r)

    def update_value(self, s: int, v_s: float) -> None:
        self.module.update_value(s, v_s)

    def reset(self) -> None:
        self._batch.clear()
        self.module.reset()
