"""Partitioned H2 scheduler with fixed Gauss-Seidel variable order."""
from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence

import numpy as np

from .standard_ps import StandardPS


class GpsH2(StandardPS):
    """Prioritize blocks with normalized H2 and solve each by Gauss-Seidel.

    ``batch_size`` is the partition size.  If ``blocks`` is omitted, states
    are split into deterministic contiguous partitions.  A modified
    topological sort fixes the intra-partition order once at initialization.
    """

    def __init__(
        self,
        P: np.ndarray,
        r: np.ndarray,
        gamma: float = 0.95,
        batch_size: int = 1,
        upsilon: float | None = None,
        blocks: Sequence[Iterable[int]] | None = None,
        solve_tolerance: float = 1e-6,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size!r}")
        if np.any(r < 0.0):
            raise ValueError("GpsH2 requires non-negative rewards for its H2 value term.")
        if not np.isfinite(solve_tolerance) or solve_tolerance <= 0.0:
            raise ValueError(
                f"solve_tolerance must be finite and positive, got {solve_tolerance!r}"
            )

        normalized_upsilon = 1.0 - gamma if upsilon is None else float(upsilon)
        if not np.isfinite(normalized_upsilon) or normalized_upsilon < 0.0:
            raise ValueError(f"upsilon must be finite and non-negative, got {upsilon!r}")

        self.batch_size = int(batch_size)
        self.upsilon = normalized_upsilon
        self.solve_tolerance = float(solve_tolerance)
        self._requested_blocks = blocks
        self._active_block: int | None = None
        self._block_sweep: deque[int] = deque()
        super().__init__(P, r, gamma=gamma)
        self.blocks = self._initialize_blocks(blocks)
        self.block_orders = tuple(self._fixed_variable_order(block) for block in self.blocks)

    def _initialize_blocks(
        self,
        blocks: Sequence[Iterable[int]] | None,
    ) -> tuple[tuple[int, ...], ...]:
        if blocks is None:
            return tuple(
                tuple(range(start, min(start + self.batch_size, self.n_states)))
                for start in range(0, self.n_states, self.batch_size)
            )

        normalized = tuple(tuple(int(state) for state in block) for block in blocks)
        if not normalized or any(not block for block in normalized):
            raise ValueError("blocks must contain at least one non-empty partition")
        flat = [state for block in normalized for state in block]
        if sorted(flat) != list(range(self.n_states)):
            raise ValueError("blocks must form an exact, duplicate-free cover of all states")
        return normalized

    def _fixed_variable_order(self, block: tuple[int, ...]) -> tuple[int, ...]:
        """Return Wingate-Seppi Algorithm 5's modified topological order."""
        remaining = set(block)
        in_degree = {
            state: int(
                sum(
                    self.P[action, source, state] > 0.0
                    for source in block
                    for action in range(self.n_actions)
                )
            )
            for state in block
        }
        final_order = [block[0]] * len(block)
        for index in range(len(block)):
            state = min(remaining, key=lambda item: (in_degree[item], item))
            remaining.remove(state)
            final_order[len(block) - 1 - index] = state
            for action in range(self.n_actions):
                for successor in block:
                    if successor in remaining and self.P[action, state, successor] > 0.0:
                        in_degree[successor] -= 1
        return tuple(final_order)

    def priority_scores(self) -> np.ndarray:
        """Return normalized H2 scores, with converged states assigned zero."""
        scores = self.xi + self.upsilon * self.V
        return np.where(self.xi > self.solve_tolerance, scores, 0.0)

    def _recompute_residuals(self) -> None:
        for state in range(self.n_states):
            self.xi[state] = self.compute_bellman_residual(
                state,
                self._P_model,
                self._r_model,
                self.gamma,
                self.V,
            )

    def _block_residual(self, block_index: int) -> float:
        return max(
            self.compute_bellman_residual(
                state,
                self._P_model,
                self._r_model,
                self.gamma,
                self.V,
            )
            for state in self.blocks[block_index]
        )

    def _select_block(self) -> int | None:
        self._recompute_residuals()
        scores = self.priority_scores()
        priorities = np.array(
            [max(float(scores[state]) for state in block) for block in self.blocks]
        )
        best = int(np.argmax(priorities))
        return best if priorities[best] > 0.0 else None

    def _refill_block_sweep(self) -> None:
        if self._active_block is not None:
            if self._block_residual(self._active_block) > self.solve_tolerance:
                self._block_sweep.extend(self.block_orders[self._active_block])
                return
            self._active_block = None

        self._active_block = self._select_block()
        if self._active_block is not None:
            self._block_sweep.extend(self.block_orders[self._active_block])

    def reward_update(self, delta_r: np.ndarray) -> None:
        """Apply a reward shift and restart partition selection."""
        next_reward = self.r + delta_r
        if np.any(next_reward < -1e-12):
            raise ValueError(
                "GpsH2 requires every shifted reward vector to remain non-negative; "
                f"received minimum {float(next_reward.min()):.6g}."
            )
        delta_r = np.maximum(next_reward, 0.0) - self.r
        self._active_block = None
        self._block_sweep.clear()
        super().reward_update(delta_r)

    def pop_and_prioritize(self) -> int | None:
        """Return the next state in the selected block's fixed GS order."""
        if not self._block_sweep:
            self._refill_block_sweep()
        return self._block_sweep.popleft() if self._block_sweep else None

    def reset(self) -> None:
        self._active_block = None
        self._block_sweep.clear()
        super().reset()
