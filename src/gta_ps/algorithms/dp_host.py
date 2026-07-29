"""
DpHost — oracle-model (Value Iteration) host, module-agnostic.

Accepts a PlanningModel (oracle P) rather than raw arrays so that
experiment code can compose host + module + model independently.

Backward-compatible: also accepts raw ndarray P via the class method
DpHost.from_arrays(P, r, gamma, module).

Uses the exact transition kernel P throughout.
Never calls module.update_model() or module.update_value().
All module pairings valid: DpHost + StandardPS, DpHost + GtaPS.
"""
from __future__ import annotations

import numpy as np

from .base import PrioritizedSweepingModule, BellmanMixin, PlanningModel

# DO NOT import StandardPS or GtaPS here.


class DpHost(BellmanMixin):
    """
    DP (distributional Bellman) host.

    Parameters
    ----------
    model  : PlanningModel        oracle transition kernel provider
    r      : ndarray (|S|,)       initial reward vector (host owns r tracking)
    gamma  : float
    module : PrioritizedSweepingModule
    """

    def __init__(
        self,
        model: PlanningModel,
        r: np.ndarray,
        gamma: float,
        module: PrioritizedSweepingModule,
    ):
        self.model = model
        self.r = r.copy()
        self.gamma = gamma
        self.n_states = model.n_states
        self.n_actions = model.n_actions
        self.module = module
        self.V = np.zeros(self.n_states)
        self.completed_backups = 0
        self.backup_counts = np.zeros(self.n_states, dtype=np.int64)

    @classmethod
    def from_arrays(
        cls,
        P: np.ndarray,
        r: np.ndarray,
        gamma: float,
        module: PrioritizedSweepingModule,
    ) -> "DpHost":
        """Backward-compatible constructor accepting raw (P, r) arrays."""
        from .oracle_model import OracleModel
        return cls(OracleModel(P, r), r, gamma, module)

    def plan(self, K: int, r: np.ndarray | None = None) -> np.ndarray:
        if r is not None:
            delta_r = r - self.r
            self.r = r.copy()
            self.module.reward_update(delta_r)

        P = self.model.P
        remaining = K
        pop_batch = getattr(self.module, "pop_batch", None)
        while remaining > 0:
            if pop_batch is not None:
                states = pop_batch(remaining)
                if not states:
                    break
                self._apply_simultaneous_batch(states, P)
                remaining -= len(states)
                continue

            s = self.module.pop_and_prioritize()
            if s is None:
                break

            delta_V = self.apply_bellman_backup(
                s, P, self.r, self.gamma, self.V
            )
            self.completed_backups += 1
            self.backup_counts[s] += 1

            # Sync module.V so push_predecessors computes xi_pred from current values.
            self.module.update_value(s, float(self.V[s]))

            # Advantage: γ·E[V|s,a] − max_a γ·E[V|s,a]  (memo §Algorithm)
            EV_all = (P[:, s, :] * self.V[None, :]).sum(axis=1)   # (|A|,)
            adv_PS_row = self.gamma * EV_all - self.gamma * EV_all.max()

            self.module.notify_backup(s, adv_PS_row, delta_V)
            self.module.push_predecessors(s, delta_V)
            remaining -= 1

        return self.V.copy()

    def _apply_simultaneous_batch(
        self,
        states: list[int],
        P: np.ndarray,
    ) -> None:
        """Apply Bellman targets for a frozen top-m batch from one V snapshot."""
        value_snapshot = self.V.copy()
        updates: list[tuple[int, float, float, np.ndarray]] = []
        for state in states:
            expected = (P[:, state, :] * value_snapshot[None, :]).sum(axis=1)
            value_new = float(self.r[state] + self.gamma * expected.max())
            delta_value = float(abs(value_new - value_snapshot[state]))
            advantage = self.gamma * expected - self.gamma * expected.max()
            updates.append((state, value_new, delta_value, advantage))

        for state, value_new, _delta_value, _advantage in updates:
            self.V[state] = value_new
            self.module.update_value(state, value_new)
        self.completed_backups += len(states)
        np.add.at(self.backup_counts, states, 1)

        for state, _value_new, delta_value, advantage in updates:
            self.module.notify_backup(state, advantage, delta_value)
            self.module.push_predecessors(state, delta_value)

    def reset(self) -> None:
        self.V = np.zeros(self.n_states)
        self.completed_backups = 0
        self.backup_counts.fill(0)
        self.module.reset()
