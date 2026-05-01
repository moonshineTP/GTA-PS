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
        for _ in range(K):
            s = self.module.pop_and_prioritize()
            if s is None:
                break

            delta_V = self.apply_bellman_backup(
                s, P, self.r, self.gamma, self.V
            )

            # Sync module.V so push_predecessors computes xi_pred from current values.
            self.module.update_value(s, float(self.V[s]))

            # Advantage: γ·E[V|s,a] − max_a γ·E[V|s,a]  (memo §Algorithm)
            EV_all = (P[:, s, :] * self.V[None, :]).sum(axis=1)   # (|A|,)
            adv_PS_row = self.gamma * EV_all - self.gamma * EV_all.max()

            self.module.notify_backup(s, adv_PS_row, delta_V)
            self.module.push_predecessors(s, delta_V)

        return self.V.copy()

    def reset(self) -> None:
        self.V = np.zeros(self.n_states)
        self.module.reset()

