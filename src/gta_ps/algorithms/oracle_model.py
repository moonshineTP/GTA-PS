"""
OracleModel — oracle distributional planning model.

Wraps a fixed oracle transition kernel P and optionally holds a reference
to the environment so that experiment loops can drive reward shifts through
the model rather than managing env and model state separately.

Design note
-----------
Reward tracking is intentionally NOT part of PlanningModel / OracleModel.
DpHost owns self.r so that multiple DpHost instances with different K values
can each track their own reward state without aliasing (as in the multi-K
experiment loop in rq1_convergence.py). OracleModel exposes r_init only
as a convenience for constructing modules — it is not updated at runtime.
"""
from __future__ import annotations

import numpy as np

from .base import PlanningModel


class OracleModel(PlanningModel):
    """
    Oracle distributional model: fixed P, env-linked reward shifts.

    Parameters
    ----------
    P   : ndarray (|A|, |S|, |S|)  exact transition kernel (copied on init)
    r   : ndarray (|S|,)            initial reward vector (stored as r_init)
    env : TabularEnv | None
        Optional environment reference. Required only if shift_from_env()
        is used by the experiment loop. DpHost.plan(r=...) does NOT need it.
    """

    def __init__(
        self,
        P: np.ndarray,
        r: np.ndarray,
        env=None,
    ) -> None:
        self._P = P.copy()
        self._r_init = r.copy()   # snapshot of initial r; never mutated
        self._env = env

    # ------------------------------------------------------------------
    # PlanningModel interface
    # ------------------------------------------------------------------

    @property
    def P(self) -> np.ndarray:
        return self._P

    @property
    def n_states(self) -> int:
        return self._P.shape[1]

    @property
    def n_actions(self) -> int:
        return self._P.shape[0]

    # ------------------------------------------------------------------
    # Convenience: initial reward snapshot
    # ------------------------------------------------------------------

    @property
    def r_init(self) -> np.ndarray:
        """Initial reward vector (read-only snapshot)."""
        return self._r_init.copy()

    # ------------------------------------------------------------------
    # Env-linked reward shift (optional convenience)
    # ------------------------------------------------------------------

    def shift_from_env(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Call env.shift_reward(), compute delta_r, return (r_new, delta_r).

        Callers should pass r_new to DpHost.plan(r=r_new) and V* to
        env.value_iteration(r_new). The model does NOT update any internal
        r state — reward tracking is the host's responsibility.

        Raises RuntimeError if no env was provided at construction.
        """
        if self._env is None:
            raise RuntimeError(
                "OracleModel.shift_from_env() requires an env. "
                "Pass env=<TabularEnv> at construction."
            )
        r_new = self._env.shift_reward()
        return r_new

