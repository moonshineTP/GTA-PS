"""
GTA-PS: Graph Topology Augmentation for Prioritized Sweeping.

Augmented priority signal (Section 4-5 of memo):
  tau(s) = |xi(s)| + beta_t * |delta_f_out(s)| + alpha_t * |delta_f_in(s)|

Variants:
  GTA-PS (base)      -- fixed alpha, beta
  GTA-PS^alpha       -- SLEM-coupled alpha_t, fixed beta
  GTA-PS^beta        -- fixed alpha, SLEM-coupled beta_t
  GTA-PS^{alpha,beta} -- both adaptive (full variant)
"""
import heapq
import math
import numpy as np
from typing import Optional

from ..utils.spectral import (
    slem_arnoldi,
    resolvent_forward,
    resolvent_backward,
)


class GTAPS:
    """
    GTA-PS planner with known model.

    Parameters
    ----------
    P : ndarray (|A|, |S|, |S|)
    r : ndarray (|S|,)
    gamma : float
    alpha : float
        Base forward->backward weight (used when adaptive_alpha=False).
    beta : float
        Base topology weight (used when adaptive_beta=False).
    beta1, beta2 : float
        GTA-PS^beta schedule parameters: beta_t = beta1 * ((1 - SLEM) - beta2).
    adaptive_alpha : bool
        If True, use alpha_t = SLEM^alpha_exp (SLEM-coupled forward/backward mix).
    adaptive_beta : bool
        If True, use beta_t = beta1 * ((1 - SLEM) - beta2).
    alpha_exp : float
        Exponent for SLEM-coupled alpha schedule.
    eps : float
        Resolvent parameter; None defaults to eps* = (1-gamma)/gamma.
    arnoldi_k : int
        Number of Arnoldi vectors for SLEM estimation.
    slem_recompute_every : int
        Recompute SLEM via Arnoldi every this many steps; use incremental otherwise.
    """

    def __init__(
        self,
        P: np.ndarray,
        r: np.ndarray,
        gamma: float = 0.99,
        alpha: float = 0.5,
        beta: float = 1.0,
        beta1: float = 1.0,
        beta2: float = 0.1,
        adaptive_alpha: bool = False,
        adaptive_beta: bool = False,
        alpha_exp: float = 1.0,
        eps: Optional[float] = None,
        arnoldi_k: int = 15,
        slem_recompute_every: Optional[int] = None,
    ):
        self.P = P
        self.r = r.copy()
        self.gamma = gamma
        self.n_states = P.shape[1]
        self.n_actions = P.shape[0]

        self.alpha_base = alpha
        self.beta_base = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.adaptive_alpha = adaptive_alpha
        self.adaptive_beta = adaptive_beta
        self.alpha_exp = alpha_exp
        self.eps = eps if eps is not None else (1.0 - gamma) / gamma
        self.arnoldi_k = arnoldi_k
        self.slem_recompute_every = slem_recompute_every

        self.V = np.zeros(self.n_states)
        self._slem: Optional[float] = None
        self._f_out_prev: Optional[np.ndarray] = None
        self._f_in_prev: Optional[np.ndarray] = None
        self._step_count = 0

    # ------------------------------------------------------------------
    # Schedule computation
    # ------------------------------------------------------------------

    def _compute_slem(self, P_pi: np.ndarray) -> float:
        """SLEM of the current policy-induced chain."""
        n = self.n_states
        recompute_every = self.slem_recompute_every or max(1, int(math.ceil(n ** 0.5)))
        if self._slem is None or self._step_count % recompute_every == 0:
            self._slem = slem_arnoldi(P_pi, k=self.arnoldi_k)
        return self._slem

    def _alpha_t(self, slem: float) -> float:
        if not self.adaptive_alpha:
            return self.alpha_base
        return slem ** self.alpha_exp

    def _beta_t(self, slem: float) -> float:
        if not self.adaptive_beta:
            return self.beta_base
        return self.beta1 * ((1.0 - slem) - self.beta2)

    # ------------------------------------------------------------------
    # Resolvent fields
    # ------------------------------------------------------------------

    def _compute_topology_signals(
        self, P_pi: np.ndarray, r: np.ndarray, alpha: float, beta: float
    ):
        """Compute delta_f_out and delta_f_in after a reward shift."""
        f_out = resolvent_forward(P_pi, r, self.eps)
        f_in = resolvent_backward(P_pi, r, self.eps)

        if self._f_out_prev is None:
            df_out = np.abs(f_out)
            df_in = np.abs(f_in)
        else:
            df_out = np.abs(f_out - self._f_out_prev)
            df_in = np.abs(f_in - self._f_in_prev)

        self._f_out_prev = f_out
        self._f_in_prev = f_in
        return df_out, df_in

    # ------------------------------------------------------------------
    # Core backup
    # ------------------------------------------------------------------

    def bellman_residual_state(self, s: int) -> float:
        EV = (self.P[:, s, :] * self.V[None, :]).sum(axis=1)
        v_new = self.r[s] + self.gamma * EV.max()
        return abs(v_new - self.V[s])

    def backup_state(self, s: int) -> float:
        EV = (self.P[:, s, :] * self.V[None, :]).sum(axis=1)
        v_new = self.r[s] + self.gamma * EV.max()
        delta = abs(v_new - self.V[s])
        self.V[s] = v_new
        return delta

    # ------------------------------------------------------------------
    # Main planning loop
    # ------------------------------------------------------------------

    def plan(self, K: int, r: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Run K augmented-priority backup steps after an optional reward shift.

        Returns V after K backups.
        """
        if r is not None:
            self.r = r.copy()

        # Derive policy-induced chain from greedy(V) for spectral signals
        Q = self.r[None, :] + self.gamma * (self.P * self.V[None, None, :]).sum(axis=2)  # (|A|,|S|)
        policy = Q.argmax(axis=0)
        P_pi = np.zeros((self.n_states, self.n_states))
        for s in range(self.n_states):
            P_pi[s] = self.P[policy[s], s]

        slem = self._compute_slem(P_pi)
        alpha = self._alpha_t(slem)
        beta = self._beta_t(slem)

        df_out, df_in = self._compute_topology_signals(P_pi, self.r, alpha, beta)

        # Augmented priority
        xi = np.array([self.bellman_residual_state(s) for s in range(self.n_states)])
        # Clamp beta: if beta < 0, topology signal is damped (GTA-PS^beta safety floor)
        if beta >= 0:
            tau = xi + beta * (df_out + alpha * df_in)
        else:
            # beta < 0: down-weight topology; tau >= xi (never below Dyna-PS)
            tau = xi + max(beta, -1.0) * (df_out + alpha * df_in)
            tau = np.maximum(tau, xi)

        heap = [(-tau[s], s) for s in range(self.n_states) if tau[s] > 0]
        heapq.heapify(heap)

        for _ in range(K):
            if not heap:
                break
            neg_pri, s = heapq.heappop(heap)
            self.backup_state(s)
            self._step_count += 1
            # Propagate residuals to predecessors
            for a in range(self.n_actions):
                for pred in np.where(self.P[a, :, s] > 0)[0]:
                    xi_pred = self.bellman_residual_state(pred)
                    if xi_pred > 0:
                        tau_pred = xi_pred + max(beta, 0) * (
                            df_out[pred] + alpha * df_in[pred]
                        )
                        heapq.heappush(heap, (-tau_pred, pred))

        return self.V.copy()

    def reset_value(self) -> None:
        self.V = np.zeros(self.n_states)
        self._f_out_prev = None
        self._f_in_prev = None
        self._slem = None
        self._step_count = 0
