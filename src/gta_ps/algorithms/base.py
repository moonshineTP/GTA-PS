"""
Abstract base and shared utilities for prioritized sweeping modules.

PrioritizedSweepingModule — ABC defining the surface the host touches.
BellmanMixin              — Stateless distributional Bellman utilities.
"""
from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np


class PrioritizedSweepingModule(ABC):
    """
    Abstract computation management module for prioritized sweeping.
    All methods here are the ONLY surface the host touches.
    Hosts must not access any module-internal attribute directly.
    """

    @abstractmethod
    def reward_update(self, delta_r: np.ndarray) -> None:
        """Called on a reward shift; updates internal topology/residual state."""

    @abstractmethod
    def pop_and_prioritize(self) -> int | None:
        """Return the next state to back up (highest aggregate priority), or None."""

    @abstractmethod
    def enqueue_priority(self, s: int, priority: float) -> None:
        """Insert or promote one state using a host-computed residual priority."""

    @abstractmethod
    def push_predecessors(self, s: int, xi_s: float) -> None:
        """
        Given that state s was backed up with residual magnitude xi_s,
        propagate priority to predecessors of s.
        The module uses its internal model reference (self._P_model) for
        predecessor lookup and residual recomputation — NOT necessarily the
        oracle P. DynaHost keeps _P_model synced to P_hat via update_model().
        """

    @abstractmethod
    def notify_backup(self, s: int, adv_PS_row: np.ndarray, xi_s: float) -> None:
        """
        Called by the host AFTER each Bellman/TD backup of state s.
        adv_PS_row = Q(s,:) - max_a Q(s,a)  shape (|A|,).
        xi_s = the TD or Bellman residual magnitude used to trigger this backup.
        StandardPS: no-op.
        GtaPSModule: triggers GtaPolicyUpdate + TopologyUpdate.
        """

    @abstractmethod
    def update_model(self, P: np.ndarray, r: np.ndarray) -> None:
        """
        Sync the module's internal transition model and reward model.
        DpHost: NEVER calls this (oracle P is fixed at construction).
        DynaHost: calls this after EVERY experience step with P_hat, r_hat
                  so that push_predecessors uses the current empirical model.
        """

    @abstractmethod
    def update_value(self, s: int, v_s: float) -> None:
        """
        Sync the module's value estimate for state s.
        DpHost: NEVER calls this (module maintains V through its own backups).
        DynaHost: calls this after each TD update with v_s = max_a Q(s,a),
                  so that push_predecessors computes residuals consistently
                  with the host's Q-value estimates.
        """

    @abstractmethod
    def reset(self) -> None:
        """Full cold-start reset of all internal state."""


class PlanningModel(ABC):
    """
    Minimal interface for a planning model consumed by host agents.

    A model provides the transition kernel P and exposes the current
    state-space dimensions. Reward tracking is deliberately left to the
    host (DpHost owns self.r so that multiple hosts can share one model
    without aliasing r across agents in multi-K experiment loops).
    """

    @property
    @abstractmethod
    def P(self) -> np.ndarray:
        """Transition kernel (|A|, |S|, |S|), row-stochastic."""

    @property
    @abstractmethod
    def n_states(self) -> int:
        """Number of states |S|."""

    @property
    @abstractmethod
    def n_actions(self) -> int:
        """Number of actions |A|."""


class BellmanMixin:
    """
    Stateless Bellman backup utilities. Accepts all arrays as arguments
    so both modules (which own V) and hosts (which own V or Q) can use them.
    """

    def compute_bellman_residual(
        self,
        s: int,
        P: np.ndarray,
        r: np.ndarray,
        gamma: float,
        V: np.ndarray,
    ) -> float:
        EV = (P[:, s, :] * V[None, :]).sum(axis=1)   # (|A|,)
        return float(abs(r[s] + gamma * EV.max() - V[s]))

    def apply_bellman_backup(
        self,
        s: int,
        P: np.ndarray,
        r: np.ndarray,
        gamma: float,
        V: np.ndarray,
    ) -> float:
        """In-place backup of V[s]. Returns |delta V(s)|."""
        EV = (P[:, s, :] * V[None, :]).sum(axis=1)
        v_new = r[s] + gamma * EV.max()
        delta = float(abs(v_new - V[s]))
        V[s] = v_new
        return delta


class TdMixin:
    """
    Stateless one-step TD utilities.  Accepts all data as arguments so
    both modules and hosts (which own Q) can use them without coupling.

    Sampled variants operate on a single transition (s, a, r_sa, s_next).
    Expected variants marginalise over the transition kernel P and are
    meaningful in the DP-host context where P is available.
    """

    def compute_td_residual(
        self,
        s: int,
        a: int,
        r_sa: float,
        s_next: int,
        gamma: float,
        Q: np.ndarray,
    ) -> float:
        """Sampled TD error magnitude for (s, a, r_sa, s') and action-value table Q.

        Q shape: (|A|, |S|).  Returns |r_sa + γ max_a' Q(a', s') − Q(a, s)|.
        """
        return float(abs(r_sa + gamma * Q[:, s_next].max() - Q[a, s]))

    def compute_expected_td_residual(
        self,
        s: int,
        P: np.ndarray,
        r: np.ndarray,
        gamma: float,
        Q: np.ndarray,
    ) -> float:
        """Distributional TD residual for state s, marginalised over P.

        Equivalent to the Bellman residual under the greedy policy:
        |r[s] + γ Σ_{s'} P(a*,s,s') max_a' Q(a',s') − max_a Q(a,s)|,
        where a* = argmax_a Q(a,s).

        Reduces to BellmanMixin.compute_bellman_residual when Q(a,s) ≡ V(s)
        for all a.
        """
        a_star = int(Q[:, s].argmax())
        v_next = (P[a_star, s, :] * Q.max(axis=0)).sum()   # E_{P(a*,s,·)}[max_a' Q]
        return float(abs(r[s] + gamma * v_next - Q[a_star, s]))

    def apply_td_backup(
        self,
        s: int,
        a: int,
        r_sa: float,
        s_next: int,
        gamma: float,
        eta: float,
        Q: np.ndarray,
    ) -> float:
        """In-place TD update of Q[a, s].  Returns |delta Q[a, s]|.

        Q[a, s] ← Q[a, s] + η · (r_sa + γ max_a' Q(a', s_next) − Q[a, s])
        """
        td_err = r_sa + gamma * Q[:, s_next].max() - Q[a, s]
        Q[a, s] += eta * td_err
        return float(abs(eta * td_err))
