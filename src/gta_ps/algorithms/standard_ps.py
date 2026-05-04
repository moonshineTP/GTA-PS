"""
Standard Prioritized Sweeping (Moore & Atkeson 1993).

No topology state. notify_backup and update_value are no-ops.
Predecessor lookup and residuals use self._P_model and self._r_model,
kept in sync with the empirical model by DynaHost via update_model().
"""
from __future__ import annotations
import heapq
import numpy as np

from .base import PrioritizedSweepingModule, BellmanMixin


class StandardPS(PrioritizedSweepingModule, BellmanMixin):
    """
    Standard prioritized sweeping module.

    Parameters
    ----------
    P : ndarray (|A|, |S|, |S|)  oracle transition kernel
    r : ndarray (|S|,)            initial reward vector
    gamma : float
    """

    def __init__(self, P: np.ndarray, r: np.ndarray, gamma: float = 0.95):
        self.P = P                  # oracle — never mutated
        self._P_model = P.copy()   # mutable; DynaHost replaces with P_hat
        self._r_model = r.copy()   # mutable; DynaHost replaces with r_hat
        self.r = r.copy()          # tracks current reward (updated by reward_update)
        self.gamma = gamma
        self.n_states = P.shape[1]
        self.n_actions = P.shape[0]

        self.V = np.zeros(self.n_states)
        self.xi = np.zeros(self.n_states)
        self._heap: list = []
        self._version: dict = {}

        self._predecessors: list[list[tuple[int, int]]] = [[] for _ in range(self.n_states)]
        self._predecessor_sets: list[set[tuple[int, int]]] = [set() for _ in range(self.n_states)]
        self._build_predecessor_cache()

        # Initialise heap from Bellman residuals with V=0
        self._rebuild_heap()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_predecessor_cache(self) -> None:
        """Precompute predecessors with thresholding."""
        n = self.n_states
        for a in range(self.n_actions):
            rows, cols = np.where(self._P_model[a] > 1e-2)
            for pred, succ in zip(rows.tolist(), cols.tolist()):
                if (pred, a) not in self._predecessor_sets[succ]:
                    self._predecessor_sets[succ].add((pred, a))
                    self._predecessors[succ].append((pred, a))

    def update_model_transition(self, s: int, a: int, p_row: np.ndarray, r_s: float) -> None:
        """Incremental model update."""
        self._P_model[a, s] = p_row
        self._r_model[s] = r_s
        for sp_idx in np.where(p_row > 1e-2)[0]:
            if (s, a) not in self._predecessor_sets[sp_idx]:
                self._predecessor_sets[sp_idx].add((s, a))
                self._predecessors[sp_idx].append((s, a))

    def _rebuild_heap(self) -> None:
        """Recompute xi for all states and rebuild heap via heapify."""
        for s in range(self.n_states):
            self.xi[s] = self.compute_bellman_residual(
                s, self._P_model, self._r_model, self.gamma, self.V
            )
        self._heap = []
        self._version = {}
        entries = []
        for s in range(self.n_states):
            if self.xi[s] > 0:
                self._version[s] = 0
                entries.append((-self.xi[s], 0, s))
        heapq.heapify(entries)
        self._heap = entries

    # ------------------------------------------------------------------
    # PrioritizedSweepingModule interface
    # ------------------------------------------------------------------

    def reward_update(self, delta_r: np.ndarray) -> None:
        self.r += delta_r
        self._r_model += delta_r
        self._rebuild_heap()

    def pop_and_prioritize(self) -> int | None:
        while self._heap:
            neg_pri, v, s = heapq.heappop(self._heap)
            if self._version.get(s, -1) == v:
                return s
        return None

    def push_predecessors(self, s: int, xi_s: float) -> None:
        for pred, _a in self._predecessors[s]:
            xi_pred = self.compute_bellman_residual(
                pred, self._P_model, self._r_model, self.gamma, self.V
            )
            self.xi[pred] = xi_pred
            if xi_pred > 1e-8:
                v = self._version.get(pred, -1) + 1
                self._version[pred] = v
                heapq.heappush(self._heap, (-xi_pred, v, pred))

    def notify_backup(self, s: int, adv_PS_row: np.ndarray, xi_s: float) -> None:
        xi_new = self.compute_bellman_residual(
            s, self._P_model, self._r_model, self.gamma, self.V
        )
        self.xi[s] = xi_new
        if xi_new > 1e-8:
            v = self._version.get(s, -1) + 1
            self._version[s] = v
            heapq.heappush(self._heap, (-xi_new, v, s))

    def update_model(self, P: np.ndarray, r: np.ndarray) -> None:
        self._P_model = P.copy()
        self._r_model = r.copy()

    def update_value(self, s: int, v_s: float) -> None:
        self.V[s] = v_s

    def reset(self) -> None:
        self.V = np.zeros(self.n_states)
        self.xi = np.zeros(self.n_states)
        self._heap = []
        self._version = {}
        self._P_model = self.P.copy()
        self._r_model = self.r.copy()
        self._rebuild_heap()

    # ------------------------------------------------------------------
    # Convenience wrappers (plan() shim, backward compat)
    # ------------------------------------------------------------------

    def bellman_residual_state(self, s: int) -> float:
        return self.compute_bellman_residual(
            s, self._P_model, self._r_model, self.gamma, self.V
        )

    def backup_state(self, s: int) -> float:
        return self.apply_bellman_backup(
            s, self._P_model, self._r_model, self.gamma, self.V
        )

    def reset_value(self) -> None:
        self.reset()

    def plan(self, K: int, r: np.ndarray | None = None) -> np.ndarray:
        """Run K Bellman backups (budget per host step). r triggers reward_update first.
        Backward-compat shim: DynaPS(P, r, gamma).plan(K, r=r_new) still works.
        """
        if r is not None:
            self.reward_update(r - self.r)
        for _ in range(K):
            s = self.pop_and_prioritize()
            if s is None:
                break
            self.backup_state(s)
            self.push_predecessors(s, self.xi[s])
        return self.V.copy()

