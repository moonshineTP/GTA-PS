"""
Dyna Prioritized Sweeping (Moore 1993 / Peng & Williams 1993) baseline.

Priority queue ordered by |Bellman residual| xi(s) = |r(s) + gamma*max_a sum_s' P(s'|s,a)*V(s') - V(s)|.
"""
import heapq
import numpy as np
from typing import Optional


class DynaPS:
    """
    Dyna-PS with known model (P, r), greedy policy, synchronous priority queue.

    Parameters
    ----------
    P : ndarray (|A|, |S|, |S|)
    r : ndarray (|S|,)
    gamma : float
    """

    def __init__(self, P: np.ndarray, r: np.ndarray, gamma: float = 0.99):
        self.P = P
        self.r = r.copy()
        self.gamma = gamma
        self.n_states = P.shape[1]
        self.n_actions = P.shape[0]
        self.V = np.zeros(self.n_states)

    # ------------------------------------------------------------------
    def bellman_residual(self) -> np.ndarray:
        """xi(s) = |r(s) + gamma*max_a E[V|s,a] - V(s)| for all s."""
        EV = (self.P * self.V[None, None, :]).sum(axis=2)  # (|A|, |S|)
        Q = self.r[None, :] + self.gamma * EV              # (|A|, |S|)
        V_new = Q.max(axis=0)                              # (|S|,)
        return np.abs(V_new - self.V)

    def backup_state(self, s: int) -> float:
        """Single Bellman backup for state s. Returns |delta V(s)|."""
        EV = (self.P[:, s, :] * self.V[None, :]).sum(axis=1)  # (|A|,)
        v_new = self.r[s] + self.gamma * EV.max()
        delta = abs(v_new - self.V[s])
        self.V[s] = v_new
        return delta

    # ------------------------------------------------------------------
    def plan(self, K: int, r: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Run K prioritized backup steps.
        Recomputes priorities from scratch after a reward shift.
        Returns V after K backups.
        """
        if r is not None:
            self.r = r.copy()

        # Initialize priority queue from Bellman residuals
        xi = self.bellman_residual()
        # Max-heap via negated priorities
        heap = [(-xi[s], s) for s in range(self.n_states) if xi[s] > 0]
        heapq.heapify(heap)

        for _ in range(K):
            if not heap:
                break
            neg_pri, s = heapq.heappop(heap)
            delta = self.backup_state(s)
            # Propagate to predecessors
            for a in range(self.n_actions):
                for pred in np.where(self.P[a, :, s] > 0)[0]:
                    xi_pred = self.bellman_residual_state(pred)
                    if xi_pred > 0:
                        heapq.heappush(heap, (-xi_pred, pred))

        return self.V.copy()

    def bellman_residual_state(self, s: int) -> float:
        EV = (self.P[:, s, :] * self.V[None, :]).sum(axis=1)
        v_new = self.r[s] + self.gamma * EV.max()
        return abs(v_new - self.V[s])

    def reset_value(self) -> None:
        self.V = np.zeros(self.n_states)
