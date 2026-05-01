"""
Abstract interface for GTA-PS tabular environments.

Both exact-model environments (FourRooms, GARNET, MiniGrid) and
learned-model environments (Reacher, CarGoal) share this interface.
The planning algorithms only consume P, r, n_states, n_actions.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class TabularEnv(ABC):
    """
    Common interface for tabular (or tabularized) MDPs.

    Attributes (set by subclass)
    ----------------------------
    P          : ndarray (|A|, |S|, |S|), row-stochastic
    r          : ndarray (|S|,)
    n_states   : int
    n_actions  : int
    gamma      : float
    """

    P: np.ndarray
    r: np.ndarray
    n_states: int
    n_actions: int
    gamma: float

    @abstractmethod
    def shift_reward(self) -> np.ndarray:
        """Resample reward (and optionally env dynamics).  Returns new r."""

    def value_iteration(
        self,
        r: Optional[np.ndarray] = None,
        tol: float = 1e-8,
        max_iter: int = 10_000,
    ) -> np.ndarray:
        """Synchronous VI on the tabular model.  Returns V*."""
        if r is None:
            r = self.r
        V = np.zeros(self.n_states)
        for _ in range(max_iter):
            Q = r[None, :] + self.gamma * (self.P * V[None, None, :]).sum(axis=2)
            V_new = Q.max(axis=0)
            if np.max(np.abs(V_new - V)) < tol:
                return V_new
            V = V_new
        return V

    def greedy_policy(self, V: np.ndarray, r: Optional[np.ndarray] = None) -> np.ndarray:
        if r is None:
            r = self.r
        Q = r[None, :] + self.gamma * (self.P * V[None, None, :]).sum(axis=2)
        return Q.argmax(axis=0)
