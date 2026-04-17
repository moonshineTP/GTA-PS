"""
GARNET (Generalized Average Reward Non-stationary Environment Task) tabular MDP.

Reference: Archibald et al. (1995). The MDP is parameterized by:
  |S|, |A|, branching factor b (non-zero transitions per (s,a)),
  reward support size, reversibility level, sparsity.

Used primarily for RQ2, RQ4, RQ5, RQ6.
"""
import numpy as np
import scipy.sparse as sp
from typing import Optional


class GARNETEnv:
    """
    Tabular GARNET MDP with nonstationary rewards.

    Parameters
    ----------
    n_states : int
        |S|
    n_actions : int
        |A| (>= 10 per memo)
    branching : int
        Number of successor states per (s, a) pair.
    reward_support : int
        Number of states with nonzero reward at each shift.
        Per memo §8: fixed in {1, 2, 4, 8} regardless of |S|.
    gamma : float
        Discount factor.
    reversible : float
        0.0 = fully directed (non-reversible), 1.0 = fully reversible.
        Interpolates: with probability `reversible`, the backward
        transition is also reinforced.
    seed : int
    """

    def __init__(
        self,
        n_states: int = 100,
        n_actions: int = 10,
        branching: int = 3,
        reward_support: int = 4,   # fixed, not a function of |S|
        gamma: float = 0.99,
        reversible: float = 0.0,
        seed: int = 0,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.branching = min(branching, n_states)
        self.reward_support = min(reward_support, n_states)
        self.gamma = gamma
        self.reversible = reversible
        self.rng = np.random.default_rng(seed)

        self.P = self._build_transition()   # (|A|, |S|, |S|) row-stochastic
        self.r = self._sample_reward()      # (|S|,)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_transition(self) -> np.ndarray:
        """
        Build |A| row-stochastic transition matrices P_a of shape (|S|, |S|).
        For each (s, a), choose `branching` successors uniformly at random
        and assign Dirichlet-sampled probabilities.
        """
        S, A, b = self.n_states, self.n_actions, self.branching
        P = np.zeros((A, S, S), dtype=np.float64)
        for a in range(A):
            for s in range(S):
                succ = self.rng.choice(S, size=b, replace=False)
                w = self.rng.dirichlet(np.ones(b))
                P[a, s, succ] = w
                if self.reversible > 0.0:
                    # reinforce reverse edges with probability `reversible`
                    for j, ns in enumerate(succ):
                        if self.rng.random() < self.reversible:
                            P[a, ns, s] += w[j] * self.reversible
            # renormalize rows
            row_sums = P[a].sum(axis=1, keepdims=True)
            P[a] /= row_sums
        return P

    def _sample_reward(self) -> np.ndarray:
        """Sparse reward: `reward_support` states get Uniform([0.5, 1.5])."""
        r = np.zeros(self.n_states)
        support = self.rng.choice(self.n_states, size=self.reward_support, replace=False)
        r[support] = self.rng.uniform(0.5, 1.5, size=self.reward_support)
        return r

    # ------------------------------------------------------------------
    # MDP interface
    # ------------------------------------------------------------------

    def shift_reward(self) -> np.ndarray:
        """Resample reward; return new r."""
        self.r = self._sample_reward()
        return self.r.copy()

    def policy_transition(self, policy: np.ndarray) -> np.ndarray:
        """
        Given a deterministic policy (|S|,) -> action index,
        return the induced (|S|, |S|) row-stochastic matrix P_pi.
        """
        S = self.n_states
        P_pi = np.zeros((S, S))
        for s in range(S):
            a = int(policy[s])
            P_pi[s] = self.P[a, s]
        return P_pi

    def value_iteration(
        self,
        r: Optional[np.ndarray] = None,
        tol: float = 1e-8,
        max_iter: int = 10_000,
    ) -> np.ndarray:
        """Full synchronous VI; returns V* for the given reward."""
        if r is None:
            r = self.r
        gamma = self.gamma
        V = np.zeros(self.n_states)
        for _ in range(max_iter):
            Q = r[None, :] + gamma * (self.P * V[None, None, :]).sum(axis=2)  # (|A|, |S|)
            V_new = Q.max(axis=0)  # (|S|,)
            if np.max(np.abs(V_new - V)) < tol:
                return V_new
            V = V_new
        return V

    def greedy_policy(self, V: np.ndarray, r: Optional[np.ndarray] = None) -> np.ndarray:
        if r is None:
            r = self.r
        Q = r[None, :] + self.gamma * (self.P * V[None, None, :]).sum(axis=2)  # (|A|, |S|)
        return Q.argmax(axis=0)
