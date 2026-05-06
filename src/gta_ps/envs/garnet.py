"""
GARNET (Generalized Average Reward Non-stationary Environment Task) tabular MDP.
"""
import numpy as np
import scipy.sparse as sp
from typing import Optional

class GarnetEnv:
    """
    Tabular GARNET MDP with nonstationary rewards.
    Supports bipartite (periodic) and reversible structures.
    """

    def __init__(
        self,
        n_states: int = 75,
        n_actions: int = 4,
        branching: int = 3,
        reward_support: int = 3,
        gamma: float = 0.95,
        n_nbr_rate: float = 0.15,
        reversible: float = 0.0,
        bipartite: bool = False,
        reward_locality: bool = True,
        seed: int = 0,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.branching = branching
        self.reward_support = reward_support
        self.gamma = gamma
        self.n_nbr = int(n_nbr_rate * n_states)
        self.reversible = reversible
        self.bipartite = bipartite
        self.reward_locality = reward_locality
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        self.P = self._build_transition()

        # Precompute adjacency (union across actions) for BFS-locality sampling
        self._adj: list | None = None
        if reward_locality:
            adj: list = [[] for _ in range(n_states)]
            for s in range(n_states):
                seen: set = set()
                for a in range(n_actions):
                    for s2 in np.where(self.P[a, s] > 0)[0]:
                        s2i = int(s2)
                        if s2i not in seen:
                            seen.add(s2i)
                            adj[s].append(s2i)
            self._adj = adj

        self.r = self._sample_reward()

    def _build_transition(self) -> np.ndarray:
        """Build transition tensor (|A|, |S|, |S|)."""
        S, A, b = self.n_states, self.n_actions, self.branching
        P = np.zeros((A, S, S), dtype=np.float64)
        
        for a in range(A):
            for s in range(S):
                # successor candidates in ring-neighbourhood
                window = np.arange(s - self.n_nbr // 2, s + self.n_nbr // 2 + 1) % S
                
                if self.bipartite:
                    # Enforce parity flip: s even -> odd successors, s odd -> even
                    target_parity = (s + 1) % 2
                    window_list = window.tolist()
                    window_filtered = [w for w in window_list if w % 2 == target_parity]
                    if not window_filtered:
                         window_filtered = [(s + 1) % S]
                    final_window = np.array(window_filtered)
                else:
                    final_window = window

                # sample branching b
                successors = self._rng.choice(final_window, size=min(b, len(final_window)), replace=False)
                for s_next in successors:
                    P[a, s, s_next] = 1.0
                    
                if self.reversible > 0.0:
                    for s_next in successors:
                        if self._rng.random() < self.reversible:
                            P[a, s_next, s] = 1.0
        
        # Ensure row-stochastic
        row_sums = P.sum(axis=2, keepdims=True)
        row_sums[row_sums == 0] = 1.0 # Avoid div zero
        P /= row_sums
        return P

    def _sample_reward(self) -> np.ndarray:
        if self._adj is not None:
            return self._sample_reward_local()
        r_vec = np.zeros(self.n_states)
        goals = self._rng.choice(self.n_states, size=self.reward_support, replace=False)
        r_vec[goals] = self._rng.uniform(0.5, 1.5)
        return r_vec

    def _sample_reward_local(self) -> np.ndarray:
        """BFS-neighborhood reward: reward_support connected states from a random root."""
        r_vec = np.zeros(self.n_states)
        root = int(self._rng.integers(self.n_states))
        cluster: list = [root]
        frontier: list = [root]
        seen: set = {root}
        while len(cluster) < self.reward_support and frontier:
            idx = int(self._rng.integers(len(frontier)))
            s = frontier[idx]
            added = False
            for nb in self._adj[s]:
                if nb not in seen:
                    seen.add(nb)
                    cluster.append(nb)
                    frontier.append(nb)
                    added = True
                    if len(cluster) >= self.reward_support:
                        break
            if not added:
                frontier.pop(idx)
        for s in cluster[:self.reward_support]:
            r_vec[s] = float(self._rng.uniform(0.5, 1.5))
        return r_vec

    def shift_reward(self) -> np.ndarray:
        self.r = self._sample_reward()
        return self.r.copy()

    def value_iteration(self, r: Optional[np.ndarray] = None) -> np.ndarray:
        if r is None: r = self.r
        V = np.zeros(self.n_states)
        for _ in range(1000):
            Q = r[None, :] + self.gamma * (self.P * V[None, None, :]).sum(axis=2)
            V_new = Q.max(axis=0)
            if np.linalg.norm(V_new - V, ord=np.inf) < 1e-4: break
            V = V_new
        return V

    def greedy_policy(self, V: np.ndarray, r: Optional[np.ndarray] = None) -> np.ndarray:
        if r is None: r = self.r
        Q = r[None, :] + self.gamma * (self.P * V[None, None, :]).sum(axis=2)
        return Q.argmax(axis=0)
