"""
FourRooms tabular MDP with adjustable corridor bottlenecks.
"""
from __future__ import annotations
import numpy as np
from typing import Optional, List, Tuple
from .base import TabularEnv

N_ACTIONS = 4
_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)] # UP, DOWN, LEFT, RIGHT

def _build_corridor_grid(room_size: int, corridor_len: int) -> frozenset[tuple[int, int]]:
    """
    Build a grid of 4 rooms connected by corridors.
    Each room is (room_size x room_size).
    Corridors are (corridor_len) cells long and 1 cell wide.
    Total grid width = 2 * room_size + corridor_len + 2 (outer walls)
    """
    passable = set()
    
    # Room offsets
    # R0: Top-Left, R1: Top-Right, R2: Bottom-Left, R3: Bottom-Right
    
    # Room 0
    for r in range(1, room_size + 1):
        for c in range(1, room_size + 1):
            passable.add((r, c))
            
    # Room 1
    c_start_r1 = room_size + corridor_len + 1
    for r in range(1, room_size + 1):
        for c in range(c_start_r1, c_start_r1 + room_size):
            passable.add((r, c))
            
    # Room 2
    r_start_r2 = room_size + corridor_len + 1
    for r in range(r_start_r2, r_start_r2 + room_size):
        for c in range(1, room_size + 1):
            passable.add((r, c))
            
    # Room 3
    for r in range(r_start_r2, r_start_r2 + room_size):
        for c in range(c_start_r1, c_start_r1 + room_size):
            passable.add((r, c))
            
    # Corridors
    mid = (room_size + 1) // 2
    
    # R0 <-> R1 (Horizontal)
    for c in range(room_size + 1, c_start_r1):
        passable.add((mid, c))
        
    # R2 <-> R3 (Horizontal)
    for c in range(room_size + 1, c_start_r1):
        passable.add((r_start_r2 + mid - 1, c))
        
    # R0 <-> R2 (Vertical)
    for r in range(room_size + 1, r_start_r2):
        passable.add((r, mid))
        
    # R1 <-> R3 (Vertical)
    for r in range(room_size + 1, r_start_r2):
        passable.add((r, c_start_r1 + mid - 1))
        
    return frozenset(passable)

class FourRoomsTabular(TabularEnv):
    def __init__(
        self,
        room_size: int = 5,
        corridor_len: int = 1,
        k: int = 1,
        gamma: float = 0.95,
        one_way_prob: float = 1.0,
        drift: Optional[tuple[int, int]] = None,
        seed: int = 0,
        n_peaks_per_room: int = 2,
        sigma_range: tuple = (0.3, 0.8),
        room_peak_ranges: Optional[List] = None,
    ):
        self.room_size = room_size
        self.corridor_len = corridor_len
        self.k = k
        self.gamma = gamma
        self.one_way_prob = one_way_prob
        self.drift = drift
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # Gaussian reward config — always active; None room_peak_ranges → equal ranges
        self.n_peaks_per_room = n_peaks_per_room
        self.sigma_range = tuple(sigma_range)
        self.room_peak_ranges = list(room_peak_ranges) if room_peak_ranges is not None else [(0.3, 1.0)] * 4

        self._passable = _build_corridor_grid(room_size, corridor_len)
        self._pos_list = sorted(list(self._passable))
        self._pos2idx = {p: i for i, p in enumerate(self._pos_list)}
        self._pos_arr  = np.array(self._pos_list, dtype=np.float64)  # cached for Gaussian

        self.n_states = len(self._pos_list)
        self.n_actions = N_ACTIONS

        self._rooms = self._detect_rooms()
        self.P = self._build_transition()
        self.r = self._sample_reward()

    def _detect_rooms(self) -> list[list[int]]:
        rooms = [[] for _ in range(4)]
        # Center gap
        c_gap = self.room_size + 1
        r_gap = self.room_size + 1
        
        for i, (r, c) in enumerate(self._pos_list):
            if r <= self.room_size and c <= self.room_size:
                rooms[0].append(i)
            elif r <= self.room_size and c >= (self.room_size + self.corridor_len + 1):
                rooms[1].append(i)
            elif r >= (self.room_size + self.corridor_len + 1) and c <= self.room_size:
                rooms[2].append(i)
            elif r >= (self.room_size + self.corridor_len + 1) and c >= (self.room_size + self.corridor_len + 1):
                rooms[3].append(i)
        return rooms

    def _sample_reward(self) -> np.ndarray:
        return self._sample_reward_gaussian()

    def _sample_reward_gaussian(self) -> np.ndarray:
        """Gaussian-mixture reward: k rooms × n_peaks_per_room bumps, spatially smooth."""
        r = np.zeros(self.n_states)
        peak_ranges = self.room_peak_ranges
        active_rooms = self._rng.choice(4, size=self.k, replace=False)
        for room_idx in active_rooms:
            room_states = self._rooms[room_idx]
            if not room_states:
                continue
            lo, hi = peak_ranges[room_idx % len(peak_ranges)]
            for _ in range(self.n_peaks_per_room):
                center_s = int(self._rng.choice(room_states))
                center   = self._pos_arr[center_s]
                sigma    = self._rng.uniform(*self.sigma_range)
                height   = self._rng.uniform(lo, hi)
                dists_sq = np.sum((self._pos_arr - center) ** 2, axis=1)
                r       += height * np.exp(-dists_sq / (2.0 * sigma ** 2))
        mx = r.max()
        if mx > 0:
            r *= 1.0 / mx
        return r

    def _build_transition(self) -> np.ndarray:
        P = np.zeros((N_ACTIONS, self.n_states, self.n_states))
        for s, (r, c) in enumerate(self._pos_list):
            for a, (dr, dc) in enumerate(_DELTAS):
                nr, nc = r + dr, c + dc
                
                if self.drift is not None and self._rng.random() < 0.2:
                    nr, nc = r + self.drift[0], c + self.drift[1]
                
                # Simple one-way logic if requested (for other RQs)
                # Not used for the base RQ5 corridor sweep
                is_blocked = False
                if self.one_way_prob < 1.0:
                    # If leaving a room into a corridor, check prob
                    in_room = any(s in room for room in self._rooms)
                    next_s_idx = self._pos2idx.get((nr, nc))
                    if next_s_idx is not None:
                        next_in_room = any(next_s_idx in room for room in self._rooms)
                        if in_room and not next_in_room and self._rng.random() > self.one_way_prob:
                            is_blocked = True

                s_next = self._pos2idx.get((nr, nc), s)
                if is_blocked: s_next = s
                
                P[a, s, s_next] += 1.0
        P /= P.sum(axis=2, keepdims=True)
        return P

    def shift_reward(self) -> np.ndarray:
        self.r = self._sample_reward()
        return self.r.copy()

    def value_iteration(self, r: Optional[np.ndarray] = None) -> np.ndarray:
        if r is None: r = self.r
        V = np.zeros(self.n_states)
        for _ in range(1000):
            Q = r[None, :] + self.gamma * (self.P * V[None, None, :]).sum(axis=2)
            V_new = Q.max(axis=0)
            if np.linalg.norm(V_new - V, ord=np.inf) < 1e-8: break
            V = V_new
        return V

    def greedy_policy(self, V: np.ndarray, r: Optional[np.ndarray] = None) -> np.ndarray:
        if r is None: r = self.r
        Q = r[None, :] + self.gamma * (self.P * V[None, None, :]).sum(axis=2)
        return Q.argmax(axis=0)
