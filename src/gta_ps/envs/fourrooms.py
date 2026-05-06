"""
FourRooms tabular MDP with adjustable corridor bottlenecks.

The environment is position-only with 4 actions:
left / up / right / down.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import TabularEnv

N_ACTIONS = 4
ACTION_LEFT = 0
ACTION_UP = 1
ACTION_RIGHT = 2
ACTION_DOWN = 3
_DRIFT_TO_ACTION = {
    (0, -1): ACTION_LEFT,
    (-1, 0): ACTION_UP,
    (0, 1): ACTION_RIGHT,
    (1, 0): ACTION_DOWN,
}
_DELTAS = [
    (0, -1),  # left
    (-1, 0),  # up
    (0, 1),   # right
    (1, 0),   # down
]


def _build_corridor_grid(
    room_size: int,
    corridor_len: int,
    corridor_width: int = 1,
    window_offset: int = 0,
) -> frozenset[tuple[int, int]]:
    """
    Build a grid of 4 rooms connected by corridors.
    Each room is (room_size x room_size).
    Corridors are (corridor_len) cells long and (corridor_width) cells wide.
    Total grid width  = 2 * room_size + corridor_len + 2 (outer walls)
    Total grid height = 2 * room_size + corridor_len + 2 (outer walls)

    corridor_width > 1 creates multiple parallel paths between rooms,
    which reduces the spectral gap (SLEM closer to 1) and speeds up mixing.

    window_offset shifts the corridor window relative to the wall:
      - window_offset=0: window centered (original behavior)
      - window_offset>0: window shifted down/right
      - window_offset<0: window shifted up/left
    The window size = corridor_len, and it slides within the room boundary.
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

    # Corridors with window_offset and corridor_width.
    mid_base = (room_size + 1) // 2
    mid = max(1, min(room_size, mid_base + window_offset))

    w_half = corridor_width // 2
    w_extra = corridor_width % 2

    # R0 <-> R1 (horizontal corridor)
    for w_off in range(-w_half, w_half + w_extra):
        row = mid + w_off
        if 1 <= row <= 2 * room_size + corridor_len:
            for c in range(room_size + 1, c_start_r1):
                passable.add((row, c))

    # R2 <-> R3 (horizontal corridor)
    base_row_r2 = r_start_r2 + mid - 1
    for w_off in range(-w_half, w_half + w_extra):
        row = base_row_r2 + w_off
        if 1 <= row <= 2 * room_size + corridor_len:
            for c in range(room_size + 1, c_start_r1):
                passable.add((row, c))

    # R0 <-> R2 (vertical corridor)
    for w_off in range(-w_half, w_half + w_extra):
        col = mid + w_off
        if 1 <= col <= 2 * room_size + corridor_len:
            for r in range(room_size + 1, r_start_r2):
                passable.add((r, col))

    # R1 <-> R3 (vertical corridor)
    base_col_r3 = c_start_r1 + mid - 1
    for w_off in range(-w_half, w_half + w_extra):
        col = base_col_r3 + w_off
        if 1 <= col <= 2 * room_size + corridor_len:
            for r in range(room_size + 1, r_start_r2):
                passable.add((r, col))

    return frozenset(passable)


class FourRoomsTabular(TabularEnv):
    ACTION_LEFT = ACTION_LEFT
    ACTION_UP = ACTION_UP
    ACTION_RIGHT = ACTION_RIGHT
    ACTION_DOWN = ACTION_DOWN
    DEFAULT_DRIFT_STRENGTH = 0.35

    def __init__(
        self,
        room_size: int = 3,
        corridor_len: int = 1,
        corridor_width: int = 1,
        window_offset: int = 0,
        k: int = 1,
        gamma: float = 0.95,
        one_way_prob: float = 1.0,
        drift: Optional[tuple[int, int]] = None,
        drift_strength: float = DEFAULT_DRIFT_STRENGTH,
        seed: int = 0,
        n_peaks_per_room: int = 2,
        sigma_range: tuple = (0.3, 0.9),
        room_peak_ranges: Optional[List] = None,
    ):
        self.room_size = room_size
        self.corridor_len = corridor_len
        self.corridor_width = corridor_width
        self.window_offset = window_offset
        self.k = k
        self.gamma = gamma
        self.one_way_prob = self._validate_one_way_prob(one_way_prob)
        self.drift = self._validate_drift(drift)
        self.drift_strength = self._validate_drift_strength(drift_strength)
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # Gaussian reward config — always active; None room_peak_ranges → equal ranges
        self.n_peaks_per_room = n_peaks_per_room
        self.sigma_range = tuple(sigma_range)
        self.room_peak_ranges = list(room_peak_ranges) if room_peak_ranges is not None else [(0.3, 1.0)] * 4

        self._passable = _build_corridor_grid(room_size, corridor_len, corridor_width, window_offset)
        self._pos_list = sorted(list(self._passable))
        self._pos2idx = {p: i for i, p in enumerate(self._pos_list)}
        self._pos_arr = np.array(self._pos_list, dtype=np.float64)  # cached for Gaussian

        self.n_positions = len(self._pos_list)
        self.n_states = self.n_positions
        self.n_actions = N_ACTIONS

        self._rooms = self._detect_rooms()
        self.base_r = self._sample_reward()
        self.r = self.base_r.copy()
        self.P = self._build_transition()

    @staticmethod
    def _validate_one_way_prob(one_way_prob: float) -> float:
        one_way_prob = float(one_way_prob)
        if not 0.0 <= one_way_prob <= 1.0:
            raise ValueError(f"one_way_prob must lie in [0, 1], got {one_way_prob!r}")
        return one_way_prob

    @staticmethod
    def _validate_drift(drift: Optional[tuple[int, int]]) -> Optional[tuple[int, int]]:
        if drift is None:
            return None
        drift = tuple(int(x) for x in drift)
        if drift not in _DRIFT_TO_ACTION:
            raise ValueError(f"Unsupported drift vector {drift!r}")
        return drift

    @staticmethod
    def _validate_drift_strength(drift_strength: float) -> float:
        drift_strength = float(drift_strength)
        if not 0.0 <= drift_strength <= 1.0:
            raise ValueError(f"drift_strength must lie in [0, 1], got {drift_strength!r}")
        return drift_strength

    def _detect_rooms(self) -> list[list[int]]:
        rooms = [[] for _ in range(4)]
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
                center = self._pos_arr[center_s]
                sigma = self._rng.uniform(*self.sigma_range)
                height = self._rng.uniform(lo, hi)
                dists_sq = np.sum((self._pos_arr - center) ** 2, axis=1)
                r += height * np.exp(-dists_sq / (2.0 * sigma ** 2))
        mx = r.max()
        if mx > 0:
            r *= 1.0 / mx
        return r

    def _next_position(self, pos: int, action: int) -> int:
        row, col = self._pos_list[pos]
        dr, dc = _DELTAS[action]
        nr, nc = row + dr, col + dc

        next_pos = self._pos2idx.get((nr, nc), pos)
        return next_pos

    def belt_action(self, state: int) -> int | None:
        row, col = self._pos_list[state]
        c_start_r1 = self.room_size + self.corridor_len + 1
        r_start_r2 = self.room_size + self.corridor_len + 1
        mid_base = (self.room_size + 1) // 2
        mid = max(1, min(self.room_size, mid_base + self.window_offset))
        w_half = self.corridor_width // 2
        w_extra = self.corridor_width % 2

        top_rows = {mid + w for w in range(-w_half, w_half + w_extra)}
        bottom_rows = {r_start_r2 + mid - 1 + w for w in range(-w_half, w_half + w_extra)}
        left_cols = {mid + w for w in range(-w_half, w_half + w_extra)}
        right_cols = {c_start_r1 + mid - 1 + w for w in range(-w_half, w_half + w_extra)}

        if row in top_rows and self.room_size < col < c_start_r1:
            return self.ACTION_RIGHT
        if col in right_cols and self.room_size < row < r_start_r2:
            return self.ACTION_DOWN
        if row in bottom_rows and self.room_size < col < c_start_r1:
            return self.ACTION_LEFT
        if col in left_cols and self.room_size < row < r_start_r2:
            return self.ACTION_UP
        return None

    def belt_cycle_states(self) -> list[int]:
        return [state for state in range(self.n_states) if self.belt_action(state) is not None]

    def _apply_transport(self, P: np.ndarray) -> np.ndarray:
        P = P.copy()

        if self.one_way_prob < 1.0:
            belt_states = self.belt_cycle_states()
            if not belt_states:
                raise ValueError("one_way_prob < 1.0 requires at least one belt state in the current geometry")
            leak_prob = self.one_way_prob
            belt_weight = 1.0 - leak_prob
            belt_P = P.copy()
            for state in belt_states:
                belt_row = P[self.belt_action(state), state]
                for action in range(self.n_actions):
                    belt_P[action, state] = leak_prob * P[action, state] + belt_weight * belt_row
            P = belt_P

        if self.drift is not None and self.drift_strength > 0.0:
            drift_action = _DRIFT_TO_ACTION[self.drift]
            drift_P = P.copy()
            for state in range(self.n_states):
                drift_row = P[drift_action, state]
                for action in range(self.n_actions):
                    drift_P[action, state] = (1.0 - self.drift_strength) * P[action, state] + self.drift_strength * drift_row
            P = drift_P

        P /= P.sum(axis=2, keepdims=True)
        return P

    def _build_transition(self) -> np.ndarray:
        P = np.zeros((self.n_actions, self.n_states, self.n_states), dtype=np.float64)
        for pos in range(self.n_positions):
            for action in range(self.n_actions):
                next_pos = self._next_position(pos, action)
                P[action, pos, next_pos] = 1.0
        return self._apply_transport(P)

    def shift_reward(self) -> np.ndarray:
        self.base_r = self._sample_reward()
        self.r = self.base_r.copy()
        return self.r.copy()
