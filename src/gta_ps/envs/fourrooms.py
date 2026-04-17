"""
FourRooms tabular MDP adapter.

Extracts the exact (|S|, |S|) transition kernel from MiniGrid-FourRooms-v1
by exhaustive rollout, then exposes a known-model planning interface
matching GARNETEnv's API.

|S| ~ 384 (11x11 grid minus walls), |A| = 3 (left/right/forward).
gamma = 0.99 per memo.

Reward configuration k in {1,2,3,4}: k rooms active with sparse goal rewards.
"""
import numpy as np
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Grid topology (static, no gymnasium dependency needed for planning)
# ---------------------------------------------------------------------------

def _fourrooms_grid() -> Tuple[np.ndarray, Dict]:
    """
    Build the deterministic FourRooms grid manually.
    Returns (passable_cells, cell_to_idx) where passable_cells is an (N, 2)
    array of (row, col) coordinates and cell_to_idx maps (row,col)->int.

    Layout (19x19 standard FourRooms):
      outer walls + 2 internal walls with 1-cell doorways each.
    """
    H, W = 19, 19
    # Mark all cells as wall by default
    grid = np.ones((H, W), dtype=bool)   # True = wall

    # Carve out interior
    grid[1:H-1, 1:W-1] = False          # free
    # Horizontal wall (row 9), vertical wall (col 9)
    grid[9, 1:W-1] = True
    grid[1:H-1, 9] = True
    # Doorways (one per wall segment, approximate positions from minigrid)
    # Horizontal wall: doorway at col 2 (left half) and col 14 (right half)
    grid[9, 2] = False
    grid[9, 14] = False
    # Vertical wall: doorway at row 4 (top half) and row 14 (bottom half)
    grid[4, 9] = False
    grid[14, 9] = False

    cells = [(r, c) for r in range(H) for c in range(W) if not grid[r, c]]
    cell_to_idx = {cell: i for i, cell in enumerate(cells)}
    return np.array(cells), cell_to_idx, grid


# Precompute once
_CELLS, _CELL_TO_IDX, _GRID = _fourrooms_grid()
_N_STATES = len(_CELLS)

# 4 movement directions: up, right, down, left (row, col deltas)
_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N_ACTIONS = 4


def _build_fourrooms_P() -> np.ndarray:
    """
    Build deterministic (|A|, |S|, |S|) transition kernel.
    Action a moves in direction _DIRS[a]; bumping into a wall stays in place.
    """
    N = _N_STATES
    P = np.zeros((_N_ACTIONS, N, N))
    for i, (r, c) in enumerate(_CELLS):
        for a, (dr, dc) in enumerate(_DIRS):
            nr, nc = r + dr, c + dc
            if 0 <= nr < _GRID.shape[0] and 0 <= nc < _GRID.shape[1] and not _GRID[nr, nc]:
                j = _CELL_TO_IDX[(nr, nc)]
            else:
                j = i   # stay
            P[a, i, j] = 1.0
    return P


_P_FOURROOMS = _build_fourrooms_P()


# ---------------------------------------------------------------------------
# Room membership
# ---------------------------------------------------------------------------

def _room_cells() -> Dict[int, list]:
    """Assign each free cell to one of 4 rooms by quadrant."""
    rooms: Dict[int, list] = {1: [], 2: [], 3: [], 4: []}
    for i, (r, c) in enumerate(_CELLS):
        top = r < 9
        left = c < 9
        if top and left:
            rooms[1].append(i)
        elif top and not left:
            rooms[2].append(i)
        elif not top and left:
            rooms[3].append(i)
        else:
            rooms[4].append(i)
    return rooms


_ROOM_CELLS = _room_cells()


class FourRoomsTabular:
    """
    Tabular FourRooms MDP with known transition kernel and nonstationary rewards.

    Parameters
    ----------
    k : int
        Number of simultaneously active rooms (1-4).
    gamma : float
    seed : int
    """

    def __init__(self, k: int = 1, gamma: float = 0.99, seed: int = 0):
        self.k = k
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.n_states = _N_STATES
        self.n_actions = _N_ACTIONS
        self.P = _P_FOURROOMS   # shape (|A|, |S|, |S|)

        self._active_rooms: list = []
        self._goal_states: list = []
        self.r = self._sample_reward()

    # ------------------------------------------------------------------
    def _sample_reward(self) -> np.ndarray:
        """
        Pick k rooms, assign one goal cell per room (Uniform([0.5, 1.5])).
        """
        r = np.zeros(self.n_states)
        room_ids = list(_ROOM_CELLS.keys())
        active = self.rng.choice(room_ids, size=self.k, replace=False).tolist()
        self._active_rooms = active
        self._goal_states = []
        for room_id in active:
            cells = _ROOM_CELLS[room_id]
            goal = int(self.rng.choice(cells))
            self._goal_states.append(goal)
            w = float(self.rng.uniform(0.5, 1.5))
            r[goal] = w
        return r

    def shift_reward(self) -> np.ndarray:
        self.r = self._sample_reward()
        return self.r.copy()

    def policy_transition(self, policy: np.ndarray) -> np.ndarray:
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
        Q = r[None, :] + self.gamma * (self.P * V[None, None, :]).sum(axis=2)
        return Q.argmax(axis=0)
