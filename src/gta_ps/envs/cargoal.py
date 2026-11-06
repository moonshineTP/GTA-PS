"""
Safety-Gymnasium CarGoal discretized tabular wrapper.

Wraps SafetyCarGoal1-v0 and builds an approximate tabular MDP by:
  1. Discretizing the 2D position state into an N×N grid.
  2. Running rollouts under all macro-actions to estimate P_hat, r_hat.

State encoding:
  s = idx_x * N + idx_y
  where idx_x, idx_y = digitize(x_pos), digitize(y_pos) into [0, N)

Macro-action set (at least 10):
  10 steering × throttle combinations from a polar grid.

Reward configuration (CarGoal):
  Goal relocated among anchor waypoints; disturbance ∈ {calm, rain, gust, slip}.

INSTALLATION NOTE
-----------------
safety-gymnasium 1.0.0 requires a specific dependency stack that conflicts
with gymnasium >= 1.0.  Install in a separate environment or pin:

    pip install "safety-gymnasium==1.0.0" "gymnasium==0.26.3" \
                "mujoco==2.3.3" "gymnasium-robotics==1.2.2"

In the default environment (gymnasium >= 1.0 + mujoco >= 3.x) this module
raises ImportError with an actionable message and falls back gracefully.

References
----------
Ji et al. (2023) Safety-Gymnasium.
"""
from __future__ import annotations

from typing import Optional
import numpy as np

from .base import TabularEnv
from ..algorithms.empirical_model import CountModel

# ---------------------------------------------------------------------------
# Macro-action table (10 steering × throttle combinations, polar grid)
# ---------------------------------------------------------------------------
# Safety-Gymnasium CarGoal action space: [steering, throttle] in [-1, 1]^2
_MACRO_ACTIONS = np.array([
    [ 0.0,  1.0],   # straight ahead
    [ 0.5,  1.0],   # slight right + full throttle
    [-0.5,  1.0],   # slight left  + full throttle
    [ 1.0,  1.0],   # hard right   + full throttle
    [-1.0,  1.0],   # hard left    + full throttle
    [ 0.0,  0.5],   # straight     + half throttle
    [ 0.5,  0.5],   # slight right + half throttle
    [-0.5,  0.5],   # slight left  + half throttle
    [ 1.0,  0.0],   # spin right   + no throttle
    [-1.0,  0.0],   # spin left    + no throttle
], dtype=np.float32)
N_MACRO_ACTIONS = len(_MACRO_ACTIONS)  # 10

# Anchor waypoints (x, y) in the CarGoal arena
_ANCHORS = np.array([
    [ 1.5,  1.5],
    [-1.5,  1.5],
    [ 1.5, -1.5],
    [-1.5, -1.5],
    [ 0.0,  2.0],
    [ 0.0, -2.0],
    [ 2.0,  0.0],
    [-2.0,  0.0],
], dtype=np.float32)

_DISTURB_MODES = ["calm", "directional_rain", "cyclic_gust", "slippery_patch"]


def _try_import_safety_gymnasium():
    """Attempt to import safety_gymnasium; raise informative error if unavailable."""
    try:
        import safety_gymnasium  # noqa: F401
        return safety_gymnasium
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "safety-gymnasium is not importable in this environment.\n"
            "CarGoal requires a separate install:\n"
            "  pip install 'safety-gymnasium==1.0.0' 'gymnasium==0.26.3' "
            "'mujoco==2.3.3' 'gymnasium-robotics==1.2.2'\n"
            f"Original error: {exc}"
        ) from exc


class CarGoalDiscrete(TabularEnv):
    """
    Discretized tabular MDP built from Safety-Gymnasium SafetyCarGoal1-v0.

    Parameters
    ----------
    n_bins : int
        Discretization bins per spatial dimension (arena is ~[-3, 3]^2).
    n_rollout_steps : int
        Total steps used to estimate P_hat.
    gamma : float
    disturbance : str
        Initial disturbance mode.
    seed : int
    """

    def __init__(
        self,
        n_bins: int = 20,
        n_rollout_steps: int = 30_000,
        gamma: float = 0.95,
        disturbance: str = "calm",
        seed: int = 0,
    ):
        sg = _try_import_safety_gymnasium()

        self.n_bins = n_bins
        self.gamma = gamma
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        self.n_states = n_bins * n_bins
        self.n_actions = N_MACRO_ACTIONS

        # Position bins (arena ≈ [-3, 3] in both axes)
        self._bins = np.linspace(-3.0, 3.0, n_bins + 1)

        self._gym_env = sg.make("SafetyCarGoal1-v0", render_mode=None)
        self._gym_env.reset(seed=seed)

        self._goal_anchor = _ANCHORS[0].copy()
        self._disturbance_mode = disturbance

        self._learner = CountModel(self.n_states, self.n_actions)
        self._collect_rollouts(n_rollout_steps)
        self.P, self.r = self._learner.get_model()


    # ------------------------------------------------------------------
    # Discretization
    # ------------------------------------------------------------------

    def _digitize(self, v: float) -> int:
        idx = int(np.digitize(v, self._bins)) - 1
        return int(np.clip(idx, 0, self.n_bins - 1))

    def _obs_to_state(self, obs: np.ndarray) -> int:
        """CarGoal obs: position is approximately obs[0:2] in ego-frame."""
        # safety_gymnasium obs layout: [lidar..., goal_dist, goal_angle, ...]
        # For planning we approximate state by agent (x, y) in world frame.
        # obs[-3:-1] is typically goal-relative position; we use raw position
        # from the underlying MuJoCo data instead.
        try:
            uw = self._gym_env.unwrapped
            x = float(uw.task.agent.pos[0])
            y = float(uw.task.agent.pos[1])
        except AttributeError:
            # Fallback: use first two obs components as proxy
            x, y = float(obs[0]), float(obs[1])
        ix = self._digitize(x)
        iy = self._digitize(y)
        return ix * self.n_bins + iy

    def _apply_disturbance(self, action: np.ndarray) -> np.ndarray:
        if self._disturbance_mode == "calm":
            return action
        elif self._disturbance_mode == "directional_rain":
            drag = float(self._rng.uniform(0.8, 1.0))
            return (action * drag).astype(np.float32)
        elif self._disturbance_mode == "cyclic_gust":
            gust = np.array([np.sin(self._rng.uniform(0, 2 * np.pi)) * 0.15, 0.0], dtype=np.float32)
            return np.clip(action + gust, -1.0, 1.0)
        elif self._disturbance_mode == "slippery_patch":
            if self._rng.random() < 0.2:
                noise = self._rng.normal(0, 0.2, size=action.shape).astype(np.float32)
                return np.clip(action + noise, -1.0, 1.0)
        return action

    def _reward_from_state(self, obs: np.ndarray) -> float:
        """Dense reward: inverse distance from agent to goal anchor."""
        try:
            uw = self._gym_env.unwrapped
            pos = np.array(uw.task.agent.pos[:2], dtype=np.float32)
        except AttributeError:
            pos = np.array(obs[:2], dtype=np.float32)
        dist = float(np.linalg.norm(pos - self._goal_anchor))
        return 1.0 / (1.0 + dist)


    # ------------------------------------------------------------------
    # Model learning
    # ------------------------------------------------------------------

    def _collect_rollouts(self, n_steps: int) -> None:
        obs, _ = self._gym_env.reset(seed=self._seed)
        s = self._obs_to_state(obs)
        self._learner.reset()

        for _ in range(n_steps):
            a = int(self._rng.integers(0, N_MACRO_ACTIONS))
            act = self._apply_disturbance(_MACRO_ACTIONS[a].copy())
            result = self._gym_env.step(act)
            obs_next = result[0]
            term = result[2]
            trunc = result[3]
            r = self._reward_from_state(obs_next)
            s_next = self._obs_to_state(obs_next)
            self._learner.update(s, a, s_next, r)
            s = s_next
            if term or trunc:
                obs, _ = self._gym_env.reset()
                s = self._obs_to_state(obs)


    # ------------------------------------------------------------------
    # TabularEnv interface
    # ------------------------------------------------------------------

    def shift_reward(self, n_rollout_steps: int = 6_000) -> np.ndarray:
        """Relocate goal to a new anchor, switch disturbance mode."""
        anchor_idx = int(self._rng.integers(0, len(_ANCHORS)))
        self._goal_anchor = _ANCHORS[anchor_idx].copy()
        self._disturbance_mode = str(self._rng.choice(_DISTURB_MODES))

        self._learner._reward_sum[:] = 0.0
        self._learner._reward_cnt[:] = 0.0

        obs, _ = self._gym_env.reset()
        s = self._obs_to_state(obs)
        for _ in range(n_rollout_steps):
            a = int(self._rng.integers(0, N_MACRO_ACTIONS))
            act = self._apply_disturbance(_MACRO_ACTIONS[a].copy())
            result = self._gym_env.step(act)
            obs_next = result[0]
            term = result[2]
            trunc = result[3]
            r = self._reward_from_state(obs_next)
            s_next = self._obs_to_state(obs_next)
            self._learner._reward_sum[s] += r
            self._learner._reward_cnt[s] += 1.0
            s = s_next
            if term or trunc:
                obs, _ = self._gym_env.reset()
                s = self._obs_to_state(obs)

        _, self.r = self._learner.get_model()
        return self.r.copy()


    def rebuild_model(self, n_steps: int = 30_000) -> None:
        """Force full model rebuild."""
        self._collect_rollouts(n_steps)
        self.P, self.r = self._learner.get_model()
