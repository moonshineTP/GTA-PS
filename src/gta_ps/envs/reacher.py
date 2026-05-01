"""
MuJoCo Reacher discretized tabular wrapper.

Wraps gymnasium Reacher-v5 and builds an approximate tabular MDP by:
  1. Discretizing the joint angle state (theta1, theta2) into an N×N grid.
  2. Running rollouts under all macro-actions to estimate P_hat and r_hat.

State encoding:
  s = idx1 * N + idx2
  where idx1, idx2 = digitize(theta1), digitize(theta2) into [0, N)

Macro-action set (≥10 per memo §8):
  All combinations of {-1, 0, +1} for each of 2 joints → 9 combinations,
  plus 4 diagonal half-steps → 13 total macro-actions.

Reward configuration (memo §8, MuJoCo Reacher):
  Goal relocated among workspace anchors; reward = 1 / (1 + dist) at goal.
  Disturbance modes: none, mild_torque, sticky_drag, directional_bias.

References
----------
Todorov et al. (2012) MuJoCo.
Towers et al. (2024) Gymnasium.
"""
from __future__ import annotations

from typing import Optional
import numpy as np

from .base import TabularEnv
from ..algorithms.empirical_model import CountModel


# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------

def _try_import_mujoco():
    """Verify gymnasium[mujoco] is available; raise informative error if not."""
    try:
        import gymnasium as gym
        import mujoco  # noqa: F401 — triggers clear error if not installed
        return gym
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "MuJoCo is not importable in this environment.\n"
            "Reacher requires the mujoco extra:\n"
            "  pip install 'gymnasium[mujoco]' 'mujoco>=3.0'\n"
            f"Original error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Macro-action table: (torque1, torque2) pairs, clipped to [-1, 1]
# ---------------------------------------------------------------------------
_MACRO_ACTIONS = np.array([
    [-1.0, -1.0],
    [-1.0,  0.0],
    [-1.0,  1.0],
    [ 0.0, -1.0],
    [ 0.0,  0.0],
    [ 0.0,  1.0],
    [ 1.0, -1.0],
    [ 1.0,  0.0],
    [ 1.0,  1.0],
    [-0.5, -0.5],
    [-0.5,  0.5],
    [ 0.5, -0.5],
    [ 0.5,  0.5],
], dtype=np.float32)
N_MACRO_ACTIONS = len(_MACRO_ACTIONS)  # 13

# Workspace anchors for goal relocation (in (x, y) end-effector space)
_ANCHORS = np.array([
    [ 0.15,  0.05],   # near-body, right
    [-0.15,  0.05],   # near-body, left
    [ 0.10,  0.10],   # mid-range, right
    [-0.10,  0.10],   # mid-range, left
    [ 0.05,  0.15],   # mid-range, far
    [ 0.18,  0.00],   # boundary, right
    [-0.18,  0.00],   # boundary, left
    [ 0.00,  0.18],   # boundary, far
], dtype=np.float32)

# Disturbance modes
_DISTURB_MODES = ["none", "mild_torque", "sticky_drag", "directional_bias"]


class ReacherDiscrete(TabularEnv):
    """
    Discretized tabular MDP built from gymnasium Reacher-v5.

    Parameters
    ----------
    n_bins : int
        Discretization bins per joint angle dimension.
    n_rollout_steps : int
        Total environment steps used to estimate P_hat.
    gamma : float
    disturbance : str
        One of {none, mild_torque, sticky_drag, directional_bias}.
    seed : int
    """

    def __init__(
        self,
        n_bins: int = 16,
        n_rollout_steps: int = 20_000,
        gamma: float = 0.95,
        disturbance: str = "none",
        seed: int = 0,
    ):
        self.n_bins = n_bins
        self.gamma = gamma
        self.disturbance = disturbance
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        self.n_states = n_bins * n_bins
        self.n_actions = N_MACRO_ACTIONS

        # Joint angle bins (Reacher joints are in [-pi, pi])
        self._bins = np.linspace(-np.pi, np.pi, n_bins + 1)

        gym = _try_import_mujoco()
        self._gym_env = gym.make("Reacher-v5", render_mode=None)
        self._gym_env.reset(seed=seed)

        # Current goal anchor
        self._goal_anchor = _ANCHORS[0].copy()
        self._disturbance_mode = disturbance

        # Build initial model via rollouts
        self._learner = CountModel(self.n_states, self.n_actions)
        self._collect_rollouts(n_rollout_steps)
        self.P, self.r = self._learner.get_model()

    # ------------------------------------------------------------------
    # Discretization
    # ------------------------------------------------------------------

    def _digitize(self, theta: float) -> int:
        idx = int(np.digitize(theta, self._bins)) - 1
        return int(np.clip(idx, 0, self.n_bins - 1))

    def _obs_to_state(self, obs: np.ndarray) -> int:
        """Reacher-v5 obs: [cos(t1), cos(t2), sin(t1), sin(t2), tgt_x, tgt_y, ...]"""
        theta1 = float(np.arctan2(obs[2], obs[0]))   # sin/cos -> angle
        theta2 = float(np.arctan2(obs[3], obs[1]))
        i1 = self._digitize(theta1)
        i2 = self._digitize(theta2)
        return i1 * self.n_bins + i2

    def _apply_disturbance(self, action: np.ndarray) -> np.ndarray:
        """Inject disturbance into macro-action torques."""
        if self._disturbance_mode == "none":
            return action
        elif self._disturbance_mode == "mild_torque":
            noise = self._rng.normal(0, 0.1, size=action.shape).astype(np.float32)
            return np.clip(action + noise, -1.0, 1.0)
        elif self._disturbance_mode == "sticky_drag":
            drag = float(self._rng.uniform(0.7, 1.0))
            return (action * drag).astype(np.float32)
        elif self._disturbance_mode == "directional_bias":
            bias = np.array([0.2, 0.0], dtype=np.float32)
            return np.clip(action + bias, -1.0, 1.0)
        return action

    def _reward_from_obs(self, obs: np.ndarray) -> float:
        """Dense reward: inverse distance from fingertip to goal anchor.

        Uses MuJoCo kinematics for absolute fingertip position.
        obs[8:10] is fingertip-minus-target (relative), not absolute — do not use.
        """
        finger_xy = self._gym_env.unwrapped.data.body("fingertip").xpos[:2]
        dist = float(np.linalg.norm(finger_xy - self._goal_anchor))
        return 1.0 / (1.0 + dist)

    # ------------------------------------------------------------------
    # Model learning
    # ------------------------------------------------------------------

    def _collect_rollouts(self, n_steps: int) -> None:
        """Collect transitions under random macro-actions to estimate P_hat."""
        obs, _ = self._gym_env.reset(seed=self._seed)
        s = self._obs_to_state(obs)
        self._learner.reset()

        for _ in range(n_steps):
            a = int(self._rng.integers(0, N_MACRO_ACTIONS))
            torques = self._apply_disturbance(_MACRO_ACTIONS[a].copy())
            obs_next, _, term, trunc, _ = self._gym_env.step(torques)
            r = self._reward_from_obs(obs_next)
            s_next = self._obs_to_state(obs_next)
            self._learner.update(s, a, s_next, r)
            s = s_next
            if term or trunc:
                obs, _ = self._gym_env.reset()
                s = self._obs_to_state(obs)

    # ------------------------------------------------------------------
    # TabularEnv interface
    # ------------------------------------------------------------------

    def shift_reward(self, n_rollout_steps: int = 5_000) -> np.ndarray:
        """
        Relocate goal to a new anchor, switch disturbance mode,
        relearn reward estimate from fresh rollouts.
        Returns updated r_hat.
        """
        anchor_idx = int(self._rng.integers(0, len(_ANCHORS)))
        self._goal_anchor = _ANCHORS[anchor_idx].copy()
        self._disturbance_mode = str(self._rng.choice(_DISTURB_MODES))

        # Only refresh the reward estimate (P_hat reuse is intentional for speed)
        self._learner._reward_sum[:] = 0.0
        self._learner._reward_cnt[:] = 0.0

        obs, _ = self._gym_env.reset()
        s = self._obs_to_state(obs)
        for _ in range(n_rollout_steps):
            a = int(self._rng.integers(0, N_MACRO_ACTIONS))
            torques = self._apply_disturbance(_MACRO_ACTIONS[a].copy())
            obs_next, _, term, trunc, _ = self._gym_env.step(torques)
            r = self._reward_from_obs(obs_next)
            s_next = self._obs_to_state(obs_next)
            self._learner._reward_sum[s] += r
            self._learner._reward_cnt[s] += 1.0
            s = s_next
            if term or trunc:
                obs, _ = self._gym_env.reset()
                s = self._obs_to_state(obs)

        _, self.r = self._learner.get_model()
        return self.r.copy()

    def rebuild_model(self, n_steps: int = 20_000) -> None:
        """Force a full model rebuild (P_hat + r_hat)."""
        self._collect_rollouts(n_steps)
        self.P, self.r = self._learner.get_model()
