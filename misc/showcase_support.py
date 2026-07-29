from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Wall

from gta_ps.algorithms import make_agent
from gta_ps.envs.fourrooms import FourRoomsTabular
from gta_ps.utils.hyperparams import load_tuned_for_env
from gta_ps.utils.metrics import value_error_l1
from plot_academic import LABELS as AGENT_LABELS

K = 10
ABLATION_T_SHIFT = 100
SHOWCASE_T_SHIFT = 100
ABLATION_ROOM_SIZE = 3
ABLATION_CORRIDOR_LEN = 1
SHOWCASE_ROOM_SIZE = 3
SHOWCASE_CORRIDOR_LEN = 1
SHOWCASE_ACTIVE_ROOMS = 2
SHOWCASE_PEAKS_PER_ROOM = 1
SHOWCASE_SIGMA_RANGE = (0.3, 0.9)
N_SHIFTS = 30
EPS_GREEDY = 0.2
GAMMA = 0.95

VARIANTS = ("standard_ps", "gta_beta")
LABELS = {variant: AGENT_LABELS[variant] for variant in VARIANTS}


def _heap_scores(template: np.ndarray, entries, versions: dict[int, int]) -> np.ndarray:
    scores = np.zeros_like(template)
    for neg_priority, version, state in entries:
        if versions.get(state, -1) == version:
            scores[state] = max(scores[state], float(-neg_priority))
    return scores


def standard_ps_queue_scores(module) -> np.ndarray:
    return _heap_scores(module.xi, module._heap, module._version)


def gta_ps_queue_scores(module) -> np.ndarray:
    scores = _heap_scores(module.xi, module._Q_xi, module._version_xi)
    topology_scores = _heap_scores(module.xi, module._H_q, module._version_hq)
    return np.maximum(scores, topology_scores)


def apply_reward_update(agent, r_new: np.ndarray) -> None:
    reward_update = getattr(agent, "reward_update", None)
    if callable(reward_update):
        reward_update(r_new)
    else:
        agent.module.reward_update(r_new - agent.module.r)


def load_config(path: str | None) -> dict:
    if path is None:
        return {}
    import yaml

    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_method_cfgs(cfg: dict) -> dict:
    if not cfg:
        return load_tuned_for_env("fourrooms")
    root_defaults = {
        key: cfg.get(key)
        for key in ["alpha", "beta", "beta1", "beta2", "eta", "b"]
        if cfg.get(key) is not None
    }
    base_params = {**root_defaults, **(cfg.get("default") or {})}
    tuned = cfg.get("tuned") or {}
    result = {"standard_ps": {}}
    for variant in ("gta_base", "gta_beta"):
        variant_cfg = {key: value for key, value in base_params.items() if value is not None}
        overrides = {key: value for key, value in (tuned.get(variant) or {}).items() if value is not None}
        result[variant] = {**variant_cfg, **overrides}
    return result


def fourrooms_env_kwargs_from_config(cfg: dict) -> dict:
    if not cfg:
        return {"gamma": GAMMA}
    env_name = cfg.get("env", "fourrooms")
    if env_name != "fourrooms":
        raise ValueError(f"showcase.py currently supports only env='fourrooms' configs, got {env_name!r}")

    env_kwargs: dict = {"gamma": cfg.get("gamma", GAMMA)}
    key_map = {
        "k_fourrooms": "k",
        "k": "k",
        "room_size": "room_size",
        "corridor_len": "corridor_len",
        "one_way_prob": "one_way_prob",
        "drift": "drift",
        "n_peaks_per_room": "n_peaks_per_room",
        "sigma_range": "sigma_range",
        "room_peak_ranges": "room_peak_ranges",
    }
    for src_key, dst_key in key_map.items():
        if src_key in cfg and cfg[src_key] is not None:
            env_kwargs[dst_key] = cfg[src_key]

    if isinstance(env_kwargs.get("drift"), list):
        env_kwargs["drift"] = tuple(env_kwargs["drift"])
    if isinstance(env_kwargs.get("sigma_range"), list):
        env_kwargs["sigma_range"] = tuple(env_kwargs["sigma_range"])
    return env_kwargs


def _resolve_base_env_kwargs(
    cfg: dict,
    *,
    room_size: int,
    corridor_len: int,
) -> dict:
    env_kwargs = fourrooms_env_kwargs_from_config(cfg)
    env_kwargs.setdefault("room_size", room_size)
    env_kwargs.setdefault("corridor_len", corridor_len)
    return env_kwargs


@dataclass(frozen=True)
class AblationOptions:
    seed_count: int
    n_shifts: int
    shift_steps: int
    host: str
    K: int
    workers: int | str | None
    tuned_cfg: dict
    env_kwargs: dict

    @classmethod
    def from_args(cls, args) -> "AblationOptions":
        cfg = load_config(args.config)
        return cls(
            seed_count=args.seeds,
            n_shifts=args.n_shifts,
            shift_steps=args.n_steps_per_shift if args.n_steps_per_shift is not None else ABLATION_T_SHIFT,
            host=args.host,
            K=args.K,
            workers=args.workers,
            tuned_cfg=build_method_cfgs(cfg),
            env_kwargs=_resolve_base_env_kwargs(
                cfg,
                room_size=ABLATION_ROOM_SIZE,
                corridor_len=ABLATION_CORRIDOR_LEN,
            ),
        )


@dataclass(frozen=True)
class ShowcaseOptions:
    seed: int
    n_shifts: int
    shift_steps: int
    n_frames: int | None
    host: str
    K: int
    tuned_cfg: dict
    env_kwargs: dict

    @property
    def total_steps(self) -> int:
        return self.n_shifts * self.shift_steps

    @property
    def frame_count(self) -> int:
        if self.n_frames is None:
            return self.total_steps
        return min(self.n_frames, self.total_steps)

    @classmethod
    def from_args(cls, args) -> "ShowcaseOptions":
        cfg = load_config(args.config)
        env_kwargs = _resolve_base_env_kwargs(
            cfg,
            room_size=SHOWCASE_ROOM_SIZE,
            corridor_len=SHOWCASE_CORRIDOR_LEN,
        )
        env_kwargs["k"] = SHOWCASE_ACTIVE_ROOMS
        env_kwargs["n_peaks_per_room"] = SHOWCASE_PEAKS_PER_ROOM
        env_kwargs["sigma_range"] = SHOWCASE_SIGMA_RANGE
        return cls(
            seed=args.seeds if args.seeds is not None else 0,
            n_shifts=args.n_shifts,
            shift_steps=args.n_steps_per_shift if args.n_steps_per_shift is not None else SHOWCASE_T_SHIFT,
            n_frames=args.n_frames,
            host=args.host,
            K=args.K,
            tuned_cfg=build_method_cfgs(cfg),
            env_kwargs=env_kwargs,
        )


class FourRoomsRenderer:
    MINIGRID_TILE_SIZE = 32
    GOAL_COLOR = "#98ff66"

    def __init__(self, env: FourRoomsTabular) -> None:
        self.env = env
        self._pos_list = env._pos_list
        self._h = 2 * env.room_size + env.corridor_len + 2
        self._w = 2 * env.room_size + env.corridor_len + 2
        self._reward_marker_count = max(1, int(getattr(env, "k", 1)) * int(getattr(env, "n_peaks_per_room", 1)))
        self._minigrid_base = Grid(self._w, self._h)
        for row in range(self._h):
            for col in range(self._w):
                self._minigrid_base.set(col, row, Wall())
        for row, col in self._pos_list:
            self._minigrid_base.set(col, row, None)

    def _to_grid(self, values: np.ndarray) -> np.ndarray:
        grid = np.full((self._h, self._w), np.nan)
        for state, (row, col) in enumerate(self._pos_list):
            grid[row, col] = values[state]
        return grid

    def _format_axis(self, ax: Axes, title: str) -> None:
        ax.set_title(title, fontsize=8)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.set_xlim(-0.5, self._w - 0.5)
        ax.set_ylim(self._h - 0.5, -0.5)
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _reward_marker_states(self, reward: np.ndarray | None) -> list[int]:
        if reward is None:
            return []
        reward_array = np.asarray(reward, dtype=np.float64)
        positive_states = np.flatnonzero(reward_array > 0)
        if positive_states.size == 0:
            return []
        marker_count = min(self._reward_marker_count, int(positive_states.size))
        ranked = positive_states[np.argsort(reward_array[positive_states])[::-1]]
        return [int(state) for state in ranked[:marker_count]]

    def _build_minigrid_frame(
        self,
        reward: np.ndarray | None = None,
        agent_pos: int | None = None,
        agent_dir: int = 3,
    ) -> np.ndarray:
        grid = self._minigrid_base.copy()
        for state in self._reward_marker_states(reward):
            row, col = self._pos_list[state]
            grid.set(col, row, Goal())

        if agent_pos is None:
            agent_xy = (-1, -1)
        else:
            row, col = self._pos_list[agent_pos]
            agent_xy = (col, row)
        return grid.render(self.MINIGRID_TILE_SIZE, agent_pos=agent_xy, agent_dir=agent_dir)

    def draw_interface(
        self,
        ax: Axes,
        reward: np.ndarray | None = None,
        agent_pos: int | None = None,
        agent_dir: int = 3,
        title: str = "Environment",
    ) -> None:
        frame = self._build_minigrid_frame(reward=reward, agent_pos=agent_pos, agent_dir=agent_dir)
        ax.imshow(
            frame,
            extent=(-0.5, self._w - 0.5, self._h - 0.5, -0.5),
            interpolation="nearest",
            zorder=0,
        )
        self._format_axis(ax, title)

    def draw_metric(
        self,
        ax: Axes,
        values: np.ndarray,
        title: str,
        cmap: str,
        colorbar: bool = False,
    ) -> None:
        grid = np.ma.masked_invalid(self._to_grid(values))
        finite_values = grid.compressed()
        vmax = float(np.max(finite_values)) if finite_values.size else 0.0
        if vmax <= 1e-8:
            vmax = 1.0

        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(color="#ffffff")
        image = ax.imshow(
            grid,
            cmap=cmap_obj,
            vmin=0.0,
            vmax=vmax,
            origin="upper",
            interpolation="nearest",
            aspect="equal",
            zorder=1,
        )
        ax.set_facecolor("#ffffff")
        self._format_axis(ax, title)
        ax.set_axisbelow(False)
        ax.set_xticks(np.arange(-0.5, self._w, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self._h, 1), minor=True)
        ax.grid(which="minor", color="#d9d9d9", linestyle="-", linewidth=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)
        if colorbar:
            plt.colorbar(image, ax=ax, fraction=0.036, pad=0.04)

    def _draw_reward_overlay(self, ax: Axes, reward: np.ndarray | None) -> None:
        for state in self._reward_marker_states(reward):
            row, col = self._pos_list[state]
            ax.add_patch(
                mpatches.Rectangle(
                    (col - 0.34, row - 0.34),
                    0.68,
                    0.68,
                    facecolor="none",
                    edgecolor=self.GOAL_COLOR,
                    linewidth=1.2,
                    zorder=2,
                )
            )


class IndependentAgentSession:
    DELTA_TO_MINIGRID_DIR = {
        (-1, 0): 3,
        (1, 0): 1,
        (0, -1): 2,
        (0, 1): 0,
    }

    def __init__(
        self,
        variant: str,
        host: str,
        tuned_cfg: dict,
        K: int = K,
        eps: float = EPS_GREEDY,
        seed: int = 0,
        env_kwargs: dict | None = None,
        initial_reward: np.ndarray | None = None,
        initial_target_V: np.ndarray | None = None,
    ) -> None:
        self.variant = variant
        self.host = host
        self.K = K
        self.eps = eps
        self._rng = np.random.default_rng(seed)

        self.env = FourRoomsTabular(seed=seed, **(env_kwargs or {"gamma": GAMMA}))
        if initial_reward is not None:
            self.r_current = np.asarray(initial_reward, dtype=np.float64).copy()
            self.env.base_r = self.r_current.copy()
            self.env.r = self.r_current.copy()
        else:
            self.r_current = self.env.r.copy()

        variant_index = VARIANTS.index(variant)
        agent_kwargs = {}
        if host == "dyna":
            agent_kwargs["model_type"] = "sample"
        self.agent = make_agent(
            variant,
            host=host,
            P=self.env.P,
            r=self.env.r,
            gamma=self.env.gamma,
            n_states=self.env.n_states,
            n_actions=self.env.n_actions,
            rng=np.random.default_rng(np.random.SeedSequence([seed, variant_index, 0xD1A])),
            **agent_kwargs,
            **(tuned_cfg.get(variant, {}).copy() if variant != "standard_ps" else {}),
        )

        self.state = int(self._rng.integers(0, self.env.n_states))
        self.agent_dir = 3
        self.step_count = 0
        self._target_V = (
            np.asarray(initial_target_V, dtype=np.float64).copy()
            if initial_target_V is not None
            else self.env.value_iteration(self.r_current)
        )

        self.err_history: list[float] = []
        self.cumul_l1: list[float] = [0.0]
        self.return_history: list[float] = []
        self.cumul_reward: list[float] = [0.0]
        self.coverage_history: list[int] = []
        self._visited: set[int] = set()

    def _pick_action(self) -> int:
        if self._rng.random() < self.eps:
            return int(self._rng.integers(0, self.env.n_actions))

        q_values = self.agent.Q[:, self.state]
        best_value = float(np.max(q_values))
        best_actions = np.flatnonzero(np.isclose(q_values, best_value, atol=1e-10))
        return int(self._rng.choice(best_actions))

    def step_once(self) -> tuple[float, float]:
        state = self.state
        action = self._pick_action()
        next_state = int(self._rng.choice(self.env.n_states, p=self.env.P[action, state, :]))
        if next_state != state:
            row_s, col_s = self.env._pos_list[state]
            row_next, col_next = self.env._pos_list[next_state]
            delta = (row_next - row_s, col_next - col_s)
            self.agent_dir = self.DELTA_TO_MINIGRID_DIR.get(delta, self.agent_dir)

        reward_value = float(self.r_current[state])
        self.agent.real_step(state, action, reward_value, next_state)
        if self.host == "dyna":
            self.agent.plan_steps(self.K)
        else:
            self.agent.plan(K=self.K, r=self.r_current)

        self.state = next_state
        self.step_count += 1
        self._visited.add(state)
        error = value_error_l1(self.agent.Q.max(axis=0), self._target_V)
        return reward_value, error

    def shift_reward(
        self,
        r_current: np.ndarray | None = None,
        target_V: np.ndarray | None = None,
    ) -> None:
        if r_current is None:
            self.r_current = self.env.shift_reward()
        else:
            self.r_current = np.asarray(r_current, dtype=np.float64).copy()
            self.env.base_r = self.r_current.copy()
            self.env.r = self.r_current.copy()
        self._target_V = (
            self.env.value_iteration(self.r_current)
            if target_V is None
            else np.asarray(target_V, dtype=np.float64).copy()
        )
        apply_reward_update(self.agent, self.r_current)
        self._visited.clear()

    def episode(self, steps: int) -> tuple[float, float, int]:
        episode_reward = 0.0
        errors: list[float] = []
        for _ in range(steps):
            reward_value, error = self.step_once()
            episode_reward += reward_value
            errors.append(error)

        mean_error = float(np.mean(errors))
        coverage = len(self._visited)
        self.return_history.append(episode_reward)
        self.cumul_reward.append(self.cumul_reward[-1] + episode_reward)
        self.err_history.append(mean_error)
        self.cumul_l1.append(self.cumul_l1[-1] + mean_error)
        self.coverage_history.append(coverage)
        self._visited.clear()
        return episode_reward, mean_error, coverage

    def position_values(self) -> np.ndarray:
        return self.agent.Q.max(axis=0).copy()

    def position_reward(self) -> np.ndarray:
        return self.r_current.copy()

    def get_queue_scores(self) -> np.ndarray:
        if self.variant == "standard_ps":
            return standard_ps_queue_scores(self.agent.module)
        return gta_ps_queue_scores(self.agent.module)

    def get_combined_queue(self) -> np.ndarray:
        return self.get_queue_scores()


def build_reference_sessions(
    *,
    seed: int,
    host: str,
    tuned_cfg: dict,
    K: int,
    env_kwargs: dict | None = None,
) -> tuple[FourRoomsTabular, dict[str, IndependentAgentSession]]:
    ref_env = FourRoomsTabular(seed=seed, **(env_kwargs or {"gamma": GAMMA}))
    initial_reward = ref_env.r.copy()
    initial_target_V = ref_env.value_iteration(ref_env.r)
    sessions = {
        variant: IndependentAgentSession(
            variant=variant,
            host=host,
            tuned_cfg=tuned_cfg,
            K=K,
            seed=seed,
            env_kwargs=env_kwargs,
            initial_reward=initial_reward,
            initial_target_V=initial_target_V,
        )
        for variant in VARIANTS
    }
    return ref_env, sessions


def shift_reference_sessions(
    ref_env: FourRoomsTabular,
    sessions: dict[str, IndependentAgentSession],
) -> tuple[np.ndarray, np.ndarray]:
    ref_reward = ref_env.shift_reward()
    ref_target = ref_env.value_iteration(ref_reward)
    for session in sessions.values():
        session.shift_reward(r_current=ref_reward, target_V=ref_target)
    return ref_reward, ref_target


def run_seed_ablation(
    seed: int,
    K: int,
    n_shifts: int,
    T_shift: int,
    host: str,
    tuned_cfg: dict,
    env_kwargs: dict | None = None,
    **_ignored,
) -> dict:
    ref_env, sessions = build_reference_sessions(
        seed=seed,
        host=host,
        tuned_cfg=tuned_cfg,
        K=K,
        env_kwargs=env_kwargs,
    )

    for episode_idx in range(n_shifts):
        if episode_idx > 0:
            shift_reference_sessions(ref_env, sessions)
        for session in sessions.values():
            session.episode(T_shift)

    return {
        "return_history": {variant: list(sessions[variant].return_history) for variant in VARIANTS},
        "cumul_reward": {variant: list(sessions[variant].cumul_reward) for variant in VARIANTS},
        "cumul_l1": {variant: list(sessions[variant].cumul_l1) for variant in VARIANTS},
        "coverage": {variant: list(sessions[variant].coverage_history) for variant in VARIANTS},
        "final_V": {variant: sessions[variant].position_values().tolist() for variant in VARIANTS},
        "final_queue": {
            "standard_ps": sessions["standard_ps"].get_queue_scores().tolist(),
            "gta_beta": sessions["gta_beta"].get_combined_queue().tolist(),
        },
        "r_current": sessions["standard_ps"].position_reward().tolist(),
    }


def _session_snapshot(session: IndependentAgentSession, *, combined_queue: bool) -> dict:
    queue = session.get_combined_queue() if combined_queue else session.get_queue_scores()
    return {
        "env": session.env,
        "state": session.state,
        "agent_dir": session.agent_dir,
        "r_current": session.position_reward(),
        "V": session.position_values(),
        "queue": queue,
    }


def simulate_showcase_frames(
    options: ShowcaseOptions,
    *,
    on_step: Callable[[], None] | None = None,
) -> tuple[dict[str, IndependentAgentSession], list[dict]]:
    ref_env, sessions = build_reference_sessions(
        seed=options.seed,
        host=options.host,
        tuned_cfg=options.tuned_cfg,
        K=options.K,
        env_kwargs=options.env_kwargs,
    )

    capture_start = max(options.total_steps - options.frame_count, 0)
    captured_data: list[dict] = []
    episode_reward = {variant: 0.0 for variant in VARIANTS}

    for step in range(options.total_steps):
        shift_idx = step // options.shift_steps
        if step > 0 and step % options.shift_steps == 0:
            shift_reference_sessions(ref_env, sessions)
            episode_reward = {variant: 0.0 for variant in VARIANTS}

        for variant in VARIANTS:
            reward_value, _ = sessions[variant].step_once()
            episode_reward[variant] += reward_value

        if step >= capture_start:
            captured_data.append(
                {
                    "step": step + 1,
                    "shift": shift_idx + 1,
                    "ep_reward": dict(episode_reward),
                    "sps": _session_snapshot(sessions["standard_ps"], combined_queue=False),
                    "gta": _session_snapshot(sessions["gta_beta"], combined_queue=True),
                }
            )
        if on_step is not None:
            on_step()
    return sessions, captured_data
