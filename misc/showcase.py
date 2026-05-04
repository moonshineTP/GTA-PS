"""
misc/showcase.py

Visual demonstration: Standard-PS vs GTA-PS^β on independent FourRooms instances.

Modes
-----
--mode ablation  (default)
    Parallel multi-seed run (RunManager). Saves a 6-panel PNG:
    Col 0: Cumulative reward, L1 error, Coverage (both agents)
    Col 1: [Summary stats]
    Col 2: Final V(s) + queue heatmaps for each agent

--mode showcase
    Single-seed animation saved as GIF (Pillow, Agg backend).
    2x3 grid:
    Col 0: Standard-PS env | GTA-PS env (two independent instances)
    Col 1: Standard-PS V(s) | GTA-PS V(s)
    Col 2: Standard-PS queue ξ(s) | GTA-PS combined queue (ξ+τ)(s)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib import animation
from matplotlib.artist import Artist
from matplotlib.axes import Axes

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Wall
from tqdm import tqdm

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "configs").exists():
    for p in Path(__file__).resolve().parents:
        if (p / "configs").exists() and (p / "src").exists():
            ROOT = p
            break
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "threads"))
sys.path.insert(0, str(ROOT / "misc"))

from gta_ps.algorithms import make_agent
from gta_ps.envs.fourrooms import FourRoomsTabular
from gta_ps.utils.hyperparams import load_tuned_for_env
from gta_ps.utils.metrics import value_error_l1
from plot_academic import LABELS as AGENT_LABELS, apply_style, get_palette
from runner import RunManager

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

K = 16
ABLATION_T_SHIFT = 125
SHOWCASE_T_SHIFT = 125
ABLATION_ROOM_SIZE = 3
ABLATION_CORRIDOR_LEN = 1
SHOWCASE_ROOM_SIZE = 3
SHOWCASE_CORRIDOR_LEN = 1
SHOWCASE_ACTIVE_ROOMS = 2
SHOWCASE_PEAKS_PER_ROOM = 1
SHOWCASE_SIGMA_RANGE = (0.3, 0.9)
N_SHIFTS = 16
EPS_GREEDY = 0.12
GAMMA = 0.95

VARIANTS = ["standard_ps", "gta_beta"]
LABELS = {v: AGENT_LABELS[v] for v in VARIANTS}


def _standardps_queue_scores(module) -> np.ndarray:
    scores = np.zeros_like(module.xi)
    for neg_pri, version, state in module._heap:
        if module._version.get(state, -1) == version:
            scores[state] = max(scores[state], float(-neg_pri))
    return scores


def _gtaps_queue_scores(module) -> np.ndarray:
    scores = np.zeros_like(module.xi)
    for neg_pri, version, state in module._Q_xi:
        if module._version_xi.get(state, -1) == version:
            scores[state] = max(scores[state], float(-neg_pri))
    for neg_pri, version, state in module._H_q:
        if module._version_hq.get(state, -1) == version:
            scores[state] = max(scores[state], float(-neg_pri))
    return scores


def _apply_reward_update(agent, r_new: np.ndarray) -> None:
    reward_update = getattr(agent, "reward_update", None)
    if callable(reward_update):
        reward_update(r_new)
    else:
        agent.module.reward_update(r_new - agent.module.r)


def _load_config(path: str | None) -> dict:
    if path is None:
        return {}
    import yaml

    with open(path) as f:
        return yaml.safe_load(f) or {}


def _build_method_cfgs(cfg: dict) -> dict:
    if not cfg:
        return load_tuned_for_env("fourrooms")
    root_defaults = {
        key: cfg.get(key)
        for key in ["alpha", "beta", "beta1", "beta2", "eta", "b"]
        if cfg.get(key) is not None
    }
    default_block = cfg.get("default") or {}
    base_params = {**root_defaults, **default_block}
    tuned = cfg.get("tuned") or {}
    result = {"standard_ps": {}}
    for variant in ("gta_base", "gta_beta"):
        variant_cfg = {key: value for key, value in base_params.items() if value is not None}
        overrides = {key: value for key, value in (tuned.get(variant) or {}).items() if value is not None}
        result[variant] = {**variant_cfg, **overrides}
    return result


def _fourrooms_env_kwargs_from_config(cfg: dict) -> dict:
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


# ---------------------------------------------------------------------------
# FourRoomsRenderer — stateless rendering plug-in
# ---------------------------------------------------------------------------

class FourRoomsRenderer:
    """Map FourRooms states onto a 2-D grid and render heatmaps."""

    MINIGRID_TILE_SIZE = 32
    GOAL_COLOR = "#98ff66"
    GOAL_EDGE_COLOR = "#f4ffe8"

    def __init__(self, env: FourRoomsTabular) -> None:
        self.env = env
        self._pos_list = env._pos_list
        self._h = 2 * env.room_size + env.corridor_len + 2
        self._w = 2 * env.room_size + env.corridor_len + 2
        self._reward_marker_count = max(1, int(getattr(env, "k", 1)) * int(getattr(env, "n_peaks_per_room", 1)))
        self._passable_mask = np.zeros((self._h, self._w), dtype=bool)
        for row, col in self._pos_list:
            self._passable_mask[row, col] = True
        self._minigrid_base = Grid(self._w, self._h)
        for row in range(self._h):
            for col in range(self._w):
                self._minigrid_base.set(col, row, Wall())
        for row, col in self._pos_list:
            self._minigrid_base.set(col, row, None)

    def _to_grid(self, values: np.ndarray) -> np.ndarray:
        grid = np.full((self._h, self._w), np.nan)
        for s, (r, c) in enumerate(self._pos_list):
            grid[r, c] = values[s]
        return grid

    def _format_axis(self, ax: Axes, title: str) -> None:
        ax.set_title(title, fontsize=8)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.set_xlim(-0.5, self._w - 0.5)
        ax.set_ylim(self._h - 0.5, -0.5)
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _build_minigrid_frame(
        self,
        r: np.ndarray | None = None,
        agent_pos: int | None = None,
        agent_dir: int = 3,
    ) -> np.ndarray:
        grid = self._minigrid_base.copy()
        for state in self._reward_marker_states(r):
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
        r: np.ndarray | None = None,
        agent_pos: int | None = None,
        agent_dir: int = 3,
        title: str = "Environment",
    ) -> None:
        frame = self._build_minigrid_frame(r=r, agent_pos=agent_pos, agent_dir=agent_dir)
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
        finite_vals = grid.compressed()
        vmax = float(np.max(finite_vals)) if finite_vals.size else 0.0
        if vmax <= 1e-8:
            vmax = 1.0

        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(color="#ffffff")
        im = ax.imshow(
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
            plt.colorbar(im, ax=ax, fraction=0.036, pad=0.04)

    def _draw_reward_overlay(self, ax: Axes, r: np.ndarray | None) -> None:
        for state in self._reward_marker_states(r):
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

    def _reward_marker_states(self, r: np.ndarray | None) -> list[int]:
        if r is None:
            return []
        reward = np.asarray(r, dtype=np.float64)
        positive_states = np.flatnonzero(reward > 0)
        if positive_states.size == 0:
            return []
        marker_count = min(self._reward_marker_count, int(positive_states.size))
        ranked = positive_states[np.argsort(reward[positive_states])[::-1]]
        return [int(state) for state in ranked[:marker_count]]

    def draw_grid(
        self,
        ax_val: Axes,
        ax_pri: Axes,
        V: np.ndarray,
        xi: np.ndarray,
        tau: np.ndarray | None = None,
        r: np.ndarray | None = None,
        agent_pos: int | None = None,
        title_val: str = "Value V(s)",
        title_pri: str = "Priority ξ(s)",
        colorbar: bool = True,
    ) -> None:
        self.draw_metric(ax_val, V, title_val, cmap="magma", colorbar=colorbar)
        self.draw_metric(
            ax_pri,
            tau if tau is not None else xi,
            title_pri if tau is None else r"Topology $\tau(s)$",
            cmap="hot",
            colorbar=colorbar,
        )
        self._draw_reward_overlay(ax_val, r)
        self._draw_reward_overlay(ax_pri, r)


# ---------------------------------------------------------------------------
# IndependentAgentSession — independent env + agent pair
# ---------------------------------------------------------------------------

class IndependentAgentSession:
    """One agent with its own environment instance."""

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

        ek = env_kwargs or {"gamma": GAMMA}
        self.env = FourRoomsTabular(seed=seed, **ek)
        if initial_reward is not None:
            self.r_current = np.asarray(initial_reward, dtype=np.float64).copy()
            self.env.base_r = self.r_current.copy()
            self.env.r = self.r_current.copy()
        else:
            self.r_current = self.env.r.copy()

        kw = tuned_cfg.get(variant, {}).copy() if variant != "standard_ps" else {}
        variant_index = VARIANTS.index(variant)
        self.agent = make_agent(
            variant,
            host=host,
            P=self.env.P,
            r=self.env.r,
            gamma=self.env.gamma,
            n_states=self.env.n_states,
            n_actions=self.env.n_actions,
            rng=np.random.default_rng(np.random.SeedSequence([seed, variant_index, 0xD1A])),
            **kw,
        )

        self.state = int(self._rng.integers(0, self.env.n_states))
        self.agent_dir = 3
        self.step_count = 0
        if initial_target_V is not None:
            self._target_V = np.asarray(initial_target_V, dtype=np.float64).copy()
        else:
            self._target_V = self.env.value_iteration(self.r_current)

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
        candidate_actions = list(range(self.env.n_actions))
        candidate_values = np.array([float(q_values[action]) for action in candidate_actions])
        best_value = float(candidate_values.max())
        best_actions = [
            action
            for action, value in zip(candidate_actions, candidate_values)
            if abs(float(value) - best_value) <= 1e-10
        ]
        return int(self._rng.choice(best_actions))

    def step_once(self) -> tuple[float, np.ndarray]:
        s = self.state
        a = self._pick_action()
        s_next = int(self._rng.choice(self.env.n_states, p=self.env.P[a, s, :]))
        if s_next != s:
            row_s, col_s = self.env._pos_list[s]
            row_next, col_next = self.env._pos_list[s_next]
            delta = (row_next - row_s, col_next - col_s)
            self.agent_dir = self.DELTA_TO_MINIGRID_DIR.get(delta, self.agent_dir)
        r_val = float(self.r_current[s])
        self.agent.real_step(s, a, r_val, s_next)
        if self.host == "dyna":
            self.agent.plan_steps(self.K)
        else:
            self.agent.plan(K=self.K, r=self.r_current)
        self.state = s_next
        self.step_count += 1
        self._visited.add(s)

        err = value_error_l1(self.agent.Q.max(axis=0), self._target_V)
        return r_val, err

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
        if target_V is None:
            self._target_V = self.env.value_iteration(self.r_current)
        else:
            self._target_V = np.asarray(target_V, dtype=np.float64).copy()
        _apply_reward_update(self.agent, self.r_current)
        self._visited.clear()

    def episode(self, T_steps: int) -> tuple[float, float, int]:
        """Run one episode of T_steps, return (episode_return, mean_l1_error, coverage)."""
        ep_reward = 0.0
        ep_errs: list[float] = []
        for _ in range(T_steps):
            r_val, err = self.step_once()
            ep_reward += r_val
            ep_errs.append(err)
        mean_err = float(np.mean(ep_errs))
        coverage = len(self._visited)

        self.return_history.append(ep_reward)
        self.cumul_reward.append(self.cumul_reward[-1] + ep_reward)
        self.err_history.append(mean_err)
        self.cumul_l1.append(self.cumul_l1[-1] + mean_err)
        self.coverage_history.append(coverage)
        self._visited.clear()
        return ep_reward, mean_err, coverage

    def position_values(self) -> np.ndarray:
        return self.agent.Q.max(axis=0).copy()

    def position_reward(self) -> np.ndarray:
        return self.r_current.copy()

    def get_queue_scores(self) -> np.ndarray:
        raw = _standardps_queue_scores(self.agent.module) if self.variant == "standard_ps" else _gtaps_queue_scores(self.agent.module)
        return raw

    def get_combined_queue(self) -> np.ndarray:
        """For GTA-PS: combined queue (ξ + τ). For Standard-PS: just ξ."""
        if self.variant == "standard_ps":
            return _standardps_queue_scores(self.agent.module)
        module = self.agent.module
        combined = np.zeros_like(module.xi)
        for neg_pri, version, state in module._Q_xi:
            if module._version_xi.get(state, -1) == version:
                combined[state] = max(combined[state], float(-neg_pri))
        for neg_pri, version, state in module._H_q:
            if module._version_hq.get(state, -1) == version:
                combined[state] = max(combined[state], float(-neg_pri))
        return combined


# ---------------------------------------------------------------------------
# Worker (top-level for pickle / RunManager)
# ---------------------------------------------------------------------------

def _run_seed_ablation(
    seed: int,
    K: int,
    n_shifts: int,
    T_shift: int,
    host: str,
    tuned_cfg: dict,
    env_kwargs: dict | None = None,
    **_ignored,
) -> dict:
    """One seed: run independent agent sessions and return metrics."""
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

    for ep in range(n_shifts):
        if ep > 0:
            ref_r = ref_env.shift_reward()
            ref_target = ref_env.value_iteration(ref_r)
            for s in sessions.values():
                s.shift_reward(r_current=ref_r, target_V=ref_target)
        for s in sessions.values():
            s.episode(T_shift)

    return {
        "return_history": {v: list(sessions[v].return_history) for v in VARIANTS},
        "cumul_reward": {v: list(sessions[v].cumul_reward) for v in VARIANTS},
        "cumul_l1": {v: list(sessions[v].cumul_l1) for v in VARIANTS},
        "coverage": {v: list(sessions[v].coverage_history) for v in VARIANTS},
        "final_V": {v: sessions[v].position_values().tolist() for v in VARIANTS},
        "final_queue": {
            "standard_ps": sessions["standard_ps"].get_queue_scores().tolist(),
            "gta_beta": sessions["gta_beta"].get_combined_queue().tolist(),
        },
        "r_current": sessions["standard_ps"].position_reward().tolist(),
    }


# ---------------------------------------------------------------------------
# Ablation mode — parallel multi-seed PNG (2x3 grid)
# ---------------------------------------------------------------------------

def run_ablation(args) -> None:
    from scipy import stats as sp_stats

    seed_count = args.seeds
    shift_steps = args.n_steps_per_shift if args.n_steps_per_shift is not None else ABLATION_T_SHIFT
    host = args.host
    cfg = _load_config(args.config)
    tuned = _build_method_cfgs(cfg)
    env_kwargs = _fourrooms_env_kwargs_from_config(cfg)
    env_kwargs.setdefault("room_size", ABLATION_ROOM_SIZE)
    env_kwargs.setdefault("corridor_len", ABLATION_CORRIDOR_LEN)

    print("=" * 62)
    print("  FourRooms ablation: Standard-PS vs GTA-PS^β")
    print("=" * 62)
    print(f"  seeds={seed_count}  n_shifts={args.n_shifts}  K={args.K}")
    print(f"  T_shift={shift_steps}  host={host}  workers={args.workers}")
    print("-" * 62)

    runner = RunManager(n_workers=args.workers)
    raw = runner.run(
        fn=_run_seed_ablation,
        seeds=list(range(seed_count)),
        fn_kwargs=dict(
            K=args.K,
            n_shifts=args.n_shifts,
            T_shift=shift_steps,
            host=host,
            tuned_cfg=tuned,
            env_kwargs=env_kwargs,
        ),
        desc="fourrooms-ablation",
    )

    def _agg(mat):
        arr = np.array(mat)
        mean = arr.mean(axis=0)
        ci = np.zeros_like(mean) if len(mat) < 2 else sp_stats.sem(arr, axis=0) * sp_stats.t.ppf(0.975, df=len(mat) - 1)
        return mean, ci

    print(f"\n{'Variant':<20}  {'Mean Return':>14}  {'Mean Coverage':>16}")
    print("-" * 62)
    all_returns = {v: [r["return_history"][v] for r in raw] for v in VARIANTS}
    all_coverage = {v: [r["coverage"][v] for r in raw] for v in VARIANTS}
    for v in VARIANTS:
        m_ret, _ = _agg(all_returns[v])
        m_cov, _ = _agg(all_coverage[v])
        print(f"  {LABELS[v]:<20}  {m_ret[-1]:>14.4f}  {m_cov[-1]:>16.1f}")

    apply_style()
    colors = {v: get_palette("line")[v] for v in VARIANTS}
    fig = plt.figure(figsize=(18, 10), layout="constrained")
    gs = gridspec.GridSpec(2, 3, figure=fig)

    episodes = np.arange(1, args.n_shifts + 1)
    cumul_ep = np.arange(0, args.n_shifts + 1)

    def _agg(mat):
        arr = np.array(mat)
        mean = arr.mean(axis=0)
        ci = np.zeros_like(mean) if len(mat) < 2 else sp_stats.sem(arr, axis=0) * sp_stats.t.ppf(0.975, df=len(mat) - 1)
        return mean, ci

    for row, v in enumerate(VARIANTS):
        ax = fig.add_subplot(gs[row, 0])
        m, c = _agg([r["cumul_reward"][v] for r in raw])
        ax.plot(cumul_ep, m, color=colors[v], lw=2, label=LABELS[v])
        ax.fill_between(cumul_ep, m - c, m + c, color=colors[v], alpha=0.18)
        ax.set_ylabel("Cumulative reward")
        ax.set_title(f"{LABELS[v]}\nCumulative reward")
        ax.grid(True, ls="--", alpha=0.3)
        if row == 1:
            ax.set_xlabel("Episode $N$")

    for row, v in enumerate(VARIANTS):
        ax = fig.add_subplot(gs[row, 1])
        m, c = _agg([r["cumul_l1"][v] for r in raw])
        ax.plot(cumul_ep, m, color=colors[v], lw=2, label=LABELS[v])
        ax.fill_between(cumul_ep, m - c, m + c, color=colors[v], alpha=0.18)
        ax.axhline(0, color="gray", ls=":", lw=1)
        ax.set_ylabel("Cumulative L1 error")
        ax.set_title(f"{LABELS[v]}\nCumulative L1 error")
        ax.grid(True, ls="--", alpha=0.3)
        if row == 1:
            ax.set_xlabel("Episode $N$")

    for row, v in enumerate(VARIANTS):
        ax = fig.add_subplot(gs[row, 2])
        m, c = _agg([r["coverage"][v] for r in raw])
        ax.plot(episodes, m, color=colors[v], lw=2, label=LABELS[v])
        ax.fill_between(episodes, m - c, m + c, color=colors[v], alpha=0.18)
        ax.set_ylabel("Unique states")
        ax.set_title(f"{LABELS[v]}\nCoverage")
        ax.grid(True, ls="--", alpha=0.3)
        if row == 1:
            ax.set_xlabel("Episode")

    fig.suptitle(
        rf"FourRooms: Standard-PS vs GTA-PS$^{{\beta}}$  —  "
        rf"$K={args.K}$, $N_{{\mathrm{{shifts}}}}={args.n_shifts}$, "
        rf"$N_{{\mathrm{{seeds}}}}={seed_count}$, host={host}",
        fontsize=11,
    )

    out_dir = Path(ROOT / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fourrooms_ablation.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nAblation plot saved to {out_path}")
    plt.close()

    out_dir_json = Path(ROOT / "results/misc/showcase")
    out_dir_json.mkdir(parents=True, exist_ok=True)
    json_path = out_dir_json / "ablation.json"

    results_payload = {
        "config": {
            "mode": "ablation",
            "host": host,
            "K": args.K,
            "n_shifts": args.n_shifts,
            "seeds": seed_count,
            "T_shift": shift_steps,
            "env_kwargs": env_kwargs,
        },
        "returns": {
            v: {"mean": _agg(all_returns[v])[0].tolist(), "ci95": _agg(all_returns[v])[1].tolist()}
            for v in VARIANTS
        },
        "coverage": {
            v: {"mean": _agg(all_coverage[v])[0].tolist(), "ci95": _agg(all_coverage[v])[1].tolist()}
            for v in VARIANTS
        },
    }
    with open(json_path, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"Ablation results saved to {json_path}")


# ---------------------------------------------------------------------------
# Showcase mode — animated GIF (2x3 grid)
# ---------------------------------------------------------------------------

def run_showcase(args) -> None:
    showcase_seed = args.seeds if args.seeds is not None else 0
    shift_steps = args.n_steps_per_shift if args.n_steps_per_shift is not None else SHOWCASE_T_SHIFT
    host = args.host
    cfg = _load_config(args.config)
    tuned = _build_method_cfgs(cfg)
    env_kwargs = _fourrooms_env_kwargs_from_config(cfg)
    env_kwargs.setdefault("room_size", SHOWCASE_ROOM_SIZE)
    env_kwargs.setdefault("corridor_len", SHOWCASE_CORRIDOR_LEN)
    env_kwargs["k"] = SHOWCASE_ACTIVE_ROOMS
    env_kwargs["n_peaks_per_room"] = SHOWCASE_PEAKS_PER_ROOM
    env_kwargs["sigma_range"] = SHOWCASE_SIGMA_RANGE

    ref_env = FourRoomsTabular(seed=showcase_seed, **env_kwargs)
    initial_reward = ref_env.r.copy()
    initial_target_V = ref_env.value_iteration(ref_env.r)
    sessions = {
        v: IndependentAgentSession(
            variant=v,
            host=host,
            tuned_cfg=tuned,
            K=args.K,
            seed=showcase_seed,
            env_kwargs=env_kwargs,
            initial_reward=initial_reward,
            initial_target_V=initial_target_V,
        )
        for v in VARIANTS
    }

    total_steps = args.n_shifts * shift_steps
    n_to_show = args.n_frames if args.n_frames is not None else total_steps
    n_to_show = min(n_to_show, total_steps)

    print("=" * 50)
    print("  FourRooms showcase animation")
    print("=" * 50)
    print(f"  seed={showcase_seed}  total_steps={total_steps}  n_frames={n_to_show}")
    print(f"  K={args.K}  T_shift={shift_steps}  host={host}")
    print("-" * 50)

    captured_data = []
    ep_reward = {v: 0.0 for v in VARIANTS}

    print(f"Simulating {total_steps} steps...")
    with tqdm(total=total_steps, desc="showcase-sim", unit="step", dynamic_ncols=True, ascii=True) as sim_pbar:
        for step in range(total_steps):
            shift_idx = step // shift_steps
            if step > 0 and step % shift_steps == 0:
                ref_r = ref_env.shift_reward()
                ref_target = ref_env.value_iteration(ref_r)
                for s in sessions.values():
                    s.shift_reward(r_current=ref_r, target_V=ref_target)
                ep_reward = {v: 0.0 for v in VARIANTS}

            for v in VARIANTS:
                r_val, _ = sessions[v].step_once()
                ep_reward[v] += r_val

            if step >= total_steps - n_to_show:
                sps_session = sessions["standard_ps"]
                gta_session = sessions["gta_beta"]
                captured_data.append(
                    {
                        "step": step + 1,
                        "shift": shift_idx + 1,
                        "ep_reward": dict(ep_reward),
                        "sps": {
                            "env": sps_session.env,
                            "state": sps_session.state,
                            "agent_dir": sps_session.agent_dir,
                            "r_current": sps_session.position_reward(),
                            "V": sps_session.position_values(),
                            "queue": sps_session.get_queue_scores(),
                        },
                        "gta": {
                            "env": gta_session.env,
                            "state": gta_session.state,
                            "agent_dir": gta_session.agent_dir,
                            "r_current": gta_session.position_reward(),
                            "V": gta_session.position_values(),
                            "queue": gta_session.get_combined_queue(),
                        },
                    }
                )
            sim_pbar.update(1)

    apply_style()
    fig = plt.figure(figsize=(18, 8), layout="constrained")
    gs = gridspec.GridSpec(2, 3, figure=fig)

    ax_env_sps = fig.add_subplot(gs[0, 0])
    ax_env_gta = fig.add_subplot(gs[1, 0])
    ax_v_sps = fig.add_subplot(gs[0, 1])
    ax_v_gta = fig.add_subplot(gs[1, 1])
    ax_q_sps = fig.add_subplot(gs[0, 2])
    ax_q_gta = fig.add_subplot(gs[1, 2])

    renderer_sps = FourRoomsRenderer(sessions["standard_ps"].env)
    renderer_gta = FourRoomsRenderer(sessions["gta_beta"].env)

    def update(frame: int) -> list[Artist]:
        data = captured_data[frame]
        for ax in [ax_env_sps, ax_env_gta, ax_v_sps, ax_v_gta, ax_q_sps, ax_q_gta]:
            ax.cla()

        sps_data = data["sps"]
        gta_data = data["gta"]

        renderer_sps.draw_interface(
            ax_env_sps,
            r=sps_data["r_current"],
            agent_pos=sps_data["state"],
            agent_dir=sps_data["agent_dir"],
            title=f"{LABELS['standard_ps']} Environment",
        )
        renderer_gta.draw_interface(
            ax_env_gta,
            r=gta_data["r_current"],
            agent_pos=gta_data["state"],
            agent_dir=gta_data["agent_dir"],
            title=f"{LABELS['gta_beta']} Environment",
        )

        renderer_sps.draw_metric(
            ax_v_sps,
            values=sps_data["V"],
            title=f"{LABELS['standard_ps']} V(s)",
            cmap="magma",
        )
        renderer_gta.draw_metric(
            ax_v_gta,
            values=gta_data["V"],
            title=f"{LABELS['gta_beta']} V(s)",
            cmap="magma",
        )

        renderer_sps.draw_metric(
            ax_q_sps,
            values=sps_data["queue"],
            title=f"{LABELS['standard_ps']} Queue ξ(s)",
            cmap="hot",
        )
        renderer_gta.draw_metric(
            ax_q_gta,
            values=gta_data["queue"],
            title=f"{LABELS['gta_beta']} Combined Queue (ξ+τ)(s)",
            cmap="hot",
        )

        fig.suptitle(
            rf"FourRooms  —  Step {data['step']}  |  Episode {data['shift']}  |  "
            rf"K={args.K}  |  host={host}",
            fontsize=11,
        )
        return []

    print(f"Animating {len(captured_data)} frames...")
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(captured_data),
        interval=120,
        blit=False,
        cache_frame_data=False,
    )
    out_dir = Path(ROOT / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fourrooms_showcase.gif"
    with tqdm(total=len(captured_data), desc="showcase-gif", unit="frame", dynamic_ncols=True, ascii=True) as gif_pbar:
        last_frame = -1

        def _on_save_progress(frame_idx: int, _total: int) -> None:
            nonlocal last_frame
            if frame_idx > last_frame:
                gif_pbar.update(frame_idx - last_frame)
                last_frame = frame_idx

        anim.save(str(out_path), writer="pillow", fps=8, progress_callback=_on_save_progress)
    print(f"Showcase GIF saved to {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="FourRooms: Standard-PS vs GTA-PS^β (2×3 showcase/ablation)")
    parser.add_argument("--config", default=None, help="YAML config path (FourRooms configs supported)")
    parser.add_argument("--mode", choices=["ablation", "showcase"], default="ablation", help="'ablation': parallel multi-seed PNG; 'showcase': animated GIF")
    parser.add_argument("--K", type=int, default=K)
    parser.add_argument("--n-steps-per-shift", type=int, default=None, help="Real steps between reward shifts")
    parser.add_argument("--shift-steps", dest="n_steps_per_shift", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--seeds", type=int, default=40, help="Seed count in ablation mode; fixed seed value in showcase mode")
    parser.add_argument("--n-shifts", type=int, default=N_SHIFTS, help="Reward-shift episodes (ablation mode)")
    parser.add_argument("--n-frames", type=int, default=300, help="Animation frames (showcase mode); defaults to entire navigation")
    parser.add_argument("--host", choices=["dyna", "dp"], default="dyna", help="Host algorithm for both agents")
    parser.add_argument("--workers", default=5, help="Parallel workers for RunManager (ablation mode)")
    args = parser.parse_args()

    if args.mode == "ablation":
        run_ablation(args)
    else:
        run_showcase(args)


if __name__ == "__main__":
    main()
