"""
misc/showcase.py

Visual demonstration: Dyna + Standard-PS vs Dyna + GTA-PS^β on FourRooms.

Modes
-----
--mode ablation  (default)
    Parallel multi-seed run (RunManager). Saves a 6-panel PNG:
    Panel 1: Mean return per reward-shift episode with 95% CI
    Panel 2: Cumulative reward
    Panels 3-6: Final value + live queue-priority heatmaps for each agent

--mode showcase
    Single-seed animation saved as GIF (pillow writer, Agg backend).
    Each frame = one real step; renders one canonical environment panel
    plus diagnostic value/priority/tau maps. By default the left panel follows
    GTA-PS^beta under the Dyna / TD(0) host.

Usage
-----
python misc/showcase.py --mode ablation --seeds 5 --n-shifts 8 --K 40
python misc/showcase.py --mode showcase --seeds 0 --n-frames 500 --K 40 --n-steps-per-shift 100
"""
from __future__ import annotations

import sys
import os
import argparse
import numpy as np
from pathlib import Path

# Agg must be set before importing pyplot (TkAgg unavailable on WSL)
import matplotlib
matplotlib.use("Agg")
from matplotlib.artist import Artist
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib import animation

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "configs").exists():
    # Try reaching up to find the real project root (e.g. if compiled in artifacts/dist/)
    for p in Path(__file__).resolve().parents:
        if (p / "configs").exists() and (p / "src").exists():
            ROOT = p
            break
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "threads"))
sys.path.insert(0, str(ROOT / "misc"))

from gta_ps.envs.fourrooms import FourRoomsTabular
from gta_ps.algorithms import make_agent
from gta_ps.utils.metrics import value_error_l1
from gta_ps.utils.hyperparams import load_tuned_for_env
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Wall
from runner import RunManager
from plot_academic import apply_style, get_palette, LABELS as AGENT_LABELS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GAMMA    = 0.95
K        = 40      # planning budget per host step
T_SHIFT  = 50      # ablation reward-shift interval (real steps)
SHOWCASE_T_SHIFT = 100
N_SHIFTS = 10
EPS      = 0.15    # epsilon-greedy exploration rate

VARIANTS = ["standard_ps", "gta_beta"]
LABELS   = {v: AGENT_LABELS[v] for v in VARIANTS}
SHOWCASE_BEHAVIOR_POLICIES = ["standard_ps", "gta_beta", "oracle_value"]

def get_showcase_colors():
    p_line = get_palette("line")
    return {v: p_line[v] for v in VARIANTS}


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
    """
    Maps FourRoomsTabular state indices onto a 2-D grid and renders
    value + priority heatmaps onto matplotlib Axes.
    """

    MINIGRID_TILE_SIZE = 32
    GOAL_COLOR = "#98ff66"
    GOAL_EDGE_COLOR = "#f4ffe8"

    def __init__(self, env: FourRoomsTabular) -> None:
        self.env = env
        self._row_min = 0
        self._col_min = 0
        self._h = 2 * env.room_size + env.corridor_len + 2
        self._w = 2 * env.room_size + env.corridor_len + 2
        self._reward_marker_count = max(1, int(getattr(env, "k", 1)) * int(getattr(env, "n_peaks_per_room", 1)))
        self._passable_mask = np.zeros((self._h, self._w), dtype=bool)
        for row, col in env._pos_list:
            self._passable_mask[row, col] = True
        self._minigrid_base = Grid(self._w, self._h)
        for row in range(self._h):
            for col in range(self._w):
                self._minigrid_base.set(col, row, Wall())
        for row, col in env._pos_list:
            self._minigrid_base.set(col, row, None)

    def _to_grid(self, values: np.ndarray) -> np.ndarray:
        grid = np.full((self._h, self._w), np.nan)
        for s, (r, c) in enumerate(self.env._pos_list):
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
            row, col = self.env._pos_list[state]
            grid.set(col, row, Goal())

        if agent_pos is None:
            agent_xy = (-1, -1)
        else:
            row, col = self.env._pos_list[agent_pos]
            agent_xy = (col, row)

        return grid.render(
            self.MINIGRID_TILE_SIZE,
            agent_pos=agent_xy,
            agent_dir=agent_dir,
        )

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
            row, col = self.env._pos_list[state]
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
        positive_states = np.flatnonzero(r > 0)
        if positive_states.size == 0:
            return []
        marker_count = min(self._reward_marker_count, int(positive_states.size))
        ranked = positive_states[np.argsort(r[positive_states])[::-1]]
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
# DualAgentSession — shared trajectory, independent models
# ---------------------------------------------------------------------------

class DualAgentSession:
    """Two Dyna agents sharing one FourRooms environment and trajectory."""

    DELTA_TO_MINIGRID_DIR = {
        (-1, 0): 3,
        (1, 0): 1,
        (0, -1): 2,
        (0, 1): 0,
    }

    def __init__(
        self,
        env: FourRoomsTabular,
        tuned_cfg: dict,
        K: int = K,
        eps: float = EPS,
        seed: int = 0,
        behavior_policy: str = "standard_ps",
    ) -> None:
        self.env = env
        self.K   = K
        self.eps = eps
        if behavior_policy not in SHOWCASE_BEHAVIOR_POLICIES:
            raise ValueError(
                f"behavior_policy must be one of {SHOWCASE_BEHAVIOR_POLICIES}, got {behavior_policy!r}"
            )
        self.behavior_policy = behavior_policy
        self._rng = np.random.default_rng(seed)

        self.agents: dict = {}
        for v in VARIANTS:
            kw = tuned_cfg.get(v, {}).copy() if v != "standard_ps" else {}
            self.agents[v] = make_agent(
                v, host="dyna",
                P=env.P, r=env.r, gamma=env.gamma,
                n_states=env.n_states, n_actions=env.n_actions,
                **kw,
            )

        self.state      = int(self._rng.integers(0, env.n_states))
        self.agent_dir  = 3
        self.step_count = 0
        self.r_current  = env.r.copy()
        self._behavior_V = env.value_iteration(self.r_current)

        self.err_history: dict[str, list[float]] = {v: [] for v in VARIANTS}
        self.cumul_te:    dict[str, list[float]] = {v: [0.0] for v in VARIANTS}

    def _next_state(self, state: int, action: int) -> int:
        return int(np.argmax(self.env.P[action, state, :]))

    def _moving_actions(self, state: int) -> list[int]:
        return [action for action in range(self.env.n_actions) if self._next_state(state, action) != state]

    def _pick_action(self) -> int:
        moving_actions = self._moving_actions(self.state)
        if self._rng.random() < self.eps:
            if moving_actions:
                return int(self._rng.choice(moving_actions))
            return int(self._rng.integers(0, self.env.n_actions))

        candidate_actions = moving_actions if moving_actions else list(range(self.env.n_actions))
        if self.behavior_policy == "oracle_value":
            candidate_values = np.array([
                float((self.env.P[action, self.state, :] * self._behavior_V).sum())
                for action in candidate_actions
            ])
        else:
            q_values = self.agents[self.behavior_policy].Q[:, self.state]
            candidate_values = np.array([float(q_values[action]) for action in candidate_actions])
        best_value = float(candidate_values.max())
        best_actions = [
            action
            for action, value in zip(candidate_actions, candidate_values)
            if abs(float(value) - best_value) <= 1e-10
        ]
        return int(self._rng.choice(best_actions))

    def step_once(self) -> tuple[dict[str, float], float]:
        s = self.state
        a = self._pick_action()
        s_next = int(self._rng.choice(self.env.n_states, p=self.env.P[a, s, :]))
        if s_next != s:
            row_s, col_s = self.env._pos_list[s]
            row_next, col_next = self.env._pos_list[s_next]
            delta = (row_next - row_s, col_next - col_s)
            self.agent_dir = self.DELTA_TO_MINIGRID_DIR.get(delta, self.agent_dir)
        r_val  = float(self.r_current[s])
        for v in VARIANTS:
            self.agents[v].real_step(s, a, r_val, s_next)
            self.agents[v].plan_steps(self.K)
        self.state      = s_next
        self.step_count += 1
        V_star = self.env.value_iteration(self.r_current)
        errs = {v: value_error_l1(self.agents[v].Q.max(axis=0), V_star) for v in VARIANTS}
        return errs, r_val

    def shift_reward(self) -> np.ndarray:
        self.r_current = self.env.shift_reward()
        self._behavior_V = self.env.value_iteration(self.r_current)
        for agent in self.agents.values():
            _apply_reward_update(agent, self.r_current)
        return self.r_current.copy()

    def episode(self, T_steps: int) -> dict[str, float]:
        ep_errs: dict[str, list[float]] = {v: [] for v in VARIANTS}
        for _ in range(T_steps):
            for v, e in self.step_once().items():
                ep_errs[v].append(e)
        means = {v: float(np.mean(ep_errs[v])) for v in VARIANTS}
        for v in VARIANTS:
            self.err_history[v].append(means[v])
            self.cumul_te[v].append(self.cumul_te[v][-1] + means[v])
        return means


class SingleAgentSession:
    """One Dyna agent controlling its own trajectory for realized-return ablations."""

    def __init__(
        self,
        env: FourRoomsTabular,
        variant: str,
        tuned_cfg: dict,
        K: int = K,
        eps: float = EPS,
        seed: int = 0,
    ) -> None:
        self.env = env
        self.variant = variant
        self.K = K
        self.eps = eps
        self._rng = np.random.default_rng(seed)

        kw = tuned_cfg.get(variant, {}).copy() if variant != "standard_ps" else {}
        self.agent = make_agent(
            variant,
            host="dyna",
            P=env.P,
            r=env.r,
            gamma=env.gamma,
            n_states=env.n_states,
            n_actions=env.n_actions,
            **kw,
        )

        self.state = int(self._rng.integers(0, env.n_states))
        self.r_current = env.r.copy()
        self.return_history: list[float] = []
        self.cumul_reward: list[float] = [0.0]

    def _pick_action(self) -> int:
        if self._rng.random() < self.eps:
            return int(self._rng.integers(0, self.env.n_actions))
        return int(self.agent.Q[:, self.state].argmax())

    def set_reward(self, r_new: np.ndarray) -> None:
        self.r_current = r_new.copy()
        self.env.r = r_new.copy()
        _apply_reward_update(self.agent, self.r_current)

    def step_once(self) -> float:
        state = self.state
        action = self._pick_action()
        next_state = int(self._rng.choice(self.env.n_states, p=self.env.P[action, state, :]))
        reward = float(self.r_current[state])
        self.agent.real_step(state, action, reward, next_state)
        self.agent.plan_steps(self.K)
        self.state = next_state
        return reward

    def episode(self, T_steps: int) -> float:
        episode_return = 0.0
        for _ in range(T_steps):
            episode_return += self.step_once()
        self.return_history.append(episode_return)
        self.cumul_reward.append(self.cumul_reward[-1] + episode_return)
        return episode_return


# ---------------------------------------------------------------------------
# Worker (top-level for pickle / RunManager)
# ---------------------------------------------------------------------------

def _run_seed_ablation(
    seed: int,
    K: int,
    n_shifts: int,
    T_shift: int,
    tuned_cfg: dict,
    env_kwargs: dict | None = None,
    **_ignored,
) -> dict:
    """One seed: run variant-controlled rollouts and return realized reward metrics."""
    base_env = FourRoomsTabular(seed=seed, **(env_kwargs or {"gamma": GAMMA}))
    sessions = {
        variant: SingleAgentSession(
            FourRoomsTabular(seed=seed, **(env_kwargs or {"gamma": GAMMA})),
            variant,
            tuned_cfg,
            K=K,
            seed=seed,
        )
        for variant in VARIANTS
    }

    current_reward = base_env.r.copy()
    for session in sessions.values():
        session.set_reward(current_reward)

    for ep in range(n_shifts):
        if ep > 0:
            current_reward = base_env.shift_reward()
            for session in sessions.values():
                session.set_reward(current_reward)
        for session in sessions.values():
            session.episode(T_shift)

    return {
        "return_history": {v: list(sessions[v].return_history) for v in VARIANTS},
        "cumul_reward": {v: list(sessions[v].cumul_reward) for v in VARIANTS},
        "final_V": {v: sessions[v].agent.Q.max(axis=0).tolist() for v in VARIANTS},
        "final_queue": {
            "standard_ps": _standardps_queue_scores(sessions["standard_ps"].agent.module).tolist(),
            "gta_beta": _gtaps_queue_scores(sessions["gta_beta"].agent.module).tolist(),
        },
        "r_current": current_reward.tolist(),
    }


# ---------------------------------------------------------------------------
# Ablation mode — parallel multi-seed PNG
# ---------------------------------------------------------------------------

def run_ablation(args) -> None:
    from scipy import stats as sp_stats

    seed_count = args.seeds
    shift_steps = args.n_steps_per_shift if args.n_steps_per_shift is not None else T_SHIFT
    cfg = _load_config(args.config)
    tuned = _build_method_cfgs(cfg)
    env_kwargs = _fourrooms_env_kwargs_from_config(cfg)

    print("=" * 62)
    print("  FourRooms ablation: Standard-PS vs GTA-PS^β")
    print("=" * 62)
    print(f"  seeds={seed_count}  n_shifts={args.n_shifts}  K={args.K}"
          f"  T_shift={shift_steps}  workers={args.workers}")
    print("-" * 62)

    runner = RunManager(n_workers=args.workers)
    raw = runner.run(
        fn=_run_seed_ablation,
        seeds=list(range(seed_count)),
        fn_kwargs=dict(K=args.K, n_shifts=args.n_shifts, T_shift=shift_steps,
                       tuned_cfg=tuned, env_kwargs=env_kwargs),
        desc="fourrooms-ablation",
    )

    def _agg(mat):
        arr  = np.array(mat)
        mean = arr.mean(axis=0)
        ci   = (np.zeros_like(mean) if len(mat) < 2
                else sp_stats.sem(arr, axis=0) * sp_stats.t.ppf(0.975, df=len(mat) - 1))
        return mean, ci

    print(f"\n{'Variant':<20}  {'Mean Return (last ep)':>22}  {'Cumul Reward':>14}")
    print("-" * 62)
    all_returns = {v: [r["return_history"][v] for r in raw] for v in VARIANTS}
    all_cumul_rewards = {v: [r["cumul_reward"][v] for r in raw] for v in VARIANTS}
    for v in VARIANTS:
        m_ret, _ = _agg(all_returns[v])
        m_cum, _ = _agg(all_cumul_rewards[v])
        print(f"  {LABELS[v]:<20}  {m_ret[-1]:>22.4f}  {m_cum[-1]:>14.4f}")

    # Plots
    apply_style()
    colors = get_showcase_colors()
    last     = raw[-1]
    env_disp = FourRoomsTabular(seed=seed_count - 1, **env_kwargs)
    renderer = FourRoomsRenderer(env_disp)

    fig = plt.figure(figsize=(16, 8), layout="constrained")
    gs  = gridspec.GridSpec(2, 4, figure=fig)

    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[1, :2])
    episodes = np.arange(1, args.n_shifts + 1)
    cumul_ep = np.arange(0, args.n_shifts + 1)

    for v in VARIANTS:
        m, c = _agg(all_returns[v])
        ax1.plot(episodes, m, color=colors[v], lw=2, label=LABELS[v])
        ax1.fill_between(episodes, m - c, m + c, color=colors[v], alpha=0.18)
        m2, c2 = _agg(all_cumul_rewards[v])
        ax2.plot(cumul_ep, m2, color=colors[v], lw=2, label=LABELS[v])
        ax2.fill_between(cumul_ep, m2 - c2, m2 + c2, color=colors[v], alpha=0.18)

    ax1.set_xlabel("Reward-shift episode")
    ax1.set_ylabel("Episode return")
    ax1.set_title("Per-episode realized return")
    ax1.legend(fontsize=9)
    ax1.grid(True, ls="--", alpha=0.3)

    ax2.set_xlabel("Episode $N$")
    ax2.set_ylabel("Cumulative reward")
    ax2.set_title("Cumulative realized reward")
    ax2.legend(fontsize=9)
    ax2.grid(True, ls="--", alpha=0.3)

    r_arr = np.array(last["r_current"])
    for col, v in enumerate(VARIANTS):
        ax_v = fig.add_subplot(gs[0, 2 + col])
        ax_p = fig.add_subplot(gs[1, 2 + col])
        V_hat = np.array(last["final_V"][v])
        queue_scores = np.array(last["final_queue"][v])
        renderer.draw_grid(
            ax_v, ax_p,
            V=V_hat, xi=queue_scores, tau=None, r=r_arr,
            title_val=f"{LABELS[v]}\nValue V(s)",
            title_pri=f"{LABELS[v]}\nqueue priority",
            colorbar=True,
        )

    fig.suptitle(
        rf"FourRooms: Dyna+Standard-PS vs Dyna+GTA-PS$^{{\beta}}$  —  "
        rf"$K={args.K}$, $N_\mathrm{{shifts}}={args.n_shifts}$, "
        rf"$N_\mathrm{{seeds}}={seed_count}$",
        fontsize=11,
    )

    out_dir = Path(ROOT / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fourrooms_ablation.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nAblation plot saved to {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Showcase mode — animated GIF (single seed)
# ---------------------------------------------------------------------------

def run_showcase(args) -> None:
    showcase_seed = args.seeds if args.seeds is not None else 0
    shift_steps = args.n_steps_per_shift if args.n_steps_per_shift is not None else SHOWCASE_T_SHIFT
    cfg = _load_config(args.config)
    tuned = _build_method_cfgs(cfg)
    env_kwargs = _fourrooms_env_kwargs_from_config(cfg)
    env   = FourRoomsTabular(seed=showcase_seed, **env_kwargs)
    sess  = DualAgentSession(
        env,
        tuned,
        K=args.K,
        seed=showcase_seed,
        behavior_policy=args.priority_algo,
    )
    renderer = FourRoomsRenderer(env)

    total_steps = args.n_shifts * shift_steps
    n_to_show = args.n_frames if args.n_frames is not None else total_steps
    n_to_show = min(n_to_show, total_steps)
    start_step = total_steps - n_to_show

    controller_title = (
        "Oracle value policy"
        if args.priority_algo == "oracle_value"
        else f"{LABELS[args.priority_algo]}  (Dyna / TD(0))"
    )

    print("=" * 50)
    print("  FourRooms showcase animation")
    print("=" * 50)
    print(f"  seed={showcase_seed}  total_steps={total_steps}  n_frames={n_to_show}")
    print(f"  K={args.K}  T_shift={shift_steps}  priority_algo={args.priority_algo}")
    print("-" * 50)

    # Pre-simulate and capture frames
    captured_data = []
    ep_reward = 0.0
    
    print(f"Simulating {total_steps} steps...")
    for step in range(total_steps):
        shift_idx = step // shift_steps
        if step > 0 and step % shift_steps == 0:
            sess.shift_reward()
            ep_reward = 0.0
        
        errs, r = sess.step_once()
        ep_reward += r
        
        if step >= start_step:
            sps = sess.agents["standard_ps"]
            gta = sess.agents["gta_beta"]
            captured_data.append({
                "step": step + 1,
                "shift": shift_idx + 1,
                "ep_reward": ep_reward,
                "agent_pos": sess.state,
                "agent_dir": sess.agent_dir,
                "r_current": sess.r_current.copy(),
                "errs": errs,
                "sps_V": sps.Q.max(axis=0).copy(),
                "gta_V": gta.Q.max(axis=0).copy(),
                "sps_queue": _standardps_queue_scores(sps.module).copy(),
                "gta_queue": _gtaps_queue_scores(gta.module).copy(),
            })

    apply_style()
    fig = plt.figure(figsize=(12.5, 8), layout="constrained")
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1.1, 1.0, 1.0])
    ax_env = fig.add_subplot(gs[:, 0])
    ax_v_sps = fig.add_subplot(gs[0, 1])
    ax_xi_sps = fig.add_subplot(gs[0, 2])
    ax_v_gta = fig.add_subplot(gs[1, 1])
    ax_tau_gta = fig.add_subplot(gs[1, 2])
    showcase_axes = (ax_env, ax_v_sps, ax_xi_sps, ax_v_gta, ax_tau_gta)

    def update(frame: int) -> list[Artist]:
        data = captured_data[frame]
        for ax in showcase_axes:
            ax.cla()
        
        renderer.draw_interface(
            ax_env,
            r=data["r_current"],
            agent_pos=data["agent_pos"],
            agent_dir=data["agent_dir"],
            title=controller_title,
        )
        renderer.draw_metric(
            ax_v_sps,
            values=data["sps_V"],
            title=f"Standard-PS  V(s)  [L1={data['errs']['standard_ps']:.3f}]",
            cmap="magma",
        )
        renderer.draw_metric(
            ax_xi_sps,
            values=data["sps_queue"],
            title=r"Standard-PS  queue priority",
            cmap="hot",
        )
        renderer.draw_metric(
            ax_v_gta,
            values=data["gta_V"],
            title=rf"GTA-PS$^{{\beta}}$  V(s)  [L1={data['errs']['gta_beta']:.3f}]",
            cmap="magma",
        )
        renderer.draw_metric(
            ax_tau_gta,
            values=data["gta_queue"],
            title=rf"GTA-PS$^{{\beta}}$  queue priority",
            cmap="hot",
        )
        fig.suptitle(
            rf"FourRooms  —  Step {data['step']}  |  Episode {data['shift']}  |  "
            rf"Ep Reward: {data['ep_reward']:.2f}  |  K={args.K}",
            fontsize=11,
        )
        return []

    print(f"Animating {len(captured_data)} frames...")
    anim = animation.FuncAnimation(
        fig, update, frames=len(captured_data), interval=120, blit=False
    )
    out_dir = Path(ROOT / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fourrooms_showcase.gif"
    anim.save(str(out_path), writer="pillow", fps=8)
    print(f"Showcase GIF saved to {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FourRooms visual demo: Standard-PS vs GTA-PS^β"
    )
    parser.add_argument("--config", default=None,
                        help="YAML config path (FourRooms configs supported)")
    parser.add_argument("--mode",     choices=["ablation", "showcase"], default="ablation",
                        help="'ablation': parallel multi-seed PNG; 'showcase': animated GIF")
    parser.add_argument("--K",        type=int,   default=K)
    parser.add_argument("--n-steps-per-shift", type=int, default=None,
                        help="Real steps between reward shifts; showcase defaults slower than ablation")
    parser.add_argument("--shift-steps", dest="n_steps_per_shift", type=int,
                        help=argparse.SUPPRESS)
    parser.add_argument("--seeds",    type=int,   default=30,
                        help="Seed count in ablation mode; fixed seed value in showcase mode")
    parser.add_argument("--n-shifts", type=int,   default=N_SHIFTS,
                        help="Reward-shift episodes (ablation mode)")
    parser.add_argument("--n-frames", type=int,   default=None,
                        help="Animation frames (showcase mode); defaults to entire navigation")
    parser.add_argument("--priority-algo", choices=SHOWCASE_BEHAVIOR_POLICIES,
                        default="gta_beta",
                        help="Controller used for the left navigation panel in showcase mode")
    parser.add_argument("--controller-variant", dest="priority_algo",
                        choices=SHOWCASE_BEHAVIOR_POLICIES,
                        help=argparse.SUPPRESS)
    parser.add_argument("--workers",  default=5,
                        help="Parallel workers for RunManager (ablation mode)"
    )
    args = parser.parse_args()

    if args.mode == "ablation":
        run_ablation(args)
    else:
        run_showcase(args)


if __name__ == "__main__":
    main()
