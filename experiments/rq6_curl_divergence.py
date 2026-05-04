"""
RQ6: Directed Curl / Divergence Analysis.

Mission:
Run two parallel mechanism-isolation experiments:
- GARNET: manually constructed path-like and cycle-like families
- FourRooms: symmetric, windy, belt-like, and mixed transport modes

This keeps the custom directed GARNET study while preserving the dedicated
FourRooms transport experiment on the right panel.
Measures convergence consistency via end-of-shift L1 error.
"""
import sys
import os
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

# Setup path
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
from gta_ps.envs.garnet import GarnetEnv
from gta_ps.envs.fourrooms import FourRoomsTabular
from gta_ps.utils.metrics import value_error_l1
from gta_ps.utils.hyperparams import load_tuned_for_env
from runner import RunManager
from plot_academic import (
    LABELS as AGENT_LABELS,
    apply_style,
    get_palette,
    get_utility_color,
)

FLOW_FAMILIES = (
    ("Path-like flow", "path"),
    ("Cycle-like flow", "cycle"),
)

FLOW_ENV_KWARGS = {
    "n_states": 35,
    "n_actions": 3,
    "branching": 2,
    "reward_support": 3,
    "reward_locality": True,
}

KERNEL_DEFAULTS = {
    "forward_hi": 0.95,
    "forward_lo": 0.85,
}

FOURROOMS_BASE_KWARGS = {
    "room_size": 3,
    "corridor_len": 1,
    "corridor_width": 1,
    "window_offset": 0,
    "k": 2,
}

FOURROOMS_MODES = (
    ("Symmetric", {}),
    ("Windy (Div)", {"drift": (0, 1)}),
    ("Belt-like (Curl)", {"one_way_prob": 0.05}),
    ("Mixed", {"drift": (0, 1), "one_way_prob": 0.05}),
)


def _load_config(path: str) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f) or {}


def _build_method_cfgs(cfg: dict) -> dict:
    root_defaults = {k: cfg.get(k) for k in ["alpha", "beta", "beta1", "beta2", "eta", "b"] if cfg.get(k) is not None}
    default_block = cfg.get("default") or {}
    base_params = {**root_defaults, **default_block}
    tuned = cfg.get("tuned") or {}
    result = {}
    for variant in ("gta_base", "gta_beta"):
        variant_cfg = {k: v for k, v in base_params.items() if v is not None}
        overrides = {k: v for k, v in (tuned.get(variant) or {}).items() if v is not None}
        result[variant] = {**variant_cfg, **overrides}
    return result


def _build_reward_adjacency(P: np.ndarray) -> List[List[int]]:
    n_states = P.shape[1]
    adj: List[List[int]] = [[] for _ in range(n_states)]
    for s in range(n_states):
        seen: set[int] = set()
        for a in range(P.shape[0]):
            for s2 in np.flatnonzero(P[a, s] > 1e-12):
                s2i = int(s2)
                if s2i not in seen:
                    seen.add(s2i)
                    adj[s].append(s2i)
    return adj


def _build_directed_kernel(
    kind: str,
    n_states: int,
    n_actions: int,
    forward_hi: float,
    forward_lo: float,
) -> np.ndarray:
    if kind not in ("path", "cycle"):
        raise ValueError(f"Unknown directed kernel kind {kind!r}")
    if n_states < 2:
        raise ValueError("n_states must be at least 2 for a directed kernel.")
    if kind == "cycle" and n_states < 3:
        raise ValueError("n_states must be at least 3 for a cycle-like kernel.")
    if n_actions < 1:
        raise ValueError("n_actions must be at least 1.")

    forward_biases = np.linspace(forward_hi, forward_lo, n_actions)
    if np.any(forward_biases <= 0.0) or np.any(forward_biases >= 1.0):
        raise ValueError("forward biases must lie strictly inside (0, 1).")
    if np.any(forward_biases <= 0.5):
        raise ValueError("forward biases must exceed 0.5 to keep upstream mass small.")

    P = np.zeros((n_actions, n_states, n_states), dtype=np.float64)
    for a, fwd in enumerate(forward_biases):
        upstream = 1.0 - fwd

        for s in range(n_states):
            if kind == "path":
                forward_target = s + 1 if s < n_states - 1 else s
                upstream_target = s - 1 if s > 0 else s
            else:
                forward_target = (s + 1) % n_states
                upstream_target = (s - 1) % n_states

            P[a, s, forward_target] += fwd
            P[a, s, upstream_target] += upstream

    return P


def _make_directed_garnet_env(
    kind: str,
    seed: int,
    gamma: float,
    env_kwargs: dict,
    kernel_kwargs: dict,
) -> GarnetEnv:
    env = GarnetEnv(seed=seed, gamma=gamma, **env_kwargs)
    env.P = _build_directed_kernel(
        kind,
        env.n_states,
        env.n_actions,
        kernel_kwargs["forward_hi"],
        kernel_kwargs["forward_lo"],
    )
    if env.reward_locality:
        env._adj = _build_reward_adjacency(env.P)
    env.r = env._sample_reward()
    return env


def _make_fourrooms_env(seed: int, gamma: float, env_kwargs: dict) -> FourRoomsTabular:
    return FourRoomsTabular(seed=seed, gamma=gamma, **env_kwargs)


def _apply_reward_update(agent, r_new: np.ndarray) -> None:
    updater = getattr(agent, "reward_update", None)
    if callable(updater):
        updater(r_new)
        return

    module = getattr(agent, "module", None)
    module_updater = getattr(module, "reward_update", None) if module is not None else None
    if callable(module_updater):
        current = getattr(module, "r", None)
        if current is None:
            raise RuntimeError("Cannot refresh rewards on this agent.")
        module_updater(r_new - np.asarray(current))
        return

    raise AttributeError("Agent does not expose a reward_update hook.")


def run_seed_worker(
    env_name: str,
    env_params: dict,
    variants: List[dict],
    n_shifts: int,
    n_steps_per_shift: int,
    k_planning: int,
    seed: int,
):
    np.random.seed(seed)
    gamma = float(env_params["gamma"])
    if env_name == "garnet":
        kind = env_params["kind"]
        env = _make_directed_garnet_env(
            kind=kind,
            seed=seed,
            gamma=gamma,
            env_kwargs=env_params["env_kwargs"],
            kernel_kwargs=env_params["kernel_kwargs"],
        )
    elif env_name == "fourrooms":
        env = _make_fourrooms_env(
            seed=seed,
            gamma=gamma,
            env_kwargs=env_params["env_kwargs"],
        )
    else:
        raise ValueError(f"Unknown env_name {env_name!r}")

    n_states = env.n_states
    seed_results = {}
    for var in variants:
        agent = make_agent(
            variant=var["variant"],
            host="dyna",
            P=env.P,
            r=env.r,
            gamma=env.gamma,
            n_states=env.n_states,
            n_actions=env.n_actions,
            **var.get("kwargs", {}),
        )

        shift_end_errors = []
        for _ in range(n_shifts):
            r_new = env.shift_reward()
            V_star = env.value_iteration(r_new)
            _apply_reward_update(agent, r_new)

            for step in range(n_steps_per_shift):
                burst_size = 5 * n_states if step == 0 else 1
                if burst_size > 1:
                    ss = np.random.randint(n_states, size=burst_size)
                    aa = np.random.randint(env.n_actions, size=burst_size)
                    sn = np.array([np.random.choice(n_states, p=env.P[aa[i], ss[i]]) for i in range(burst_size)])
                    rr = r_new[ss]
                    agent.real_steps_batch(ss, aa, rr, sn)
                else:
                    s_rand = np.random.randint(n_states)
                    a_rand = np.random.randint(env.n_actions)
                    s_next_rand = np.random.choice(n_states, p=env.P[a_rand, s_rand])
                    r_rand = r_new[s_rand]
                    agent.real_step(s_rand, a_rand, r_rand, s_next_rand)

                agent.plan_steps(K=k_planning)
                if step == n_steps_per_shift - 1:
                    V_after = agent.Q.max(axis=0)
                    shift_end_errors.append(value_error_l1(V_after, V_star))

        seed_results[var["label"]] = float(np.mean(shift_end_errors))
    return seed_results


def get_stats(seed_results: List[Dict[str, float]], labels: List[str]):
    stats = {}
    for label in labels:
        vals = [res[label] for res in seed_results]
        mean = float(np.mean(vals))
        if len(vals) < 2:
            ci95 = 0.0
        else:
            ci95 = float(1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals)))
        stats[label] = {"mean": mean, "ci95": ci95}
    return stats


def plot_rq6_results(garnet_stats: dict, fourrooms_stats: dict):
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.2, 5.0))

    point_colors = get_palette("point")
    grid_color = get_utility_color("grid")
    label_to_key = {
        AGENT_LABELS["standard_ps"]: "standard_ps",
        AGENT_LABELS["gta_base"]: "gta_base",
        AGENT_LABELS["gta_beta"]: "gta_beta",
    }
    labels = [AGENT_LABELS["standard_ps"], AGENT_LABELS["gta_base"], AGENT_LABELS["gta_beta"]]

    garnet_order = [name for name, _ in FLOW_FAMILIES]
    garnet_ticks = np.arange(len(garnet_order))
    garnet_display = ["Path", "Cycle"]
    offsets = [-0.18, 0.0, 0.18]
    markers = ["o", "^", "s"]

    for i, label in enumerate(labels):
        key = label_to_key[label]
        means = [garnet_stats[name][label]["mean"] for name in garnet_order]
        cis = [garnet_stats[name][label]["ci95"] for name in garnet_order]
        ax1.errorbar(
            garnet_ticks + offsets[i],
            means,
            yerr=cis,
            fmt=markers[i],
            capsize=4,
            color=point_colors[key],
            linewidth=1.7,
            markersize=7,
            label=label,
        )

    ax1.set_title("Directed GARNET Families", fontsize=11, pad=8)
    ax1.set_xlabel("Transition family", fontsize=10)
    ax1.set_xticks(garnet_ticks)
    ax1.set_xticklabels(garnet_display)
    ax1.set_ylabel("L1 error", fontsize=9)
    ax1.tick_params(axis="both", labelsize=8)
    ax1.legend(frameon=False, fontsize=7)
    ax1.grid(True, alpha=0.3, color=grid_color)

    fr_order = [name for name, _ in FOURROOMS_MODES]
    fr_ticks = np.arange(len(fr_order))
    for i, label in enumerate(labels):
        key = label_to_key[label]
        means = [fourrooms_stats[name][label]["mean"] for name in fr_order]
        cis = [fourrooms_stats[name][label]["ci95"] for name in fr_order]
        ax2.errorbar(
            fr_ticks + offsets[i],
            means,
            yerr=cis,
            fmt=markers[i],
            capsize=4,
            color=point_colors[key],
            linewidth=1.7,
            markersize=7,
            label=label,
        )

    ax2.set_title("FourRooms Transport Sweep", fontsize=11, pad=8)
    ax2.set_xlabel("Transport mode", fontsize=10)
    ax2.set_xticks(fr_ticks)
    ax2.set_xticklabels(["Symmetric", "Windy", "Belt", "Mixed"])
    ax2.set_ylabel("L1 error", fontsize=9)
    ax2.tick_params(axis="both", labelsize=8)
    ax2.legend(frameon=False, fontsize=7)
    ax2.grid(True, alpha=0.3, linestyle="--", color=grid_color)

    fig.tight_layout(pad=1.0, w_pad=2.0)
    plot_path = ROOT / "plots" / "rq6_curl_divergence.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--n-shifts", type=int, default=3)
    parser.add_argument("--n-steps-per-shift", type=int, default=40)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--n-states", type=int, default=FLOW_ENV_KWARGS["n_states"])
    parser.add_argument("--n-actions", type=int, default=FLOW_ENV_KWARGS["n_actions"])
    parser.add_argument("--branching", type=int, default=FLOW_ENV_KWARGS["branching"])
    parser.add_argument("--reward-support", type=int, default=FLOW_ENV_KWARGS["reward_support"])
    parser.add_argument("--reward-locality", action="store_true", default=FLOW_ENV_KWARGS["reward_locality"])
    parser.add_argument("--no-reward-locality", dest="reward_locality", action="store_false")
    parser.add_argument("--forward-hi", type=float, default=KERNEL_DEFAULTS["forward_hi"])
    parser.add_argument("--forward-lo", type=float, default=KERNEL_DEFAULTS["forward_lo"])
    parser.add_argument("--room-size", type=int, default=FOURROOMS_BASE_KWARGS["room_size"])
    parser.add_argument("--corridor-len", type=int, default=FOURROOMS_BASE_KWARGS["corridor_len"])
    parser.add_argument("--corridor-width", type=int, default=FOURROOMS_BASE_KWARGS["corridor_width"])
    parser.add_argument("--window-offset", type=int, default=FOURROOMS_BASE_KWARGS["window_offset"])
    parser.add_argument("--k-fourrooms", type=int, default=FOURROOMS_BASE_KWARGS["k"])
    args = parser.parse_args()

    if args.config:
        method_cfgs = _build_method_cfgs(_load_config(args.config))
    else:
        method_cfgs = load_tuned_for_env("garnet")

    variants = [
        {"label": AGENT_LABELS["standard_ps"], "variant": "standard_ps", "kwargs": {}},
        {"label": AGENT_LABELS["gta_base"], "variant": "gta_base", "kwargs": {**method_cfgs.get("gta_base", {}), "alpha": 0.0}},
        {"label": AGENT_LABELS["gta_beta"], "variant": "gta_beta", "kwargs": method_cfgs.get("gta_beta", {})},
    ]
    variant_labels = [v["label"] for v in variants]
    runner = RunManager(n_workers=args.workers)

    garnet_env_kwargs = {
        "n_states": args.n_states,
        "n_actions": args.n_actions,
        "branching": args.branching,
        "reward_support": args.reward_support,
        "reward_locality": args.reward_locality,
    }
    kernel_kwargs = {
        "forward_hi": args.forward_hi,
        "forward_lo": args.forward_lo,
    }
    fourrooms_base = {
        "room_size": args.room_size,
        "corridor_len": args.corridor_len,
        "corridor_width": args.corridor_width,
        "window_offset": args.window_offset,
        "k": args.k_fourrooms,
    }

    garnet_results = {}
    for family_name, kind in FLOW_FAMILIES:
        print(f"Running {family_name.lower()} family (Seeds: {args.seeds})...")
        res = runner.run(
            run_seed_worker,
            list(range(args.seeds)),
            dict(
                env_name="garnet",
                env_params={
                    "kind": kind,
                    "gamma": args.gamma,
                    "env_kwargs": garnet_env_kwargs,
                    "kernel_kwargs": kernel_kwargs,
                },
                variants=variants,
                n_shifts=args.n_shifts,
                n_steps_per_shift=args.n_steps_per_shift,
                k_planning=args.K,
            ),
            desc=f"rq6_garnet_{kind}",
        )
        garnet_results[family_name] = get_stats(res, variant_labels)

    fourrooms_results = {}
    print(f"Running FourRooms transport sweep (Seeds: {args.seeds})...")
    for mode_name, mode_kwargs in FOURROOMS_MODES:
        env_kwargs = dict(fourrooms_base)
        env_kwargs.update(mode_kwargs)
        res = runner.run(
            run_seed_worker,
            list(range(args.seeds)),
            dict(
                env_name="fourrooms",
                env_params={
                    "gamma": args.gamma,
                    "env_kwargs": env_kwargs,
                },
                variants=variants,
                n_shifts=args.n_shifts,
                n_steps_per_shift=args.n_steps_per_shift,
                k_planning=args.K,
            ),
            desc=f"rq6_fourrooms_{mode_name}",
        )
        fourrooms_results[mode_name] = get_stats(res, variant_labels)

    plot_rq6_results(garnet_results, fourrooms_results)

    os.makedirs(ROOT / "results", exist_ok=True)
    with open(ROOT / "results" / "rq6_curl_divergence.json", "w") as f:
        json.dump(
            {
                "garnet": garnet_results,
                "fourrooms": fourrooms_results,
                "config": {
                    **vars(args),
                    "flow_families": [{"label": label, "kind": kind} for label, kind in FLOW_FAMILIES],
                    "flow_env_kwargs": garnet_env_kwargs,
                    "kernel_kwargs": kernel_kwargs,
                    "fourrooms_modes": [{"label": label, "kwargs": kwargs} for label, kwargs in FOURROOMS_MODES],
                    "fourrooms_base_kwargs": fourrooms_base,
                },
            },
            f,
            indent=2,
        )

    print("\nRQ6 Directed Flow Audit Complete.")
