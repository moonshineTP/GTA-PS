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

from gta_ps.algorithms import make_agent, DynaHost
from gta_ps.envs.garnet import GarnetEnv
from gta_ps.envs.fourrooms import FourRoomsTabular
from gta_ps.utils.metrics import value_error_l1
from gta_ps.utils.hyperparams import load_tuned_for_env
from gta_ps.utils import operator_split, slem_arnoldi
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

FOURROOMS_BELT_LEAK = 0.05


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


def _geometry_stats(env) -> dict:
    P_mean = env.P.mean(axis=0)
    S, A = operator_split(P_mean)
    curl_norm = float(np.linalg.norm(A, ord="fro"))
    div_norm = float(np.linalg.norm(np.eye(P_mean.shape[0]) - S, ord="fro"))
    slem, _ = slem_arnoldi(P_mean)
    return {
        "curl_norm": curl_norm,
        "div_norm": div_norm,
        "curl_over_div": float(curl_norm / max(div_norm, 1e-12)),
        "slem": float(slem),
    }


def _make_fourrooms_env(
    *,
    seed: int,
    gamma: float,
    env_kwargs: dict,
) -> FourRoomsTabular:
    env = FourRoomsTabular(
        seed=seed,
        gamma=gamma,
        **env_kwargs,
    )
    return env


def _fourrooms_transition_delta(gamma: float, base_kwargs: dict, mode_kwargs: dict) -> float:
    symmetric_env = _make_fourrooms_env(seed=0, gamma=gamma, env_kwargs=base_kwargs)
    env_kwargs = dict(base_kwargs)
    env_kwargs.update(mode_kwargs)
    env = _make_fourrooms_env(seed=0, gamma=gamma, env_kwargs=env_kwargs)
    return float(np.abs(env.P - symmetric_env.P).sum())


def _match_drift_strength_to_belt(gamma: float, base_kwargs: dict, one_way_prob: float) -> float:
    target_delta = _fourrooms_transition_delta(gamma, base_kwargs, {"one_way_prob": one_way_prob})
    lo = 0.0
    hi = 1.0
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        delta = _fourrooms_transition_delta(gamma, base_kwargs, {"drift": (0, 1), "drift_strength": mid})
        if delta < target_delta:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _resolve_fourrooms_modes(gamma: float, base_kwargs: dict) -> List[tuple[str, dict]]:
    matched_drift_strength = _match_drift_strength_to_belt(
        gamma=gamma,
        base_kwargs=base_kwargs,
        one_way_prob=FOURROOMS_BELT_LEAK,
    )
    return [
        ("Symmetric", {}),
        ("Windy (Div)", {"drift": (0, 1), "drift_strength": matched_drift_strength}),
        ("Cyclic (Curl)", {"one_way_prob": FOURROOMS_BELT_LEAK}),
        ("Mixed", {"drift": (0, 1), "drift_strength": matched_drift_strength, "one_way_prob": FOURROOMS_BELT_LEAK}),
    ]


def _fourrooms_transport_diagnostics(
    *,
    gamma: float,
    base_kwargs: dict,
    modes: List[tuple[str, dict]],
) -> dict:
    symmetric_env = _make_fourrooms_env(seed=0, gamma=gamma, env_kwargs=base_kwargs)
    symmetric_P = symmetric_env.P.copy()
    belt_positions = [list(symmetric_env._pos_list[s]) for s in symmetric_env.belt_cycle_states()]
    if not belt_positions:
        raise RuntimeError("FourRooms transport diagnostics found no belt cycle states for the current geometry.")

    diagnostics = {}
    mode_kernels = {"Symmetric": symmetric_P}
    for mode_name, mode_kwargs in modes:
        env_kwargs = dict(base_kwargs)
        env_kwargs.update(mode_kwargs)
        env = _make_fourrooms_env(seed=0, gamma=gamma, env_kwargs=env_kwargs)
        diff = np.abs(env.P - symmetric_P)
        changed_states = int(np.count_nonzero(np.any(diff > 1e-12, axis=(0, 2))))
        diagnostics[mode_name] = {
            "transport_l1_vs_symmetric": float(diff.sum()),
            "changed_states": changed_states,
            "geometry": _geometry_stats(env),
        }
        if mode_name != "Symmetric" and diagnostics[mode_name]["transport_l1_vs_symmetric"] <= 1e-12:
            raise RuntimeError(f"FourRooms mode {mode_name!r} did not modify the transition kernel.")
        mode_kernels[mode_name] = env.P.copy()

    pairwise_l1 = {}
    mode_names = [mode_name for mode_name, _ in modes]
    for i, lhs in enumerate(mode_names):
        for rhs in mode_names[i + 1:]:
            pair_key = f"{lhs} vs {rhs}"
            pair_delta = float(np.abs(mode_kernels[lhs] - mode_kernels[rhs]).sum())
            if pair_delta <= 1e-12:
                raise RuntimeError(f"FourRooms modes {lhs!r} and {rhs!r} produced the same transition kernel.")
            pairwise_l1[pair_key] = pair_delta

    diagnostics["belt_cycle_positions"] = belt_positions
    diagnostics["pairwise_transport_l1"] = pairwise_l1
    return diagnostics


def _print_fourrooms_transport_diagnostics(diagnostics: dict) -> None:
    print("FourRooms transport diagnostics:")
    for mode_name, _ in diagnostics["resolved_modes"]:
        stats = diagnostics[mode_name]
        geom = stats["geometry"]
        print(
            f"  {mode_name}: delta_vs_symmetric={stats['transport_l1_vs_symmetric']:.3f}, "
            f"changed_states={stats['changed_states']}, "
            f"curl={geom['curl_norm']:.3f}, div={geom['div_norm']:.3f}, slem={geom['slem']:.4f}"
        )
    print(f"  belt_cycle_positions: {diagnostics['belt_cycle_positions']}")
    print(f"  pairwise_transport_l1: {diagnostics['pairwise_transport_l1']}")


def _build_variants(method_cfgs: dict) -> List[dict]:
    return [
        {"label": AGENT_LABELS["standard_ps"], "variant": "standard_ps", "kwargs": {}},
        {"label": AGENT_LABELS["gta_base"], "variant": "gta_base", "kwargs": {**method_cfgs.get("gta_base", {}), "alpha": 0.0}},
        {"label": AGENT_LABELS["gta_beta"], "variant": "gta_beta", "kwargs": method_cfgs.get("gta_beta", {})},
    ]


def run_seed_worker(
    env_name: str,
    env_params: dict,
    variants: List[dict],
    n_shifts: int,
    n_steps_per_shift: int,
    k_planning: int,
    seed: int,
):
    if env_name == "garnet":
        env = _make_directed_garnet_env(
            kind=env_params["kind"],
            seed=seed,
            gamma=env_params["gamma"],
            env_kwargs=env_params["env_kwargs"],
            kernel_kwargs=env_params["kernel_kwargs"],
        )
    elif env_name == "fourrooms":
        env = _make_fourrooms_env(
            seed=seed,
            gamma=env_params["gamma"],
            env_kwargs=env_params["env_kwargs"],
        )
    else:
        raise ValueError(f"Unknown env_name {env_name!r}")

    sim_rng = np.random.default_rng(np.random.SeedSequence([seed, 0xC06E, 6]))
    n_states = env.n_states
    seed_results = {}
    for idx, var in enumerate(variants):
        agent: DynaHost = make_agent(
            variant=var["variant"],
            host="dyna",
            P=env.P,
            r=env.r,
            gamma=env.gamma,
            rng=np.random.default_rng(np.random.SeedSequence([seed, idx, 0xC06E])),
            **var.get("kwargs", {}),
        )

        shift_end_errors = []
        for _ in range(n_shifts):
            r_new = env.shift_reward()
            V_star = env.value_iteration(r_new)
            agent.reward_update(r_new)

            for step in range(n_steps_per_shift):
                burst_size = 5 * n_states if step == 0 else 1
                if burst_size > 1:
                    ss = sim_rng.integers(n_states, size=burst_size)
                    aa = sim_rng.integers(env.n_actions, size=burst_size)
                    sn = np.array([sim_rng.choice(n_states, p=env.P[aa[i], ss[i]]) for i in range(burst_size)])
                    rr = r_new[ss]
                    agent.real_steps_batch(ss, aa, rr, sn)
                else:
                    s_rand = sim_rng.integers(n_states)
                    a_rand = sim_rng.integers(env.n_actions)
                    s_next_rand = sim_rng.choice(n_states, p=env.P[a_rand, s_rand])
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


def plot_rq6_results(garnet_stats: dict, fourrooms_stats: dict) -> bool:
    if plt is None:
        return False
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

    fr_order = list(fourrooms_stats.keys())
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
    ax2.set_xticklabels(fr_order)
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
    return True


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
        garnet_method_cfgs = _build_method_cfgs(_load_config(args.config))
        fourrooms_method_cfgs = garnet_method_cfgs
    else:
        garnet_method_cfgs = load_tuned_for_env("garnet")
        fourrooms_method_cfgs = load_tuned_for_env("fourrooms")

    garnet_variants = _build_variants(garnet_method_cfgs)
    fourrooms_variants = _build_variants(fourrooms_method_cfgs)
    variant_labels = [v["label"] for v in garnet_variants]
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
    fourrooms_modes = _resolve_fourrooms_modes(
        gamma=args.gamma,
        base_kwargs=fourrooms_base,
    )
    fourrooms_diagnostics = _fourrooms_transport_diagnostics(
        gamma=args.gamma,
        base_kwargs=fourrooms_base,
        modes=fourrooms_modes,
    )
    fourrooms_diagnostics["resolved_modes"] = [
        [label, kwargs]
        for label, kwargs in fourrooms_modes
    ]
    _print_fourrooms_transport_diagnostics(fourrooms_diagnostics)

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
                variants=garnet_variants,
                n_shifts=args.n_shifts,
                n_steps_per_shift=args.n_steps_per_shift,
                k_planning=args.K,
            ),
            desc=f"rq6_garnet_{kind}",
        )
        garnet_results[family_name] = get_stats(res, variant_labels)

    fourrooms_results = {}
    print(f"Running FourRooms transport sweep (Seeds: {args.seeds})...")
    for mode_name, mode_kwargs in fourrooms_modes:
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
                variants=fourrooms_variants,
                n_shifts=args.n_shifts,
                n_steps_per_shift=args.n_steps_per_shift,
                k_planning=args.K,
            ),
            desc=f"rq6_fourrooms_{mode_name}",
        )
        fourrooms_results[mode_name] = get_stats(res, variant_labels)

    plot_written = plot_rq6_results(garnet_results, fourrooms_results)

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
                    "fourrooms_modes": [{"label": label, "kwargs": kwargs} for label, kwargs in fourrooms_modes],
                    "fourrooms_base_kwargs": fourrooms_base,
                    "fourrooms_transport_diagnostics": fourrooms_diagnostics,
                },
            },
            f,
            indent=2,
        )

    print("\nRQ6 Directed Flow Audit Complete.")
    if not plot_written:
        print("Plot skipped: matplotlib is not installed in the active interpreter.")
