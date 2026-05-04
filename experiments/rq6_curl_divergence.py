"""
RQ6: Mutual Curl / Divergence Analysis.

Mission:
Compare two directed GARNET families that separate source-sink flow from
cyclic circulation:
- a tight path-wise kernel with one-way forward motion
- a cyclic kernel with the same directional bias on a ring

This is meant to isolate curl/divergence geometry, not bottleneck severity.
Measures convergence consistency via end-of-shift L1 error.
"""
import sys
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, List

# Setup path
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

from gta_ps.algorithms import make_agent
from gta_ps.envs.garnet import GarnetEnv
from gta_ps.utils.metrics import value_error_l1
from gta_ps.utils.hyperparams import load_tuned_for_env
from gta_ps.utils.spectral import operator_split, slem_arnoldi
from runner import RunManager
from plot_academic import (
    apply_style,
    get_palette,
    get_utility_color,
)

FLOW_FAMILIES = (
    ("Path-wise flow", "path"),
    ("Cyclic flow", "cycle"),
)

FLOW_ENV_KWARGS = {
    "n_states": 75,
    "n_actions": 4,
    "branching": 2,
    "reward_support": 3,
    "reward_locality": False,
}

def _load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}

def _build_method_cfgs(cfg: dict) -> dict:
    root_defaults = {k: cfg.get(k) for k in ["alpha", "beta", "beta1", "beta2", "eta", "b"] if cfg.get(k) is not None}
    default_block = cfg.get("default") or {}
    base_params = {**root_defaults, **default_block}
    tuned = cfg.get("tuned") or {}
    result  = {}
    for variant in ("gta_base", "gta_beta"):
        variant_cfg = {k: v for k, v in base_params.items() if v is not None}
        overrides = {k: v for k, v in (tuned.get(variant) or {}).items() if v is not None}
        result[variant] = {**variant_cfg, **overrides}
    return result


def _build_directed_kernel(kind: str, n_states: int, n_actions: int) -> np.ndarray:
    """
    Build a directed transition tensor with a consistent forward bias.

    ``kind='path'`` makes the terminal state absorbing so flow remains path-wise.
    ``kind='cycle'`` wraps the same bias around a ring to create circulation.
    """
    if kind not in ("path", "cycle"):
        raise ValueError(f"Unknown directed kernel kind {kind!r}")

    P = np.zeros((n_actions, n_states, n_states), dtype=np.float64)
    forward_biases = np.linspace(0.90, 0.78, n_actions)
    reverse_mass = 0.05

    for a, fwd in enumerate(forward_biases):
        stay = 1.0 - fwd - reverse_mass
        if stay < 0.0:
            raise ValueError("Forward and reverse masses must sum to at most 1.")

        for s in range(n_states):
            if kind == "path":
                if s < n_states - 1:
                    P[a, s, s + 1] += fwd
                else:
                    P[a, s, s] += fwd

                if s > 0:
                    P[a, s, s - 1] += reverse_mass
                else:
                    P[a, s, s] += reverse_mass
            else:
                P[a, s, (s + 1) % n_states] += fwd
                P[a, s, (s - 1) % n_states] += reverse_mass

            P[a, s, s] += stay

    return P


def _make_directed_garnet_env(kind: str, seed: int, gamma: float) -> GarnetEnv:
    env = GarnetEnv(
        seed=seed,
        gamma=gamma,
        **FLOW_ENV_KWARGS,
    )
    env.P = _build_directed_kernel(kind, env.n_states, env.n_actions)
    return env


def _geometry_stats(env: GarnetEnv) -> dict:
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

def run_seed_worker(kind: str, gamma: float, variants: List[dict], n_shifts: int, n_steps_per_shift: int, k_planning: int, seed: int):
    env = _make_directed_garnet_env(kind, seed=seed, gamma=gamma)
    sim_rng = np.random.default_rng(np.random.SeedSequence([seed, 0xC06E, 6]))
    n_states = env.n_states

    seed_results = {}
    for idx, var in enumerate(variants):
        agent = make_agent(
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

        seed_results[var["label"]] = np.mean(shift_end_errors)
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

def plot_rq6_results(flow_stats: dict, geometry_stats: dict):
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    p_line  = get_palette("line")
    p_point = get_palette("point")
    grid_color = get_utility_color("grid")

    label_to_key = {
        "Standard-PS": "standard_ps",
        "Forward-GTA": "forward_gta",
        "Full-GTA":    "full_gta"
    }

    labels = ["Standard-PS", "Forward-GTA", "Full-GTA"]
    family_order = [name for name, _ in FLOW_FAMILIES]
    offsets = [-0.25, 0.0, 0.25]
    bar_width = 0.22

    # 1. Directed GARNET families: path vs cycle
    x = np.arange(len(family_order))
    for i, label in enumerate(labels):
        key = label_to_key.get(label, "standard_ps")
        means = [flow_stats[fam][label]['mean'] for fam in family_order]
        cis = [flow_stats[fam][label]['ci95'] for fam in family_order]
        ax1.bar(x + offsets[i], means, width=bar_width, yerr=cis, capsize=4,
                label=label, color=p_point[key], alpha=0.95)

    ax1.set_title("Directed GARNET Families")
    ax1.set_xlabel("Transition family")
    ax1.set_xticks(x)
    ax1.set_xticklabels(family_order)
    ax1.set_ylabel(r"End-of-shift $L_1$ Error")
    ax1.legend()
    ax1.grid(True, alpha=0.3, color=grid_color)

    # 2. Geometry diagnostics for the same kernels
    curl_vals = [geometry_stats[fam]["curl_norm"] for fam in family_order]
    div_vals = [geometry_stats[fam]["div_norm"] for fam in family_order]
    ratio_vals = [geometry_stats[fam]["curl_over_div"] for fam in family_order]
    width = 0.32
    ax2.bar(x - width / 2, curl_vals, width=width, label=r"$\|A\|_F$", color=p_line["gta_base"], alpha=0.9)
    ax2.bar(x + width / 2, div_vals, width=width, label=r"$\|I-S\|_F$", color=p_line["standard_ps"], alpha=0.9)
    for i, fam in enumerate(family_order):
        top = max(curl_vals[i], div_vals[i])
        ax2.text(
            x[i],
            top * 1.04,
            f"ratio={ratio_vals[i]:.2f}\nSLEM={geometry_stats[fam]['slem']:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax2.set_title("Curl / Divergence Geometry")
    ax2.set_xlabel("Transition family")
    ax2.set_xticks(x)
    ax2.set_xticklabels(family_order)
    ax2.set_ylabel("Frobenius norm")
    ax2.legend()
    ax2.grid(True, alpha=0.3, linestyle='--', color=grid_color)

    plt.tight_layout()
    plot_path = ROOT / "plots" / "rq6_curl_divergence.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--n-shifts", type=int, default=4)
    parser.add_argument("--n-steps-per-shift", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.95)
    args = parser.parse_args()

    if args.config:
        method_cfgs = _build_method_cfgs(_load_config(args.config))
    else:
        method_cfgs = load_tuned_for_env("garnet")

    variants = [
        {"label": "Standard-PS", "variant": "standard_ps", "kwargs": {}},
        {"label": "Forward-GTA",  "variant": "gta_base",    "kwargs": {**method_cfgs.get("gta_base", {}), "alpha": 0.0}},
        {"label": "Full-GTA",     "variant": "gta_beta",    "kwargs": method_cfgs.get("gta_beta", {})},
    ]
    runner = RunManager(n_workers=args.workers)

    family_stats = {}
    geometry_stats = {}
    variant_labels = [v["label"] for v in variants]

    for family_name, kind in FLOW_FAMILIES:
        print(f"Running {family_name.lower()} family (Seeds: {args.seeds})...")
        env = _make_directed_garnet_env(kind, seed=0, gamma=args.gamma)
        geometry_stats[family_name] = _geometry_stats(env)
        geom = geometry_stats[family_name]
        print(
            f"  geometry: curl={geom['curl_norm']:.4f}, div={geom['div_norm']:.4f}, "
            f"ratio={geom['curl_over_div']:.4f}, slem={geom['slem']:.4f}"
        )
        res = runner.run(
            run_seed_worker,
            list(range(args.seeds)),
            dict(
                kind=kind,
                gamma=args.gamma,
                variants=variants,
                n_shifts=args.n_shifts,
                n_steps_per_shift=args.n_steps_per_shift,
                k_planning=args.K,
            ),
            desc=f"rq6_{kind}",
        )
        family_stats[family_name] = get_stats(res, variant_labels)

    plot_rq6_results(family_stats, geometry_stats)

    os.makedirs(ROOT / "results", exist_ok=True)
    with open(ROOT / "results" / "rq6_curl_divergence.json", "w") as f:
        json.dump(
            {
                "families": family_stats,
                "geometry": geometry_stats,
                "config": {
                    **vars(args),
                    "flow_families": [
                        {"label": label, "kind": kind}
                        for label, kind in FLOW_FAMILIES
                    ],
                    "flow_env_kwargs": FLOW_ENV_KWARGS,
                },
            },
            f,
            indent=2,
        )
    
    print("\nRQ6 Directed Flow Audit Complete.")
