"""
RQ6: Mutual Curl / Divergence Analysis (Redesigned).

Mission:
Disentangle the performance contributions of Curl (circulation) and 
Divergence (source-sink structure) by sweeping bottleneck severity
under directed dynamics (reversible=0.0).
Measures convergence consistency (L1 error at shift-end).
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
from gta_ps.envs.fourrooms import FourRoomsTabular
from gta_ps.utils.metrics import value_error_l1
from gta_ps.utils.hyperparams import load_tuned_for_env
from runner import RunManager
from plot_academic import (
    LABELS as AGENT_LABELS,
    apply_style,
    get_palette,
    get_publication_defaults,
    get_utility_color,
    warn_if_nonpublication_run,
)

PUBLICATION_DEFAULTS = get_publication_defaults()

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

def run_seed_worker(env_name: str, env_params: dict, variants: List[dict], n_shifts: int, n_steps_per_shift: int, k_planning: int, seed: int):
    if env_name == "garnet":
        env = GarnetEnv(**env_params, seed=seed)
    else:
        env = FourRoomsTabular(**env_params, seed=seed)
        
    n_states = env.n_states
    
    seed_results = {}
    for var in variants:
        agent = make_agent(variant=var['variant'], host="dyna", P=env.P, r=env.r, gamma=env.gamma, **var.get('kwargs', {}))
        
        shift_end_errors = []
        for shift in range(n_shifts):
            r_new = env.shift_reward()
            V_star = env.value_iteration(r_new)
            
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
            
        seed_results[var['label']] = np.mean(shift_end_errors)
    return seed_results

def get_stats(seed_results: List[Dict[str, float]], labels: List[str]):
    stats = {}
    for label in labels:
        vals = [res[label] for res in seed_results]
        stats[label] = {"mean": np.mean(vals), "ci95": 1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals))}
    return stats

def plot_rq6_results(garnet_stats: dict, fr_results: dict, seeds: int, k: int, steps: int):
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
    markers = ['o', '^', 's']
    offsets = [-0.15, 0, 0.15]

    # 1. GARNET Plot: Sweep n_nbr_rate at reversible=0.0 (Line plot)
    nbr_rates = sorted(garnet_stats.keys(), reverse=True)
    for i, label in enumerate(labels):
        key = label_to_key.get(label, "standard_ps")
        means = [garnet_stats[rate][label]['mean'] for rate in nbr_rates]
        cis = [garnet_stats[rate][label]['ci95'] for rate in nbr_rates]
        ax1.errorbar(nbr_rates, means, yerr=cis, marker=markers[i], capsize=5,
                     label=label, color=p_line[key], linewidth=2)

    ax1.set_title("GARNET Bottleneck Sweep")
    ax1.set_xlabel("n_nbr_rate (1.0=Global, 0.05=Localized)")
    ax1.set_ylabel(r"Final $L_1$ Error")
    ax1.invert_xaxis()
    ax1.legend()
    ax1.grid(True, alpha=0.3, color=grid_color)

    # 2. FourRooms Plot: Mechanism Comparison (Point-wise plot)
    f_modes = list(fr_results.keys())
    fx = np.arange(len(f_modes))
    for i, label in enumerate(labels):
        key = label_to_key.get(label, "standard_ps")
        means = [fr_results[m][label]['mean'] for m in f_modes]
        cis = [fr_results[m][label]['ci95'] for m in f_modes]
        ax2.errorbar(fx + offsets[i], means, yerr=cis, fmt=markers[i], capsize=5,
                     label=label, color=p_point[key], markersize=7)
    ax2.set_title("FourRooms Mechanism Sweep")
    ax2.set_xticks(fx)
    ax2.set_xticklabels(f_modes)
    ax2.set_ylabel(r"Final $L_1$ Error")
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
    parser.add_argument("--workers", default=PUBLICATION_DEFAULTS["workers"])
    parser.add_argument("--seeds", type=int, default=PUBLICATION_DEFAULTS["seeds"])
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--n-shifts", type=int, default=4)
    parser.add_argument("--n-steps-per-shift", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.95)
    args = parser.parse_args()
    warn_if_nonpublication_run(args.seeds, args.workers)

    if args.config:
        method_cfgs = _build_method_cfgs(_load_config(args.config))
    else:
        method_cfgs = load_tuned_for_env("garnet")

    variants = [
        {"label": "Standard-PS", "variant": "standard_ps", "kwargs": {}},
        {"label": "Forward-GTA",  "variant": "gta_base",    "kwargs": {**method_cfgs.get("gta_base", {}), "alpha": 0.0}},
        {"label": "Full-GTA",     "variant": "gta_beta",    "kwargs": method_cfgs.get("gta_beta", {})},
    ]
    variant_labels = [v['label'] for v in variants]
    runner = RunManager(n_workers=args.workers)

    # 1. GARNET Sweep (Directed Bottlenecks)
    print(f"Running GARNET sweep at reversible=0.2 (Seeds: {args.seeds})...")
    nbr_rates = [0.8, 0.4, 0.2, 0.1, 0.05]
    garnet_final = {}
    for rate in nbr_rates:
        res = runner.run(run_seed_worker, list(range(args.seeds)),
                         dict(env_name="garnet", env_params={"n_states": 75, "n_nbr_rate": rate, "reversible": 0.2, "gamma": args.gamma},
                              variants=variants, n_shifts=args.n_shifts, n_steps_per_shift=args.n_steps_per_shift, k_planning=args.K),
                         desc=f"garnet_nbr_{rate}")
        garnet_final[rate] = get_stats(res, variant_labels)
            
    # 2. FourRooms Mechanism Isolation
    print(f"Running FourRooms mechanism isolation (Seeds: {args.seeds})...")
    fr_results = {}
    modes = [("Symmetric", {}), ("Windy (Div)", {"drift": (0, 1)}), ("Cyclic (Curl)", {"one_way_prob": 0.05}), ("Mixed", {"drift": (0, 1), "one_way_prob": 0.05})]
    for mode_name, p in modes:
        res = runner.run(run_seed_worker, list(range(args.seeds)),
                         dict(env_name="fourrooms", env_params={**p, "gamma": args.gamma},
                              variants=variants, n_shifts=args.n_shifts, n_steps_per_shift=args.n_steps_per_shift, k_planning=args.K),
                         desc=f"fr_{mode_name}")
        fr_results[mode_name] = get_stats(res, variant_labels)
        
    plot_rq6_results(garnet_final, fr_results, args.seeds, args.K, args.n_steps_per_shift)
    
    # Save JSON
    os.makedirs(ROOT / "results", exist_ok=True)
    with open(ROOT / "results" / "rq6_curl_divergence.json", "w") as f:
        json.dump({"garnet": {str(k): v for k, v in garnet_final.items()}, "fourrooms": fr_results, "config": vars(args)}, f, indent=2)
    
    print("\nRQ6 Redesigned Audit Complete.")
