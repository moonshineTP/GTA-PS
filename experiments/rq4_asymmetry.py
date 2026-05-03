"""
RQ4: Symmetric vs. Asymmetric Environment Effects on GTA-PS.

Mission:
Quantify the performance gain of directional GTA-PS (forward + backward signals)
as environment induced dynamics move from reversible to highly asymmetric.
Measures convergence consistency at the end of each reward shift.
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
    """
    Worker function for RunManager.
    """
    if env_name == "garnet":
        env = GarnetEnv(**env_params, seed=seed)
    else:
        env = FourRoomsTabular(**env_params, seed=seed)
        
    n_states = env.n_states
    n_actions = env.n_actions
    
    seed_results = {}
    for var in variants:
        agent = make_agent(
            variant=var['variant'],
            host="dyna",
            P=env.P,
            r=env.r,
            gamma=env.gamma,
            **var.get('kwargs', {})
        )
        
        # We track the error at the END of each shift to measure convergence consistency
        shift_end_errors = []
        for shift in range(n_shifts):
            r_new = env.shift_reward()
            V_star = env.value_iteration(r_new)
            agent.reward_update(r_new)
            
            for step in range(n_steps_per_shift):
                # Dyna interaction
                burst_size = 5 * n_states if step == 0 else 1
                if burst_size > 1:
                    ss = np.random.randint(n_states, size=burst_size)
                    aa = np.random.randint(n_actions, size=burst_size)
                    sn = np.array([np.random.choice(n_states, p=env.P[aa[i], ss[i]]) for i in range(burst_size)])
                    rr = r_new[ss]
                    agent.real_steps_batch(ss, aa, rr, sn)
                else:
                    s_rand = np.random.randint(n_states)
                    a_rand = np.random.randint(n_actions)
                    s_next_rand = np.random.choice(n_states, p=env.P[a_rand, s_rand])
                    r_rand = r_new[s_rand]
                    agent.real_step(s_rand, a_rand, r_rand, s_next_rand)
                
                # Planning
                agent.plan_steps(K=k_planning)
                
                # If last step of shift, record error
                if step == n_steps_per_shift - 1:
                    V_after = agent.Q.max(axis=0)
                    error = value_error_l1(V_after, V_star)
                    shift_end_errors.append(error)
            
        seed_results[var['label']] = np.mean(shift_end_errors)
        
    return seed_results

def plot_rq4_schema(garnet_stats: Dict[float, Dict[str, dict]], fourrooms_stats: Dict[str, Dict[str, dict]], seeds: int, k: int, steps: int):
    """
    Revamped plot schema using end-of-shift L1 convergence.
    """
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    p_line  = get_palette("line")
    p_point = get_palette("point")
    grid_color = get_utility_color("grid")
    
    # Robust mapping from display label back to palette keys for color lookup
    label_to_key = {
        AGENT_LABELS["standard_ps"]: "standard_ps",
        AGENT_LABELS["forward_gta"]: "forward_gta",
        AGENT_LABELS["full_gta"]:    "full_gta",
        AGENT_LABELS["gta_base"]:    "gta_base",
        AGENT_LABELS["gta_beta"]:    "gta_beta",
    }
    
    # --- 1. GARNET Plot (Connected Line) ---
    rev_levels = sorted(garnet_stats.keys())
    labels = list(next(iter(garnet_stats.values())).keys())
    for i, label in enumerate(labels):
        key = label_to_key.get(label, "standard_ps")
        means = [garnet_stats[rev][label]['mean'] for rev in rev_levels]
        cis = [garnet_stats[rev][label]['ci95'] for rev in rev_levels]
        ax1.errorbar(rev_levels, means, yerr=cis, marker='o', capsize=4, 
                     label=label, color=p_point[key], linewidth=1.5, markersize=5)
    
    ax1.set_title("GARNET Asymmetry Sweep")
    ax1.set_xlabel("Reversibility (0.0=Directed, 1.0=Symmetric)")
    ax1.set_ylabel(r"Final $L_1$ Error")
    ax1.legend()
    ax1.grid(True, alpha=0.3, color=grid_color)
    
    # --- 2. FourRooms Plot (Point-wise) ---
    modes = list(fourrooms_stats.keys())
    x = np.arange(len(modes))
    offsets = [-0.2, 0.0, 0.2]
    for i, label in enumerate(labels):
        key = label_to_key.get(label, "standard_ps")
        means = [fourrooms_stats[m][label]['mean'] for m in modes]
        cis = [fourrooms_stats[m][label]['ci95'] for m in modes]
        ax2.errorbar(x + offsets[i], means, yerr=cis, fmt='o', capsize=5,
                     label=label, color=p_point[key], markersize=7, elinewidth=2)
        
    ax2.set_title("FourRooms Asymmetry Sweep")
    ax2.set_xticks(x)
    ax2.set_xticklabels(modes)
    ax2.set_ylabel(r"Final $L_1$ Error")
    
    all_means = [fourrooms_stats[m][l]['mean'] for m in modes for l in labels]
    if all_means:
        ax2.set_ylim(min(all_means) * 0.9, max(all_means) * 1.1)
    
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--', color=grid_color)
    
    plt.tight_layout()
    plot_path = ROOT / "plots" / "rq4_asymmetry.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")

def get_stats(seed_results: List[Dict[str, float]], labels: List[str]):
    stats = {}
    for label in labels:
        vals = [res[label] for res in seed_results]
        mean = np.mean(vals)
        std = np.std(vals, ddof=1)
        ci95 = 1.96 * std / np.sqrt(len(vals))
        stats[label] = {"mean": mean, "ci95": ci95}
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--workers", default=PUBLICATION_DEFAULTS["workers"])
    parser.add_argument("--seeds", type=int, default=PUBLICATION_DEFAULTS["seeds"])
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--n-shifts", type=int, default=4)
    parser.add_argument("--n-steps-per-shift", type=int, default=40)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--n-states", type=int, default=75)
    args = parser.parse_args()
    warn_if_nonpublication_run(args.seeds, args.workers)

    if args.config:
        method_cfgs = _build_method_cfgs(_load_config(args.config))
    else:
        method_cfgs = load_tuned_for_env("garnet")

    print(f"Starting RQ4 Consistency Sweep...")
    
    variants = [
        {"label": AGENT_LABELS["standard_ps"], "variant": "standard_ps", "kwargs": {}},
        {"label": AGENT_LABELS["gta_base"],    "variant": "gta_base",    "kwargs": {**method_cfgs.get("gta_base", {}), "alpha": 0.0}},
        {"label": AGENT_LABELS["gta_beta"],    "variant": "gta_beta",    "kwargs": method_cfgs.get("gta_beta", {})},
    ]
    variant_labels = [v['label'] for v in variants]
    runner = RunManager(n_workers=args.workers)
    
    # 1. GARNET
    print(f"Running GARNET sweep...")
    rev_points = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    garnet_final_stats = {}
    for rev in rev_points:
        seed_results = runner.run(
            fn=run_seed_worker,
            seeds=list(range(args.seeds)),
            fn_kwargs=dict(
                env_name="garnet", 
                env_params={"n_states": args.n_states, "reversible": rev, "gamma": args.gamma},
                variants=variants, n_shifts=args.n_shifts, 
                n_steps_per_shift=args.n_steps_per_shift, k_planning=args.K
            ),
            desc=f"garnet_rev_{rev}"
        )
        garnet_final_stats[rev] = get_stats(seed_results, variant_labels)
        
    # 2. FourRooms
    print(f"Running FourRooms sweep...")
    fourrooms_final_stats = {}
    modes = [("Symmetric", {"one_way_prob": 1.0}), ("One-Way", {"one_way_prob": 0.05}), ("Windy", {"drift": (0, 1)})]
    for mode_name, params in modes:
        p_copy = params.copy()
        p_copy["gamma"] = args.gamma
        seed_results = runner.run(
            fn=run_seed_worker,
            seeds=list(range(args.seeds)),
            fn_kwargs=dict(
                env_name="fourrooms", env_params=p_copy,
                variants=variants, n_shifts=args.n_shifts,
                n_steps_per_shift=args.n_steps_per_shift, k_planning=args.K
            ),
            desc=f"fourrooms_{mode_name}"
        )
        fourrooms_final_stats[mode_name] = get_stats(seed_results, variant_labels)
    
    plot_rq4_schema(garnet_final_stats, fourrooms_final_stats, args.seeds, args.K, args.n_steps_per_shift)
    
    # Save JSON results
    results_dir = ROOT / "results"
    os.makedirs(results_dir, exist_ok=True)
    with open(results_dir / "rq4_asymmetry.json", "w") as f:
        json.dump({
            "experiment": "rq4_asymmetry",
            "config": vars(args),
            "results": {"garnet": {str(k): v for k, v in garnet_final_stats.items()}, "fourrooms": fourrooms_final_stats}
        }, f, indent=2)
    
    print("\nRQ4 Consistency Sweep Complete.")
