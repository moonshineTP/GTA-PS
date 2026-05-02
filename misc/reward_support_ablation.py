"""
GARNET Ablation: Impact of Reward Support Size on Planning Efficiency.

Mission:
Quantify how the density of reward attractors (support size) affects the 
relative advantage of GTA-augmented priority signals over Standard-PS.
Measures convergence consistency at the end of each reward shift.

Setup:
- Environment: GARNET (|S|=100 default)
- Support Sweep: {1, 2, 4, 8, 16} active reward states.
- Agent: Dyna host with Count-Based Model learning.
- Metric: L1 Value Error at shift-end.
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

from gta_ps.algorithms import make_agent
from gta_ps.envs.garnet import GarnetEnv
from gta_ps.utils.metrics import value_error_l1
from gta_ps.utils.hyperparams import load_tuned_for_env
from runner import RunManager
from plot_academic import apply_style, get_palette

def run_seed_worker(support: int, n_states: int, variants: List[dict], n_shifts: int, n_steps_per_shift: int, k_planning: int, seed: int):
    """
    Worker function for RunManager.
    """
    env = GarnetEnv(n_states=n_states, reward_support=support, seed=seed)
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
        
        shift_end_errors = []
        for shift in range(n_shifts):
            r_new = env.shift_reward()
            V_star = env.value_iteration(r_new)
            
            for step in range(n_steps_per_shift):
                # Dyna interaction
                burst_size = 5 * n_states if step == 0 else 1
                for _ in range(burst_size):
                    s_rand = np.random.randint(n_states)
                    a_rand = np.random.randint(n_actions)
                    s_next_rand = np.random.choice(n_states, p=env.P[a_rand, s_rand])
                    r_rand = r_new[s_rand]
                    agent.real_step(s_rand, a_rand, r_rand, s_next_rand)
                
                agent.plan_steps(K=k_planning)
                
                if step == n_steps_per_shift - 1:
                    V_after = agent.Q.max(axis=0)
                    error = value_error_l1(V_after, V_star)
                    shift_end_errors.append(error)
            
        seed_results[var['label']] = np.mean(shift_end_errors)
        
    return seed_results

def plot_ablation(stats: Dict[int, Dict[str, dict]], seeds: int, k: int, steps: int):
    """
    Plot Value Error vs. Reward Support Size using end-of-shift L1.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    p_line = get_palette("line")
    label_to_key = {
        "Standard-PS": "standard_ps",
        "GTA-Base":    "gta_base",
        "GTA-Beta":    "gta_beta"
    }
    
    support_sizes = sorted(stats.keys())
    labels = list(next(iter(stats.values())).keys())
    
    for i, label in enumerate(labels):
        key = label_to_key.get(label, "standard_ps")
        means = [stats[s][label]['mean'] for s in support_sizes]
        cis = [stats[s][label]['ci95'] for s in support_sizes]
        
        ax.errorbar(support_sizes, means, yerr=cis, marker='o', capsize=5, 
                     label=label, color=p_line[key], linewidth=1.5)
    
    ax.set_title(f"GARNET Ablation (N={seeds}, K={k}, T={steps})\nMean L1 Error at Shift End")
    ax.set_xlabel("Reward Support Size (Number of non-zero reward states)")
    ax.set_ylabel(r"Final $L_1$ Error")
    ax.set_xscale('log', base=2)
    ax.set_xticks(support_sizes)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = ROOT / "plots" / "reward_support_ablation.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

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
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--K", type=int, default=20)

    parser.add_argument("--n-shifts", type=int, default=5)
    parser.add_argument("--n-steps-per-shift", type=int, default=40)
    parser.add_argument("--n-states", type=int, default=60)
    args = parser.parse_args()

    # Load configuration
    import yaml
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        root_defaults = {k: cfg.get(k) for k in ["alpha", "beta", "beta1", "beta2", "eta", "b"] if cfg.get(k) is not None}
        default_block = cfg.get("default") or {}
        base_params = {**root_defaults, **default_block}
        tuned = cfg.get("tuned") or {}
        method_cfgs = {}
        for variant in ("gta_base", "gta_beta"):
            variant_cfg = {k: v for k, v in base_params.items() if v is not None}
            overrides = {k: v for k, v in (tuned.get(variant) or {}).items() if v is not None}
            method_cfgs[variant] = {**variant_cfg, **overrides}
    else:
        method_cfgs = load_tuned_for_env("garnet")

    print(f"Starting Reward Support Ablation...")
    
    variants = [
        {"label": "Standard-PS", "variant": "standard_ps", "kwargs": {}},
        {"label": "GTA-Base",    "variant": "gta_base",    "kwargs": method_cfgs.get("gta_base", {})},
        {"label": "GTA-Beta",    "variant": "gta_beta",    "kwargs": method_cfgs.get("gta_beta", {})},
    ]
    variant_labels = [v['label'] for v in variants]
    
    n_seeds = args.seeds
    support_points = [1, 2, 4, 8, 16, 32]
    runner = RunManager(n_workers=args.workers)
    
    final_stats = {}
    for support in support_points:
        print(f"  Testing Support Size: {support}")
        seed_results = runner.run(
            fn=run_seed_worker,
            seeds=list(range(n_seeds)),
            fn_kwargs=dict(
                support=support, n_states=args.n_states,
                variants=variants, n_shifts=args.n_shifts,
                n_steps_per_shift=args.n_steps_per_shift, k_planning=args.K
            ),
            desc=f"supp_{support}"
        )
        final_stats[support] = get_stats(seed_results, variant_labels)
        
    # Plot
    os.makedirs(ROOT / "plots", exist_ok=True)
    plot_ablation(final_stats, args.seeds, args.K, args.n_steps_per_shift)
    
    # Save JSON
    results_dir = ROOT / "results"
    os.makedirs(results_dir, exist_ok=True)
    output_data = {
        "experiment": "reward_support_ablation",
        "config": vars(args),
        "results": {str(k): v for k, v in final_stats.items()}
    }
    with open(results_dir / "reward_support_ablation.json", "w") as f:
        json.dump(output_data, f, indent=2)
    
    print("\nReward Support Ablation Complete.")
