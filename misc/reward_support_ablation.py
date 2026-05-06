"""
GARNET Ablation: Impact of Reward Support Size on Planning Efficiency.

Mission:
Quantify how the density of reward attractors (support size) affects the 
relative advantage of GTA-augmented priority signals over Standard-PS.
Measures convergence consistency at the end of each reward shift.

Setup:
- Environment: GARNET (|S|=75 default)
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
from gta_ps.utils.hyperparams import ENV_DEFAULT_CONFIG
from runner import RunManager
from plot_academic import apply_style, get_palette

def _load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}

def _load_default_env_config(env_name: str) -> dict:
    filename = ENV_DEFAULT_CONFIG.get(env_name)
    if filename is None:
        raise ValueError(f"Reward support ablation has no default config mapping for env '{env_name}'.")
    cfg_path = ROOT / "configs" / filename
    if not cfg_path.exists():
        raise FileNotFoundError(f"Reward support ablation expected default config at {cfg_path}.")
    return _load_config(str(cfg_path))

def _resolve_tuned_variants(tuned: dict) -> dict:
    if not tuned:
        return {"forward_gta": {}, "full_gta": {}}

    if "forward_gta" in tuned or "full_gta" in tuned:
        if "forward_gta" not in tuned or "full_gta" not in tuned:
            raise ValueError("Reward support ablation expects both tuned.forward_gta and tuned.full_gta when using explicit method names.")
        forward_cfg = dict(tuned.get("forward_gta") or {})
        full_cfg = dict(tuned.get("full_gta") or {})
    elif "gta_base" in tuned or "gta_beta" in tuned:
        if "gta_beta" not in tuned:
            raise ValueError("Reward support ablation expects tuned.gta_beta when using legacy method names.")
        # Forward-GTA is the full GTA-Beta family with alpha fixed to zero.
        forward_cfg = dict(tuned.get("gta_beta") or {})
        full_cfg = dict(tuned.get("gta_beta") or {})
    else:
        raise ValueError("Reward support ablation expected tuned.forward_gta/full_gta or tuned.gta_base/gta_beta.")

    forward_cfg["alpha"] = 0.0
    return {
        "forward_gta": forward_cfg,
        "full_gta": full_cfg,
    }

def _build_method_cfgs(cfg: dict) -> dict:
    root_defaults = {k: cfg.get(k) for k in ["alpha", "beta", "beta1", "beta2", "eta", "b"] if cfg.get(k) is not None}
    default_block = cfg.get("default") or {}
    base_params = {**root_defaults, **default_block}
    tuned = _resolve_tuned_variants(cfg.get("tuned") or {})
    result = {}
    for variant in ("forward_gta", "full_gta"):
        variant_cfg = {k: v for k, v in base_params.items() if v is not None}
        overrides = {k: v for k, v in (tuned.get(variant) or {}).items() if v is not None}
        result[variant] = {**variant_cfg, **overrides}
    result["forward_gta"]["alpha"] = 0.0
    return result

def run_seed_worker(support: int, n_states: int, variants: List[dict], n_shifts: int, n_steps_per_shift: int, k_planning: int, seed: int):
    """
    Worker function for RunManager.
    """
    env = GarnetEnv(n_states=n_states, reward_support=support, seed=seed)
    n_actions = env.n_actions
    sim_rng = np.random.default_rng(np.random.SeedSequence([seed, support, 0x5EED]))
    
    seed_results = {}
    for idx, var in enumerate(variants):
        agent_rng = np.random.default_rng(np.random.SeedSequence([seed, support, idx, 0x5EED]))
        agent = make_agent(
            variant=var['variant'],
            host="dp",
            P=env.P,
            r=env.r,
            gamma=env.gamma,
            model_type="count",
            rng=agent_rng,
            **var.get('kwargs', {})
        )
        
        shift_end_errors = []
        for shift in range(n_shifts):
            r_new = env.shift_reward()
            V_star = env.value_iteration(r_new)
            
            for step in range(n_steps_per_shift):
                # DP agent planning with reward update on first step
                if step == 0:
                    agent.plan(K=k_planning, r=r_new)
                else:
                    agent.plan(K=k_planning)
                
                # If last step of shift, record error
                if step == n_steps_per_shift - 1:
                    V_after = agent.V
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
    
    p_point = get_palette("point")
    label_to_key = {
        "Standard-PS": "standard_ps",
        "Forward-GTA": "forward_gta",
        "Full-GTA":    "full_gta"
    }
    
    support_sizes = sorted(stats.keys())
    labels = list(next(iter(stats.values())).keys())
    
    for i, label in enumerate(labels):
        key = label_to_key.get(label, "standard_ps")
        means = [stats[s][label]['mean'] for s in support_sizes]
        cis = [stats[s][label]['ci95'] for s in support_sizes]
        
        ax.errorbar(support_sizes, means, yerr=cis, marker='o', capsize=5, 
                     label=label, color=p_point[key], linewidth=1.5)
    
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

    if args.config:
        method_cfgs = _build_method_cfgs(_load_config(args.config))
    else:
        method_cfgs = _build_method_cfgs(_load_default_env_config("garnet"))

    print(f"Starting Reward Support Ablation...")
    
    variants = [
        {"label": "Standard-PS", "variant": "standard_ps", "kwargs": {}},
        {"label": "Forward-GTA", "variant": "forward_gta", "kwargs": method_cfgs.get("forward_gta", {"alpha": 0.0})},
        {"label": "Full-GTA",    "variant": "full_gta",    "kwargs": method_cfgs.get("full_gta", {})},
    ]
    variant_labels = [v['label'] for v in variants]
    
    n_seeds = args.seeds
    support_points = [1, 2, 4, 8, 16]
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
