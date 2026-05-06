"""
RQ5: Bottleneck and SCC Geometry (Refined Corridor Sweep).

Mission:
Investigate how soft geometric bottlenecks (Mean First-Passage Time barriers)
affect GTA-PS performance. The bottleneck is explicitly defined by the 
length of one-cell wide corridors connecting rooms.

Setup:
- Metric: L1 Value Error at shift-end.
- Environments: 
    - GARNET: n_nbr_rate sweep (localized vs. global connectivity).
    - FourRooms: Corridor length sweep (MFPT barrier scaling).
- Agent: Dyna host (count-based on GARNET, sample-based on FourRooms).
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
    get_utility_color,
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
    result  = {}
    for variant in ("gta_base", "gta_beta"):
        variant_cfg = {k: v for k, v in base_params.items() if v is not None}
        overrides = {k: v for k, v in (tuned.get(variant) or {}).items() if v is not None}
        result[variant] = {**variant_cfg, **overrides}
    return result

def run_seed_worker(env_name: str, env_params: dict, variants: List[dict], n_shifts: int, n_steps_per_shift: int, k_planning: int, seed: int):
    if env_name == "garnet":
        env = GarnetEnv(**env_params, seed=seed)
        model_type = "count"
    else:
        env = FourRoomsTabular(**env_params, seed=seed)
        model_type = "sample"
    sim_rng = np.random.default_rng(np.random.SeedSequence([seed, 0xB07, 5]))
        
    n_states = env.n_states
    
    seed_results = {}
    for idx, var in enumerate(variants):
        agent = make_agent(
            variant=var['variant'],
            host="dyna",
            P=env.P,
            r=env.r,
            gamma=env.gamma,
            model_type=model_type,
            rng=np.random.default_rng(np.random.SeedSequence([seed, idx, 0xB07])),
            **var.get('kwargs', {})
        )
        
        shift_end_errors = []
        for shift in range(n_shifts):
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
            
        seed_results[var['label']] = np.mean(shift_end_errors)
    return seed_results

def plot_rq5_schema(garnet_stats: Dict[float, Dict[str, dict]], fourrooms_stats: Dict[int, Dict[str, dict]], seeds: int, k: int, steps: int):
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = get_palette("point")
    grid_color = get_utility_color("grid")
    # Robust mapping from display label back to palette keys
    label_to_key = {
        AGENT_LABELS["standard_ps"]: "standard_ps",
        AGENT_LABELS["gta_beta"]:    "gta_beta",
        AGENT_LABELS["full_gta"]:    "full_gta",
    }
    
    # --- 1. GARNET Plot ---
    nbr_rates = sorted(garnet_stats.keys(), reverse=True)
    labels = list(next(iter(garnet_stats.values())).keys())
    for i, label in enumerate(labels):
        key = label_to_key.get(label, "standard_ps")
        means = [garnet_stats[rate][label]['mean'] for rate in nbr_rates]
        cis = [garnet_stats[rate][label]['ci95'] for rate in nbr_rates]
        ax1.errorbar(nbr_rates, means, yerr=cis, marker='o', capsize=4, label=label, color=colors[key])
    
    ax1.set_title("GARNET Connectivity Sweep")
    ax1.set_xlabel("n_nbr_rate (1.0=Global, 0.05=Localized)")
    ax1.set_ylabel(r"Final $L_1$ Error")
    ax1.invert_xaxis()
    ax1.legend()
    ax1.grid(True, alpha=0.3, color=grid_color)
    
    # --- 2. FourRooms Plot ---
    corridor_lens = sorted(fourrooms_stats.keys())
    for i, label in enumerate(labels):
        key = label_to_key.get(label, "standard_ps")
        means = [fourrooms_stats[cl][label]['mean'] for cl in corridor_lens]
        cis = [fourrooms_stats[cl][label]['ci95'] for cl in corridor_lens]
        ax2.errorbar(corridor_lens, means, yerr=cis, marker='s', capsize=4, label=label, color=colors[key])
        
    ax2.set_title("FourRooms Corridor Sweep")
    ax2.set_xlabel("Corridor Length (Number of cells between rooms)")
    ax2.set_ylabel(r"Final $L_1$ Error")
    ax2.set_xticks(corridor_lens)
    
    all_means = [fourrooms_stats[cl][l]['mean'] for cl in corridor_lens for l in labels]
    if all_means:
        ax2.set_ylim(min(all_means) * 0.8, max(all_means) * 1.2)
    
    ax2.legend()
    ax2.grid(True, alpha=0.3, color=grid_color)
    
    plt.tight_layout()
    plot_path = ROOT / "plots" / "rq5_bottlenecks_corridor.png"
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
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--n-shifts", type=int, default=4)
    parser.add_argument("--n-steps-per-shift", type=int, default=40)
    parser.add_argument("--gamma", type=float, default=0.95)
    args = parser.parse_args()

    if args.config:
        method_cfgs = _build_method_cfgs(_load_config(args.config))
    else:
        method_cfgs = load_tuned_for_env("garnet")

    print(f"Starting RQ5 Corridor Sweep...")
    
    variants = [
        {"label": AGENT_LABELS["standard_ps"], "variant": "standard_ps", "kwargs": {}},
        {"label": AGENT_LABELS["gta_beta"],    "variant": "gta_beta",    "kwargs": method_cfgs.get("gta_beta", {})},
    ]
    variant_labels = [v['label'] for v in variants]
    runner = RunManager(n_workers=args.workers)
    
    # 1. GARNET
    print(f"Running GARNET sweep...")
    nbr_rates = [0.8, 0.4, 0.2, 0.1, 0.05]
    garnet_final_stats = {}
    for rate in nbr_rates:
        seed_results = runner.run(fn=run_seed_worker, seeds=list(range(args.seeds)), 
                                  fn_kwargs=dict(env_name="garnet", env_params={"n_states": 75, "n_nbr_rate": rate, "reversible": 1.0, "gamma": args.gamma},
                                                 variants=variants, n_shifts=args.n_shifts, n_steps_per_shift=args.n_steps_per_shift, k_planning=args.K),
                                  desc=f"garnet_{rate}")
        garnet_final_stats[rate] = get_stats(seed_results, variant_labels)
        
    # 2. FourRooms (Corridor Sweep)
    print("Running FourRooms Corridor sweep...")
    fourrooms_final_stats = {}

    # Load corridor lens from config if provided, else default
    corridor_lens = [1, 3, 5, 7]
    room_size = 3
    if args.config:
        cfg = _load_config(args.config)
        corridor_lens = cfg.get("corridor_lens", corridor_lens)
        room_size = cfg.get("room_size", room_size)

    for cl in corridor_lens:
        print(f"  Corridor Length: {cl}")
        seed_results = runner.run(
            fn=run_seed_worker,
            seeds=list(range(args.seeds)),
            fn_kwargs=dict(
                env_name="fourrooms", 
                env_params={"room_size": room_size, "corridor_len": cl, "k": 1, "gamma": args.gamma},
                variants=variants, n_shifts=args.n_shifts, 
                n_steps_per_shift=args.n_steps_per_shift, k_planning=args.K
            ),
            desc=f"fr_cl_{cl}"
        )
        fourrooms_final_stats[cl] = get_stats(seed_results, variant_labels)

    
    plot_rq5_schema(garnet_final_stats, fourrooms_final_stats, args.seeds, args.K, args.n_steps_per_shift)
    
    # Save JSON
    os.makedirs(ROOT / "results", exist_ok=True)
    with open(ROOT / "results" / "rq5_bottlenecks_corridor.json", "w") as f:
        json.dump({"garnet": {str(k): v for k, v in garnet_final_stats.items()}, 
                   "fourrooms": {str(k): v for k, v in fourrooms_final_stats.items()}, "config": vars(args)}, f, indent=2)
    
    print("\nRQ5 Refinement Complete.")
