"""
Aperiodic Ablation: Impact of Periodicity (Bipartite Structure) on GTA-PS.

Mission:
Investigate if GTA-PS's spectral signal is robust to periodic Induced Markov Chains.
Verifies strict bipartite structure (no same-group transitions) and SLEM=1.0.
"""
import sys
import os
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
from plot_academic import apply_style, get_palette, LABELS as AGENT_LABELS

def get_true_slem(P_avg):
    """Correctly compute SLEM for periodic chains by only skipping the eigenvalue at 1.0."""
    import scipy.linalg as sla
    vals = sla.eigvals(P_avg)
    mods = np.sort(np.abs(vals))[::-1]
    # mods[0] is the stationary 1.0. mods[1] is the SLEM.
    return float(mods[1])

def verify_bipartite(P):
    """Assert that even states only move to odd and vice versa."""
    S = P.shape[1]
    evens = np.arange(0, S, 2)
    odds = np.arange(1, S, 2)
    # Check even -> even transitions
    ee = np.sum(P[:, evens, :][:, :, evens])
    # Check odd -> odd transitions
    oo = np.sum(P[:, odds, :][:, :, odds])
    return (ee + oo) == 0

def run_seed_worker(env_params: dict, variants: List[dict], n_shifts: int, n_steps_per_shift: int, k_planning: int, seed: int):
    mode = env_params.pop("mode", "normal")
    gamma = env_params.get("gamma", 0.95)
    env = GarnetEnv(**env_params, seed=seed)
    
    if mode == "lazy":
        for a in range(env.n_actions):
            env.P[a] = 0.9 * env.P[a] + 0.1 * np.eye(env.n_states)
            
    # Diagnostic
    P_avg = env.P.mean(axis=0)
    slem = get_true_slem(P_avg)
    is_bipartite = verify_bipartite(env.P) if env.bipartite and mode != "lazy" else False

    seed_results = {"slem": slem, "is_bipartite": is_bipartite}
    for var in variants:
        agent = make_agent(variant=var['variant'], host="dp", P=env.P, r=env.r, gamma=gamma, **var['kwargs'])
        shift_end_errors = []
        for shift in range(n_shifts):
            r_new = env.shift_reward()
            V_star = env.value_iteration(r_new)
            for step in range(n_steps_per_shift):
                agent.plan(K=k_planning, r=(r_new if step == 0 else None))
                if step == n_steps_per_shift - 1:
                    V_after = agent.module.V
                    shift_end_errors.append(value_error_l1(V_after, V_star))
        seed_results[var['label']] = np.mean(shift_end_errors)
    return seed_results

def get_stats(seed_results: List[Dict[str, any]], labels: List[str]):
    stats = {"slem": np.mean([r["slem"] for r in seed_results])}
    for label in labels:
        vals = [res[label] for res in seed_results]
        stats[label] = {"mean": np.mean(vals), "ci95": 1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals))}
    return stats

def plot_periodicity(stats: Dict[str, Dict[str, dict]], seeds: int, k: int, n_states: int = 40):
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    p_point = get_palette("point")
    label_to_key = {
        "Standard-PS": "standard_ps",
        "Full-GTA":    "full_gta"
    }
    
    modes = list(stats.keys())
    x = np.arange(len(modes))
    labels = ["Standard-PS", "Full-GTA"]
    offsets = [-0.15, 0.15]
    for i, label in enumerate(labels):
        key = label_to_key.get(label, "standard_ps")
        means = [stats[m][label]['mean'] for m in modes]
        cis = [stats[m][label]['ci95'] for m in modes]
        ax.errorbar(x + offsets[i], means, yerr=cis, fmt='o', capsize=5, label=label, color=p_point[key])
    ax.set_title(f"Impact of Periodicity (N={seeds}, K={k})\nGARNET Bipartite Ablation ($|S|={n_states}$, rev=0.0)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\nSLEM={stats[m]['slem']:.3f}" for m in modes])
    ax.set_ylabel("Final L1 Error")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(ROOT / "plots" / "periodicity_ablation.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", default=3, type=int)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--K", type=int, default=20)
    args = parser.parse_args()

    method_cfgs = load_tuned_for_env("garnet")
    variants = [
        {"label": "Standard-PS", "variant": "standard_ps", "kwargs": {}},
        {"label": "Full-GTA",     "variant": "full_gta",    "kwargs": method_cfgs.get("gta_beta", {})},
    ]
    variant_labels = [v['label'] for v in variants]
    runner = RunManager(n_workers=args.workers)
    
    print(f"Starting Refined Bipartite Ablation (S=40, K={args.K}, seeds={args.seeds})...")
    
    results = {}
    modes = [
        ("Aperiodic", {"n_states": 40, "reversible": 0.0, "n_nbr_rate": 0.15, "bipartite": False, "mode": "normal"}),
        ("Periodic", {"n_states": 40, "reversible": 0.0, "n_nbr_rate": 0.15, "bipartite": True, "mode": "normal"}),
        ("Lazy", {"n_states": 40, "reversible": 0.0, "n_nbr_rate": 0.15, "bipartite": True, "mode": "lazy"}),
    ]
    
    for name, params in modes:
        print(f"  Testing Mode: {name}")
        res = runner.run(run_seed_worker, list(range(args.seeds)),
                         dict(env_params=params, variants=variants, n_shifts=5, 
                              n_steps_per_shift=40, k_planning=args.K),
                         desc=f"{name}")
        results[name] = get_stats(res, variant_labels)
        
    plot_periodicity(results, args.seeds, args.K, n_states=40)
    
    # Save JSON
    os.makedirs(ROOT / "results", exist_ok=True)
    with open(ROOT / "results" / "periodicity_ablation.json", "w") as f:
        json.dump({"experiment": "periodicity_ablation", "results": results, "config": vars(args)}, f, indent=2)
    
    print("\nPeriodicity Ablation Complete.")
    for mode, stats in results.items():
        print(f"\nMode: {mode}")
        print(f"  Topology SLEM: {stats['slem']:.4f}")
        for label in variant_labels:
            m, c = stats[label]['mean'], stats[label]['ci95']
            print(f"    {label}: {m:.4f} +/- {c:.4f}")
