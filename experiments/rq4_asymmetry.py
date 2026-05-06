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
from gta_ps.utils.hyperparams import ENV_DEFAULT_CONFIG
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

def _load_default_env_config(env_name: str) -> dict:
    filename = ENV_DEFAULT_CONFIG.get(env_name)
    if filename is None:
        raise ValueError(f"RQ4 has no default config mapping for env '{env_name}'.")
    cfg_path = ROOT / "configs" / filename
    if not cfg_path.exists():
        raise FileNotFoundError(f"RQ4 expected default config at {cfg_path}.")
    return _load_config(str(cfg_path))

def _resolve_tuned_variants(tuned: dict) -> dict:
    if not tuned:
        return {"forward_gta": {}, "full_gta": {}}

    if "forward_gta" in tuned or "full_gta" in tuned:
        if "forward_gta" not in tuned or "full_gta" not in tuned:
            raise ValueError("RQ4 expects both tuned.forward_gta and tuned.full_gta when using explicit method names.")
        forward_cfg = dict(tuned.get("forward_gta") or {})
        full_cfg = dict(tuned.get("full_gta") or {})
    else:
        forward_cfg = None
        full_cfg = None
        for cfg in tuned.values():
            cfg_dict = dict(cfg or {})
            is_full = any(cfg_dict.get(key) is not None for key in ("beta1", "beta2")) or bool(cfg_dict.get("adaptive_beta"))
            is_forward = cfg_dict.get("beta") is not None and not is_full
            if is_full:
                if full_cfg is not None:
                    raise ValueError("RQ4 could not infer a unique Full-GTA tuned block.")
                full_cfg = cfg_dict
            elif is_forward:
                if forward_cfg is not None:
                    raise ValueError("RQ4 could not infer a unique Forward-GTA tuned block.")
                forward_cfg = cfg_dict
        if forward_cfg is None or full_cfg is None:
            raise ValueError("RQ4 could not infer Forward-GTA / Full-GTA tuned blocks from the config.")

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

def _fourrooms_mode_diagnostics(gamma: float, modes: List[tuple[str, dict]]) -> dict:
    symmetric_env = FourRoomsTabular(gamma=gamma, seed=0, one_way_prob=1.0)
    symmetric_P = symmetric_env.P.copy()
    diagnostics = {}
    kernels = {"Symmetric": symmetric_P}

    for mode_name, mode_kwargs in modes:
        env = FourRoomsTabular(gamma=gamma, seed=0, **mode_kwargs)
        diff = np.abs(env.P - symmetric_P)
        diagnostics[mode_name] = {
            "transport_l1_vs_symmetric": float(diff.sum()),
            "changed_states": int(np.count_nonzero(np.any(diff > 1e-6, axis=(0, 2)))),
        }
        if mode_name != "Symmetric" and diagnostics[mode_name]["transport_l1_vs_symmetric"] <= 1e-6:
            raise RuntimeError(f"FourRooms mode {mode_name!r} did not modify the transition kernel.")
        kernels[mode_name] = env.P.copy()

    pairwise_l1 = {}
    mode_names = [mode_name for mode_name, _ in modes]
    for i, lhs in enumerate(mode_names):
        for rhs in mode_names[i + 1:]:
            pair_key = f"{lhs} vs {rhs}"
            pair_delta = float(np.abs(kernels[lhs] - kernels[rhs]).sum())
            if pair_delta <= 1e-6:
                raise RuntimeError(f"FourRooms modes {lhs!r} and {rhs!r} produced the same transition kernel.")
            pairwise_l1[pair_key] = pair_delta

    diagnostics["pairwise_transport_l1"] = pairwise_l1
    diagnostics["belt_cycle_positions"] = [
        list(symmetric_env._pos_list[state])
        for state in symmetric_env.belt_cycle_states()
    ]
    return diagnostics

def run_seed_worker(env_name: str, env_params: dict, variants: List[dict], n_shifts: int, n_steps_per_shift: int, k_planning: int, seed: int):
    """
    Worker function for RunManager.
    """
    if env_name == "garnet":
        env = GarnetEnv(**env_params, seed=seed)
        model_type = "count"
    else:
        env = FourRoomsTabular(**env_params, seed=seed)
        model_type = "sample"
    sim_rng = np.random.default_rng(np.random.SeedSequence([seed, 0xA54, 4]))
        
    n_states = env.n_states
    n_actions = env.n_actions
    
    seed_results = {}
    for idx, var in enumerate(variants):
        agent = make_agent(
            variant=var['variant'],
            host="dyna",
            P=env.P,
            r=env.r,
            gamma=env.gamma,
            model_type=model_type,
            rng=np.random.default_rng(np.random.SeedSequence([seed, idx, 0xA54])),
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
                    ss = sim_rng.integers(n_states, size=burst_size)
                    aa = sim_rng.integers(n_actions, size=burst_size)
                    sn = np.array([sim_rng.choice(n_states, p=env.P[aa[i], ss[i]]) for i in range(burst_size)])
                    rr = r_new[ss]
                    agent.real_steps_batch(ss, aa, rr, sn)
                else:
                    s_rand = sim_rng.integers(n_states)
                    a_rand = sim_rng.integers(n_actions)
                    s_next_rand = sim_rng.choice(n_states, p=env.P[a_rand, s_rand])
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
    
    label_to_key = {
        AGENT_LABELS["standard_ps"]: "standard_ps",
        AGENT_LABELS["forward_gta"]: "forward_gta",
        AGENT_LABELS["full_gta"]:    "full_gta",
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
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--n-shifts", type=int, default=4)
    parser.add_argument("--n-steps-per-shift", type=int, default=40)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--n-states", type=int, default=75)
    args = parser.parse_args()

    if args.config:
        method_cfgs = _build_method_cfgs(_load_config(args.config))
    else:
        method_cfgs = _build_method_cfgs(_load_default_env_config("garnet"))

    print(f"Starting RQ4 Consistency Sweep...")
    
    variants = [
        {"label": AGENT_LABELS["standard_ps"],  "variant": "standard_ps", "kwargs": {}},
        {"label": AGENT_LABELS["forward_gta"], "variant": "forward_gta", "kwargs": method_cfgs.get("forward_gta", {"alpha": 0.0})},
        {"label": AGENT_LABELS["full_gta"],    "variant": "full_gta", "kwargs": method_cfgs.get("full_gta", {})},
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
    fourrooms_diagnostics = _fourrooms_mode_diagnostics(args.gamma, modes)
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
            "config": {
                **vars(args),
                "method_variants": [
                    {"label": variant["label"], "variant": variant["variant"], "kwargs": variant["kwargs"]}
                    for variant in variants
                ],
                "fourrooms_modes": [{"label": label, "kwargs": kwargs} for label, kwargs in modes],
                "fourrooms_transport_diagnostics": fourrooms_diagnostics,
            },
            "results": {"garnet": {str(k): v for k, v in garnet_final_stats.items()}, "fourrooms": fourrooms_final_stats}
        }, f, indent=2)
    
    print("\nRQ4 Consistency Sweep Complete.")
