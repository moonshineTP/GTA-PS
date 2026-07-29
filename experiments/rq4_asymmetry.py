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

FOURROOMS_BELT_LEAK = 0.05


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

    if "gta_beta" not in tuned:
        raise ValueError("RQ4 expects tuned.gta_beta when using legacy method names.")
    else:
        # Forward-GTA is the full GTA-Beta family with alpha fixed to zero.
        forward_cfg = dict(tuned.get("gta_beta") or {})
        full_cfg = dict(tuned.get("gta_beta") or {})

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

def _build_variants(method_cfgs: dict) -> List[dict]:
    return [
        {"label": AGENT_LABELS["standard_ps"], "variant": "standard_ps", "kwargs": {}},
        {"label": AGENT_LABELS["forward_gta"], "variant": "forward_gta", "kwargs": method_cfgs.get("forward_gta", {"alpha": 0.0})},
        {"label": AGENT_LABELS["full_gta"], "variant": "full_gta", "kwargs": method_cfgs.get("full_gta", {})},
    ]

def _resolve_env_method_cfgs(config_path: str | None) -> tuple[dict, dict]:
    if config_path:
        method_cfgs = _build_method_cfgs(_load_config(config_path))
        return method_cfgs, method_cfgs
    return (
        _build_method_cfgs(_load_default_env_config("garnet")),
        _build_method_cfgs(_load_default_env_config("fourrooms")),
    )


def _build_fourrooms_base_kwargs(cfg: dict) -> dict:
    base = {}
    direct_keys = (
        "room_size",
        "corridor_len",
        "corridor_width",
        "window_offset",
        "n_peaks_per_room",
        "sigma_range",
        "room_peak_ranges",
    )
    for key in direct_keys:
        value = cfg.get(key)
        if value is not None:
            base[key] = value
    if cfg.get("k_fourrooms") is not None:
        base["k"] = cfg["k_fourrooms"]
    elif cfg.get("k") is not None:
        base["k"] = cfg["k"]
    return base


def _resolve_fourrooms_base_kwargs(config_path: str | None) -> dict:
    base_kwargs = _build_fourrooms_base_kwargs(_load_default_env_config("fourrooms"))
    if config_path:
        base_kwargs.update(_build_fourrooms_base_kwargs(_load_config(config_path)))
    return base_kwargs


def _make_fourrooms_env(*, seed: int, gamma: float, env_kwargs: dict) -> FourRoomsTabular:
    return FourRoomsTabular(seed=seed, gamma=gamma, **env_kwargs)


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
        delta = _fourrooms_transition_delta(
            gamma,
            base_kwargs,
            {"drift": (0, 1), "drift_strength": mid},
        )
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
        ("One-Way", {"one_way_prob": FOURROOMS_BELT_LEAK}),
        ("Windy", {"drift": (0, 1), "drift_strength": matched_drift_strength}),
    ]


def _fourrooms_mode_diagnostics(gamma: float, base_kwargs: dict, modes: List[tuple[str, dict]]) -> dict:
    symmetric_env = _make_fourrooms_env(seed=0, gamma=gamma, env_kwargs=base_kwargs)
    symmetric_P = symmetric_env.P.copy()
    diagnostics = {}
    kernels = {"Symmetric": symmetric_P}

    for mode_name, mode_kwargs in modes:
        env_kwargs = dict(base_kwargs)
        env_kwargs.update(mode_kwargs)
        env = _make_fourrooms_env(seed=0, gamma=gamma, env_kwargs=env_kwargs)
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
            host="dp",
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

    garnet_method_cfgs, fourrooms_method_cfgs = _resolve_env_method_cfgs(args.config)

    print(f"Starting RQ4 Consistency Sweep...")
    print("Resolved method configs:")
    print("  GARNET:", json.dumps(garnet_method_cfgs, indent=2, sort_keys=True))
    print("  FourRooms:", json.dumps(fourrooms_method_cfgs, indent=2, sort_keys=True))

    fourrooms_base_kwargs = _resolve_fourrooms_base_kwargs(args.config)
    fourrooms_modes = _resolve_fourrooms_modes(args.gamma, fourrooms_base_kwargs)
    print("Resolved FourRooms base kwargs:")
    print(json.dumps(fourrooms_base_kwargs, indent=2, sort_keys=True))

    garnet_variants = _build_variants(garnet_method_cfgs)
    fourrooms_variants = _build_variants(fourrooms_method_cfgs)
    variant_labels = [v['label'] for v in garnet_variants]
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
                variants=garnet_variants, n_shifts=args.n_shifts, 
                n_steps_per_shift=args.n_steps_per_shift, k_planning=args.K
            ),
            desc=f"garnet_rev_{rev}"
        )
        garnet_final_stats[rev] = get_stats(seed_results, variant_labels)
        
    # 2. FourRooms
    print(f"Running FourRooms sweep...")
    fourrooms_final_stats = {}
    modes = fourrooms_modes
    fourrooms_diagnostics = _fourrooms_mode_diagnostics(args.gamma, fourrooms_base_kwargs, modes)
    print("FourRooms transport diagnostics:")
    print(json.dumps(fourrooms_diagnostics, indent=2, sort_keys=True))
    for mode_name, params in modes:
        p_copy = dict(fourrooms_base_kwargs)
        p_copy.update(params)
        p_copy["gamma"] = args.gamma
        seed_results = runner.run(
            fn=run_seed_worker,
            seeds=list(range(args.seeds)),
            fn_kwargs=dict(
                env_name="fourrooms", env_params=p_copy,
                variants=fourrooms_variants, n_shifts=args.n_shifts,
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
                "garnet_method_variants": [
                    {"label": variant["label"], "variant": variant["variant"], "kwargs": variant["kwargs"]}
                    for variant in garnet_variants
                ],
                "fourrooms_method_variants": [
                    {"label": variant["label"], "variant": variant["variant"], "kwargs": variant["kwargs"]}
                    for variant in fourrooms_variants
                ],
                "fourrooms_base_kwargs": fourrooms_base_kwargs,
                "fourrooms_modes": [{"label": label, "kwargs": kwargs} for label, kwargs in modes],
                "fourrooms_transport_diagnostics": fourrooms_diagnostics,
            },
            "results": {"garnet": {str(k): v for k, v in garnet_final_stats.items()}, "fourrooms": fourrooms_final_stats}
        }, f, indent=2)
    
    print("\nRQ4 Consistency Sweep Complete.")
