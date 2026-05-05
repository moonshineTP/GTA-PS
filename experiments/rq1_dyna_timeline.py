"""
RQ1 Dyna-Timeline: per-action-step value error under trajectory constraints.

In this experiment, a "host step" is a physical action in the environment.
The agent moves cell-by-cell, which poses a significant hindrance to 
information propagation compared to the global DP setting.
"""
from __future__ import annotations

import json
import os
import sys
import argparse
from pathlib import Path

# Pin BLAS threads
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

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

from gta_ps.envs.fourrooms import FourRoomsTabular
from gta_ps.algorithms import make_agent
from gta_ps.utils.metrics import REGISTRY
from runner import RunManager
from plot_academic import (
    LABELS as AGENT_LABELS,
    apply_style,
    get_palette,
    get_utility_color,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VARIANTS = ["standard_ps", "gta_base", "gta_beta"]
DEFAULT_METRIC_NAMES = ["l1", "inf"]

# ---------------------------------------------------------------------------
# Config Helpers (Duplicated to avoid import issues)
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        raise SystemExit("PyYAML required: pip install pyyaml")
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

# ---------------------------------------------------------------------------
# Gaussian-mixture reward generator (reuse from rq1_timeline)
# ---------------------------------------------------------------------------


def _module_kwargs(variant: str, method_cfgs: dict | None, arnoldi_k: int) -> dict:
    kw: dict = {"arnoldi_k": arnoldi_k}
    if method_cfgs and variant in method_cfgs:
        kw.update({k: v for k, v in method_cfgs[variant].items() if v is not None})
    return kw

# ---------------------------------------------------------------------------
# Per-seed worker
# ---------------------------------------------------------------------------

def _run_seed(
    seed: int,
    grid_size: int,
    k: int,
    gamma: float,
    K_budgets: list,
    n_shifts: int,
    n_steps_per_shift: int,
    tail_steps: int,
    arnoldi_k: int,
    method_cfgs: dict | None,
    room_peak_ranges: list,
    n_peaks_per_room: int,
    sigma_range: list,
    metric_names: list[str],
) -> dict:
    env = FourRoomsTabular(
        k=k, gamma=gamma, seed=seed, room_size=grid_size,
        n_peaks_per_room=n_peaks_per_room,
        sigma_range=sigma_range,
        room_peak_ranges=room_peak_ranges,
    )
    sim_rng = np.random.default_rng(seed + 0xABCD_1234)  # trajectory only
    metrics = [REGISTRY[m] for m in metric_names]
    variant_index = {v: i for i, v in enumerate(VARIANTS)}

    agents = {
        (K, v): make_agent(
            v, host="dyna", P=env.P, r=env.r, gamma=gamma, q=20,
            rng=np.random.default_rng(np.random.SeedSequence([seed, K, variant_index[v], 0xD1A])),
            **_module_kwargs(v, method_cfgs, arnoldi_k)
        )
        for K in K_budgets
        for v in VARIANTS
    }

    traces: dict = {f"{v}_K{K}_{m.name}": [] for v in VARIANTS for K in K_budgets for m in metrics}
    shift_steps: list[int] = []
    r_new = env.r.copy()
    V_star = env.value_iteration(r_new)

    # Start all compared agents from the same state so early-step differences
    # come from the planner, not from mismatched initial conditions.
    start_state = int(sim_rng.integers(env.n_states))
    current_states = {(K, v): start_state for K in K_budgets for v in VARIANTS}

    # -- Shift phase --
    for shift_i in range(n_shifts):
        r_new = env.shift_reward()
        V_star = env.value_iteration(r_new)
        shift_steps.append(len(traces[f"standard_ps_K{K_budgets[0]}_{metrics[0].name}"]))
        for agent in agents.values():
            agent.reward_update(r_new)

        for step in range(n_steps_per_shift):
            for K in K_budgets:
                for v in VARIANTS:
                    agent = agents[(K, v)]
                    s = current_states[(K, v)]
                    
                    # 1. Choose action (epsilon-greedy)
                    if sim_rng.random() < 0.1:
                        a = sim_rng.integers(env.n_actions)
                    else:
                        # Use the host Q-table for behavior; module.V is only the planner's
                        # internal cache and can lag behind on unvisited states.
                        a = int(agent.Q[:, s].argmax())
                    
                    # 2. Step environment
                    probs = env.P[a, s]
                    s_next = int(sim_rng.choice(len(probs), p=probs))
                    # FourRoomsTabular uses state-based rewards r[s].
                    r_env = float(r_new[s])
                    current_states[(K, v)] = s_next
                    
                    # 3. Dyna Process: real step -> planning steps
                    agent.real_step(s, a, r_env, s_next)
                    agent.plan_steps(K=K)
                    
                    # Estimate V from Q for metric collection
                    V = agent.Q.max(axis=0)
                    for m in metrics:
                        traces[f"{v}_K{K}_{m.name}"].append(m(V, V_star))

    # -- Tail phase --
    # Use the last V_star
    for _ in range(tail_steps):
        for K in K_budgets:
            for v in VARIANTS:
                agent = agents[(K, v)]
                s = current_states[(K, v)]
                
                # 1. Choose action (epsilon-greedy)
                if sim_rng.random() < 0.1:
                    a = sim_rng.integers(env.n_actions)
                else:
                    a = int(agent.Q[:, s].argmax())
                
                # 2. Step environment
                probs = env.P[a, s]
                s_next = int(sim_rng.choice(len(probs), p=probs))
                # FourRoomsTabular uses state-based rewards r[s].
                r_env = float(r_new[s]) # Still uses last r_new
                current_states[(K, v)] = s_next
                
                # 3. Dyna Process
                agent.real_step(s, a, r_env, s_next)
                agent.plan_steps(K=K)
                
                V = agent.Q.max(axis=0)
                for m in metrics:
                    traces[f"{v}_K{K}_{m.name}"].append(m(V, V_star))

    return {"traces": traces, "shift_steps": shift_steps}

# ---------------------------------------------------------------------------
# Aggregation & Plotting (Shared logic)
# ---------------------------------------------------------------------------

def _aggregate(results):
    from scipy import stats
    keys = list(results[0]["traces"].keys())
    agg = {"shift_steps": results[0]["shift_steps"], "traces": {}}
    for key in keys:
        mat = np.array([r["traces"][key] for r in results])
        n = mat.shape[0]
        mean = mat.mean(axis=0)
        se = stats.sem(mat, axis=0)
        ci95 = (se * stats.t.ppf(0.975, df=n-1)) if n > 1 else np.zeros_like(mean)
        agg["traces"][key] = {"mean": mean.tolist(), "ci95": ci95.tolist()}
    return agg

def _plot(agg_by_gs, grid_sizes, n_states_list, K_budgets, metric_names, n_shift_steps, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    apply_style()
    p_line = get_palette("line")
    bg_color = get_utility_color("bg_gray")
    grid_color = get_utility_color("grid")
    text_color = get_utility_color("text_main")
    
    metrics = [REGISTRY[m] for m in metric_names]
    nrows, ncols = len(grid_sizes), len(K_budgets)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.0 * nrows), sharex=False, sharey=False)
    if nrows == 1: axes = axes[np.newaxis, :]
    if ncols == 1: axes = axes[:, np.newaxis]
    _STYLES = ["-", "--", ":", "-."]

    for ri, (gs, n_s) in enumerate(zip(grid_sizes, n_states_list)):
        agg = agg_by_gs[f"gs{gs}"]
        shift_steps = agg["shift_steps"]
        T = len(agg["traces"][list(agg["traces"].keys())[0]]["mean"])
        steps = np.arange(T)

        for ci, K in enumerate(K_budgets):
            ax = axes[ri, ci]
            ax.axvspan(n_shift_steps, T, color=bg_color, alpha=0.8, zorder=0)
            for ss in shift_steps:
                ax.axvline(ss, color=grid_color, linewidth=0.75, linestyle=":", alpha=0.9, zorder=1)

            for v in VARIANTS:
                for mi, m in enumerate(metrics):
                    key = f"{v}_K{K}_{m.name}"
                    mean = np.array(agg["traces"][key]["mean"])
                    ci95 = np.array(agg["traces"][key]["ci95"])
                    color = p_line.get(v, text_color)
                    ax.plot(steps, mean, color=color, linewidth=1.6, linestyle=_STYLES[mi % len(_STYLES)], zorder=3)
                    ax.fill_between(steps, mean - ci95, mean + ci95, color=color, alpha=0.1, zorder=2)

            ax.set_yscale("log")
            ax.set_title(f"|S| = {n_s},  K = {K}", fontsize=9)
            ax.set_xlabel("Environment Steps")
            if ci == 0: ax.set_ylabel("Value Error (Dyna Setting)")
            ax.grid(True, axis="y", color=grid_color, alpha=0.25, linewidth=0.75)

    # Add shared legend
    legend_handles: list[object] = [
        mlines.Line2D([], [], color=p_line.get(v, text_color), linewidth=2, label=AGENT_LABELS.get(v, v))
        for v in VARIANTS
    ]
    for mi, m in enumerate(metrics):
        legend_handles.append(
            mlines.Line2D([], [], color=text_color, linestyle=_STYLES[mi % len(_STYLES)], linewidth=2, label=m.label)
        )
    legend_handles.append(mpatches.Patch(facecolor=bg_color, edgecolor=grid_color, label="Post-shift tail"))

    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=9)
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=300, bbox_inches='tight')

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fourrooms_k2_timeline.yaml")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--n-steps", type=int, help="Override steps per shift")
    parser.add_argument("--n-shifts", type=int, help="Override number of shifts")
    parser.add_argument("--tail-steps", type=int, help="Override tail steps")
    args = parser.parse_args()

    # Load config
    cfg = _load_config(ROOT / args.config)
    method_cfgs = _build_method_cfgs(cfg)

    k = cfg.get("k_fourrooms", 2)
    gamma = cfg.get("gamma", 0.95)
    grid_sizes = list(cfg["grid_sizes"])
    K_budgets = list(cfg["budgets"])
    n_shifts = args.n_shifts if args.n_shifts is not None else cfg.get("n_shifts", 6)
    n_steps_shift = args.n_steps if args.n_steps is not None else cfg.get("n_steps_per_shift", 100)
    tail_steps = args.tail_steps if args.tail_steps is not None else cfg.get("tail_steps", 240)
    seeds = args.seeds

    runner = RunManager(n_workers=args.workers)
    agg_by_gs = {}
    n_states_list = []

    for gs in grid_sizes:
        room_size = (gs - 3) // 2
        n_s = FourRoomsTabular(room_size=room_size).n_states
        n_states_list.append(n_s)
        print(f"Starting experiment for gs={gs} (room={room_size}) (|S|={n_s})...", flush=True)

        seed_results = runner.run(
            fn=_run_seed,
            seeds=[s for s in range(seeds)],
            fn_kwargs=dict(
                grid_size=room_size, k=k, gamma=gamma, K_budgets=K_budgets,
                n_shifts=n_shifts, n_steps_per_shift=n_steps_shift,
                tail_steps=tail_steps, arnoldi_k=cfg.get("arnoldi_k", 10),
                method_cfgs=method_cfgs, room_peak_ranges=cfg.get("room_peak_ranges"),
                n_peaks_per_room=cfg.get("n_peaks_per_room", 2),
                sigma_range=cfg.get("sigma_range", [0.8, 3.0]),
                metric_names=DEFAULT_METRIC_NAMES,
            ),
            desc=f"gs{gs}"
        )
        agg_by_gs[f"gs{gs}"] = _aggregate(seed_results)

    tag = f"rq1_dyna_timeline_k{k}_" + "_".join(str(g) for g in grid_sizes)
    out_dir = ROOT / "results"
    with open(out_dir / f"{tag}.json", "w") as f:
        json.dump({"experiment": "rq1_dyna_timeline", "config": cfg, "results": agg_by_gs}, f, indent=2)

    out_plot = ROOT / "plots" / f"{tag}.png"
    _plot(agg_by_gs, grid_sizes, n_states_list, K_budgets, DEFAULT_METRIC_NAMES, n_shifts * n_steps_shift, str(out_plot))
    print(f"Dyna Timeline experiment complete. Plot -> {out_plot}")

if __name__ == "__main__":
    main()
