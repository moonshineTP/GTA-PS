"""
RQ1 Timeline: per-host-step value error across reward shifts.

Protocol
--------
- FourRooms tabular MDP, k=2, oracle DP+VI host (DpHost + OracleModel)
- 3 grid sizes: grid sizes 7, 9, 11 (|S| ~ 20, 40, 68)
- K budgets: 10, 50, 100  (Bellman backups per host step)
- 3 PS modules: standard_ps, gta_base, gta_beta
- Gaussian-mixture reward shift protocol (25 steps/shift, 10 shifts)
- Mean ± 95% CI over N seeds
"""
from __future__ import annotations

import json
import os
import sys
import argparse
from pathlib import Path

import numpy as np

# Pin BLAS threads before numpy import in each worker
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

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
from gta_ps.utils.metrics import REGISTRY, ValueMetric
from runner import RunManager
from plot_academic import (
    LABELS as AGENT_LABELS,
    apply_style,
    get_palette,
    get_publication_defaults,
    get_utility_color,
    warn_if_nonpublication_run,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VARIANTS = ["standard_ps", "gta_base", "gta_beta"]
DEFAULT_METRIC_NAMES = ["l1", "inf"]
PUBLICATION_DEFAULTS = get_publication_defaults()


# ---------------------------------------------------------------------------
# Gaussian-mixture reward generator
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Agent kwarg builder
# ---------------------------------------------------------------------------

def _module_kwargs(variant: str, method_cfgs: dict | None, arnoldi_k: int) -> dict:
    kw: dict = {"arnoldi_k": arnoldi_k}
    if method_cfgs and variant in method_cfgs:
        kw.update({k: v for k, v in method_cfgs[variant].items() if v is not None})
    
    if variant == "gta_beta":
        kw.setdefault("beta1", 1.0)
        kw.setdefault("beta2", 0.1)
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
    metrics = [REGISTRY[m] for m in metric_names]

    agents = {
        (K, v): make_agent(
            v, P=env.P, r=env.r, gamma=gamma, q=20,
            **_module_kwargs(v, method_cfgs, arnoldi_k),
        )
        for K in K_budgets
        for v in VARIANTS
    }

    traces: dict = {f"{v}_K{K}_{m.name}": [] for v in VARIANTS for K in K_budgets for m in metrics}
    shift_steps: list[int] = []
    r_new = env.r.copy()
    V_star = env.value_iteration(r_new)

    # ── Shift phase ──────────────────────────────────────────────────────
    for shift_i in range(n_shifts):
        r_new  = env.shift_reward()
        V_star = env.value_iteration(r_new)
        shift_steps.append(len(traces[f"standard_ps_K{K_budgets[0]}_{metrics[0].name}"]))

        for step in range(n_steps_per_shift):
            r_arg = r_new if step == 0 else None
            for K in K_budgets:
                for v in VARIANTS:
                    V = agents[(K, v)].plan(K=K, r=r_arg)
                    for m in metrics:
                        traces[f"{v}_K{K}_{m.name}"].append(m(V, V_star))

    # ── Post-last-shift convergence tail ─────────────────────────────────
    for _ in range(tail_steps):
        for K in K_budgets:
            for v in VARIANTS:
                V = agents[(K, v)].plan(K=K)
                for m in metrics:
                    traces[f"{v}_K{K}_{m.name}"].append(m(V, V_star))

    return {"traces": traces, "shift_steps": shift_steps}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(all_seed_results: list) -> dict:
    from scipy import stats
    keys        = list(all_seed_results[0]["traces"].keys())
    shift_steps = all_seed_results[0]["shift_steps"]
    agg: dict   = {"shift_steps": shift_steps, "traces": {}}

    for key in keys:
        mat  = np.array([r["traces"][key] for r in all_seed_results])
        n    = mat.shape[0]
        mean = mat.mean(axis=0)
        se   = stats.sem(mat, axis=0)
        ci95 = (se * stats.t.ppf(0.975, df=n - 1)) if n > 1 else np.zeros_like(mean)
        agg["traces"][key] = {"mean": mean.tolist(), "ci95": ci95.tolist()}
    return agg


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(
    agg_by_gs: dict,
    grid_sizes: list,
    n_states_list: list,
    K_budgets: list,
    metric_names: list[str],
    n_shift_steps: int,
    out_path: str,
) -> None:
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
    nrows = len(grid_sizes)
    ncols = len(K_budgets)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.2 * ncols, 4.0 * nrows),
        sharex=False, sharey=False,
    )
    if nrows == 1: axes = axes[np.newaxis, :]
    if ncols == 1: axes = axes[:, np.newaxis]

    # Dynamically assign styles to metrics
    _STYLES = ["-", "--", ":", "-."]

    for ri, (gs, n_s) in enumerate(zip(grid_sizes, n_states_list)):
        agg         = agg_by_gs[f"gs{gs}"]
        shift_steps = agg["shift_steps"]
        T           = len(agg["traces"][list(agg["traces"].keys())[0]]["mean"])
        steps       = np.arange(T)

        for ci, K in enumerate(K_budgets):
            ax = axes[ri, ci]
            ax.axvspan(n_shift_steps, T, color=bg_color, alpha=0.8, zorder=0)
            for ss in shift_steps:
                ax.axvline(ss, color=grid_color, linewidth=0.75, linestyle=":", alpha=0.9, zorder=1)

            for v in VARIANTS:
                for mi, m in enumerate(metrics):
                    key  = f"{v}_K{K}_{m.name}"
                    mean = np.array(agg["traces"][key]["mean"])
                    ci95 = np.array(agg["traces"][key]["ci95"])
                    col  = p_line.get(v, text_color)
                    ls   = _STYLES[mi % len(_STYLES)]
                    
                    alpha_fill = 0.12 if mi == 0 else 0.05

                    ax.plot(steps, mean, color=col, linewidth=1.6, linestyle=ls, zorder=3)
                    ax.fill_between(steps, mean - ci95, mean + ci95, color=col, alpha=alpha_fill, zorder=2)

            ax.set_yscale("log")
            ax.set_xlim(0, T - 1)
            ax.set_xlabel("Host step", fontsize=8)
            if ci == 0: ax.set_ylabel("Value Error (MAE/RMSE)", fontsize=9)
            ax.set_title(f"|S| = {n_s},  K = {K}", fontsize=9, pad=4)
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(True, axis="y", color=grid_color, alpha=0.25, linewidth=0.75)

    legend_handles: list[object] = [mlines.Line2D([], [], color=p_line.get(v, text_color), linewidth=2, label=AGENT_LABELS.get(v, v)) for v in VARIANTS]
    for mi, m in enumerate(metrics):
        legend_handles.append(mlines.Line2D([], [], color=text_color, linestyle=_STYLES[mi % len(_STYLES)], label=m.label))
    legend_handles.append(mpatches.Patch(facecolor=bg_color, edgecolor=grid_color, label="Post-shift tail"))

    fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles)//2 + 1, fontsize=8, bbox_to_anchor=(0.5, -0.05), frameon=True)
    fig.suptitle("RQ1 Timeline — Multi-Metric Convergence Comparison", fontsize=12, y=1.02)
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Config helpers
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs/fourrooms_k2_timeline.yaml"))
    parser.add_argument("--workers", default=PUBLICATION_DEFAULTS["workers"])
    parser.add_argument("--seeds", type=int, default=PUBLICATION_DEFAULTS["seeds"])
    parser.add_argument("--n-steps", type=int, help="Override steps per shift")
    parser.add_argument("--n-shifts", type=int, help="Override number of shifts")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRIC_NAMES, help="Metrics from REGISTRY")
    parser.add_argument("--out-dir", default=str(ROOT / "results"))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    warn_if_nonpublication_run(args.seeds, args.workers)

    # Validate metrics
    for m in args.metrics:
        if m not in REGISTRY:
            raise ValueError(f"Metric '{m}' not found in registry. Options: {list(REGISTRY.keys())}")

    config_path = Path(args.config).resolve()
    cfg          = _load_config(str(config_path))
    method_cfgs  = _build_method_cfgs(cfg)

    print(f"\n--- Hyperparameter Configuration ({args.config}) ---")
    for variant, params in method_cfgs.items():
        print(f"  {variant:10s}: {', '.join(f'{k}={v}' for k, v in params.items())}")
    print(f"  Metrics   : {', '.join(args.metrics)}")
    print("--------------------------------------------------\n")

    k             = cfg.get("k_fourrooms", 2)
    gamma         = cfg.get("gamma", 0.95)
    grid_sizes    = list(cfg["grid_sizes"])
    K_budgets     = list(cfg["budgets"])
    n_shifts      = cfg.get("n_shifts", 10)
    n_steps_shift = cfg.get("n_steps_per_shift", 25)
    tail_steps    = cfg.get("tail_steps", 60)
    seeds         = args.seeds

    n_total   = n_shifts * n_steps_shift + tail_steps
    runner    = RunManager(n_workers=args.workers)

    agg_by_gs:    dict = {}
    n_states_list: list = []

    for gs in grid_sizes:
        _tmp = FourRoomsTabular(k=k, gamma=gamma, seed=0, room_size=gs)
        n_s  = _tmp.n_states
        n_states_list.append(n_s)
        print(f"\n── grid_size={gs}  |S|={n_s} ─────────────────────────────")

        seed_results = runner.run(
            fn=_run_seed,
            seeds=[s for s in range(seeds)],
            fn_kwargs=dict(
                grid_size=gs, k=k, gamma=gamma, K_budgets=K_budgets,
                n_shifts=n_shifts, n_steps_per_shift=n_steps_shift,
                tail_steps=tail_steps, arnoldi_k=cfg.get("arnoldi_k", 10),
                method_cfgs=method_cfgs, room_peak_ranges=cfg.get("room_peak_ranges"),
                n_peaks_per_room=cfg.get("n_peaks_per_room", 2),
                sigma_range=cfg.get("sigma_range", [0.8, 3.0]),
                metric_names=args.metrics,
            ),
            desc=f"gs{gs}",
        )

        agg = _aggregate(seed_results)
        agg_by_gs[f"gs{gs}"] = agg

        for v in VARIANTS:
            row = f"  {v:12s}"
            for K in K_budgets:
                row += f"  K={K:3d}: {agg['traces'][f'{v}_K{K}_{args.metrics[0]}']['mean'][-1]:.4f}"
            print(row)

    tag = f"rq1_dp_timeline_k{k}_" + "_".join(str(g) for g in grid_sizes)
    out_dir = Path(args.out_dir).resolve()
    with open(out_dir / f"{tag}.json", "w") as f:
        json.dump({"experiment": "rq1_dp_timeline", "config": cfg, "results": agg_by_gs}, f, indent=2)

    if not args.no_plot:
        _plot(agg_by_gs, grid_sizes, n_states_list, K_budgets, args.metrics, n_shifts * n_steps_shift, str(Path(ROOT / "plots" / f"{tag}.png").resolve()))

if __name__ == "__main__":
    main()
