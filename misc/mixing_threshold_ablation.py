"""
misc/mixing_threshold_ablation.py

K-vs-t_mix threshold ablation — validates Prop. prop:mfpt(i).

Claim: whenever K < t_mix(1/4), the set B_K of MFPT-blocked states is
non-empty, and GTA-PS's full-support resolvent signal yields a cumulative
tracking-error advantage over Standard-PS (Thm. thm:tracking).

Protocol:
  Fix room_size=3 (9x9 grid, ~40 passable states).
  Sweep K ∈ [50, 100, 200] — planning budget per shift.
  Sweep corridor_width ∈ [1, 2, 3] — controls connectivity/mixing.
  For each (K, corridor_width) × seed:
    1. Build FourRoomsTabular with DpHost (oracle transitions).
    2. Measure SLEM after first plan -> compute K/t_mix analytically.
    3. Run N_SHIFTS reward-shift episodes:
       shift reward, agent.plan(K=K, r=r_new), record L1 error vs V*.
    4. Compute cumulative TE(N) = sum_{t=1}^{N} err_t per variant.
  Aggregate across seeds; report advantage ΔTE = TE^PS - TE^GTA.

Why DpHost (not Dyna):
  Dyna confounds the threshold test: fast-mixing environments are also
  easier to learn (model quality and t_mix move together). The oracle model
  isolates the K effect cleanly.

Output:
  plots/mixing_threshold_ablation.png
  results/misc/mixing_threshold_ablation/threshold.json
"""
from __future__ import annotations

import sys
import os
import argparse
import json
import math
import numpy as np
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "configs").exists():
    for p in Path(__file__).resolve().parents:
        if (p / "configs").exists() and (p / "src").exists():
            ROOT = p
            break
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "threads"))
sys.path.insert(0, str(ROOT / "misc"))

from gta_ps.envs.fourrooms import FourRoomsTabular
from gta_ps.algorithms import make_agent
from gta_ps.utils.metrics import value_error_l1
from runner import RunManager
from plot_academic import apply_style, get_palette, LABELS as AGENT_LABELS

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    plt = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

K_SWEEP            = [20, 50, 100]  # planning budget sweep (GTA needs K>=100 for topology)
N_SHIFTS            = 5              # number of reward-shift episodes per seed
N_STEPS_PER_SHIFT   = 100            # env steps between shifts (monitors error over time)

# Fixed room_size=3 -> 9x9 grid (2*3 + 1 + 2 = 9), ~40 passable states.
# Sweep corridor_width to vary SLEM/t_mix.
ROOM_SIZE          = 3
CORRIDOR_SWEEP     = [1, 2, 3]      # corridor_width values (max = room_size)

VARIANTS    = ["standard_ps", "gta_beta"]
LABELS      = {v: AGENT_LABELS[v] for v in VARIANTS}

# Colormap for corridor_width lines
C_CMAP    = plt.cm.viridis if plt is not None else None


def _tmix(slem: float, n_states: int, eps: float = 0.25) -> float:
    """Spectral upper bound on t_mix(eps): log(2n/eps) / (1-SLEM)."""
    gap = max(1.0 - slem, 1e-10)
    return math.log(2.0 * n_states / eps) / gap


# ---------------------------------------------------------------------------
# Worker: one (K, corridor_width, seed) pair
# ---------------------------------------------------------------------------

def _run_seed(
    seed: int,
    K: int,
    corridor_width: int,
    n_shifts: int,
    gamma: float,
    n_steps_per_shift: int,
    method_cfgs: dict,
    **_ignored,
) -> dict:
    env = FourRoomsTabular(room_size=ROOM_SIZE, corridor_len=1,
                           corridor_width=corridor_width,
                           gamma=gamma, seed=seed)
    n_states = env.P.shape[0]  # actual state count

    agents = {}
    for v in VARIANTS:
        # Use default params for dp host (tuned params are optimized for dyna, not dp)
        kw = {}
        agents[v] = make_agent(
            v, host="dp", P=env.P, r=env.r, gamma=gamma, q=20, **kw
        )

    # Warm-up plan with initial reward to stabilize topology and measure SLEM
    for v in VARIANTS:
        agents[v].plan(K=K, r=env.r)

    # SLEM after warm-up (use gta_beta module; standard_ps does not track SLEM)
    slem_val  = float(agents["gta_beta"].module.slem)
    tmix_val  = _tmix(slem_val, n_states)
    k_tmix    = K / tmix_val

    per_shift: dict[str, list[float]] = {v: [] for v in VARIANTS}
    cumul:     dict[str, list[float]] = {v: [0.0] for v in VARIANTS}

    for _ in range(n_shifts):
        r_new  = env.shift_reward()
        V_star = env.value_iteration(r_new)

        for v in VARIANTS:
            V_out = agents[v].plan(K=K, r=r_new)
            err   = value_error_l1(V_out, V_star)
            per_shift[v].append(err)
            cumul[v].append(cumul[v][-1] + err)

    return {
        "slem":      slem_val,
        "tmix":      tmix_val,
        "k_tmix":    k_tmix,
        "per_shift": per_shift,
        "cumul":     cumul,
    }


# ---------------------------------------------------------------------------
# Plotting: 3×3 grid (K × corridor_width)
# ---------------------------------------------------------------------------

def plot_results(
    agg:           dict,
    K_sweep:      list,
    corridor_sweep: list,
    n_shifts:    int,
    seeds:      int,
    out_path:   str,
) -> None:
    if plt is None:
        print("matplotlib not available — skipping plot.")
        return

    apply_style()
    
    # 3×3 grid: rows=K values, cols=corridor_width values
    n_rows = len(K_sweep)
    n_cols = len(corridor_sweep)
    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows), layout="constrained")
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig)
    
    # Colors for variants
    variant_colors = {"standard_ps": "steelblue", "gta_beta": "darkorange"}
    variant_ls    = {"standard_ps": "--", "gta_beta": "-"}
    variant_lw    = {"standard_ps": 1.2, "gta_beta": 1.8}

    shifts = np.arange(1, n_shifts + 1)
    cumul_shifts = np.arange(0, n_shifts + 1)

    for row, K in enumerate(K_sweep):
        for col, cw in enumerate(corridor_sweep):
            ax = fig.add_subplot(gs[row, col])
            
            k_str = str(K)
            cw_str = str(cw)
            
            if k_str in agg and cw_str in agg[k_str]:
                data = agg[k_str][cw_str]
                
                # Plot per-shift L1 error for each variant
                for v in VARIANTS:
                    if v in data["per_shift_mean"]:
                        m = np.array(data["per_shift_mean"][v])
                        ci = np.array(data["per_shift_ci"][v])
                        ax.plot(shifts, m, color=variant_colors[v], 
                                ls=variant_ls[v], lw=variant_lw[v],
                                label=LABELS[v])
                        ax.fill_between(shifts, m - ci, m + ci, 
                                    color=variant_colors[v], alpha=0.15)
                
                # Plot cumulative DTE
                if "diff_cumul_mean" in data:
                    diff_mean = np.array(data["diff_cumul_mean"])
                    diff_ci   = np.array(data["diff_cumul_ci"])
                    ax_twin = ax.twinx()
                    ax_twin.plot(cumul_shifts, diff_mean, color="gray", 
                                ls=":", lw=1.0, label="ΔTE")
                    ax_twin.fill_between(cumul_shifts, diff_mean - diff_ci, 
                                        diff_mean + diff_ci, color="gray", alpha=0.1)
                    ax_twin.set_ylabel("ΔTE", fontsize=8, color="gray")
                    ax_twin.tick_params(labelsize=7, colors="gray")
                
                # Title with K/t_mix
                k_tmix = data.get("k_tmix_mean", 0)
                regime = ("K<<t_mix" if k_tmix < 0.4 
                         else ("K~t_mix" if k_tmix < 1.2 else "K>t_mix"))
                ax.set_title(f"K={K}, cw={cw}\nK/t_mix={k_tmix:.3f} ({regime})", 
                           fontsize=8)
                
                ax.set_xlabel("Shift", fontsize=7)
                ax.set_ylabel("L1 error", fontsize=7)
                ax.set_yscale("log")
                ax.grid(True, which="both", ls="--", alpha=0.25)
                ax.tick_params(labelsize=6)
                
                if row == 0 and col == 0:
                    ax.legend(fontsize=6, loc="upper right")
                    if 'ax_twin' in locals():
                        ax_twin.legend(fontsize=6, loc="lower right")

    fig.suptitle(
        rf"$K$-vs-$t_\mathrm{{mix}}$ Threshold Ablation (9×9 Grid, $N_{{\mathrm{{seeds}}}}={seeds}$)",
        fontsize=10,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="K-vs-t_mix threshold ablation — validates Prop. prop:mfpt(i)"
    )
    parser.add_argument("--n-shifts",        dest="n_shifts",        type=int,   default=N_SHIFTS)
    parser.add_argument("--n-steps-per-shift", dest="n_steps_per_shift", type=int,   default=N_STEPS_PER_SHIFT)
    parser.add_argument("--K",          type=int,   default=None)  # Override for single K
    parser.add_argument("--gamma",      type=float, default=0.95)
    parser.add_argument("--seeds",      type=int,   default=5)
    parser.add_argument("--workers",    default="auto")
    parser.add_argument("--out-dir",
                        default=str(ROOT / "results/misc/mixing_threshold_ablation"))
    parser.add_argument("--plot-dir",   default=str(ROOT / "plots"))
    args = parser.parse_args()

    from scipy import stats as sp_stats

    runner = RunManager(n_workers=args.workers)

    print("=" * 62)
    print("  K-vs-t_mix threshold ablation — Prop. prop:mfpt(i)")
    print("=" * 62)
    print(f"  seeds={args.seeds}  n_shifts={args.n_shifts}"
          f"  n_steps_per_shift={args.n_steps_per_shift}"
          f"  gamma={args.gamma}  workers={args.workers}")
    print(f"  room_size={ROOM_SIZE}  K_sweep={K_SWEEP}  corridor_sweep={CORRIDOR_SWEEP}")
    print("-" * 62)

    # Determine K values to sweep
    K_values = [args.K] if args.K is not None else K_SWEEP

    all_agg: dict = {}  # all_agg[K][cw] = data

    for K in K_values:
        all_agg[str(K)] = {}
        print(f"\n{'K='+str(K):<12} {'cw':>4}  {'SLEM':>6}  {'K/t_mix':>8}  {'regime':>12}")
        print("-" * 62)

        for cw in CORRIDOR_SWEEP:
            # Pre-compute mixing stats from a single seed before the full trial
            pre = _run_seed(
                seed=0,
                K=K,
                corridor_width=cw,
                n_shifts=args.n_shifts,
                gamma=args.gamma,
                n_steps_per_shift=args.n_steps_per_shift,
                method_cfgs={},  # Use default params for DpHost
            )
            slem_pre = pre["slem"]
            k_tm_pre = pre["k_tmix"]
            regime_pre = ("K << t_mix" if k_tm_pre < 0.4
                          else ("K ~ t_mix" if k_tm_pre < 1.2 else "K > t_mix"))
            print(f"{' ':<12} {cw:>4d}  {slem_pre:>6.4f}  {k_tm_pre:>8.3f}  {regime_pre:>12}")

            raw = runner.run(
                _run_seed, range(args.seeds),
                dict(K=K,
                     corridor_width=cw,
                     n_shifts=args.n_shifts,
                     gamma=args.gamma,
                     n_steps_per_shift=args.n_steps_per_shift,
                     method_cfgs={}),  # Use default params for DpHost
                desc=f"K{K}_cw{cw}",
            )

            # Per-seed scalar summaries
            slems   = np.array([r["slem"]   for r in raw])
            k_tmixs = np.array([r["k_tmix"] for r in raw])

            # Cumulative difference curves (seeds x shifts+1)
            diff_mat = np.array([
                np.array(r["cumul"]["standard_ps"]) - np.array(r["cumul"]["gta_beta"])
                for r in raw
            ])
            diff_mean = diff_mat.mean(axis=0)
            diff_ci   = (sp_stats.sem(diff_mat, axis=0)
                         * sp_stats.t.ppf(0.975, df=len(raw) - 1))

            diff_end      = float(diff_mean[-1])
            diff_end_ci_v = float(diff_ci[-1])

            # Per-shift error curves
            def _agg_ep(key: str, v: str):
                mat  = np.array([r[key][v] for r in raw])  # (seeds, n_shifts)
                mean = mat.mean(axis=0)
                ci   = sp_stats.sem(mat, axis=0) * sp_stats.t.ppf(0.975, df=len(mat) - 1)
                return mean.tolist(), ci.tolist()

            per_shift_mean: dict = {}
            per_shift_ci:   dict = {}
            for v in VARIANTS:
                per_shift_mean[v], per_shift_ci[v] = _agg_ep("per_shift", v)

            all_agg[str(K)][str(cw)] = {
                "slem_mean":        float(slems.mean()),
                "k_tmix_mean":      float(k_tmixs.mean()),
                "diff_cumul_mean":  diff_mean.tolist(),
                "diff_cumul_ci":    diff_ci.tolist(),
                "diff_end_mean":    diff_end,
                "diff_end_ci":      diff_end_ci_v,
                "per_shift_mean":   per_shift_mean,
                "per_shift_ci":     per_shift_ci,
            }

            sign    = "+ GROWING" if diff_end > 0 else "- SHRINKING"
            regime  = ("K << t_mix" if k_tm_pre < 0.4
                      else ("K ~ t_mix" if k_tm_pre < 1.2 else "K > t_mix"))
            print(f"{' ':<12} {' ':>4}  {' ':>6}  {' ':>8}  "
                  f"{diff_end:>12.4f}  {regime} {sign}")

    # -- Persist --
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "threshold.json"
    payload = {
        "config": {
            "room_size":         ROOM_SIZE,
            "gamma":             args.gamma,
            "K_sweep":          K_values,
            "n_shifts":          args.n_shifts,
            "n_steps_per_shift": args.n_steps_per_shift,
            "seeds":             args.seeds,
            "corridor_sweep":    CORRIDOR_SWEEP,
        },
        "agg": all_agg,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {out_path}")

    plot_path = str(Path(args.plot_dir).resolve() / "mixing_threshold_ablation.png")
    plot_results(all_agg, K_values, CORRIDOR_SWEEP, args.n_shifts, args.seeds, plot_path)


if __name__ == "__main__":
    main()
