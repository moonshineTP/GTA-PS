"""
misc/smoothness_ablation.py

Smoothness ablation: graph Rayleigh quotient of the reward perturbation
delta_r versus the diffused perturbation delta_f^out.

Validates Lemma 6.9 (Poincaré inequality):
  The resolvent acts as a graph low-pass filter:
    delta_f^out = (L^out + eta* I)^{-1} delta_r  has lower graph-frequency
    content than delta_r, measured by the Rayleigh quotient E(f)/Var_mu(f).
  Quantitatively:
    Var_mu(delta V^pi) <= 8 gamma^2 Delta_r^2 / ((1-gamma)^2 * (1-SLEM))

Note: delta_r is highly sparse (reward_support states out of N_STATES), so
comparing absolute E or Var values is misleading — the resolvent diffuses
a sparse signal to all states, necessarily increasing absolute amplitude.
The scale-invariant Rayleigh quotient E/Var correctly captures frequency.

At each reward shift we record TWO signals:
  delta_r      : reward perturbation  r_new - r_old  (sparse; reward_support nonzero)
  delta_f_out  : potential perturbation  f^out_new - f^out_old  (smoothed by resolvent)

Panels:
  1. Rayleigh quotient E(delta)/Var_mu(delta) per shift for delta_r vs delta_f^out.
     Lower = smoother (dominated by low-frequency graph modes).
     Claim: Rq(delta_f) << Rq(delta_r), showing resolvent is a low-pass filter.
  2. Var_mu(delta_f^out) per shift vs per-shift Lemma 6.9 bound
     8*gamma^4*||delta_r||_inf^2 / ((1-gamma)^2 * (1-SLEM)).
     Empirical variance must stay below the bound at each shift.
  3. Scatter: Var_mu(delta_f^out) vs spectral gap (1-SLEM) across GARNET
     environments + Lemma 6.9 theoretical upper-bound curve.

Environments: GARNET n_states=75, oracle model (DpHost), varying n_nbr_rate.

Output:
  plots/smoothness_ablation.png
  results/misc/smoothness_ablation/smoothness.json
"""
from __future__ import annotations

import sys
import os
import argparse
import json
import numpy as np
from pathlib import Path

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

from gta_ps.envs.garnet import GarnetEnv
from gta_ps.algorithms import make_agent
from gta_ps.utils.metrics import graph_dirichlet_energy, graph_variance
from gta_ps.utils.hyperparams import load_tuned_for_env
from runner import RunManager
from plot_academic import apply_style, get_palette

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    plt = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_STATES    = 75
N_ACTIONS   = 5
BRANCHING   = 3
REWARD_SUP  = 4
GAMMA       = 0.95
N_SHIFTS    = 8        # shifts for time-series panels (1 & 2)
K_PLAN      = 5        # one plan step triggers reward_update → f_out refresh
SEEDS       = 50

# n_nbr_rate sweep for cross-SLEM panel (panel 3)
NBR_SWEEP      = [0.05, 0.07, 0.10, 0.20, 0.40, 0.50]
N_SHIFTS_CROSS = 15
NBR_RATE_TS    = 0.10   # SLEM ~ 0.97 (time-series env)

# Measurement keys returned by _measure_shift
KEYS = ["E_dr", "E_df", "Var_dr", "Var_df", "Rq_dr", "Rq_df", "Delta_r", "slem"]


# ---------------------------------------------------------------------------
# Core measurement: one reward shift → perturbation smoothness
# ---------------------------------------------------------------------------

def _measure_shift(agent, env: GarnetEnv, r_prev: np.ndarray) -> dict:
    """
    Shift reward once and return smoothness of delta_r vs delta_f_out.

    The reward perturbation is  delta_r = r_new - r_prev  (sparse, high-freq).
    The potential perturbation is  delta_f_out = f_out_new - f_out_old  (smoothed
    by the resolvent; = gamma * delta V^pi at eta = eta*).
    """
    f_out_prev = agent.module.f_out.copy()
    r_new = env.shift_reward()
    agent.plan(K=K_PLAN, r=r_new)

    mod     = agent.module
    delta_r = r_new - r_prev
    delta_f = mod.f_out - f_out_prev
    P_pi    = mod.P_pi
    slem    = float(mod.slem)

    E_dr    = graph_dirichlet_energy(delta_r, P_pi)
    E_df    = graph_dirichlet_energy(delta_f, P_pi)
    Var_dr  = graph_variance(delta_r, P_pi)
    Var_df  = graph_variance(delta_f, P_pi)
    Rq_dr   = E_dr / max(Var_dr, 1e-14)
    Rq_df   = E_df / max(Var_df, 1e-14)
    Delta_r = float(np.abs(delta_r).max())

    return dict(
        r_new=r_new,
        slem=slem,
        E_dr=E_dr, E_df=E_df,
        Var_dr=Var_dr, Var_df=Var_df,
        Rq_dr=Rq_dr, Rq_df=Rq_df,
        Delta_r=Delta_r,
    )


# ---------------------------------------------------------------------------
# Workers (top-level for pickle / RunManager)
# ---------------------------------------------------------------------------

def _run_ts_seed(
    seed: int,
    n_shifts: int,
    gamma: float,
    method_cfgs: dict,
    **_ignored,
) -> dict:
    """Time-series worker: one seed × n_shifts shifts at NBR_RATE_TS."""
    env = GarnetEnv(n_states=N_STATES, n_actions=N_ACTIONS,
                    branching=BRANCHING, n_nbr_rate=NBR_RATE_TS,
                    reward_support=REWARD_SUP, gamma=gamma, seed=seed)
    kw = method_cfgs.get("gta_beta", {})
    ag = make_agent("gta_beta", host="dp", P=env.P, r=env.r, gamma=gamma, q=20, **kw)
    r_cur = env.r.copy()
    rows: dict[str, list] = {k: [] for k in KEYS}
    for _ in range(n_shifts):
        m = _measure_shift(ag, env, r_cur)
        r_cur = m["r_new"]
        for k in KEYS:
            rows[k].append(m[k])
    return rows


def _run_cross_seed(
    seed: int,
    nbr_rate: float,
    n_shifts: int,
    gamma: float,
    method_cfgs: dict,
    **_ignored,
) -> dict:
    """Cross-SLEM worker: one seed × n_shifts shifts at a given nbr_rate."""
    env = GarnetEnv(n_states=N_STATES, n_actions=N_ACTIONS,
                    branching=BRANCHING, n_nbr_rate=nbr_rate,
                    reward_support=REWARD_SUP, gamma=gamma, seed=seed)
    kw = method_cfgs.get("gta_beta", {})
    ag = make_agent("gta_beta", host="dp", P=env.P, r=env.r, gamma=gamma, q=20, **kw)
    r_cur = env.r.copy()
    slems:   list[float] = []
    var_dfs: list[float] = []
    for _ in range(n_shifts):
        m = _measure_shift(ag, env, r_cur)
        r_cur = m["r_new"]
        slems.append(m["slem"])
        var_dfs.append(m["Var_df"])
    return {"slem": float(np.mean(slems)), "var_df": float(np.mean(var_dfs))}


# ---------------------------------------------------------------------------
# Aggregation (mirrors rq4/rq5 get_stats pattern)
# ---------------------------------------------------------------------------

def _agg_ts(raw: list[dict]) -> dict:
    """Aggregate time-series results across seeds."""
    from scipy import stats as sp_stats
    result = {}
    for k in KEYS:
        mat  = np.array([r[k] for r in raw])     # (seeds, n_shifts)
        mean = mat.mean(axis=0)
        ci95 = (sp_stats.sem(mat, axis=0)
                * sp_stats.t.ppf(0.975, df=len(raw) - 1))
        result[k] = {"mean": mean.tolist(), "ci95": ci95.tolist()}
    return result


def _agg_cross(raw: list[dict]) -> tuple[float, float, float]:
    """Return (slem_mean, var_mean, var_ci95) from one nbr_rate batch."""
    from scipy import stats as sp_stats
    slems   = [r["slem"]   for r in raw]
    var_dfs = [r["var_df"] for r in raw]
    slem_m  = float(np.mean(slems))
    var_m   = float(np.mean(var_dfs))
    var_ci  = float(sp_stats.sem(var_dfs) * sp_stats.t.ppf(0.975, df=len(var_dfs) - 1))
    return slem_m, var_m, var_ci


# ---------------------------------------------------------------------------
# Experiment drivers
# ---------------------------------------------------------------------------

def run_timeseries(
    n_shifts: int,
    seeds: int,
    gamma: float,
    method_cfgs: dict,
    workers: int | str,
) -> dict:
    runner = RunManager(n_workers=workers)
    raw = runner.run(
        fn=_run_ts_seed,
        seeds=list(range(seeds)),
        fn_kwargs=dict(n_shifts=n_shifts, gamma=gamma, method_cfgs=method_cfgs),
        desc="smoothness-ts",
    )
    return _agg_ts(raw)


def run_cross_slem(
    n_shifts: int,
    seeds: int,
    gamma: float,
    method_cfgs: dict,
    workers: int | str,
) -> list:
    runner = RunManager(n_workers=workers)
    rows = []
    for nbr in NBR_SWEEP:
        raw = runner.run(
            fn=_run_cross_seed,
            seeds=list(range(seeds)),
            fn_kwargs=dict(nbr_rate=nbr, n_shifts=n_shifts, gamma=gamma,
                           method_cfgs=method_cfgs),
            desc=f"smoothness-cross-{nbr:.2f}",
        )
        slem_m, var_m, var_ci = _agg_cross(raw)
        print(f"  n_nbr={nbr:.2f}  SLEM={slem_m:.4f}  "
              f"Var(δf)={var_m:.5f} ± {var_ci:.5f}")
        rows.append({"n_nbr_rate": nbr, "slem": slem_m,
                     "Var_df_mean": var_m, "Var_df_ci": var_ci})
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    ts:       dict,
    cross:    list,
    n_shifts: int,
    seeds:    int,
    gamma:    float,
    out_path: str,
) -> None:
    if plt is None:
        print("matplotlib not available — skipping plot.")
        return

    apply_style()
    p_line = get_palette("line")
    p_point = get_palette("point")

    shifts  = np.arange(1, n_shifts + 1)
    C_RAW_TS       = p_line["standard_ps"]
    C_SIGNAL_TS    = p_line["gta_base"]
    C_THEORY_TS    = p_line["gta_beta"]
    C_SIGNAL_POINT = p_point["gta_beta"]
    C_THEORY_POINT = p_point["standard_ps"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.subplots_adjust(wspace=0.40)

    # ── Panel 1: Rayleigh quotient ────────────────────────────────────────────
    ax1 = axes[0]
    for key, label, color, ls in [
        ("Rq_dr", "Raw Reward", C_RAW_TS, "--"),
        ("Rq_df", "Diffused Signal", C_SIGNAL_TS, "-"),
    ]:
        m = np.array(ts[key]["mean"])
        c = np.array(ts[key]["ci95"])
        ax1.plot(shifts, m, label=label, color=color, lw=1.8, ls=ls)
        ax1.fill_between(shifts, np.maximum(m - c, 1e-10), m + c, color=color, alpha=0.15)
    ax1.set_xlabel("Reward-shift index")
    ax1.set_ylabel(r"Rayleigh quotient $\mathcal{E}(f)/\mathrm{Var}_\mu(f)$")
    ax1.set_title("Graph frequency: raw vs diffused")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", ls="--", alpha=0.25)
    ax1.legend(fontsize=8)

    # ── Panel 2: Var(delta_f) vs Lemma 6.9 bound ─────────────────────────────
    ax2 = axes[1]
    Var_df_m  = np.array(ts["Var_df"]["mean"])
    Var_df_c  = np.array(ts["Var_df"]["ci95"])
    Delta_r_m = np.array(ts["Delta_r"]["mean"])
    slem_m    = np.array(ts["slem"]["mean"])
    gap_m     = np.maximum(1.0 - slem_m, 1e-6)
    per_shift_bound = (8.0 * gamma**4 * Delta_r_m**2) / ((1.0 - gamma)**2 * gap_m)

    ax2.plot(shifts, Var_df_m, color=C_SIGNAL_TS, lw=1.8,
             label="Empirical Variance")
    ax2.fill_between(shifts, np.maximum(Var_df_m - Var_df_c, 1e-10),
                     Var_df_m + Var_df_c, color=C_SIGNAL_TS, alpha=0.15)
    ax2.plot(shifts, per_shift_bound, color=C_THEORY_TS, lw=1.6, ls="--",
             label="Theoretical Bound")
    ax2.set_xlabel("Reward-shift index")
    ax2.set_ylabel(r"$\mathrm{Var}_\mu(\delta f^{out}_t)$")
    ax2.set_title("Poincaré bound")
    ax2.set_yscale("log")
    ax2.grid(True, which="both", ls="--", alpha=0.25)
    ax2.legend(fontsize=8)

    frac_ok = float(np.mean(Var_df_m <= per_shift_bound))
    ax2.text(0.97, 0.05, f"{frac_ok*100:.0f}% shifts below bound",
             transform=ax2.transAxes, ha="right", fontsize=8,
             color="#333333",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    # ── Panel 3: cross-SLEM scatter ───────────────────────────────────────────
    ax3 = axes[2]
    slems     = np.array([r["slem"]        for r in cross])
    var_means = np.array([r["Var_df_mean"] for r in cross])
    var_cis   = np.array([r["Var_df_ci"]   for r in cross])
    gaps      = 1.0 - slems

    ax3.errorbar(gaps, var_means, yerr=var_cis, fmt="o", capsize=4,
                 color=C_SIGNAL_POINT, zorder=3,
                 label="Empirical")

    Delta_r_med = float(np.median(Delta_r_m)) if len(Delta_r_m) > 0 else 1.0
    x_grid = np.linspace(max(gaps.min() * 0.4, 1e-4), gaps.max() * 1.5, 300)
    bound  = (8.0 * gamma**4 * Delta_r_med**2) / ((1.0 - gamma)**2 * x_grid)
    ax3.plot(x_grid, bound, ls="--", color=C_THEORY_POINT, lw=1.6,
             label="Theory")

    ax3.set_xlabel(r"Spectral gap $1 - \mathrm{SLEM}$")
    ax3.set_ylabel(r"$\mathrm{Var}_\mu(\delta f^{out}_t)$")
    ax3.set_title("Var(δf) vs spectral gap across GARNETs")
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.grid(True, which="both", ls="--", alpha=0.25)
    ax3.legend(fontsize=8)

    fig.suptitle(
        rf"Smoothness Ablation — GARNET $|S|={N_STATES}$, $\gamma={gamma}$, "
        rf"$N_\mathrm{{seeds}}={seeds}$  |  "
        rf"Rq $= \mathcal{{E}}/\mathrm{{Var}}_\mu$ (scale-invariant frequency)",
        fontsize=10,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoothness ablation: δr vs δf^out, validates Lemma 6.9"
    )
    parser.add_argument("--n-shifts",       type=int,   default=N_SHIFTS)
    parser.add_argument("--n-shifts-cross", type=int,   default=N_SHIFTS_CROSS)
    parser.add_argument("--gamma",          type=float, default=GAMMA)
    parser.add_argument("--seeds",          type=int,   default=SEEDS)
    parser.add_argument("--workers",                    default=5)
    parser.add_argument("--out-dir",
                        default=str(ROOT / "results/misc/smoothness_ablation"))
    parser.add_argument("--plot-dir",       default=str(ROOT / "plots"))
    args = parser.parse_args()

    tuned = load_tuned_for_env("garnet")

    print("=" * 62)
    print("  Smoothness ablation — Lemma 6.9 validation")
    print("=" * 62)
    print(f"  seeds={args.seeds}  n_shifts={args.n_shifts}"
          f"  n_shifts_cross={args.n_shifts_cross}"
          f"  gamma={args.gamma}  workers={args.workers}")
    print(f"  n_nbr_rate_ts={NBR_RATE_TS}  nbr_sweep={NBR_SWEEP}")
    print("-" * 62)

    # ── Time-series ──────────────────────────────────────────────────────────
    print(f"\nTime-series: {args.seeds} seeds × {args.n_shifts} shifts "
          f"(n_nbr_rate={NBR_RATE_TS}) …")
    ts = run_timeseries(args.n_shifts, args.seeds, args.gamma, tuned, args.workers)

    mean_Rq_dr  = float(np.mean(ts["Rq_dr"]["mean"]))
    mean_Rq_df  = float(np.mean(ts["Rq_df"]["mean"]))
    mean_Var_df = float(np.mean(ts["Var_df"]["mean"]))
    mean_Delta  = float(np.mean(ts["Delta_r"]["mean"]))
    mean_slem   = float(np.mean(ts["slem"]["mean"]))
    gap         = max(1.0 - mean_slem, 1e-6)
    theory_bound = (8.0 * args.gamma**4 * mean_Delta**2) / ((1.0 - args.gamma)**2 * gap)
    print(f"  Mean Rq(δr):    {mean_Rq_dr:.5f}")
    print(f"  Mean Rq(δf):    {mean_Rq_df:.5f}")
    print(f"  Rq ratio δf/δr: {mean_Rq_df / max(mean_Rq_dr, 1e-12):.4f}"
          f"  {'✓ smoothed' if mean_Rq_df < mean_Rq_dr else '✗ NOT smoothed'}")
    print(f"  Mean Var(δf):   {mean_Var_df:.5f}")
    print(f"  Lemma 6.9 bound (SLEM={mean_slem:.4f}, Δr={mean_Delta:.4f}): "
          f"{theory_bound:.4f}"
          f"  {'✓ Var ≤ bound' if mean_Var_df <= theory_bound else '✗ VIOLATION'}")

    # ── Cross-SLEM ───────────────────────────────────────────────────────────
    print(f"\nCross-SLEM sweep: {len(NBR_SWEEP)} envs × {args.seeds} seeds "
          f"× {args.n_shifts_cross} shifts …")
    cross = run_cross_slem(args.n_shifts_cross, args.seeds, args.gamma, tuned, args.workers)

    # ── Persist ──────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "smoothness.json"
    payload = {
        "config": {
            "n_states":      N_STATES,
            "n_nbr_rate_ts": NBR_RATE_TS,
            "gamma":         args.gamma,
            "seeds":         args.seeds,
            "n_shifts":      args.n_shifts,
            "nbr_sweep":     NBR_SWEEP,
        },
        "timeseries": ts,
        "cross_slem": cross,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {out_path}")

    plot_path = str(Path(args.plot_dir).resolve() / "smoothness_ablation.png")
    plot_results(ts, cross, args.n_shifts, args.seeds, args.gamma, plot_path)


if __name__ == "__main__":
    main()
