"""
misc/mixing_threshold_ablation.py

K-vs-t_mix threshold ablation — validates Prop. prop:mfpt(i).

Claim: whenever K < t_mix(1/4), the set B_K of MFPT-blocked states is
non-empty, and GTA-PS's full-support resolvent signal yields a cumulative
tracking-error advantage over Standard-PS (Thm. thm:tracking).

Protocol:
  Fix K=40 planning steps per reward shift.
  Sweep n_nbr_rate ∈ NBR_SWEEP — this moves t_mix(1/4) across K=40:
    small n_nbr → slow mixing → t_mix >> K   (K << t_mix regime)
    large n_nbr → fast mixing → t_mix ≈ K    (K ~ t_mix, threshold)
  For each n_nbr_rate × seed:
    1. Build GARNET with DpHost (oracle transitions; no warm-up needed).
    2. Measure SLEM after first plan → compute K/t_mix analytically.
    3. Run N_EPISODES reward-shift episodes:
         shift reward, agent.plan(K=K, r=r_new), record L1 error vs V*.
    4. Compute cumulative TE(N) = sum_{t=1}^{N} err_t per variant.
  Aggregate across seeds; report advantage ΔTE = TE^PS - TE^GTA per n_nbr_rate.

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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "threads"))
sys.path.insert(0, str(ROOT / "misc"))

from gta_ps.envs.garnet import GarnetEnv
from gta_ps.algorithms import make_agent
from gta_ps.utils.metrics import value_error_l1
from gta_ps.utils.hyperparams import load_tuned_for_env
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

N_STATES    = 100
N_ACTIONS   = 5
BRANCHING   = 3
REWARD_SUP  = 4
GAMMA       = 0.95

K           = 40            # fixed planning budget per shift
N_EPISODES  = 20            # reward-shift episodes per seed

# n_nbr_rate sweep: moves t_mix(1/4) across K=40
# K/t_mix ratios (approximate): 0.018, 0.042, 0.102, 0.197, 0.563, 1.472
NBR_SWEEP   = [0.05, 0.07, 0.10, 0.15, 0.25, 0.40]

VARIANTS    = ["standard_ps", "gta_beta"]
LABELS      = {v: AGENT_LABELS[v] for v in VARIANTS}

# Colormap for n_nbr_rate lines (Panel 1)
NBR_CMAP    = plt.cm.viridis if plt is not None else None


def _tmix(slem: float, n_states: int, eps: float = 0.25) -> float:
    """Spectral upper bound on t_mix(eps): log(2n/eps) / (1-SLEM)."""
    gap = max(1.0 - slem, 1e-10)
    return math.log(2.0 * n_states / eps) / gap


# ---------------------------------------------------------------------------
# Worker: one (n_nbr_rate, seed) pair
# ---------------------------------------------------------------------------

def _run_seed(
    seed: int,
    nbr_rate: float,
    n_episodes: int,
    K: int,
    gamma: float,
    method_cfgs: dict,
    **_ignored,
) -> dict:
    env = GarnetEnv(
        n_states=N_STATES, n_actions=N_ACTIONS, branching=BRANCHING,
        n_nbr_rate=nbr_rate, reward_support=REWARD_SUP, gamma=gamma, seed=seed,
    )

    agents = {}
    for v in VARIANTS:
        kw = method_cfgs.get(v, {}).copy() if v != "standard_ps" else {}
        agents[v] = make_agent(
            v, host="dp", P=env.P, r=env.r, gamma=gamma, q=20, **kw
        )

    # Warm-up plan with initial reward to stabilise topology and measure SLEM
    for v in VARIANTS:
        agents[v].plan(K=K, r=env.r)

    # SLEM after warm-up (use gta_beta module; standard_ps does not track SLEM)
    slem_val  = float(agents["gta_beta"].module.slem)
    tmix_val  = _tmix(slem_val, N_STATES)
    k_tmix    = K / tmix_val

    per_ep: dict[str, list[float]] = {v: [] for v in VARIANTS}
    cumul:  dict[str, list[float]] = {v: [0.0] for v in VARIANTS}

    for _ in range(n_episodes):
        r_new  = env.shift_reward()
        V_star = env.value_iteration(r_new)

        for v in VARIANTS:
            V_out = agents[v].plan(K=K, r=r_new)
            err   = value_error_l1(V_out, V_star)
            per_ep[v].append(err)
            cumul[v].append(cumul[v][-1] + err)

    return {
        "slem":   slem_val,
        "tmix":   tmix_val,
        "k_tmix": k_tmix,
        "per_ep": per_ep,
        "cumul":  cumul,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    agg:        dict,
    nbr_sweep:  list,
    n_episodes: int,
    seeds:      int,
    K:          int,
    out_path:   str,
) -> None:
    if plt is None:
        print("matplotlib not available — skipping plot.")
        return

    apply_style()
    episodes = np.arange(1, n_episodes + 1)
    cumul_ep = np.arange(0, n_episodes + 1)

    k_tmix_vals  = [agg[str(nbr)]["k_tmix_mean"]  for nbr in nbr_sweep]
    diff_end     = [agg[str(nbr)]["diff_end_mean"] for nbr in nbr_sweep]
    diff_end_ci  = [agg[str(nbr)]["diff_end_ci"]   for nbr in nbr_sweep]

    colors_nbr = [NBR_CMAP(i / max(len(nbr_sweep) - 1, 1)) for i in range(len(nbr_sweep))]

    fig = plt.figure(figsize=(16, 4.5), layout="constrained")
    gs  = gridspec.GridSpec(1, 3, figure=fig)

    # ── Panel 1: cumulative advantage ΔTE(N) vs episode, one line per n_nbr ──
    ax1 = fig.add_subplot(gs[0])
    for i, nbr in enumerate(nbr_sweep):
        ratio  = agg[str(nbr)]["k_tmix_mean"]
        m_diff = np.array(agg[str(nbr)]["diff_cumul_mean"])
        c_diff = np.array(agg[str(nbr)]["diff_cumul_ci"])
        lbl    = f"Ratio {ratio:.2f}"
        ax1.plot(cumul_ep, m_diff, color=colors_nbr[i], lw=1.6, label=lbl)
        ax1.fill_between(cumul_ep, m_diff - c_diff, m_diff + c_diff,
                         color=colors_nbr[i], alpha=0.12)
    ax1.axhline(0, color="gray", ls=":", lw=1)
    ax1.set_xlabel("Episode $N$")
    ax1.set_ylabel(r"$\mathrm{TE}^{\mathrm{PS}}(N) - \mathrm{TE}^{\mathrm{GTA}}(N)$")
    ax1.set_title("Cumulative advantage vs episode\n(one line per mixing regime)")
    ax1.grid(True, ls="--", alpha=0.25)
    ax1.legend(fontsize=7, loc="upper left")

    # ── Panel 2: end-of-run ΔTE vs K/t_mix ratio ─────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    for i, (kt, dm, dc) in enumerate(zip(k_tmix_vals, diff_end, diff_end_ci)):
        ax2.errorbar(kt, dm, yerr=dc, fmt="o", capsize=4,
                     color=colors_nbr[i], markersize=7)
    ax2.axvline(1.0, color="#ff7f0e", ls="--", lw=1.5,
                label="Threshold")
    ax2.axhline(0, color="gray", ls=":", lw=1)
    ax2.set_xlabel(r"$K / t_{\mathrm{mix}}(1/4)$")
    ax2.set_ylabel(rf"$\Delta\mathrm{{TE}}(N={n_episodes})$")
    ax2.set_title("Advantage vs Mixing Ratio")
    ax2.grid(True, ls="--", alpha=0.25)
    ax2.legend(fontsize=9)

    # ── Panel 3: per-episode L1 error, all variants, one line per n_nbr ──────
    ax3 = fig.add_subplot(gs[2])
    for v in VARIANTS:
        for i, nbr in enumerate(nbr_sweep):
            m = np.array(agg[str(nbr)]["per_ep_mean"][v])
            c = np.array(agg[str(nbr)]["per_ep_ci"][v])
            ratio = agg[str(nbr)]["k_tmix_mean"]
            ls = "-" if v == "gta_beta" else "--"
            lw = 1.5 if v == "gta_beta" else 1.1
            ax3.plot(episodes, m, color=colors_nbr[i], ls=ls, lw=lw,
                     alpha=0.85)
            ax3.fill_between(episodes, m - c, m + c, color=colors_nbr[i], alpha=0.08)
    # Legend proxies
    from matplotlib.lines import Line2D
    proxy = [
        Line2D([0], [0], color="k", ls="-",  lw=1.5, label=LABELS["gta_beta"]),
        Line2D([0], [0], color="k", ls="--", lw=1.1, label=LABELS["standard_ps"]),
    ]
    ax3.legend(handles=proxy, fontsize=8, loc="upper right")
    ax3.set_xlabel("Episode (reward-shift index)")
    ax3.set_ylabel(r"Per-episode $L_1$ error")
    ax3.set_title("Per-episode error by mixing regime\n(darker = faster mixing)")
    ax3.set_yscale("log")
    ax3.grid(True, which="both", ls="--", alpha=0.25)

    fig.suptitle(
        rf"$K$-vs-$t_\mathrm{{mix}}$ Threshold Ablation — "
        rf"GARNET $|S|={N_STATES}$, DpHost, $K={K}$, $N_\mathrm{{seeds}}={seeds}$  "
        rf"(Prop.~\texttt{{prop:mfpt}}(i))",
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
    parser.add_argument("--n-episodes", type=int,   default=N_EPISODES)
    parser.add_argument("--K",          type=int,   default=K)
    parser.add_argument("--gamma",      type=float, default=GAMMA)
    parser.add_argument("--seeds",      type=int,   default=5)
    parser.add_argument("--workers",    default="auto")
    parser.add_argument("--out-dir",
                        default=str(ROOT / "results/misc/mixing_threshold_ablation"))
    parser.add_argument("--plot-dir",   default=str(ROOT / "plots"))
    args = parser.parse_args()

    from scipy import stats as sp_stats

    tuned  = load_tuned_for_env("garnet")
    runner = RunManager(n_workers=args.workers)

    print("=" * 62)
    print("  K-vs-t_mix threshold ablation — Prop. prop:mfpt(i)")
    print("=" * 62)
    print(f"  seeds={args.seeds}  K={args.K}  n_episodes={args.n_episodes}"
          f"  gamma={args.gamma}  workers={args.workers}")
    print(f"  nbr_sweep={NBR_SWEEP}")
    print("-" * 62)

    all_agg: dict = {}

    print(f"\n{'n_nbr':>6}  {'SLEM':>6}  {'K/t_mix':>8}  "
          f"{'ΔTE_PS-GTA':>12}  {'regime'}")
    print("-" * 60)

    for nbr in NBR_SWEEP:
        raw = runner.run(
            _run_seed, range(args.seeds),
            dict(nbr_rate=nbr, n_episodes=args.n_episodes, K=args.K,
                 gamma=args.gamma, method_cfgs=tuned),
            desc=f"nbr{nbr}",
        )

        # Per-seed scalar summaries
        slems   = np.array([r["slem"]   for r in raw])
        k_tmixs = np.array([r["k_tmix"] for r in raw])

        # Cumulative difference curves (seeds × episodes+1)
        diff_mat = np.array([
            np.array(r["cumul"]["standard_ps"]) - np.array(r["cumul"]["gta_beta"])
            for r in raw
        ])
        diff_mean = diff_mat.mean(axis=0)
        diff_ci   = (sp_stats.sem(diff_mat, axis=0)
                     * sp_stats.t.ppf(0.975, df=len(raw) - 1))

        diff_end      = float(diff_mean[-1])
        diff_end_ci_v = float(diff_ci[-1])

        # Per-episode error curves
        def _agg_ep(key: str, v: str):
            mat  = np.array([r[key][v] for r in raw])  # (seeds, n_episodes)
            mean = mat.mean(axis=0)
            ci   = sp_stats.sem(mat, axis=0) * sp_stats.t.ppf(0.975, df=len(mat) - 1)
            return mean.tolist(), ci.tolist()

        per_ep_mean: dict = {}
        per_ep_ci:   dict = {}
        for v in VARIANTS:
            per_ep_mean[v], per_ep_ci[v] = _agg_ep("per_ep", v)

        all_agg[str(nbr)] = {
            "slem_mean":       float(slems.mean()),
            "k_tmix_mean":     float(k_tmixs.mean()),
            "diff_cumul_mean": diff_mean.tolist(),
            "diff_cumul_ci":   diff_ci.tolist(),
            "diff_end_mean":   diff_end,
            "diff_end_ci":     diff_end_ci_v,
            "per_ep_mean":     per_ep_mean,
            "per_ep_ci":       per_ep_ci,
        }

        slem_m  = float(slems.mean())
        k_tm    = float(k_tmixs.mean())
        regime  = ("K << t_mix" if k_tm < 0.4
                   else ("K ~ t_mix" if k_tm < 1.2 else "K > t_mix"))
        sign    = "✓ GROWING" if diff_end > 0 else "✗ SHRINKING"
        print(f"{nbr:>6.2f}  {slem_m:>6.4f}  {k_tm:>8.3f}  "
              f"{diff_end:>12.4f}  {regime} {sign}")

    # ── Persist ────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "threshold.json"
    payload = {
        "config": {
            "n_states":   N_STATES,
            "gamma":      args.gamma,
            "K":          args.K,
            "n_episodes": args.n_episodes,
            "seeds":      args.seeds,
            "nbr_sweep":  NBR_SWEEP,
        },
        "agg": all_agg,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {out_path}")

    plot_path = str(Path(args.plot_dir).resolve() / "mixing_threshold_ablation.png")
    plot_results(all_agg, NBR_SWEEP, args.n_episodes, args.seeds, args.K, plot_path)


if __name__ == "__main__":
    main()
