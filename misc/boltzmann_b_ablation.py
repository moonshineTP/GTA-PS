"""
misc/boltzmann_b_ablation.py

Ablation: Effect of Boltzmann temperature b on GTA-PS performance (Assumption 8).

Assumption 8 requires b in (0, b_max] so that the priority policy
pi^PS_t(a|s) = softmax(b * A_t)(a|s) >= pi_act,min > 0 for all (s,a,t).

This ablation sweeps b across B_SWEEP and measures value-error for GTA-PS
variants vs Standard-PS (which does not use b, so its error is a fixed
horizontal baseline).  An advantage that is robust across b values validates
Assm 8: the topology signal adds value for any bounded temperature.

Output:
  plots/boltzmann_b_ablation.png
  results/misc/boltzmann_b_ablation/ablation.json
"""
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "threads"))

from gta_ps.envs.garnet import GarnetEnv
from gta_ps.algorithms import make_agent
from gta_ps.utils.hyperparams import load_tuned_for_env
from runner import RunManager
from plot_academic import apply_style, get_palette, LABELS as AGENT_LABELS

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

B_SWEEP       = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
VARIANTS_GTA  = ["gta_base", "gta_beta"]
GAMMA         = 0.95

# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

def _run_gta(
    b: float,
    variant: str,
    seed: int,
    n_states: int,
    n_shifts: int,
    K: int,
    gamma: float,
    base_kwargs: dict,
    **_ignored,
) -> dict:
    env   = GarnetEnv(n_states=n_states, n_actions=5, branching=3,
                      reward_support=4, n_nbr_rate=0.15, gamma=gamma, seed=seed)
    kw    = {k: v for k, v in base_kwargs.items() if k != "b"}
    kw["b"] = b
    agent = make_agent(variant, host="dp", P=env.P, r=env.r, gamma=gamma, q=20, **kw)

    errs_l1 = []
    errs_inf = []
    for _ in range(n_shifts):
        r_new  = env.shift_reward()
        V_star = env.value_iteration(r_new)
        V      = agent.plan(K=K, r=r_new)
        diff   = np.abs(V - V_star)
        errs_l1.append(float(diff.mean()))
        errs_inf.append(float(diff.max()))
    return {"l1": float(np.mean(errs_l1)), "inf": float(np.mean(errs_inf))}


def _run_baseline(
    seed: int,
    n_states: int,
    n_shifts: int,
    K: int,
    gamma: float,
    **_ignored,
) -> dict:
    env   = GarnetEnv(n_states=n_states, n_actions=5, branching=3,
                      reward_support=4, n_nbr_rate=0.15, gamma=gamma, seed=seed)
    agent = make_agent("standard_ps", host="dp", P=env.P, r=env.r, gamma=gamma, q=20)

    errs_l1 = []
    errs_inf = []
    for _ in range(n_shifts):
        r_new  = env.shift_reward()
        V_star = env.value_iteration(r_new)
        V      = agent.plan(K=K, r=r_new)
        diff   = np.abs(V - V_star)
        errs_l1.append(float(diff.mean()))
        errs_inf.append(float(diff.max()))
    return {"l1": float(np.mean(errs_l1)), "inf": float(np.mean(errs_inf))}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ablation(results: dict, baseline: dict, b_sweep: list,
                  seeds: int, K: int, out_path: str) -> None:
    if plt is None:
        print("matplotlib not available — skipping plot.")
        return

    apply_style()
    p_line = get_palette("line")
    MARKERS = {"gta_base": "o", "gta_beta": "^"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(b_sweep))

    for ax, norm in zip(axes, ["l1", "inf"]):
        norm_tex = r"$L_1$" if norm == "l1" else r"$L_\infty$"

        for v in VARIANTS_GTA:
            means = [results[v][str(b)]["mean"][norm] for b in b_sweep]
            cis   = [results[v][str(b)]["ci95"][norm] for b in b_sweep]
            ax.errorbar(x, means, yerr=cis, marker=MARKERS[v], linewidth=1.8,
                        capsize=4, label=AGENT_LABELS.get(v, v), color=p_line.get(v, "#444444"))

        bl_m  = baseline["mean"][norm]
        bl_ci = baseline["ci95"][norm]
        ax.axhline(bl_m, color=p_line.get("standard_ps", "#333333"), linestyle="--", linewidth=1.5,
                   label=AGENT_LABELS.get("standard_ps", "Standard-PS"))
        ax.fill_between([-0.5, len(b_sweep) - 0.5],
                        bl_m - bl_ci, bl_m + bl_ci, color=p_line.get("standard_ps", "#333333"), alpha=0.08)

        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in b_sweep])
        ax.set_xlabel("Boltzmann temperature $b$")
        ax.set_ylabel(f"Value error ({norm_tex})")
        ax.set_title(f"{norm_tex} error vs temperature $b$")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", alpha=0.2)
        ax.legend()

    fig.suptitle(
        f"Boltzmann Temperature Ablation — Assumption 8\n"
        f"GARNET $|S|=100$,  $N={seeds}$ seeds,  $K={K}$ backups/shift",
        fontsize=12,
    )
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation: Boltzmann temperature b vs GTA-PS performance (Assm 8)"
    )
    parser.add_argument("--n-states", type=int,   default=100)
    parser.add_argument("--n-shifts", type=int,   default=8)
    parser.add_argument("--K",        type=int,   default=40)
    parser.add_argument("--seeds",    type=int,   default=10)
    parser.add_argument("--gamma",    type=float, default=GAMMA)
    parser.add_argument("--workers",  default="auto")
    parser.add_argument("--out-dir",
                        default=str(ROOT / "results/misc/boltzmann_b_ablation"))
    parser.add_argument("--plot-dir", default=str(ROOT / "plots"))
    args = parser.parse_args()

    from scipy import stats as sp_stats

    tuned  = load_tuned_for_env("garnet")
    runner = RunManager(n_workers=args.workers)

    # ── Baseline (standard_ps, b-independent) ────────────────────────────────
    print("Running standard_ps baseline …")
    bl_raw = runner.run(
        _run_baseline, range(args.seeds),
        dict(n_states=args.n_states, n_shifts=args.n_shifts,
             K=args.K, gamma=args.gamma),
        desc="baseline",
    )
    bl_l1  = np.array([r["l1"]  for r in bl_raw])
    bl_inf = np.array([r["inf"] for r in bl_raw])

    def _ci(arr: np.ndarray) -> float:
        return float(sp_stats.sem(arr) * sp_stats.t.ppf(0.975, df=len(arr) - 1))

    baseline = {
        "mean": {"l1": float(bl_l1.mean()),  "inf": float(bl_inf.mean())},
        "ci95": {"l1": _ci(bl_l1),           "inf": _ci(bl_inf)},
    }
    print(f"  Standard-PS:  L1={baseline['mean']['l1']:.4f}  "
          f"Linf={baseline['mean']['inf']:.4f}")

    # ── GTA variants × b sweep ───────────────────────────────────────────────
    all_results: dict = {}

    for variant in VARIANTS_GTA:
        all_results[variant] = {}
        base_kw = {k: v for k, v in tuned.get(variant, {}).items() if k != "b"}

        print(f"\n{'='*60}")
        print(f"  Variant: {variant}   "
              f"(baseline L1={baseline['mean']['l1']:.4f}  "
              f"Linf={baseline['mean']['inf']:.4f})")
        print(f"{'='*60}")
        print(f"  {'b':>5}  {'L1 err':>9}  {'Linf err':>10}  {'ΔL1%':>8}")
        print(f"  {'-'*5}  {'-'*9}  {'-'*10}  {'-'*8}")

        for b in B_SWEEP:
            raw = runner.run(
                _run_gta, range(args.seeds),
                dict(b=b, variant=variant, n_states=args.n_states,
                     n_shifts=args.n_shifts, K=args.K, gamma=args.gamma,
                     base_kwargs=base_kw),
                desc=f"{variant}_b{b}",
            )
            l1_arr  = np.array([r["l1"]  for r in raw])
            inf_arr = np.array([r["inf"] for r in raw])
            mean_l1  = float(l1_arr.mean())
            mean_inf = float(inf_arr.mean())
            delta    = (mean_l1 - baseline["mean"]["l1"]) / baseline["mean"]["l1"] * 100
            all_results[variant][str(b)] = {
                "mean": {"l1": mean_l1,   "inf": mean_inf},
                "ci95": {"l1": _ci(l1_arr), "inf": _ci(inf_arr)},
            }
            marker = " *" if mean_l1 < baseline["mean"]["l1"] else ""
            print(f"  {b:>5.1f}  {mean_l1:>9.4f}  {mean_inf:>10.4f}  "
                  f"{delta:>+7.1f}%{marker}")

    # ── Persist ───────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ablation.json"
    with open(out_path, "w") as f:
        json.dump(
            {"b_sweep": B_SWEEP, "baseline": baseline,
             "results": all_results, "config": vars(args)},
            f, indent=2,
        )
    print(f"\nResults saved to {out_path}")

    plot_path = str(Path(args.plot_dir).resolve() / "boltzmann_b_ablation.png")
    plot_ablation(all_results, baseline, B_SWEEP, args.seeds, args.K, plot_path)


if __name__ == "__main__":
    main()
