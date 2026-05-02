"""
misc/shift_rate_ablation.py

Ablation: Effect of reward-shift frequency on GTA-PS advantage (Assumption 4b).

Assumption 4b: the sequence {(s_{r,t}, Delta_{r,t})} is stationary-ergodic
under mu_R, with mean shift amplitude Delta_bar_r > 0.

Varying T — the number of planning steps the agent gets between successive
reward shifts — probes whether the GTA-PS advantage is preserved across
different ergodic shift rates:
  * Small T (fast shifting, little adaptation time): hard regime.
  * Large T (slow shifting, near-convergence per shift): easy regime.

The ablation reports value error AT THE END of each T-step window (shift-end
residual error).  If GTA-PS consistently beats Standard-PS across the full
T range, Assm 4b is empirically supported.

Output:
  plots/shift_rate_ablation.png
  results/misc/shift_rate_ablation/ablation.json
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
if not (ROOT / "configs").exists():
    # Try reaching up to find the real project root (e.g. if compiled in artifacts/dist/)
    for p in Path(__file__).resolve().parents:
        if (p / "configs").exists() and (p / "src").exists():
            ROOT = p
            break
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

T_SWEEP  = [5, 10, 25, 50, 100]   # host steps per reward shift
VARIANTS = ["standard_ps", "gta_base", "gta_beta"]
GAMMA    = 0.95


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _run_worker(
    T: int,
    seed: int,
    n_states: int,
    n_shifts: int,
    K: int,
    gamma: float,
    method_cfgs: dict,
    **_ignored,
) -> dict:
    """
    Run one seed for shift interval T.

    Protocol per shift window:
      1. Shift reward (env.shift_reward()).
      2. Give each agent T planning steps of K backups each.
         Only the first step carries r=r_new (triggers reward_update);
         subsequent steps use r=None (no redundant update).
      3. Record value error against V* at the end of the window.
    """
    env = GarnetEnv(n_states=n_states, n_actions=5, branching=3,
                    reward_support=4, n_nbr_rate=0.15, gamma=gamma, seed=seed)

    agents = {}
    for v in VARIANTS:
        kw = method_cfgs.get(v, {}).copy()
        if v == "standard_ps":
            agents[v] = make_agent(v, host="dp", P=env.P, r=env.r, gamma=gamma, q=20)
        else:
            agents[v] = make_agent(v, host="dp", P=env.P, r=env.r, gamma=gamma, q=20, **kw)

    errs: dict = {v: {"l1": [], "inf": []} for v in VARIANTS}

    for _ in range(n_shifts):
        r_new  = env.shift_reward()
        V_star = env.value_iteration(r_new)

        V_last: dict = {}
        for step in range(T):
            r_arg = r_new if step == 0 else None
            for v in VARIANTS:
                V_out = agents[v].plan(K=K, r=r_arg)
                if step == T - 1:
                    V_last[v] = V_out

        for v in VARIANTS:
            diff = np.abs(V_last[v] - V_star)
            errs[v]["l1"].append(float(diff.mean()))
            errs[v]["inf"].append(float(diff.max()))

    return {v: {"l1": float(np.mean(errs[v]["l1"])),
                "inf": float(np.mean(errs[v]["inf"]))}
            for v in VARIANTS}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ablation(results: dict, t_sweep: list,
                  seeds: int, K: int, out_path: str) -> None:
    if plt is None:
        print("matplotlib not available — skipping plot.")
        return

    apply_style()
    p_line  = get_palette("point")
    MARKERS = {"standard_ps": "s",       "gta_base": "o",       "gta_beta": "^"}
    LINES   = {"standard_ps": "--",      "gta_base": "-",       "gta_beta": "-"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(t_sweep))

    for ax, norm in zip(axes, ["l1", "inf"]):
        norm_tex = r"$L_1$" if norm == "l1" else r"$L_\infty$"
        for v in VARIANTS:
            means = [results[v][str(T)]["mean"][norm] for T in t_sweep]
            cis   = [results[v][str(T)]["ci95"][norm] for T in t_sweep]
            ax.errorbar(x, means, yerr=cis, marker=MARKERS[v], linestyle=LINES[v],
                        linewidth=1.8, capsize=4, label=AGENT_LABELS.get(v, v), color=p_line.get(v, "#444444"))

        ax.set_xticks(x)
        ax.set_xticklabels([str(T) for T in t_sweep])
        ax.set_xlabel("Planning steps per shift window ($T$)")
        ax.set_ylabel(f"Shift-end value error ({norm_tex})")
        ax.set_title(f"{norm_tex} error vs shift interval $T$")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", alpha=0.2)
        ax.legend()

    fig.suptitle(
        r"Shift-Rate Ablation — Assumption 4b (Ergodic Reward Shifts)"
        f"\nGARNET $|S|=100$,  $N={seeds}$ seeds,  $K={K}$ backups/step",
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
        description="Ablation: shift-rate T vs GTA-PS advantage (Assm 4b)"
    )
    parser.add_argument("--n-states", type=int,   default=100)
    parser.add_argument("--n-shifts", type=int,   default=6)
    parser.add_argument("--K",        type=int,   default=15)
    parser.add_argument("--seeds",    type=int,   default=50)
    parser.add_argument("--gamma",    type=float, default=GAMMA)
    parser.add_argument("--workers",  default="auto")
    parser.add_argument("--out-dir",
                        default=str(ROOT / "results/misc/shift_rate_ablation"))
    parser.add_argument("--plot-dir", default=str(ROOT / "plots"))
    args = parser.parse_args()

    from scipy import stats as sp_stats

    tuned  = load_tuned_for_env("garnet")
    runner = RunManager(n_workers=args.workers)

    all_results: dict = {v: {} for v in VARIANTS}

    for T in T_SWEEP:
        print(f"\nT={T:>4}  ({args.seeds} seeds × {args.n_shifts} shifts × "
              f"{T} steps × K={args.K} backups) …")
        raw = runner.run(
            _run_worker, range(args.seeds),
            dict(T=T, n_states=args.n_states, n_shifts=args.n_shifts,
                 K=args.K, gamma=args.gamma, method_cfgs=tuned),
            desc=f"shift_T{T}",
        )

        std_l1 = float(np.mean([r["standard_ps"]["l1"] for r in raw]))

        print(f"  {'variant':<12}  {'L1':>9}  {'Linf':>10}  {'ΔL1 vs Std':>12}")
        print(f"  {'-'*12}  {'-'*9}  {'-'*10}  {'-'*12}")

        for v in VARIANTS:
            l1_arr  = np.array([r[v]["l1"]  for r in raw])
            inf_arr = np.array([r[v]["inf"] for r in raw])
            mean_l1  = float(l1_arr.mean())
            mean_inf = float(inf_arr.mean())

            def _ci(arr: np.ndarray) -> float:
                return float(sp_stats.sem(arr) * sp_stats.t.ppf(0.975, df=len(arr) - 1))

            all_results[v][str(T)] = {
                "mean": {"l1": mean_l1,   "inf": mean_inf},
                "ci95": {"l1": _ci(l1_arr), "inf": _ci(inf_arr)},
            }
            delta  = (mean_l1 - std_l1) / std_l1 * 100 if v != "standard_ps" else 0.0
            marker = " *" if mean_l1 < std_l1 and v != "standard_ps" else ""
            print(f"  {v:<12}  {mean_l1:>9.4f}  {mean_inf:>10.4f}  "
                  f"{delta:>+11.1f}%{marker}")

    # ── Persist ───────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ablation.json"
    with open(out_path, "w") as f:
        json.dump(
            {"T_sweep": T_SWEEP, "results": all_results, "config": vars(args)},
            f, indent=2,
        )
    print(f"\nResults saved to {out_path}")

    plot_path = str(Path(args.plot_dir).resolve() / "shift_rate_ablation.png")
    plot_ablation(all_results, T_SWEEP, args.seeds, args.K, plot_path)


if __name__ == "__main__":
    main()
