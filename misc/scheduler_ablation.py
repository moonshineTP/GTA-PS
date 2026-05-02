"""
misc/scheduler_ablation.py

Ablation: two beta_t scheduler families vs standard-PS baseline.

  linear : beta_t = beta1 * ((1-SLEM) + beta2),  beta2 in (-0.05, 0.25)
  log    : beta_t = beta1 * (-log(SLEM) + beta2), beta2 in (-0.05, 0.25)

Each scheduler is swept over its beta2 grid with beta1=1 fixed.
Variants: gta_beta.  Envs: fourrooms, garnet.

Output:
  results/misc/scheduler_ablation/ablation.json
  printed tables + cross-env summary
"""
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gta_ps.envs.fourrooms import FourRoomsTabular
from gta_ps.envs.garnet import GarnetEnv
from gta_ps.algorithms import make_agent
from threads.runner import RunManager

# ---------------------------------------------------------------------------
# Per-scheduler beta2 grids
# ---------------------------------------------------------------------------

_GRID_LINEAR = [-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]

SCHEDULER_GRIDS = {
    "linear": _GRID_LINEAR,
    "log":    _GRID_LINEAR,
}

VARIANTS = ["gta_beta"]
BETA1    = 0.5
GAMMA    = 0.95


# ---------------------------------------------------------------------------
# Workers (must be top-level for pickling)
# ---------------------------------------------------------------------------

def _make_env(env_name, seed, gamma):
    if env_name == "fourrooms":
        return FourRoomsTabular(k=1, gamma=gamma, seed=seed)
    return GarnetEnv(n_states=100, n_actions=5, branching=3,
                     reward_support=4, n_nbr_rate=0.15, gamma=gamma, seed=seed)


def _run_seed(job):
    """Run one seed for a (scheduler, beta2) config."""
    env_name  = job["env_name"]
    variant   = job["variant"]
    scheduler = job["scheduler"]
    beta2     = job["beta2"]
    n_shifts  = job["n_shifts"]
    K         = job["K"]
    gamma     = job["gamma"]
    seed      = job["seed"]

    env   = _make_env(env_name, seed, gamma)
    agent = make_agent(
        variant, P=env.P, r=env.r, gamma=gamma, q=20,
        beta1=BETA1, beta2=beta2, beta_mode=scheduler,
    )
    mod = agent.module
    l1_errs  = []
    inf_errs = []
    betas    = []
    for _ in range(n_shifts):
        r_new  = env.shift_reward()
        V_star = env.value_iteration(r_new)
        V      = agent.plan(K=K, r=r_new)
        diff   = np.abs(V - V_star)
        l1_errs.append(float(np.mean(diff)))
        inf_errs.append(float(np.max(diff)))
        betas.append(float(mod._last_beta))
    return l1_errs, inf_errs, betas


def _run_baseline_seed(job):
    """Run one seed for baseline."""
    env_name = job["env_name"]
    n_shifts = job["n_shifts"]
    K        = job["K"]
    gamma    = job["gamma"]
    seed     = job["seed"]

    env = _make_env(env_name, seed, gamma)
    ag  = make_agent("standard_ps", P=env.P, r=env.r, gamma=gamma, q=20)
    l1_errs  = []
    inf_errs = []
    for _ in range(n_shifts):
        r_new  = env.shift_reward()
        V_star = env.value_iteration(r_new)
        V      = ag.plan(K=K, r=r_new)
        diff   = np.abs(V - V_star)
        l1_errs.append(float(np.mean(diff)))
        inf_errs.append(float(np.max(diff)))
    return l1_errs, inf_errs


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _eval_one(env_name, variant, scheduler, beta2, n_seeds, n_shifts, K, gamma, workers):
    """Evaluate one (scheduler, beta2) config across seeds (parallel)."""
    rm = RunManager(n_workers=workers)
    seeds = list(range(n_seeds))
    jobs = []
    for s in seeds:
        jobs.append({
            "env_name":  env_name,
            "variant":   variant,
            "scheduler": scheduler,
            "beta2":     beta2,
            "n_shifts":  n_shifts,
            "K":         K,
            "gamma":     gamma,
            "seed":      s,
        })

    results = rm.run_jobs(_run_seed, jobs,
                          desc="%s beta2=%+.3f" % (scheduler, beta2))

    errors_l1  = []
    errors_inf = []
    beta_ts    = []
    for l1, inf, bt in results:
        errors_l1.extend(l1)
        errors_inf.extend(inf)
        beta_ts.extend(bt)

    return {
        "scheduler":    scheduler,
        "beta2":        beta2,
        "mean_error":   round(float(np.mean(errors_l1)),  4),
        "std_error":    round(float(np.std(errors_l1)),   4),
        "mean_error_inf": round(float(np.mean(errors_inf)), 4),
        "std_error_inf":  round(float(np.std(errors_inf)),  4),
        "beta_t_mean":  round(float(np.mean(beta_ts)), 6),
        "pct_zero":     round(100.0 * sum(b == 0.0 for b in beta_ts) / len(beta_ts), 1),
    }


def _baseline_error(env_name, n_seeds, n_shifts, K, gamma, workers):
    """Compute baseline error across seeds (parallel)."""
    rm = RunManager(n_workers=workers)
    seeds = list(range(n_seeds))
    jobs = []
    for s in seeds:
        jobs.append({
            "env_name": env_name,
            "n_shifts": n_shifts,
            "K":        K,
            "gamma":    gamma,
            "seed":     s,
        })

    results = rm.run_jobs(_run_baseline_seed, jobs,
                          desc="Baseline %s" % env_name)

    errs_l1  = []
    errs_inf = []
    for l1, inf in results:
        errs_l1.extend(l1)
        errs_inf.extend(inf)

    return {
        "l1":  round(float(np.mean(errs_l1)),  4),
        "inf": round(float(np.mean(errs_inf)), 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs",      nargs="+", default=["fourrooms", "garnet"])
    parser.add_argument("--variants",  nargs="+", default=VARIANTS)
    parser.add_argument("--seeds",    type=int,  default=10)
    parser.add_argument("--n-shifts", type=int,  default=8)
    parser.add_argument("--K",        type=int,  default=40)
    parser.add_argument("--gamma",    type=float, default=GAMMA)
    parser.add_argument("--workers",  type=int,  default=1,
                        help="Number of worker processes for parallel seed evaluation")
    parser.add_argument("--out", default=str(ROOT / "results/misc/scheduler_ablation/ablation.json"))
    args = parser.parse_args()

    all_results: Dict = {}

    for env_name in args.envs:
        all_results[env_name] = {}
        baseline = _baseline_error(
            env_name, args.seeds, args.n_shifts, args.K, args.gamma, args.workers
        )

        for variant in args.variants:
            all_results[env_name][variant] = []

            print("\n" + "=" * 90)
            print("  [%s]  %s   (baseline standard_ps  L1=%.4f  Linf=%.4f)" %
                  (env_name.upper(), variant, baseline["l1"], baseline["inf"]))
            print("=" * 90)
            print("  %-8s  %7s  %8s  %9s  %7s  %9s  %5s" %
                  ("scheduler", "beta2", "L1 err", "Linf err", "DL1%", "beta_t mean", "%beta=0"))
            print("  " + "-" * 8 + "  " + "-" * 7 + "  " + "-" * 8 +
                  "  " + "-" * 9 + "  " + "-" * 7 + "  " + "-" * 9 + "  " + "-" * 5)

            best_per_sched: Dict[str, Tuple] = {}

            for scheduler, grid in SCHEDULER_GRIDS.items():
                for beta2 in grid:
                    row = _eval_one(
                        env_name, variant, scheduler, beta2,
                        args.seeds, args.n_shifts, args.K, args.gamma, args.workers,
                    )
                    row["baseline_l1"]  = baseline["l1"]
                    row["baseline_inf"] = baseline["inf"]
                    row["vs_baseline_pct"] = round(
                        (row["mean_error"] - baseline["l1"]) / baseline["l1"] * 100, 1)
                    all_results[env_name][variant].append(row)

                    if (scheduler not in best_per_sched or
                            row["mean_error"] < best_per_sched[scheduler][0]["mean_error"]):
                        best_per_sched[scheduler] = (row, beta2)

                    marker = " *" if row["mean_error"] < baseline["l1"] else ""
                    print("  %-8s  %+7.3f  %8.4f  %9.4f  %+6.1f%%  %9.6f  %4.1f%%%s" %
                          (scheduler, beta2, row["mean_error"], row["mean_error_inf"],
                           row["vs_baseline_pct"], row["beta_t_mean"], row["pct_zero"], marker))

                print("  " + "-" * 8 + "  " + "-" * 7 + "  " + "-" * 8 +
                      "  " + "-" * 9 + "  " + "-" * 7 + "  " + "-" * 9 + "  " + "-" * 5)

            print()
            for sched, (best, b2) in best_per_sched.items():
                print("  * best %-8s: beta2=%+.3f  L1=%.4f  Linf=%.4f  (%.1f%% vs L1 baseline)  beta_t=%.5f  %%zero=%.1f%%" %
                      (sched, b2, best["mean_error"], best["mean_error_inf"],
                       best["vs_baseline_pct"], best["beta_t_mean"], best["pct_zero"]))

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to %s" % out)

    # -- Cross-env summary
    print("\n" + "=" * 90)
    print("  CROSS-ENV SUMMARY  (best beta2 per scheduler x env x variant, ranked by L1)")
    print("=" * 90)
    print("  %-10s  %-10s  %-8s  %8s  %8s  %9s  %9s  %9s  %5s" %
          ("env", "variant", "scheduler", "best beta2", "L1 err", "Linf err", "DL1%", "beta_t mean", "%beta=0"))
    print("  " + "-" * 10 + "  " + "-" * 10 + "  " + "-" * 8 + "  " + "-" * 8 +
          "  " + "-" * 8 + "  " + "-" * 9 + "  " + "-" * 9 + "  " + "-" * 9 + "  " + "-" * 5)
    for env_name in args.envs:
        for variant in args.variants:
            rows = all_results[env_name][variant]
            for scheduler in SCHEDULER_GRIDS:
                srows = [r for r in rows if r["scheduler"] == scheduler]
                if not srows:
                    continue
                best = min(srows, key=lambda r: r["mean_error"])
                print("  %-10s  %-10s  %-8s  %+8.3f  %8.4f  %9.4f  %+9.1f%%  %9.5f  %4.1f%%" %
                      (env_name, variant, scheduler, best["beta2"], best["mean_error"],
                       best["mean_error_inf"], best["vs_baseline_pct"], best["beta_t_mean"], best["pct_zero"]))


if __name__ == "__main__":
    main()
