"""
misc/scheduler_ablation.py

Ablation: three beta_t scheduler families vs standard-PS baseline.

  linear : beta_t = beta1 * ((1-SLEM) + beta2),  beta2 in (-0.05, 0.25)
           Unified replacement for clip_linear / additive / linear.
           beta2 < 0 introduces a dead-zone (dampen); beta2 > 0 adds a
           floor (always active).  No ReLU.
  power  : beta_t = beta1 * SLEM**beta2,          beta2 > 0 (decay exponent).
           Always in (0, beta1]; larger beta2 -> stronger SLEM-attenuation.
  log    : beta_t = beta1 * (-log(SLEM) + beta2), beta2 in (-0.05, 0.25)
           -log(SLEM) ~= (1-SLEM) near 1 but saturates as SLEM->0,
           preventing topology overdrive on fast-mixing chains.

Each scheduler is swept over its natural beta2 grid with beta1=1 fixed.
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

# ---------------------------------------------------------------------------
# Per-scheduler β₂ grids
# ---------------------------------------------------------------------------

# linear / log:  offset in spectral-gap space; negative damps, positive floors
_GRID_LINEAR = [-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.25]
# power:  β₂ = exponent; larger → slower decay, smaller → fast decay to 0
_GRID_POWER  = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0]

SCHEDULER_GRIDS = {
    "linear": _GRID_LINEAR,
    "power":  _GRID_POWER,
    "log":    _GRID_LINEAR,   # same offset semantics as linear
}

VARIANTS = ["gta_beta"]
BETA1    = 1.0
GAMMA    = 0.95


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(env_name: str, seed: int, gamma: float):
    if env_name == "fourrooms":
        return FourRoomsTabular(k=1, gamma=gamma, seed=seed)
    return GarnetEnv(n_states=100, n_actions=5, branching=3,
                     reward_support=4, n_nbr_rate=0.15, gamma=gamma, seed=seed)


def _eval_one(
    env_name: str,
    variant: str,
    scheduler: str,
    beta2: float,
    n_seeds: int,
    n_shifts: int,
    K: int,
    gamma: float,
) -> Dict:
    errors_l1:  List[float] = []
    errors_inf: List[float] = []
    beta_ts:    List[float] = []

    for seed in range(n_seeds):
        env   = _make_env(env_name, seed, gamma)
        agent = make_agent(
            variant, P=env.P, r=env.r, gamma=gamma, q=20,
            beta1=BETA1, beta2=beta2, beta_mode=scheduler,
        )
        mod = agent.module

        for _ in range(n_shifts):
            r_new  = env.shift_reward()
            V_star = env.value_iteration(r_new)
            V      = agent.plan(K=K, r=r_new)
            diff = np.abs(V - V_star)
            errors_l1.append(float(np.mean(diff)))
            errors_inf.append(float(np.max(diff)))
            beta_ts.append(float(mod._last_beta))

    return {
        "scheduler":    scheduler,
        "beta2":        beta2,
        "mean_error":   round(float(np.mean(errors_l1)),  4),   # L1 primary
        "std_error":    round(float(np.std(errors_l1)),   4),
        "mean_error_inf": round(float(np.mean(errors_inf)), 4),
        "std_error_inf":  round(float(np.std(errors_inf)),  4),
        "beta_t_mean":  round(float(np.mean(beta_ts)), 6),
        "pct_zero":     round(100.0 * sum(b == 0.0 for b in beta_ts) / len(beta_ts), 1),
    }


def _baseline_error(env_name: str, n_seeds: int, n_shifts: int, K: int, gamma: float) -> dict:
    errs_l1:  List[float] = []
    errs_inf: List[float] = []
    for seed in range(n_seeds):
        env = _make_env(env_name, seed, gamma)
        ag  = make_agent("standard_ps", P=env.P, r=env.r, gamma=gamma, q=20)
        for _ in range(n_shifts):
            r_new  = env.shift_reward()
            V_star = env.value_iteration(r_new)
            V      = ag.plan(K=K, r=r_new)
            diff = np.abs(V - V_star)
            errs_l1.append(float(np.mean(diff)))
            errs_inf.append(float(np.max(diff)))
    return {
        "l1":  round(float(np.mean(errs_l1)),  4),
        "inf": round(float(np.mean(errs_inf)), 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs",     nargs="+", default=["fourrooms", "garnet"])
    parser.add_argument("--variants", nargs="+", default=VARIANTS)
    parser.add_argument("--seeds",  type=int,  default=10)
    parser.add_argument("--n-shifts", type=int,  default=8)
    parser.add_argument("--K",        type=int,  default=40)
    parser.add_argument("--gamma",    type=float, default=GAMMA)
    parser.add_argument("--out", default=str(ROOT / "results/misc/scheduler_ablation/ablation.json"))
    args = parser.parse_args()

    all_results: Dict = {}

    for env_name in args.envs:
        all_results[env_name] = {}
        baseline = _baseline_error(env_name, args.n_seeds, args.n_shifts, args.K, args.gamma)

        for variant in args.variants:
            all_results[env_name][variant] = []

            print(f"\n{'='*90}")
            print(f"  [{env_name.upper()}]  {variant}   "
                  f"(baseline standard_ps  L1={baseline['l1']:.4f}  Linf={baseline['inf']:.4f})")
            print(f"{'='*90}")
            print(f"  {'scheduler':<8}  {'β₂':>7}  {'L1 err':>8}  {'Linf err':>9}  "
                  f"{'ΔL1%':>7}  {'β_t mean':>9}  {'%β=0':>5}")
            print(f"  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*9}  {'-'*5}")

            best_per_sched: Dict[str, Tuple] = {}  # scheduler → (row, beta2)

            for scheduler, grid in SCHEDULER_GRIDS.items():
                for beta2 in grid:
                    row = _eval_one(
                        env_name, variant, scheduler, beta2,
                        args.n_seeds, args.n_shifts, args.K, args.gamma,
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
                    print(
                        f"  {scheduler:<8}  {beta2:>7.3f}  "
                        f"{row['mean_error']:>8.4f}  "
                        f"{row['mean_error_inf']:>9.4f}  "
                        f"{row['vs_baseline_pct']:>+6.1f}%  "
                        f"{row['beta_t_mean']:>9.6f}  "
                        f"{row['pct_zero']:>4.1f}%{marker}"
                    )

                print(f"  {'·'*8}  {'·'*7}  {'·'*8}  {'·'*9}  {'·'*7}  {'·'*9}  {'·'*5}")

            print()
            for sched, (best, b2) in best_per_sched.items():
                print(f"  ★ best {sched:<8}: β₂={b2:+.3f}  "
                      f"L1={best['mean_error']:.4f}  Linf={best['mean_error_inf']:.4f}  "
                      f"({best['vs_baseline_pct']:+.1f}% vs L1 baseline)  "
                      f"β_t={best['beta_t_mean']:.5f}  %zero={best['pct_zero']:.1f}%")

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out}")

    # ── Cross-env summary ────────────────────────────────────────────────────
    print("\n" + "="*90)
    print("  CROSS-ENV SUMMARY  (best β₂ per scheduler × env × variant, ranked by L1)")
    print("="*90)
    print(f"  {'env':<10}  {'variant':<10}  {'scheduler':<8}  "
          f"{'best β₂':>8}  {'L1 err':>8}  {'Linf err':>9}  {'ΔL1%':>9}  "
          f"{'β_t mean':>9}  {'%β=0':>5}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}  "
          f"{'-'*8}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*5}")
    for env_name in args.envs:
        for variant in args.variants:
            rows = all_results[env_name][variant]
            for scheduler in SCHEDULER_GRIDS:
                srows = [r for r in rows if r["scheduler"] == scheduler]
                if not srows:
                    continue
                best = min(srows, key=lambda r: r["mean_error"])
                print(
                    f"  {env_name:<10}  {variant:<10}  {scheduler:<8}  "
                    f"{best['beta2']:>+8.3f}  {best['mean_error']:>8.4f}  "
                    f"{best['mean_error_inf']:>9.4f}  "
                    f"{best['vs_baseline_pct']:>+9.1f}%  "
                    f"{best['beta_t_mean']:>9.5f}  {best['pct_zero']:>4.1f}%"
                )


if __name__ == "__main__":
    main()

