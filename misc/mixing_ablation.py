"""
misc/mixing_ablation.py

Ablation: GTA-PS (linear scheduler) vs standard-PS across slow, medium,
and fast-mixing GARNET environments.

Environments (neighbourhood-GARNET, S=100, A=5, b=3):
  slow   : n_nbr=7   -> SLEM ~0.994  (comparable to FourRooms)
  medium : n_nbr=20  -> SLEM ~0.933
  fast   : n_nbr=100 -> SLEM ~0.327  (standard random GARNET)

For each mixing class we sweep beta2 over the linear-scheduler range
(-0.05, 0.25) and also report the best log-scheduler config.
Variants: gta_beta.

Output:
  results/misc/mixing_ablation/mixing.json
  printed tables + cross-mixing-class summary
"""
import os
# Pin BLAS threads before numpy import — worker processes inherit this limit
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "threads"))

from gta_ps.algorithms import make_agent
from gta_ps.utils.spectral import slem_arnoldi
from runner import RunManager

# ---------------------------------------------------------------------------
# Neighbourhood-GARNET builder (inline; does not modify GarnetEnv)
# ---------------------------------------------------------------------------

class NbrGarnetEnv:
    """
    GARNET with ring-neighbourhood successor sampling.

    Parameters
    ----------
    n_states   : |S|
    n_actions  : |A|
    branching  : successors per (s,a)
    n_nbr      : ring window size; n_nbr=n_states -> fully random
    reward_support : nonzero reward states per shift
    gamma      : discount factor
    seed       : RNG seed
    """
    def __init__(self, n_states=100, n_actions=5, branching=3,
                 n_nbr=100, reward_support=4, gamma=0.95, seed=0):
        self.n_states  = n_states
        self.n_actions = n_actions
        self.branching = branching
        self.n_nbr     = n_nbr
        self.reward_support = reward_support
        self.gamma     = gamma
        self.rng       = np.random.default_rng(seed)

        self.P = self._build_transition()
        self.r = self._sample_reward()

    def _build_transition(self):
        S, A, b = self.n_states, self.n_actions, self.branching
        P = np.zeros((A, S, S), dtype=np.float64)
        for a in range(A):
            for s in range(S):
                window = np.arange(s - self.n_nbr // 2,
                                   s + self.n_nbr // 2 + 1) % S
                window = np.unique(window)
                size   = min(b, len(window))
                succ   = self.rng.choice(window, size=size, replace=False)
                w      = self.rng.dirichlet(np.ones(size))
                for ni, ns in enumerate(succ):
                    P[a, s, int(ns)] += w[ni]
        return P

    def _sample_reward(self):
        r = np.zeros(self.n_states)
        idx = self.rng.choice(self.n_states,
                              size=min(self.reward_support, self.n_states),
                              replace=False)
        r[idx] = self.rng.random(len(idx))
        return r

    def shift_reward(self):
        self.r = self._sample_reward()
        return self.r.copy()

    def value_iteration(self, r=None, tol=1e-8, max_iter=10000):
        r_ = self.r if r is None else r
        gamma = self.gamma
        V = np.zeros(self.n_states)
        for _ in range(max_iter):
            EV  = np.einsum('ast,t->as', self.P, V)
            V_new = (r_[None, :] + gamma * EV).max(axis=0)
            if np.max(np.abs(V_new - V)) < tol:
                return V_new
            V = V_new
        return V


# ---------------------------------------------------------------------------
# Mixing configs
# ---------------------------------------------------------------------------

MIXING_CONFIGS = {
    "slow":   dict(n_nbr=7,   label="slow   (SLEM~0.994)"),
    "medium": dict(n_nbr=20,  label="medium (SLEM~0.933)"),
    "fast":   dict(n_nbr=50, label="fast   (SLEM~0.327)"),
}

# beta2 grids for each scheduler type
# linear: beta_t = beta1 * (1 - SLEM + beta2), beta2 can be negative or positive
_GRID_LINEAR = [-0.05, -0.02, 0.02, 0.05, 0.10, 0.20]
# log: beta_t = beta1 * SLEM^{beta2}, beta2 > 0 for power-law decay
_GRID_LOG = [0.2, 0.5, 1.0, 2.0, 5.0]
# fixed: beta2 is ignored, beta1 is used directly
_GRID_FIXED = [0.3]

VARIANTS = ["gta_beta"]
BETA1    = 0.3
GAMMA    = 0.95


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(n_nbr: int, seed: int) -> NbrGarnetEnv:
    return NbrGarnetEnv(n_states=100, n_actions=5, branching=3,
                        n_nbr=n_nbr, reward_support=4, gamma=GAMMA, seed=seed)


def _measure_slem(n_nbr: int, n_seeds: int = 5) -> float:
    slems = []
    for seed in range(n_seeds):
        env = _make_env(n_nbr, seed)
        P_uni = np.einsum('sa,ast->st',
                          np.ones((env.n_states, env.n_actions)) / env.n_actions,
                          env.P)
        sv, _ = slem_arnoldi(P_uni, k=min(15, env.n_states - 2))
        slems.append(sv)
    return float(np.mean(slems))


def _eval_worker(seed: int, mixing_key: str, variant: str, scheduler: str, beta2: float,
                 n_shifts: int, K: int) -> dict:
    n_nbr = MIXING_CONFIGS[mixing_key]["n_nbr"]
    errors_l1:  List[float] = []
    errors_inf: List[float] = []
    beta_ts:    List[float] = []

    env   = _make_env(n_nbr, seed)
    agent = make_agent(
        variant, P=env.P, r=env.r, gamma=GAMMA, q=20,
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
    return {"l1": errors_l1, "inf": errors_inf, "beta_t": beta_ts}


def _eval_one(mixing_key, variant, scheduler, beta2,
              n_seeds, n_shifts, K, runner: RunManager) -> Dict:
    seed_results = runner.run(
        _eval_worker, range(n_seeds),
        dict(mixing_key=mixing_key, variant=variant, scheduler=scheduler,
             beta2=beta2, n_shifts=n_shifts, K=K)
    )

    errors_l1  = [e for r in seed_results for e in r["l1"]]
    errors_inf = [e for r in seed_results for e in r["inf"]]
    beta_ts    = [b for r in seed_results for b in r["beta_t"]]

    return {
        "mixing":         mixing_key,
        "scheduler":      scheduler,
        "beta2":          beta2,
        "mean_error":     round(float(np.mean(errors_l1)),  4),   # L1 primary
        "std_error":      round(float(np.std(errors_l1)),   4),
        "mean_error_inf": round(float(np.mean(errors_inf)), 4),
        "std_error_inf":  round(float(np.std(errors_inf)),  4),
        "beta_t_mean":    round(float(np.mean(beta_ts)), 5),
    }


def _baseline_worker(seed: int, mixing_key: str, n_shifts: int, K: int) -> dict:
    n_nbr = MIXING_CONFIGS[mixing_key]["n_nbr"]
    errs_l1:  List[float] = []
    errs_inf: List[float] = []
    env = _make_env(n_nbr, seed)
    ag  = make_agent("standard_ps", P=env.P, r=env.r, gamma=GAMMA, q=20)
    for _ in range(n_shifts):
        r_new  = env.shift_reward()
        V_star = env.value_iteration(r_new)
        V      = ag.plan(K=K, r=r_new)
        diff = np.abs(V - V_star)
        errs_l1.append(float(np.mean(diff)))
        errs_inf.append(float(np.max(diff)))
    return {"l1": errs_l1, "inf": errs_inf}


def _baseline_error(mixing_key, n_seeds, n_shifts, K, runner: RunManager) -> dict:
    seed_results = runner.run(
        _baseline_worker, range(n_seeds),
        dict(mixing_key=mixing_key, n_shifts=n_shifts, K=K)
    )
    all_l1  = [e for r in seed_results for e in r["l1"]]
    all_inf = [e for r in seed_results for e in r["inf"]]
    return {
        "l1":  round(float(np.mean(all_l1)),  4),
        "inf": round(float(np.mean(all_inf)), 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=VARIANTS)
    parser.add_argument("--seeds",    type=int, default=10)
    parser.add_argument("--n-shifts", type=int, default=8)
    parser.add_argument("--K",        type=int, default=40)
    parser.add_argument("--workers",  type=int, default=1)
    parser.add_argument("--out", default=str(ROOT / "results/misc/mixing_ablation/mixing.json"))
    args = parser.parse_args()

    runner = RunManager(n_workers=args.workers)

    # Measure actual SLEMs
    print("Measuring environment SLEMs...")
    slem_map = {}
    for key, cfg in MIXING_CONFIGS.items():
        sv = _measure_slem(cfg["n_nbr"])
        slem_map[key] = sv
        print(f"  {key:8s} (n_nbr={cfg['n_nbr']:3d}): SLEM={sv:.5f}")
    print()

    all_results: Dict = {}

    for mixing_key, cfg in MIXING_CONFIGS.items():
        all_results[mixing_key] = {}
        baseline = _baseline_error(mixing_key, args.seeds, args.n_shifts, args.K, runner)
        slem_val = slem_map[mixing_key]

        for variant in args.variants:
            all_results[mixing_key][variant] = []

            print(f"\n{'='*86}")
            print(f"  [{mixing_key.upper():6s}] {cfg['label']}  variant={variant}  "
                  f"baseline L1={baseline['l1']:.4f}  Linf={baseline['inf']:.4f}")
            print(f"{'='*86}")
            print(f"  {'scheduler':<8}  {'b2':>7}  {'L1 err':>8}  {'Linf err':>9}  "
                  f"{'ΔL1%':>7}  {'bt_mean':>8}  note")
            print(f"  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*8}  ----")

            best_per_sched: Dict = {}

            for scheduler, grid in [("fixed", _GRID_FIXED),
                                     ("linear", _GRID_LINEAR),
                                     ("log", _GRID_LOG)]:
                for beta2 in grid:
                    row = _eval_one(mixing_key, variant, scheduler, beta2,
                                    args.seeds, args.n_shifts, args.K, runner)
                    row["baseline_l1"]  = baseline["l1"]
                    row["baseline_inf"] = baseline["inf"]
                    row["slem"]         = round(slem_val, 5)
                    row["vs_baseline_pct"] = round(
                        (row["mean_error"] - baseline["l1"]) / baseline["l1"] * 100, 1)
                    all_results[mixing_key][variant].append(row)

                    if (scheduler not in best_per_sched or
                            row["mean_error"] < best_per_sched[scheduler]["mean_error"]):
                        best_per_sched[scheduler] = row

                    marker = " <--" if row["mean_error"] < baseline["l1"] else ""
                    print(
                        f"  {scheduler:<8}  {beta2:>+7.3f}  "
                        f"{row['mean_error']:>8.4f}  "
                        f"{row['mean_error_inf']:>9.4f}  "
                        f"{row['vs_baseline_pct']:>+6.1f}%  "
                        f"{row['beta_t_mean']:>8.5f}"
                        f"{marker}"
                    )

                print(f"  {'·'*8}  {'·'*7}  {'·'*8}  {'·'*9}  {'·'*7}  {'·'*8}")

            print()
            for sched, best in best_per_sched.items():
                print(f"  best {sched:<8}: b2={best['beta2']:+.3f}  "
                      f"L1={best['mean_error']:.4f}  Linf={best['mean_error_inf']:.4f}  "
                      f"({best['vs_baseline_pct']:+.1f}% vs L1 baseline)  "
                      f"bt={best['beta_t_mean']:.5f}")

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out}")

    # ── Cross-mixing summary ─────────────────────────────────────────────────
    print("\n" + "="*86)
    print("  CROSS-MIXING SUMMARY  (best b2 per scheduler x mixing x variant, ranked by L1)")
    print("="*86)
    print(f"  {'mixing':<8}  {'SLEM':>7}  {'variant':<10}  {'scheduler':<8}  "
          f"{'best_b2':>7}  {'L1 err':>8}  {'Linf err':>9}  {'ΔL1%':>9}  {'bt_mean':>8}")
    print(f"  {'-'*8}  {'-'*7}  {'-'*10}  {'-'*8}  "
          f"{'-'*7}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*8}")
    for mixing_key in MIXING_CONFIGS:
        slem_val = slem_map[mixing_key]
        for variant in args.variants:
            rows = all_results[mixing_key][variant]
            for scheduler in ["fixed", "linear", "log"]:
                srows = [r for r in rows if r["scheduler"] == scheduler]
                if not srows:
                    continue
                best = min(srows, key=lambda r: r["mean_error"])
                print(
                    f"  {mixing_key:<8}  {slem_val:>7.4f}  {variant:<10}  {scheduler:<8}  "
                    f"{best['beta2']:>+7.3f}  {best['mean_error']:>8.4f}  "
                    f"{best['mean_error_inf']:>9.4f}  "
                    f"{best['vs_baseline_pct']:>+9.1f}%  {best['beta_t_mean']:>8.5f}"
                )


if __name__ == "__main__":
    main()
