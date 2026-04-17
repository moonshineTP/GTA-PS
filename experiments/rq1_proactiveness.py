"""
RQ1: Proactiveness under reward shifts.

Does GTA-PS reduce post-shift value error and K_eps relative to Dyna-PS,
especially at small replanning budgets K < t_mix(1/4)?

Protocol (memo §8):
  - FourRooms k=1,2,3,4 + GARNET |S|=100,500
  - K in {|S|/10, |S|/2, |S|}
  - 1000 seeds; mean ± 95% CI
  - Primary metric: ||V^K - V*||_inf  and  K_eps (eps=0.1)

Usage:
  python experiments/rq1_proactiveness.py --env garnet --n-states 100 --seeds 10
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gta_ps.envs.garnet import GARNETEnv
from gta_ps.envs.fourrooms import FourRoomsTabular
from gta_ps.algorithms.dyna_ps import DynaPS
from gta_ps.algorithms.gta_ps import GTAPS
from gta_ps.utils.metrics import value_error_inf, steps_to_epsilon


def run_single_seed(
    env_name: str,
    n_states: int,
    k_fourrooms: int,
    reward_support: int,
    seed: int,
    gamma: float,
    n_shifts: int,
    eps_metric: float,
) -> dict:
    """Run one seed; return dict of metrics."""
    rng = np.random.default_rng(seed)

    if env_name == "garnet":
        # reward_support is fixed in {1,2,4,8} per memo §8 — NOT scaled by |S|
        env = GARNETEnv(n_states=n_states, n_actions=10, branching=4,
                        reward_support=reward_support,
                        gamma=gamma, seed=seed)
    else:
        env = FourRoomsTabular(k=k_fourrooms, gamma=gamma, seed=seed)
        n_states = env.n_states

    budgets = [
        max(1, n_states // 10),
        max(1, n_states // 2),
        n_states,
    ]

    results: dict = {name: {K: [] for K in budgets}
                     for name in ["dyna_ps", "gta_base", "gta_alpha", "gta_beta", "gta_full"]}

    for shift_idx in range(n_shifts):
        r = env.shift_reward()
        V_star = env.value_iteration(r)

        for K in budgets:
            for name, cls, kwargs in [
                ("dyna_ps",  DynaPS, {}),
                ("gta_base", GTAPS,  {}),
                ("gta_alpha",GTAPS,  {"adaptive_alpha": True}),
                ("gta_beta", GTAPS,  {"adaptive_beta":  True}),
                ("gta_full", GTAPS,  {"adaptive_alpha": True, "adaptive_beta": True}),
            ]:
                if name == "dyna_ps":
                    alg = DynaPS(env.P, env.r, gamma=gamma)
                else:
                    alg = GTAPS(env.P, env.r, gamma=gamma, **kwargs)
                V = alg.plan(K=K, r=r)
                err = value_error_inf(V, V_star)
                results[name][K].append(err)

    return results


def aggregate(all_results: list, budgets: list) -> dict:
    """Average across seeds; return mean and 95% CI per (method, K)."""
    from scipy import stats

    methods = list(all_results[0].keys())
    out = {}
    for m in methods:
        out[m] = {}
        for K in budgets:
            vals = np.array([r[m][K] for r in all_results]).ravel()
            mean = float(np.mean(vals))
            ci = float(stats.sem(vals) * stats.t.ppf(0.975, df=len(vals) - 1))
            out[m][K] = {"mean": mean, "ci95": ci, "n": len(vals)}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="garnet", choices=["garnet", "fourrooms"])
    parser.add_argument("--n-states", type=int, default=100)
    parser.add_argument("--k-fourrooms", type=int, default=1)
    # GARNET reward support: fixed in {1,2,4,8}, never a function of |S| (memo §8)
    parser.add_argument("--reward-support", type=int, default=4,
                        choices=[1, 2, 4, 8],
                        help="GARNET |supp(r_t)|; ignored for fourrooms")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--n-shifts", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps-metric", type=float, default=0.1)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    n_states = args.n_states if args.env == "garnet" else None
    budgets = [
        max(1, (n_states or 384) // 10),
        max(1, (n_states or 384) // 2),
        (n_states or 384),
    ]

    all_results = []
    for seed in tqdm(range(args.seeds), desc="seeds"):
        r = run_single_seed(
            env_name=args.env,
            n_states=args.n_states,
            k_fourrooms=args.k_fourrooms,
            reward_support=args.reward_support,
            seed=seed,
            gamma=args.gamma,
            n_shifts=args.n_shifts,
            eps_metric=args.eps_metric,
        )
        all_results.append(r)

    summary = aggregate(all_results, budgets)

    out_path = Path(args.out_dir) / f"rq1_{args.env}_S{args.n_states or 'fr'}_supp{args.reward_support}_k{args.k_fourrooms}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print("\nValue error ||V^K - V*||_inf  (mean ± 95%CI):")
    for method, kdict in summary.items():
        row = "  " + f"{method:12s}"
        for K, stats in kdict.items():
            row += f"  K={K}: {stats['mean']:.4f}±{stats['ci95']:.4f}"
        print(row)


if __name__ == "__main__":
    main()
