"""
RQ3: Convergence vs. Propagation Tradeoff (Accuracy-Coverage).

Claims:
1. Small alpha (fixed) keeps the ranking close to the forward decision oracle (accuracy).
2. Larger alpha broadens predecessor-side coverage (propagation).
3. There exists a tradeoff yielding the best finite-K replanning performance.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "threads"))
sys.path.insert(0, str(ROOT / "misc"))

from gta_ps.envs.fourrooms import FourRoomsTabular
from gta_ps.envs.garnet import GarnetEnv
from gta_ps.algorithms import make_agent
from gta_ps.utils.metrics import value_error_inf, value_error_l1
from runner import RunManager
from plot_academic import get_publication_defaults, warn_if_nonpublication_run

PUBLICATION_DEFAULTS = get_publication_defaults()

def compute_policy_transition(env, policy):
    S = env.n_states
    P_pi = np.zeros((S, S))
    for s in range(S):
        a = int(policy[s])
        P_pi[s] = env.P[a, s]
    return P_pi

def compute_greedy_oracle(env, r):
    """
    psi = gamma * (I - gamma * P_star)^{-1} * delta_r
    where P_star is the greedy policy kernel.
    """
    V_star = env.value_iteration(r)
    pi_star = env.greedy_policy(V_star, r)
    P_star = compute_policy_transition(env, pi_star)
    
    n = env.n_states
    I = np.eye(n)
    # Using delta_r = r because we shift from 0 to r
    psi = env.gamma * np.linalg.solve(I - env.gamma * P_star, r)
    return psi, V_star

def run_alpha_sweep(
    env_name: str,
    seed: int,
    gamma: float,
    alpha_vals: list,
    beta: float,
    budgets: list,
    **env_kwargs
) -> dict:
    if env_name == "garnet":
        env = GarnetEnv(gamma=gamma, seed=seed, **env_kwargs)
    else:
        env = FourRoomsTabular(gamma=gamma, seed=seed, **env_kwargs)
        
    n_states = env.n_states
    
    # Force reward shift from 0 to r
    r = env.shift_reward()
    psi, V_star = compute_greedy_oracle(env, r)
    
    results = []
    for alpha in alpha_vals:
        # Create GTA-PS agent
        agent = make_agent(
            "gta_base", host="dp", P=env.P, r=np.zeros(n_states), gamma=gamma,
            beta=beta, alpha=alpha
        )
        
        # Apply reward shift
        agent.module.reward_update(r)
        tau = agent.module.tau.copy()
        
        # Metrics 1: Top-1 Accuracy (overlap with argmax psi)
        top1_tau = np.argmax(tau)
        top1_psi = np.argmax(psi)
        accuracy_top1 = 1.0 if top1_tau == top1_psi else 0.0
        
        # Metrics 2: Spearman Rho with oracle
        if np.all(tau == tau[0]) or np.max(np.abs(tau)) < 1e-12:
            rho = 0.0
        else:
            rho_raw = stats.spearmanr(tau, psi)[0]
            if isinstance(rho_raw, tuple):
                rho = 0.0
            else:
                rho = float(str(rho_raw))
                if np.isnan(rho):
                    rho = 0.0
            
        # Metrics 3: Predecessor Coverage
        # For Garnet, identify states that have high psi but are "far" from reward
        # We define coverage as Top-20 states in tau overlap with states 
        # that have high oracle psi but are NOT the goal states.
        goal_states = set(np.where(r > 0)[0])
        predecessor_states = np.argsort(-psi)[:40] # Top 40 in oracle
        predecessor_states = [s for s in predecessor_states if s not in goal_states]
        
        top20_states = np.argsort(-tau)[:20]
        coverage = len(set(top20_states) & set(predecessor_states)) / 20.0
        
        # Metrics 4: Value Error after planning (L1 replanning + L_inf consistency)
        value_errors = {}
        for K in budgets:
            agent_plan = make_agent(
                "gta_base", host="dp", P=env.P, r=np.zeros(n_states), gamma=gamma,
                beta=beta, alpha=alpha
            )
            # Use host.plan(K, r=r) to correctly update both host and module reward
            V = agent_plan.plan(K=K, r=r)
            value_errors[K] = {
                "inf": float(value_error_inf(V, V_star)),
                "l1":  float(value_error_l1(V, V_star)),
            }
            
        results.append({
            "alpha": alpha,
            "accuracy_top1": accuracy_top1,
            "rho": rho,
            "coverage": coverage,
            "value_errors": value_errors
        })
        
    return {"results": results}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="garnet", choices=["garnet", "fourrooms"])
    parser.add_argument("--k", type=int, default=1, help="FourRooms k")
    parser.add_argument("--seeds", type=int, default=PUBLICATION_DEFAULTS["seeds"])
    parser.add_argument("--workers", default=PUBLICATION_DEFAULTS["workers"])
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()
    warn_if_nonpublication_run(args.seeds, args.workers)

    alpha_vals = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    budgets = [25, 100]
    
    if args.env == "garnet":
        env_kwargs = {"n_states": 75, "n_actions": 10, "branching": 3, "reward_support": 4, "reversible": 0.0}
    else:
        env_kwargs = {"k_fourrooms": args.k}

    runner = RunManager(n_workers=args.workers)
    fn_kwargs = {
        "env_name": args.env,
        "alpha_vals": alpha_vals,
        "beta": args.beta,
        "budgets": budgets,
        "gamma": 0.95,
        **env_kwargs
    }
    
    print(f"Running RQ3 tradeoff on {args.env} with beta={args.beta}...")
    all_raw = runner.run(run_alpha_sweep, list(range(args.seeds)), fn_kwargs)
    
    # Aggregate
    summary = {a: {"acc": [], "rho": [], "cov": [],
                   "err_inf": {K: [] for K in budgets},
                   "err_l1":  {K: [] for K in budgets}} for a in alpha_vals}
    for seed_res in all_raw:
        for res in seed_res["results"]:
            a = res["alpha"]
            summary[a]["acc"].append(res["accuracy_top1"])
            summary[a]["rho"].append(res["rho"])
            summary[a]["cov"].append(res["coverage"])
            for K in budgets:
                summary[a]["err_inf"][K].append(res["value_errors"][K]["inf"])
                summary[a]["err_l1"][K].append(res["value_errors"][K]["l1"])
                
    print(f"\nResults for {args.env}:")
    header = (f"{'Alpha':>6} | {'Top-1 Acc':>10} | {'Rho':>10} | {'Coverage':>10} | "
              f"{'L1(K=25)':>10} | {'L1(K=100)':>10} | {'Linf(K=25)':>11} | {'Linf(K=100)':>11}")
    print(header)
    print("-" * len(header))

    final_data = []
    for a in alpha_vals:
        m_acc    = np.mean(summary[a]["acc"])
        m_rho    = np.mean(summary[a]["rho"])
        m_cov    = np.mean(summary[a]["cov"])
        m_l1_25  = np.mean(summary[a]["err_l1"][25])
        m_l1_100 = np.mean(summary[a]["err_l1"][100])
        m_inf_25  = np.mean(summary[a]["err_inf"][25])
        m_inf_100 = np.mean(summary[a]["err_inf"][100])
        print(f"{a:6.2f} | {m_acc:10.4f} | {m_rho:10.4f} | {m_cov:10.4f} | "
              f"{m_l1_25:10.4f} | {m_l1_100:10.4f} | {m_inf_25:11.4f} | {m_inf_100:11.4f}")

        final_data.append({
            "alpha": a,
            "top1_acc": m_acc,
            "rho": m_rho,
            "coverage": m_cov,
            "err_l1_25": m_l1_25,
            "err_l1_100": m_l1_100,
            "err_inf_25": m_inf_25,
            "err_inf_100": m_inf_100,
        })

    # Output to JSON
    tag = f"{args.env}" if args.env == "garnet" else f"fourrooms_k{args.k}"
    out_path = (ROOT / "results" / f"rq3_tradeoff_{tag}.json").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(final_data, f, indent=2)

if __name__ == "__main__":
    main()
