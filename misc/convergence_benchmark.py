"""
GTA-PS convergence benchmark (misc/convergence_benchmark.py).

Measures backup steps needed to reach ||V - V*||_inf < epsilon after each
reward shift, using warm V from the preceding shift (realistic replanning).

Agents: dyna_ps, gta_base, gta_beta.
Envs  : garnet (n=100), fourrooms.
Default epsilon = 5.0.

Use --config <yaml> to load tuned hyperparameters from a YAML experiment
config (the tuned.<variant> block).  Without --config the benchmark runs
with code defaults, which are not representative of the tuned algorithm.
"""
import os
# Pin BLAS threads before numpy import — worker processes inherit this limit
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
import json
from pathlib import Path
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "threads"))


from gta_ps.envs.garnet import GarnetEnv
from gta_ps.envs.fourrooms import FourRoomsTabular as FourRooms
from gta_ps.algorithms import make_agent
from gta_ps.utils.hyperparams import load_tuned_for_env
from runner import RunManager



# ---------------------------------------------------------------------------
# YAML config loader  (explicit --config override; auto-load used by default)
# ---------------------------------------------------------------------------

def _load_tuned(config_path: str | None) -> Dict[str, Dict]:
    """Return {variant: kwargs_dict} from the YAML tuned block, or {} if no config."""
    if config_path is None:
        return {}
    try:
        import yaml
    except ImportError:
        raise SystemExit("PyYAML required for --config: pip install pyyaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("tuned") or {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(env_name: str, gamma: float, seed: int):
    if env_name == "garnet":
        return GarnetEnv(n_states=100, n_actions=5, branching=3,
                         reward_support=4, n_nbr_rate=0.15, gamma=gamma, seed=seed)
    if env_name == "fourrooms":
        return FourRooms(k=1, gamma=gamma, seed=seed)
    raise ValueError(f"Unknown env: {env_name!r}")


def _make_agent(agent_name: str, env, gamma: float, tuned_kwargs: Dict | None = None):
    """Assemble a DpHost from (env, variant) using the canonical factory.

    tuned_kwargs, when provided, are passed as module kwargs so that the
    tuned hyperparameters from the YAML config are used instead of defaults.
    """
    return make_agent(
        agent_name, P=env.P, r=env.r, gamma=gamma, q=20,
        **(tuned_kwargs or {}),
    )


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------

def steps_to_converge(
    agent,
    env,
    r_new: np.ndarray,
    epsilon: float,
    budget_cap: int,
    K: int,  # Ignored for loop logic to allow exact step detection
) -> int:
    """
    Advance agent.plan() one backup at a time until ||V - V*||_inf < epsilon.

    V* is computed once via value_iteration for r_new. The agent retains its
    warm V from the previous shift (realistic replanning setting).
    """
    V_star = env.value_iteration(r_new)

    # Apply the reward shift first (K=0 ensures no backups are performed yet)
    V = agent.plan(K=0, r=r_new)
    
    if float(np.max(np.abs(V - V_star))) <= epsilon:
        return 0

    # Perform backups one by one to find the exact convergence step
    for k in range(1, budget_cap + 1):
        V = agent.plan(K=1)
        if float(np.max(np.abs(V - V_star))) <= epsilon:
            return k

    return budget_cap  # cap — did not converge within budget


def run_convergence_worker(
    seed: int,
    env_name: str,
    agent_name: str,
    gamma: float,
    epsilon: float,
    budget_cap: int,
    K: int,
    n_shifts: int,
    tuned_kwargs: Dict | None = None,
) -> list[int]:
    """Run convergence for a single seed (multiple shifts)."""
    env = _make_env(env_name, gamma, seed)
    agent = _make_agent(agent_name, env, gamma, tuned_kwargs)
    seed_k = []
    for _ in range(n_shifts):
        r_new = env.shift_reward()
        k = steps_to_converge(agent, env, r_new, epsilon, budget_cap, K)
        seed_k.append(k)
    return seed_k


def run_convergence(
    env_name: str,
    agent_name: str,
    gamma: float,
    epsilon: float,
    budget_cap: int,
    K: int,
    seeds: int,
    n_shifts: int,
    runner: RunManager,
    tuned_kwargs: Dict | None = None,
) -> Dict:
    """Average steps-to-epsilon over seeds × shifts (warm-start across shifts)."""
    fn_kwargs = dict(
        env_name=env_name, agent_name=agent_name, gamma=gamma,
        epsilon=epsilon, budget_cap=budget_cap, K=K,
        n_shifts=n_shifts, tuned_kwargs=tuned_kwargs
    )
    
    seed_results = runner.run(run_convergence_worker, range(seeds), fn_kwargs)
    all_k = [k for res in seed_results for k in res]

    not_converged = sum(k == budget_cap for k in all_k)
    return {
        "mean_K":        round(float(np.mean(all_k)), 2),
        "median_K":      round(float(np.median(all_k)), 2),
        "std_K":         round(float(np.std(all_k)), 2),
        "converged_pct": round(float((len(all_k) - not_converged) / len(all_k) * 100), 1),
        "n_trials":      len(all_k),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

AGENTS = ["standard_ps", "gta_base", "gta_beta"]
ENVS   = ["garnet", "fourrooms"]


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convergence benchmark: steps to ||V - V*||_inf < epsilon"
    )
    parser.add_argument("--envs",       nargs="+", default=ENVS)
    parser.add_argument("--agents",     nargs="+", default=AGENTS)
    parser.add_argument("--epsilon",    type=float, default=4.0,
                        help="Absolute inf-norm convergence threshold (default: 4.0)")
    parser.add_argument("--budget-cap", type=int,   default=2000,
                        help="Total backup ceiling across all host steps (default: 2000)")
    parser.add_argument("--K",          type=int,   default=50,
                        help="Bellman backups per host step (default: 50)")
    parser.add_argument("--seeds",    type=int,   default=50)
    parser.add_argument("--n-shifts",   type=int,   default=5)
    parser.add_argument("--gamma",      type=float, default=0.95)
    parser.add_argument("--workers",    type=int,   default=1)
    parser.add_argument("--out",        default=str(ROOT / "results/misc/convergence_benchmark/convergence.json"))
    parser.add_argument(
        "--config", default=None, metavar="YAML",
        help="Path to a YAML experiment config file.  The tuned.<variant> "
             "block is loaded and passed as agent kwargs.  Without this flag "
             "the benchmark runs with code defaults (not representative).",
    )
    args = parser.parse_args()

    all_results: Dict = {}
    runner = RunManager(n_workers=args.workers)

    for env_name in args.envs:
        all_results[env_name] = {}
        K = args.K

        # Tuned params: explicit --config wins; otherwise auto-load per-env default.
        if args.config is not None:
            config_path = Path(args.config).resolve()
            tuned_by_variant = _load_tuned(str(config_path))
        else:
            tuned_by_variant = load_tuned_for_env(env_name)

        print(f"\n[{env_name}]  eps={args.epsilon}  budget_cap={args.budget_cap}  K={K}")
        print(f"  {'agent':<12}  {'mean_K':>8}  {'median_K':>10}  {'std_K':>8}  {'converged':>10}")
        print(f"  {'-'*12}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*10}")

        for agent_name in args.agents:
            tuned_kwargs = tuned_by_variant.get(agent_name)  # None for standard_ps
            stats = run_convergence(
                env_name, agent_name, args.gamma,
                args.epsilon, args.budget_cap, K,
                args.seeds, args.n_shifts, runner,
                tuned_kwargs=tuned_kwargs,
            )
            all_results[env_name][agent_name] = stats
            print(
                f"  {agent_name:<12}  {stats['mean_K']:>8.1f}  {stats['median_K']:>10.1f}"
                f"  {stats['std_K']:>8.1f}  {stats['converged_pct']:>9.1f}%"
            )

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
