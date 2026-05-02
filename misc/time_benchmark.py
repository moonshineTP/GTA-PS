"""
Time complexity measurement and implementation benchmark.

Reports median wall-clock time over N_REPS repeated runs per agent, with a
warmup rep that is excluded from statistics. Each rep builds a fresh agent
(including _init_topology precompute) and plans for K backups after one
reward shift.

Parallelized with RunManager: all (env, agent) configurations run concurrently.

Results: median ± std (ms), excluding warmup.
"""
import sys
import json
import time
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "configs").exists():
    for p in Path(__file__).resolve().parents:
        if (p / "configs").exists() and (p / "src").exists():
            ROOT = p
            break
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "threads"))

from gta_ps.envs.garnet import GarnetEnv
from gta_ps.envs.fourrooms import FourRoomsTabular
from gta_ps.algorithms import make_agent
from gta_ps.utils.hyperparams import load_tuned_for_env
from runner import RunManager

# Number of timed repetitions (warmup rep excluded)
N_REPS = 8

AGENTS = ["standard_ps", "gta_base", "gta_beta"]

# All benchmark configurations: (env_name, n_states, k_fourrooms)
BENCHMARK_ENVS = [
    ("fourrooms", None, 2),
    ("garnet", 75, 3),
    ("garnet", 200, 5),
]

# Flatten into (env_name, n_states, k_fourrooms, agent_name) tuples
def _build_configs():
    configs = []
    for env_name, n_states, k_fourrooms in BENCHMARK_ENVS:
        for agent_name in AGENTS:
            configs.append((env_name, n_states, k_fourrooms, agent_name))
    return configs


def _make_env(env_name: str, n_states: int | None, k_fourrooms: int, gamma: float, seed: int):
    if env_name == "garnet":
        return GarnetEnv(
            n_states=n_states, n_actions=10, branching=4,
            reward_support=4, gamma=gamma, seed=seed,
        )
    elif env_name == "fourrooms":
        return FourRoomsTabular(k=k_fourrooms, gamma=gamma, seed=seed)
    raise ValueError(f"Unknown env: {env_name}")


def _run_single_benchmark(
    env_name: str,
    n_states: int | None,
    k_fourrooms: int,
    agent_name: str,
    gamma: float = 0.95,
    n_reps: int = N_REPS,
) -> dict:
    """Run benchmark for a single (env, agent) configuration."""
    tuned_by_variant = load_tuned_for_env(env_name)
    tuned_kwargs = tuned_by_variant.get(agent_name) or {}

    times_ms = []
    for rep in range(n_reps + 1):
        env = _make_env(env_name, n_states, k_fourrooms, gamma, rep)
        actual_n_states = env.n_states
        K = max(1, actual_n_states // 4)
        r_new = env.shift_reward()
        agent = make_agent(
            agent_name, P=env.P, r=r_new, gamma=gamma, q=20,
            **tuned_kwargs,
        )
        t0 = time.perf_counter()
        agent.plan(K=K, r=r_new)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if rep > 0:
            times_ms.append(elapsed_ms)

    med = round(statistics.median(times_ms), 4)
    std = round(statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0, 4)
    return {
        "env_name": env_name,
        "agent_name": agent_name,
        "n_states": actual_n_states,
        "K": K,
        "median_ms": med,
        "std_ms": std,
        "n_reps": n_reps,
    }


def _run_benchmark_by_index(seed: int, configs: list, n_reps: int = N_REPS, gamma: float = 0.95) -> dict:
    """Top-level worker function for RunManager. seed indexes into configs."""
    env_name, n_states, k_fourrooms, agent_name = configs[seed]
    return _run_single_benchmark(env_name, n_states, k_fourrooms, agent_name, gamma, n_reps)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Time Complexity Benchmark (parallel)")
    parser.add_argument("--workers", type=int, default=5, help="Parallel workers for RunManager")
    parser.add_argument("--seeds", type=int, default=N_REPS, help="Timed repetitions per agent (excl. warmup)")
    args = parser.parse_args()

    configs = _build_configs()
    n_reps = args.seeds

    print(f"=== Time Complexity Benchmark ({n_reps} reps + 1 warmup, {len(configs)} configs) ===")

    runner = RunManager(n_workers=args.workers)
    raw = runner.run(
        fn=_run_benchmark_by_index,
        seeds=list(range(len(configs))),
        fn_kwargs=dict(configs=configs, n_reps=n_reps, gamma=0.95),
        desc="time-benchmark",
    )

    # Organize results
    results = {}
    for res in raw:
        env_key = res["env_name"]
        if env_key == "fourrooms":
            env_key = f"fourrooms_S{res['n_states']}"
        elif res["env_name"] == "garnet":
            env_key = f"garnet_S{res['n_states']}"
        if env_key not in results:
            results[env_key] = {}
        results[env_key][res["agent_name"]] = {
            "median_ms": res["median_ms"],
            "std_ms": res["std_ms"],
            "n_reps": res["n_reps"],
            "K": res["K"],
        }

    # Print results
    print("-" * 70)
    for env_key, agents_dict in results.items():
        for agent_name, metrics in agents_dict.items():
            print(
                f"[{env_key.upper():14s}] "
                f"{agent_name:12s} K={metrics.get('K', '?'):<4d}  "
                f"median={metrics['median_ms']:.2f}ms  std={metrics['std_ms']:.2f}ms  ({metrics['n_reps']} reps)"
            )
        print("-" * 70)

    out_path = (ROOT / "results" / "misc" / "time_benchmark" / "time.json").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
