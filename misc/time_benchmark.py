"""
Time complexity measurement and implementation benchmark.

Reports median wall-clock time over N_REPS repeated runs per agent, with a
warmup rep that is excluded from statistics. Each rep builds a fresh agent
(including _init_topology precompute) and plans for K backups after one
reward shift.

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
    # Try reaching up to find the real project root (e.g. if compiled in artifacts/dist/)
    for p in Path(__file__).resolve().parents:
        if (p / "configs").exists() and (p / "src").exists():
            ROOT = p
            break
sys.path.insert(0, str(ROOT / "src"))

from gta_ps.envs.garnet import GarnetEnv
from gta_ps.envs.fourrooms import FourRoomsTabular
from gta_ps.algorithms import make_agent
from gta_ps.utils.hyperparams import load_tuned_for_env

# Number of timed repetitions (warmup rep excluded)
N_REPS = 8


class TimeBenchmark:
    def __init__(self, env_name: str, n_states: int = 100, k_fourrooms: int = 1, gamma: float = 0.95):
        self.env_name = env_name
        self.gamma = gamma
        if env_name == "garnet":
            self._make_env = lambda seed: GarnetEnv(
                n_states=n_states, n_actions=10, branching=4,
                reward_support=4, gamma=gamma, seed=seed,
            )
        elif env_name == "fourrooms":
            self._make_env = lambda seed: FourRoomsTabular(k=k_fourrooms, gamma=gamma, seed=seed)
        else:
            raise ValueError(f"Unknown env: {env_name}")

        self.n_states = self._make_env(0).n_states

    def run(self, agent_name: str, K: int) -> dict:
        """Time agent_name over N_REPS reps. Returns dict with median/std/reps."""
        # Auto-load best-tuned params for this env; {} for standard_ps (no entry).
        tuned_by_variant = load_tuned_for_env(self.env_name)
        tuned_kwargs = tuned_by_variant.get(agent_name) or {}

        times_ms = []
        for rep in range(N_REPS + 1):  # +1 for warmup
            env = self._make_env(rep)
            r_new = env.shift_reward()
            agent = make_agent(
                agent_name, P=env.P, r=r_new, gamma=self.gamma, q=20,
                **tuned_kwargs,
            )
            t0 = time.perf_counter()
            agent.plan(K=K, r=r_new)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if rep > 0:  # discard warmup
                times_ms.append(elapsed_ms)

        med = round(statistics.median(times_ms), 4)
        std = round(statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0, 4)
        print(
            f"[{self.env_name.upper():9s} | |S|={self.n_states:<4d}] "
            f"{agent_name:12s} K={K:<4d}  "
            f"median={med:.2f}ms  std={std:.2f}ms  ({N_REPS} reps)"
        )
        return {"median_ms": med, "std_ms": std, "n_reps": N_REPS}


def main():
    print(f"=== Time Complexity Benchmark  ({N_REPS} reps + 1 warmup) ===")
    results = {}

    def safe_run(bench, agent_name, K):
        try:
            return bench.run(agent_name, K=K)
        except Exception as e:
            print(f"  Error during {agent_name}: {e}")
            return None

    agents = ["standard_ps", "gta_base", "gta_beta"]

    # 1. FourRooms
    bench_fr = TimeBenchmark(env_name="fourrooms")
    K_fr = max(1, bench_fr.n_states // 4)
    results["fourrooms"] = {a: safe_run(bench_fr, a, K_fr) for a in agents}
    print("-" * 70)

    # 2. GARNET S75
    bench_gr_small = TimeBenchmark(env_name="garnet", n_states=75)
    K_gr_small = max(1, bench_gr_small.n_states // 4)
    results["garnet_S75"] = {a: safe_run(bench_gr_small, a, K_gr_small) for a in agents}
    print("-" * 70)

    # 3. GARNET S200
    bench_gr_med = TimeBenchmark(env_name="garnet", n_states=200)
    K_gr_med = max(1, bench_gr_med.n_states // 4)
    results["garnet_S200"] = {a: safe_run(bench_gr_med, a, K_gr_med) for a in agents}

    out_path = (ROOT / "results" / "misc" / "time_benchmark" / "time.json").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
