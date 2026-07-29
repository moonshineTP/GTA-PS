"""GPS-H2 versus GTA-PS batch updates on the established GARNET-75 task."""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import yaml
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "misc"))

from gta_ps.algorithms import BatchedPriorityModule, DpHost, GpsH2, make_agent
from gta_ps.envs.garnet import GarnetEnv
from gta_ps.utils.metrics import value_error_l1
from plot_academic import apply_style, get_palette, get_utility_color


BATCH_SIZES = (1, 2, 4, 8, 16)
PLOT_METRIC = "shift_end_l1"
GTA_TOP_Q: int | None = None


def load_garnet_config(path: Path) -> dict[str, Any]:
    """Load the environment and tuned GTA-beta parameters from GARNET-75 YAML."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if payload.get("env") != "garnet":
        raise ValueError(f"expected a GARNET config, got {payload.get('env')!r}")

    environment = {
        key: payload[key]
        for key in (
            "n_states",
            "n_actions",
            "branching",
            "n_nbr_rate",
            "reward_support",
            "reward_locality",
        )
    }
    tuned = payload["tuned"]["gta_beta"]
    environment["gamma"] = float(payload["gamma"])
    environment["gta_beta"] = {
        key: float(tuned[key])
        for key in ("alpha", "beta1", "beta2", "eta", "b")
    }
    environment["gta_beta"]["arnoldi_k"] = int(payload["arnoldi_k"])
    return environment


def _make_environment(config: dict[str, Any], seed: int) -> GarnetEnv:
    return GarnetEnv(
        n_states=int(config["n_states"]),
        n_actions=int(config["n_actions"]),
        branching=int(config["branching"]),
        reward_support=int(config["reward_support"]),
        gamma=float(config["gamma"]),
        n_nbr_rate=float(config["n_nbr_rate"]),
        reward_locality=bool(config["reward_locality"]),
        seed=seed,
    )


def _make_agents(
    env: GarnetEnv,
    batch_sizes: tuple[int, ...],
    gta_beta_kwargs: dict[str, float],
) -> dict[str, DpHost]:
    agents: dict[str, DpHost] = {}
    for size in batch_sizes:
        agents[f"gps_h2_block_{size}"] = DpHost.from_arrays(
            env.P,
            env.r,
            env.gamma,
            GpsH2(env.P, env.r, gamma=env.gamma, batch_size=size),
        )
        gta_agent = make_agent(
            "gta_beta",
            P=env.P,
            r=env.r,
            gamma=env.gamma,
            q=GTA_TOP_Q,
            **gta_beta_kwargs,
        )
        gta_agent.module = BatchedPriorityModule(gta_agent.module, batch_size=size)
        agents[f"gta_beta_batch_{size}"] = gta_agent
    return agents


def run_seed(
    seed: int,
    *,
    config: dict[str, Any],
    gamma: float,
    budget: int,
    n_shifts: int,
    steps_per_shift: int,
    batch_sizes: tuple[int, ...],
) -> dict[str, Any]:
    """Run paired GPS-H2 and GTA-PS traces on one GARNET-75 instance."""
    if not np.isclose(gamma, float(config["gamma"])):
        raise ValueError("gamma must match the established GARNET-75 config")
    env = _make_environment(config, seed)
    agents = _make_agents(env, batch_sizes, config["gta_beta"])
    pre_shift_value = env.value_iteration(env.r)
    for agent in agents.values():
        agent.V = pre_shift_value.copy()
        agent.module.V = pre_shift_value.copy()

    errors = {
        name: {"first_plan_l1": [], "shift_end_l1": []}
        for name in agents
    }
    reward_sequence: list[list[float]] = []
    for _ in range(n_shifts):
        reward = env.shift_reward()
        reward_sequence.append(reward.tolist())
        value_star = env.value_iteration(reward)
        for name, agent in agents.items():
            value = agent.plan(K=budget, r=reward)
            errors[name]["first_plan_l1"].append(value_error_l1(value, value_star))
            for _ in range(steps_per_shift - 1):
                value = agent.plan(K=budget)
            errors[name]["shift_end_l1"].append(value_error_l1(value, value_star))
    return {
        "seed": seed,
        "reward_sequence": reward_sequence,
        "metrics": errors,
    }


def _aggregate(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for method, method_metrics in seed_results[0]["metrics"].items():
        aggregate[method] = {}
        for metric in method_metrics:
            per_seed = np.asarray(
                [
                    np.mean(result["metrics"][method][metric])
                    for result in seed_results
                ],
                dtype=float,
            )
            sample_std = float(np.std(per_seed, ddof=1))
            critical = float(student_t.ppf(0.975, df=per_seed.size - 1))
            aggregate[method][metric] = {
                "mean": float(np.mean(per_seed)),
                "std": sample_std,
                "ci95": critical * sample_std / np.sqrt(per_seed.size),
                "n": int(per_seed.size),
            }
    return aggregate


def _plot(
    aggregate: dict[str, Any],
    batch_sizes: tuple[int, ...],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    palette = get_palette("line", extra_keys=["gps_h2"])
    figure, axis = plt.subplots(figsize=(5.1, 3.5))
    axis.set_facecolor(get_utility_color("bg_gray"))
    axis.grid(True, color=get_utility_color("grid"))
    for color_key, prefix, label, marker in (
        ("gps_h2", "gps_h2_block_", "GPS H2", "s"),
        ("gta_beta", "gta_beta_batch_", "GTA-PS", "^"),
    ):
        means = np.asarray(
            [aggregate[f"{prefix}{size}"][PLOT_METRIC]["mean"] for size in batch_sizes]
        )
        intervals = np.asarray(
            [aggregate[f"{prefix}{size}"][PLOT_METRIC]["ci95"] for size in batch_sizes]
        )
        axis.errorbar(
            batch_sizes,
            means,
            yerr=intervals,
            color=palette[color_key],
            marker=marker,
            capsize=3,
            linewidth=1.2,
            elinewidth=0.8,
            markersize=4.5,
            label=label,
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(batch_sizes)
    axis.set_xticklabels([str(size) for size in batch_sizes])
    axis.set_xlabel("Batch or block size $m$")
    axis.set_ylabel("Mean post-shift $L_1$ error")
    axis.set_title("GARNET-75 priority schedulers")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path.with_suffix(".pdf"))
    figure.savefig(output_path.with_suffix(".png"))
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "garnet_s75.yaml",
    )
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--n-shifts", type=int, default=6)
    parser.add_argument("--steps-per-shift", type=int, default=25)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=BATCH_SIZES)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "rebuttal" / "gps_batch_garnet75",
    )
    parser.add_argument(
        "--output-stem",
        default="rq7_gps_batch_garnet75_s100_c25_m1to16",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seeds < 2:
        raise ValueError("seeds must be at least 2")
    if args.seed_start < 0 or args.workers < 1:
        raise ValueError("seed-start must be non-negative and workers must be positive")
    if args.budget < 1 or args.n_shifts < 1 or args.steps_per_shift < 1:
        raise ValueError("budget, n-shifts, and steps-per-shift must be positive")
    if any(size < 1 for size in args.batch_sizes):
        raise ValueError("all batch sizes must be positive")

    output_path = args.out_dir / args.output_stem
    artifact_paths = [
        output_path.with_suffix(suffix)
        for suffix in (".json", ".pdf", ".png")
    ]
    existing = [path for path in artifact_paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite artifacts: {existing}")

    config = load_garnet_config(args.config)
    batch_sizes = tuple(args.batch_sizes)
    seeds = tuple(range(args.seed_start, args.seed_start + args.seeds))
    worker = partial(
        run_seed,
        config=config,
        gamma=float(config["gamma"]),
        budget=args.budget,
        n_shifts=args.n_shifts,
        steps_per_shift=args.steps_per_shift,
        batch_sizes=batch_sizes,
    )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        seed_results = list(executor.map(worker, seeds))
    aggregate = _aggregate(seed_results)

    output = {
        "protocol": {
            "host": "oracle_dp",
            "env": "GarnetEnv",
            "config_source": str(args.config.resolve()),
            "states": int(config["n_states"]),
            "actions": int(config["n_actions"]),
            "branching": int(config["branching"]),
            "n_nbr_rate": float(config["n_nbr_rate"]),
            "reward_support": int(config["reward_support"]),
            "reward_locality": bool(config["reward_locality"]),
            "gamma": float(config["gamma"]),
            "budget": args.budget,
            "n_shifts": args.n_shifts,
            "steps_per_shift": args.steps_per_shift,
            "seeds": seeds,
            "workers": args.workers,
            "batch_sizes": batch_sizes,
            "gps_upsilon": 1.0 - float(config["gamma"]),
            "gta_top_q": GTA_TOP_Q,
            "gta_beta_parameters": config["gta_beta"],
            "initialization": "exact pre-shift value function",
            "plot_metric": PLOT_METRIC,
        },
        "results": aggregate,
        "seed_results": seed_results,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path.with_suffix(".json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    _plot(aggregate, batch_sizes, output_path)
    print(f"results={output_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
