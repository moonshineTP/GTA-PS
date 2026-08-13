"""Tune the LU-Woodbury to GMRES crossover on sparse localized GARNETs."""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "misc"))

from gta_ps.algorithms.gta_ps import GtaPS
from gta_ps.envs.garnet import GarnetEnv
from plot_academic import apply_style, get_palette, get_utility_color


def _calibration_metrics(env: GarnetEnv) -> dict[str, float]:
    policy_kernel = env.P.mean(axis=0)
    degree = np.count_nonzero(policy_kernel > 0.0, axis=1)
    midpoint = env.n_states // 2
    cut_mass = (
        policy_kernel[:midpoint, midpoint:].sum()
        + policy_kernel[midpoint:, :midpoint].sum()
    ) / env.n_states
    return {
        "mean_out_degree": float(np.mean(degree)),
        "matrix_density": float(np.count_nonzero(policy_kernel) / policy_kernel.size),
        "half_cut_mass_per_state": float(cut_mass),
    }


def _apply_rank_k_change(module: GtaPS, rank: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    states = rng.choice(module.n_states, size=min(rank, module.n_states), replace=False)
    for state in states:
        advantage = rng.normal(size=module.n_actions)
        delta_pi = module.gta_policy_update(int(state), advantage)
        module._pending_delta_pi[state] += delta_pi
    module._has_pending_topo = True


def _time_solver(
    env: GarnetEnv,
    *,
    mode: str,
    rank: int,
    repeats: int,
    seed: int,
) -> list[float]:
    threshold = env.n_states if mode == "lu_woodbury" else 0
    timings: list[float] = []
    for repeat in range(repeats):
        module = GtaPS(
            env.P,
            env.r,
            gamma=env.gamma,
            q=None,
            arnoldi_k=min(8, max(1, env.n_states - 2)),
            solver_threshold=threshold,
        )
        _apply_rank_k_change(module, rank, seed + 10_007 * repeat)
        started = time.perf_counter()
        module.topology_update_lazy()
        timings.append(time.perf_counter() - started)
    return timings


def _benchmark_seed(
    seed: int,
    *,
    n_states: int,
    repeats: int,
    rank: int,
    branching: int,
    n_nbr_rate: float,
    reversible: float,
    gamma: float,
) -> dict[str, Any]:
    env = GarnetEnv(
        n_states=n_states,
        n_actions=4,
        branching=branching,
        reward_support=max(3, n_states // 50),
        gamma=gamma,
        n_nbr_rate=n_nbr_rate,
        reversible=reversible,
        reward_locality=True,
        seed=seed,
    )
    result: dict[str, Any] = {
        "seed": seed,
        "calibration": _calibration_metrics(env),
    }
    for mode in ("lu_woodbury", "gmres"):
        timings = _time_solver(
            env,
            mode=mode,
            rank=rank,
            repeats=repeats,
            seed=seed,
        )
        result[mode] = float(np.mean(timings))
    return result


def _benchmark_size(
    n_states: int,
    *,
    seeds: tuple[int, ...],
    repeats: int,
    rank: int,
    branching: int,
    n_nbr_rate: float,
    reversible: float,
    gamma: float,
    executor: ProcessPoolExecutor,
) -> dict[str, Any]:
    benchmark_seed = partial(
        _benchmark_seed,
        n_states=n_states,
        repeats=repeats,
        rank=rank,
        branching=branching,
        n_nbr_rate=n_nbr_rate,
        reversible=reversible,
        gamma=gamma,
    )
    seed_records = list(executor.map(benchmark_seed, seeds))
    records = {
        mode: [float(record[mode]) for record in seed_records]
        for mode in ("lu_woodbury", "gmres")
    }
    calibrations = [record["calibration"] for record in seed_records]

    summary: dict[str, Any] = {}
    for mode, values in records.items():
        sample_std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        critical = (
            float(student_t.ppf(0.975, df=len(values) - 1))
            if len(values) > 1
            else 0.0
        )
        summary[mode] = {
            "mean_seconds": float(np.mean(values)),
            "median_seconds": float(np.median(values)),
            "std_seconds": sample_std,
            "ci95_seconds": critical * sample_std / np.sqrt(len(values)),
            "seed_samples": len(values),
            "repeats_per_seed": repeats,
        }
    summary["calibration"] = {
        key: float(np.mean([item[key] for item in calibrations]))
        for key in calibrations[0]
    }
    summary["seed_records"] = seed_records
    summary["rank_k"] = rank
    lu_time = summary["lu_woodbury"]["mean_seconds"]
    gmres_time = summary["gmres"]["mean_seconds"]
    summary["log_time_ratio"] = float(np.log(lu_time / gmres_time))
    return summary


def _ternary_search(
    sizes: tuple[int, ...],
    evaluate,
) -> tuple[int, dict[int, dict[str, Any]]]:
    cache: dict[int, dict[str, Any]] = {}

    def objective(index: int) -> float:
        size = sizes[index]
        if size not in cache:
            cache[size] = evaluate(size)
        return abs(cache[size]["log_time_ratio"])

    low, high = 0, len(sizes) - 1
    while high - low > 3:
        third = (high - low) // 3
        left = low + third
        right = high - third
        if objective(left) <= objective(right):
            high = right - 1
        else:
            low = left + 1
    for index in range(low, high + 1):
        objective(index)

    best = min(cache, key=lambda size: abs(cache[size]["log_time_ratio"]))
    return best, cache


def _plot(
    measurements: dict[int, dict[str, Any]],
    crossover: int,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    palette = get_palette("line", extra_keys=["lu_woodbury", "gmres"])
    sizes = np.array(sorted(measurements))
    fig, axis = plt.subplots(figsize=(5.5, 3.3))
    axis.set_facecolor(get_utility_color("bg_gray"))
    axis.grid(True, color=get_utility_color("grid"))
    axis.tick_params(labelsize=8)
    for mode, label, marker in (
        ("lu_woodbury", "LU + Woodbury", "o"),
        ("gmres", "GMRES", "s"),
    ):
        means = np.array(
            [measurements[int(size)][mode]["mean_seconds"] * 1e3 for size in sizes]
        )
        intervals = np.array(
            [measurements[int(size)][mode]["ci95_seconds"] * 1e3 for size in sizes]
        )
        axis.fill_between(
            sizes,
            means - intervals,
            means + intervals,
            color=palette[mode],
            alpha=0.16,
            linewidth=0,
        )
        axis.errorbar(
            sizes,
            means,
            yerr=intervals,
            color=palette[mode],
            marker=marker,
            capsize=3,
            elinewidth=0.8,
            linewidth=1.2,
            markersize=4.8,
            label=label,
        )
    crossover_record = measurements[crossover]
    crossover_time = np.sqrt(
        crossover_record["lu_woodbury"]["mean_seconds"]
        * crossover_record["gmres"]["mean_seconds"]
    ) * 1e3
    axis.scatter(
        [crossover],
        [crossover_time],
        marker="*",
        s=95,
        color=palette["lu_woodbury"],
        edgecolor="black",
        linewidth=0.5,
        zorder=5,
        label=f"crossover candidate: {crossover}",
    )
    axis.set_yscale("log")
    axis.set_xlabel("GARNET states", fontsize=9)
    axis.set_ylabel("Mean topology refresh time (ms)", fontsize=9)
    axis.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".pdf"))
    fig.savefig(output_path.with_suffix(".png"))
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-states", type=int, default=100)
    parser.add_argument("--max-states", type=int, default=1_000)
    parser.add_argument("--step", type=int, default=50)
    parser.add_argument(
        "--state-sizes",
        type=int,
        nargs="+",
        help="Evaluate this exact state-size grid instead of ternary search.",
    )
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--branching", type=int, default=3)
    parser.add_argument("--n-nbr-rate", type=float, default=0.10)
    parser.add_argument("--reversible", type=float, default=0.20)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "additional_evidence" / "solver_threshold",
    )
    parser.add_argument(
        "--plot-from",
        type=Path,
        help="Regenerate the figure from an existing threshold JSON artifact.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.plot_from is not None:
        saved = json.loads(args.plot_from.read_text(encoding="utf-8"))
        measurements = {
            int(size): record for size, record in saved["measurements"].items()
        }
        args.out_dir.mkdir(parents=True, exist_ok=True)
        _plot(
            measurements,
            int(saved["crossover_candidate"]),
            args.out_dir / "solver_threshold_ablation",
        )
        print(f"plot_source={args.plot_from}")
        return

    if args.min_states < 4 or args.max_states < args.min_states or args.step < 1:
        raise ValueError("state-size range is invalid")
    if (
        args.seeds < 1
        or args.repeats < 1
        or args.rank < 1
        or args.workers < 1
    ):
        raise ValueError("seeds, repeats, rank, and workers must be positive")
    if args.seed_start < 0:
        raise ValueError("seed-start must be non-negative")
    sizes = (
        tuple(args.state_sizes)
        if args.state_sizes is not None
        else tuple(range(args.min_states, args.max_states + 1, args.step))
    )
    if any(size < 4 for size in sizes) or len(set(sizes)) != len(sizes):
        raise ValueError("state sizes must be unique and at least four")
    if args.state_sizes is None and len(sizes) < 4:
        raise ValueError("ternary search requires at least four candidate state sizes")
    seeds = tuple(range(args.seed_start, args.seed_start + args.seeds))

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        def evaluate(size: int) -> dict[str, Any]:
            record = _benchmark_size(
                size,
                seeds=seeds,
                repeats=args.repeats,
                rank=args.rank,
                branching=args.branching,
                n_nbr_rate=args.n_nbr_rate,
                reversible=args.reversible,
                gamma=args.gamma,
                executor=executor,
            )
            print(
                f"n={size} "
                f"rank={args.rank} "
                f"lu={record['lu_woodbury']['mean_seconds'] * 1e3:.3f}ms "
                f"gmres={record['gmres']['mean_seconds'] * 1e3:.3f}ms"
            )
            return record

        if args.state_sizes is None:
            crossover, measurements = _ternary_search(sizes, evaluate)
        else:
            measurements = {size: evaluate(size) for size in sizes}
            crossover = min(
                measurements,
                key=lambda size: abs(measurements[size]["log_time_ratio"]),
            )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "protocol": {
            "search": (
                "discrete ternary search minimizing absolute log timing ratio"
                if args.state_sizes is None
                else "explicit state-size grid minimizing absolute log timing ratio"
            ),
            "candidate_sizes": sizes,
            "seeds": seeds,
            "workers": args.workers,
            "repeats": args.repeats,
            "rank_k": args.rank,
            "branching": args.branching,
            "n_nbr_rate": args.n_nbr_rate,
            "reversible": args.reversible,
            "gamma": args.gamma,
            "timed_region": "topology_update_lazy including tau refresh",
        },
        "crossover_candidate": crossover,
        "measurements": {str(size): record for size, record in sorted(measurements.items())},
    }
    results_path = args.out_dir / "solver_threshold_ablation.json"
    results_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    _plot(measurements, crossover, args.out_dir / "solver_threshold_ablation")
    print(f"crossover_candidate={crossover}")
    print(f"results={results_path}")


if __name__ == "__main__":
    main()
