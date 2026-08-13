"""RQ7: GPS block updates versus simultaneous top-m PS updates."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "misc"))

from gta_ps.algorithms import (
    BatchedPriorityModule,
    DpHost,
    GpsH2,
    GtaPsFa,
    LinearStateProjection,
    StandardPS,
    VaST,
    make_agent,
)
from gta_ps.envs.fourrooms import FourRoomsTabular
from gta_ps.utils.metrics import value_error_l1
from plot_academic import apply_style, get_palette, get_utility_color


BATCH_SIZES = (1, 2, 4, 8, 16)
ABSTRACTION_SIZES = (24, 40, 64)
ABSTRACTION_BATCH_SIZE = 1
PLOT_METRIC = "shift_end_l1"
GTA_TOP_Q: int | None = None
N_PEAKS_PER_ROOM = 2
SIGMA_RANGE = (0.5, 2.0)
ROOM_PEAK_RANGES = (
    (0.6, 1.2),
    (0.3, 0.6),
    (0.2, 0.8),
    (0.25, 0.5),
)
GTA_BETA_KWARGS = {
    "alpha": 0.051719,
    "beta1": 0.159673,
    "beta2": 1.361556,
    "eta": 0.018037,
    "b": 2.849322,
    "arnoldi_k": 15,
}
GTA_FA_KWARGS = {
    "alpha": 0.15,
    "beta1": 0.24,
    "beta2": 1.0,
    "eta": 0.026315789473684237,
    "gta_b": 4.0,
    "arnoldi_k": 15,
    "adaptive_beta": True,
}


def _warm_start(agent: DpHost | VaST | GtaPsFa, value: np.ndarray) -> None:
    if isinstance(agent, DpHost):
        agent.V = value.copy()
        agent.module.V = value.copy()
    else:
        agent.warm_start(value)


def _make_agents(
    env: FourRoomsTabular,
    gamma: float,
    batch_sizes: tuple[int, ...],
    abstraction_sizes: tuple[int, ...],
    gta_beta_kwargs: dict[str, float],
) -> dict[str, DpHost | VaST | GtaPsFa]:
    agents: dict[str, DpHost | VaST | GtaPsFa] = {}
    for size in batch_sizes:
        agents[f"standard_ps_batch_{size}"] = DpHost.from_arrays(
            env.P,
            env.r,
            gamma,
            BatchedPriorityModule(
                StandardPS(env.P, env.r, gamma=gamma),
                batch_size=size,
            ),
        )
        agents[f"gps_h2_block_{size}"] = DpHost.from_arrays(
            env.P,
            env.r,
            gamma,
            GpsH2(env.P, env.r, gamma=gamma, batch_size=size),
        )
        gta_agent = make_agent(
            "gta_beta",
            P=env.P,
            r=env.r,
            gamma=gamma,
            q=GTA_TOP_Q,
            **gta_beta_kwargs,
        )
        gta_agent.module = BatchedPriorityModule(gta_agent.module, batch_size=size)
        agents[f"gta_beta_batch_{size}"] = gta_agent

    state_features = np.asarray(env._pos_arr, dtype=float)
    for abstraction_size in abstraction_sizes:
        agents[
            f"vast_vi_a_{abstraction_size}_batch_{ABSTRACTION_BATCH_SIZE}"
        ] = VaST(
            env.P,
            env.r,
            state_features,
            gamma=gamma,
            a=abstraction_size,
            batch_size=ABSTRACTION_BATCH_SIZE,
        )
        agents[
            f"gta_ps_fa_a_{abstraction_size}_batch_{ABSTRACTION_BATCH_SIZE}"
        ] = GtaPsFa(
            env.P,
            env.r,
            state_features,
            gamma=gamma,
            a=abstraction_size,
            batch_size=ABSTRACTION_BATCH_SIZE,
            q=GTA_TOP_Q,
            **GTA_FA_KWARGS,
        )
    return agents


def _run_seed(
    seed: int,
    room_size: int,
    k: int,
    gamma: float,
    budget: int,
    n_shifts: int,
    steps_per_shift: int,
    batch_sizes: tuple[int, ...],
    abstraction_sizes: tuple[int, ...],
    gta_beta_kwargs: dict[str, float],
) -> dict[str, Any]:
    env = FourRoomsTabular(
        k=k,
        gamma=gamma,
        room_size=room_size,
        seed=seed,
        n_peaks_per_room=N_PEAKS_PER_ROOM,
        sigma_range=SIGMA_RANGE,
        room_peak_ranges=ROOM_PEAK_RANGES,
    )
    agents = _make_agents(
        env,
        gamma,
        batch_sizes,
        abstraction_sizes,
        gta_beta_kwargs,
    )
    pre_shift_value = env.value_iteration(env.r)
    for agent in agents.values():
        _warm_start(agent, pre_shift_value)

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


def _aggregate(
    seed_results: list[dict[str, Any]],
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    first_metrics = seed_results[0]["metrics"]
    for method in first_metrics:
        aggregate[method] = {}
        for metric in first_metrics[method]:
            per_seed = np.array(
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
    abstraction_sizes: tuple[int, ...],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    palette = get_palette("line", extra_keys=["gps_h2"])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    for axis in axes:
        axis.set_facecolor(get_utility_color("bg_gray"))
        axis.grid(True, color=get_utility_color("grid"))
        axis.tick_params(labelsize=8)

    scheduler_axis, abstraction_axis = axes
    series = (
        ("standard_ps", "standard_ps_batch_", "Standard-PS", "o"),
        ("gps_h2", "gps_h2_block_", "GPS H2", "s"),
        ("gta_beta", "gta_beta_batch_", "GTA-PS", "^"),
    )
    for color_key, prefix, label, marker in series:
        means = np.array(
            [aggregate[f"{prefix}{size}"][PLOT_METRIC]["mean"] for size in batch_sizes]
        )
        errors = np.array(
            [
                aggregate[f"{prefix}{size}"][PLOT_METRIC].get(
                    "ci95",
                    1.96
                    * aggregate[f"{prefix}{size}"][PLOT_METRIC]["std"]
                    / np.sqrt(aggregate[f"{prefix}{size}"][PLOT_METRIC]["n"]),
                )
                for size in batch_sizes
            ]
        )
        scheduler_axis.errorbar(
            batch_sizes,
            means,
            yerr=errors,
            color=palette[color_key],
            marker=marker,
            capsize=3,
            linewidth=1.2,
            elinewidth=0.8,
            markersize=4.5,
            label=label,
        )
    scheduler_axis.set_xscale("log", base=2)
    scheduler_axis.set_xticks(batch_sizes)
    scheduler_axis.set_xticklabels([str(size) for size in batch_sizes])
    scheduler_axis.set_xlabel("Batch or block size $m$", fontsize=9)
    scheduler_axis.set_ylabel("Mean post-shift $L_1$ error", fontsize=9)
    scheduler_axis.set_title("Priority schedulers", fontsize=10)
    scheduler_axis.legend(fontsize=7)

    abstraction_palette = get_palette(
        "line",
        extra_keys=["standard_ps_fa", "gta_ps_fa"],
    )
    for method, label, marker, linestyle, color_key in (
        ("vast_vi", "Std-PS-FA", "D", "--", "standard_ps_fa"),
        ("gta_ps_fa", "GTA-PS-FA", "P", "-", "gta_ps_fa"),
    ):
        means = np.array(
            [
                aggregate[
                    f"{method}_a_{abstraction_size}_batch_{ABSTRACTION_BATCH_SIZE}"
                ][PLOT_METRIC]["mean"]
                for abstraction_size in abstraction_sizes
            ]
        )
        errors = np.array(
            [
                aggregate[
                    f"{method}_a_{abstraction_size}_batch_{ABSTRACTION_BATCH_SIZE}"
                ][PLOT_METRIC]["ci95"]
                for abstraction_size in abstraction_sizes
            ]
        )
        abstraction_axis.errorbar(
            abstraction_sizes,
            means,
            yerr=errors,
            color=abstraction_palette[color_key],
            marker=marker,
            linestyle=linestyle,
            capsize=2,
            linewidth=1.1,
            elinewidth=0.7,
            markersize=4,
            label=label,
        )
    abstraction_axis.set_xticks(abstraction_sizes)
    abstraction_axis.set_xticklabels([str(size) for size in abstraction_sizes])
    abstraction_axis.set_xlabel("Abstract states $a$", fontsize=9)
    abstraction_axis.set_ylabel("Mean post-shift $L_1$ error", fontsize=9)
    abstraction_axis.set_title("State abstraction", fontsize=10)
    abstraction_axis.legend(fontsize=5.8, ncol=2)

    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".pdf"))
    fig.savefig(output_path.with_suffix(".png"))
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--room-size", type=int, default=5)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--n-shifts", type=int, default=6)
    parser.add_argument("--steps-per-shift", type=int, default=8)
    parser.add_argument("--gta-alpha", type=float, default=GTA_BETA_KWARGS["alpha"])
    parser.add_argument("--gta-beta1", type=float, default=GTA_BETA_KWARGS["beta1"])
    parser.add_argument("--gta-beta2", type=float, default=GTA_BETA_KWARGS["beta2"])
    parser.add_argument("--gta-eta", type=float, default=GTA_BETA_KWARGS["eta"])
    parser.add_argument("--gta-b", type=float, default=GTA_BETA_KWARGS["b"])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=BATCH_SIZES)
    parser.add_argument(
        "--abstraction-sizes",
        type=int,
        nargs="+",
        default=ABSTRACTION_SIZES,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "additional_evidence" / "gps_batch_topq",
    )
    parser.add_argument(
        "--output-stem",
        default="rq7_gps_batch_topq",
        help="Non-overwriting stem for JSON, PDF, and PNG artifacts.",
    )
    parser.add_argument(
        "--plot-from",
        type=Path,
        help="Regenerate the figure from an existing RQ7 JSON artifact.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.plot_from is not None:
        saved = json.loads(args.plot_from.read_text(encoding="utf-8"))
        protocol = saved["protocol"]
        args.out_dir.mkdir(parents=True, exist_ok=True)
        _plot(
            saved["results"],
            tuple(protocol["batch_sizes"]),
            tuple(protocol["abstraction_sizes"]),
            args.out_dir / args.output_stem,
        )
        print(f"plot_source={args.plot_from}")
        return

    if args.seeds < 2:
        raise ValueError("seeds must be at least 2 to report a standard deviation")
    if args.seed_start < 0 or args.workers < 1:
        raise ValueError("seed-start must be non-negative and workers must be positive")
    if args.budget < 1 or args.n_shifts < 1 or args.steps_per_shift < 1:
        raise ValueError("budget, n-shifts, and steps-per-shift must be positive")
    if any(size < 1 for size in args.batch_sizes):
        raise ValueError("all batch sizes must be positive")
    if any(size < 1 for size in args.abstraction_sizes):
        raise ValueError("all abstraction sizes must be positive")

    batch_sizes = tuple(args.batch_sizes)
    abstraction_sizes = tuple(args.abstraction_sizes)
    gta_beta_kwargs = {
        "alpha": args.gta_alpha,
        "beta1": args.gta_beta1,
        "beta2": args.gta_beta2,
        "eta": args.gta_eta,
        "b": args.gta_b,
        "arnoldi_k": GTA_BETA_KWARGS["arnoldi_k"],
    }
    seeds = tuple(range(args.seed_start, args.seed_start + args.seeds))
    run_seed = partial(
        _run_seed,
        room_size=args.room_size,
        k=args.k,
        gamma=args.gamma,
        budget=args.budget,
        n_shifts=args.n_shifts,
        steps_per_shift=args.steps_per_shift,
        batch_sizes=batch_sizes,
        abstraction_sizes=abstraction_sizes,
        gta_beta_kwargs=gta_beta_kwargs,
    )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        seed_results = list(executor.map(run_seed, seeds))
    aggregate = _aggregate(seed_results)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reference_env = FourRoomsTabular(
        k=args.k,
        gamma=args.gamma,
        room_size=args.room_size,
        seed=0,
        n_peaks_per_room=N_PEAKS_PER_ROOM,
        sigma_range=SIGMA_RANGE,
        room_peak_ranges=ROOM_PEAK_RANGES,
    )
    abstract_state_counts = {
        str(size): int(
            np.unique(
                LinearStateProjection(size).fit_transform(reference_env._pos_arr)
            ).size
        )
        for size in abstraction_sizes
    }
    output = {
        "protocol": {
            "host": "oracle_dp",
            "env": "FourRoomsTabular",
            "states": reference_env.n_states,
            "room_size": args.room_size,
            "k": args.k,
            "gamma": args.gamma,
            "budget": args.budget,
            "n_shifts": args.n_shifts,
            "steps_per_shift": args.steps_per_shift,
            "seeds": seeds,
            "workers": args.workers,
            "batch_sizes": batch_sizes,
            "abstraction_batch_sizes": [ABSTRACTION_BATCH_SIZE],
            "abstraction_sizes": abstraction_sizes,
            "abstract_state_counts": abstract_state_counts,
            "standard_ps_update": "frozen top-m simultaneous",
            "gps_h2_update": "H2-prioritized block, fixed-order Gauss-Seidel",
            "gta_ps_update": "frozen top-m simultaneous",
            "gps_upsilon": 1.0 - args.gamma,
            "gta_top_q": GTA_TOP_Q,
            "gta_beta_parameters": gta_beta_kwargs,
            "gta_ps_fa_parameters": GTA_FA_KWARGS,
            "reward_profile": {
                "n_peaks_per_room": N_PEAKS_PER_ROOM,
                "sigma_range": SIGMA_RANGE,
                "room_peak_ranges": ROOM_PEAK_RANGES,
            },
            "initialization": "exact pre-shift value function",
            "plot_metric": PLOT_METRIC,
        },
        "results": aggregate,
        "seed_results": seed_results,
    }
    output_path = args.out_dir / args.output_stem
    artifact_paths = [
        output_path.with_suffix(suffix)
        for suffix in (".json", ".pdf", ".png")
    ]
    existing = [path for path in artifact_paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite artifacts: {existing}")
    results_path = output_path.with_suffix(".json")
    results_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    _plot(
        aggregate,
        batch_sizes,
        abstraction_sizes,
        output_path,
    )

    print(f"states={reference_env.n_states}")
    print(f"abstract_states={abstract_state_counts}")
    print(f"results={results_path}")


if __name__ == "__main__":
    main()
