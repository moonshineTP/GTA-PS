"""Measure GTA-PS residual and topology signal AUC by reward-shift hop distance."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "misc"))

from gta_ps.algorithms import DpHost, make_agent
from gta_ps.envs.fourrooms import FourRoomsTabular
from plot_academic import apply_style, get_utility_color


HOP_BINS = (
    ("0-2", 0, 2),
    ("3-5", 3, 5),
    ("6-8", 6, 8),
    ("9-11", 9, 11),
    ("12+", 12, None),
)
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
METRICS = ("xi_auc", "tau_auc", "beta_tau_auc", "topology_share")
COLORS = {
    "xi_auc": "#D55E00",
    "tau_auc": "#0072B2",
    "topology_share": "#009E73",
}


def shortest_hops_to_source(P: np.ndarray, source: int) -> np.ndarray:
    """Return directed shortest-path lengths from every state to ``source``."""
    transition = np.asarray(P, dtype=float)
    if transition.ndim != 3 or transition.shape[1] != transition.shape[2]:
        raise ValueError("P must have shape (actions, states, states)")
    n_states = transition.shape[1]
    if not 0 <= source < n_states:
        raise ValueError(f"source must lie in [0, {n_states}), got {source}")

    adjacency = np.any(transition > 0.0, axis=0)
    distances = np.full(n_states, -1, dtype=int)
    distances[source] = 0
    frontier = deque([source])
    while frontier:
        state = frontier.popleft()
        for predecessor in np.flatnonzero(adjacency[:, state]):
            predecessor = int(predecessor)
            if distances[predecessor] < 0:
                distances[predecessor] = distances[state] + 1
                frontier.append(predecessor)
    return distances


def normalized_auc(values: Iterable[float]) -> float:
    """Return trapezoidal AUC divided by the observed call horizon."""
    array = np.asarray(tuple(values), dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")
    if array.size == 1:
        return float(array[0])
    return float(np.trapezoid(array) / (array.size - 1))


def _bin_mask(distances: np.ndarray, lower: int, upper: int | None) -> np.ndarray:
    mask = distances >= lower
    if upper is not None:
        mask &= distances <= upper
    return mask


def _snapshot(module: Any, distances: np.ndarray) -> dict[str, dict[str, float]]:
    xi = np.abs(np.asarray(module.xi, dtype=float))
    tau = np.abs(np.asarray(module.tau, dtype=float))
    beta_tau = abs(float(module.beta_t)) * tau
    snapshot: dict[str, dict[str, float]] = {}
    for label, lower, upper in HOP_BINS:
        mask = _bin_mask(distances, lower, upper)
        if np.any(mask):
            snapshot[label] = {
                "xi": float(np.mean(xi[mask])),
                "tau": float(np.mean(tau[mask])),
                "beta_tau": float(np.mean(beta_tau[mask])),
                "states": int(np.count_nonzero(mask)),
            }
    return snapshot


def _warm_start(agent: DpHost, value: np.ndarray) -> None:
    agent.V = value.copy()
    agent.module.V = value.copy()


def run_seed(
    seed: int,
    *,
    room_size: int,
    k: int,
    gamma: float,
    budget: int,
    n_shifts: int,
    calls_per_shift: int,
) -> dict[str, Any]:
    """Run one E.2-matched GTA-PS trajectory and aggregate its six shifts."""
    env = FourRoomsTabular(
        room_size=room_size,
        k=k,
        gamma=gamma,
        seed=seed,
        n_peaks_per_room=N_PEAKS_PER_ROOM,
        sigma_range=SIGMA_RANGE,
        room_peak_ranges=ROOM_PEAK_RANGES,
    )
    agent = make_agent(
        "gta_beta",
        P=env.P,
        r=env.r,
        gamma=gamma,
        q=None,
        **GTA_BETA_KWARGS,
    )
    _warm_start(agent, env.value_iteration(env.r))

    previous_reward = env.r.copy()
    bin_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    shift_metadata: list[dict[str, Any]] = []
    for shift in range(n_shifts):
        reward = env.shift_reward()
        reward_delta = np.abs(reward - previous_reward)
        source = int(np.argmax(reward_delta))
        distances = shortest_hops_to_source(env.P, source)
        if np.any(distances < 0):
            raise RuntimeError("FourRooms transition graph is not source-reachable")

        agent.plan(K=0, r=reward)
        traces: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        state_counts: dict[str, int] = {}

        for call in range(calls_per_shift + 1):
            snapshot = _snapshot(agent.module, distances)
            for label, values in snapshot.items():
                traces[label]["xi"].append(values["xi"])
                traces[label]["tau"].append(values["tau"])
                traces[label]["beta_tau"].append(values["beta_tau"])
                state_counts[label] = int(values["states"])
            if call < calls_per_shift:
                agent.plan(K=budget)

        for label, signals in traces.items():
            xi_auc = normalized_auc(signals["xi"])
            tau_auc = normalized_auc(signals["tau"])
            beta_tau_auc = normalized_auc(signals["beta_tau"])
            bin_values[label]["xi_auc"].append(xi_auc)
            bin_values[label]["tau_auc"].append(tau_auc)
            bin_values[label]["beta_tau_auc"].append(beta_tau_auc)
            bin_values[label]["states"].append(float(state_counts[label]))

        shift_metadata.append(
            {
                "shift": shift,
                "source_state": source,
                "source_position": [int(x) for x in env._pos_list[source]],
                "max_reward_change": float(reward_delta[source]),
                "max_hop": int(np.max(distances)),
            }
        )
        previous_reward = reward.copy()

    bins: dict[str, dict[str, float]] = {}
    for label, values in bin_values.items():
        xi_auc = float(np.mean(values["xi_auc"]))
        tau_auc = float(np.mean(values["tau_auc"]))
        beta_tau_auc = float(np.mean(values["beta_tau_auc"]))
        bins[label] = {
            "xi_auc": xi_auc,
            "tau_auc": tau_auc,
            "beta_tau_auc": beta_tau_auc,
            "topology_share": beta_tau_auc / max(xi_auc + beta_tau_auc, 1e-12),
            "mean_states": float(np.mean(values["states"])),
            "observed_shifts": int(len(values["xi_auc"])),
        }

    return {
        "seed": seed,
        "bins": bins,
        "shifts": shift_metadata,
    }


def _mean_ci(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(tuple(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2:
        raise ValueError("at least two finite seed values are required")
    mean = float(np.mean(array))
    radius = float(
        student_t.ppf(0.975, df=array.size - 1)
        * np.std(array, ddof=1)
        / np.sqrt(array.size)
    )
    return {
        "mean": mean,
        "ci95": radius,
        "lower": mean - radius,
        "upper": mean + radius,
        "n": int(array.size),
    }


def summarize_seed_results(
    seed_results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build cross-seed intervals from per-seed, shift-averaged measurements."""
    summary: dict[str, dict[str, Any]] = {}
    for label, _lower, _upper in HOP_BINS:
        rows = [result["bins"][label] for result in seed_results if label in result["bins"]]
        if not rows:
            continue
        summary[label] = {
            metric: _mean_ci(row[metric] for row in rows)
            for metric in METRICS
        }
        summary[label]["mean_states"] = float(
            np.mean([row.get("mean_states", np.nan) for row in rows])
        )
    return summary


def _plot(summary: dict[str, dict[str, Any]], output_stem: Path) -> None:
    apply_style()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )
    labels = [label for label, _lower, _upper in HOP_BINS if label in summary]
    x = np.arange(len(labels), dtype=float)
    fig, (signal_axis, share_axis) = plt.subplots(
        1,
        2,
        figsize=(5.5, 2.35),
        gridspec_kw={"width_ratios": [1.15, 0.85]},
    )
    for axis in (signal_axis, share_axis):
        axis.set_facecolor(get_utility_color("bg_gray"))
        axis.grid(True, color=get_utility_color("grid"), alpha=0.35)

    for metric, marker, label in (
        ("xi_auc", "o", r"$A_\xi$"),
        ("tau_auc", "s", r"$A_\tau$"),
    ):
        means = np.asarray([summary[item][metric]["mean"] for item in labels])
        errors = np.asarray([summary[item][metric]["ci95"] for item in labels])
        signal_axis.errorbar(
            x,
            means,
            yerr=errors,
            color=COLORS[metric],
            marker=marker,
            capsize=2,
            label=label,
        )
    signal_axis.set_yscale("log")
    signal_axis.set_xticks(x, labels)
    signal_axis.set_xlabel("Directed hops to shift source")
    signal_axis.set_ylabel("Mean per-state signal AUC")
    signal_axis.legend()

    share_means = 100.0 * np.asarray(
        [summary[item]["topology_share"]["mean"] for item in labels]
    )
    share_errors = 100.0 * np.asarray(
        [summary[item]["topology_share"]["ci95"] for item in labels]
    )
    share_axis.errorbar(
        x,
        share_means,
        yerr=share_errors,
        color=COLORS["topology_share"],
        marker="^",
        capsize=2,
    )
    share_axis.set_xticks(x, labels)
    share_axis.set_xlabel("Directed hops to shift source")
    share_axis.set_ylabel(r"$A_{\beta\tau}/(A_\xi+A_{\beta\tau})$ (%)")
    share_axis.set_ylim(0.0, min(100.0, max(10.0, 1.15 * np.max(share_means + share_errors))))

    fig.tight_layout(pad=0.5, w_pad=1.2)
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def _write_csv(summary: dict[str, dict[str, Any]], path: Path) -> None:
    fields = ["hop_bin"]
    for metric in METRICS:
        fields.extend((f"{metric}_mean", f"{metric}_ci95", f"{metric}_n"))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label, _lower, _upper in HOP_BINS:
            if label not in summary:
                continue
            row: dict[str, Any] = {"hop_bin": label}
            for metric in METRICS:
                row[f"{metric}_mean"] = summary[label][metric]["mean"]
                row[f"{metric}_ci95"] = summary[label][metric]["ci95"]
                row[f"{metric}_n"] = summary[label][metric]["n"]
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--room-size", type=int, default=5)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--n-shifts", type=int, default=6)
    parser.add_argument("--calls-per-shift", type=int, default=25)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "rebuttal" / "xi_tau_hop_distance",
    )
    parser.add_argument("--output-stem", default="xi_tau_hop_distance")
    parser.add_argument(
        "--plot-from",
        type=Path,
        help="Regenerate PDF and PNG from an existing JSON artifact.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_stem = args.out_dir / args.output_stem

    if args.plot_from is not None:
        saved = json.loads(args.plot_from.read_text(encoding="utf-8"))
        _plot(saved["summary"], output_stem)
        print(f"plot_source={args.plot_from}")
        return

    if args.seeds < 2:
        raise ValueError("seeds must be at least 2")
    if args.seed_start < 0 or args.workers < 1:
        raise ValueError("seed-start must be non-negative and workers must be positive")
    if min(args.room_size, args.k, args.budget, args.n_shifts, args.calls_per_shift) < 1:
        raise ValueError("room size, k, budget, shifts, and calls must be positive")
    if not 0.0 < args.gamma < 1.0:
        raise ValueError("gamma must lie in (0, 1)")

    artifact_paths = [
        output_stem.with_suffix(suffix)
        for suffix in (".json", ".csv", ".pdf", ".png")
    ]
    existing = [path for path in artifact_paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite artifacts: {existing}")

    seeds = tuple(range(args.seed_start, args.seed_start + args.seeds))
    worker = partial(
        run_seed,
        room_size=args.room_size,
        k=args.k,
        gamma=args.gamma,
        budget=args.budget,
        n_shifts=args.n_shifts,
        calls_per_shift=args.calls_per_shift,
    )
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        seed_results = list(executor.map(worker, seeds))
    elapsed = time.perf_counter() - started
    summary = summarize_seed_results(seed_results)

    reference_env = FourRoomsTabular(
        room_size=args.room_size,
        k=args.k,
        gamma=args.gamma,
        seed=0,
        n_peaks_per_room=N_PEAKS_PER_ROOM,
        sigma_range=SIGMA_RANGE,
        room_peak_ranges=ROOM_PEAK_RANGES,
    )
    output = {
        "protocol": {
            "host": "oracle_dp",
            "environment": "FourRoomsTabular",
            "states": reference_env.n_states,
            "room_size": args.room_size,
            "k": args.k,
            "gamma": args.gamma,
            "budget_K": args.budget,
            "n_shifts": args.n_shifts,
            "calls_per_shift": args.calls_per_shift,
            "seeds": list(seeds),
            "workers": args.workers,
            "batch_size_m": 1,
            "gta_top_q": None,
            "gta_parameters": GTA_BETA_KWARGS,
            "reward_profile": {
                "n_peaks_per_room": N_PEAKS_PER_ROOM,
                "sigma_range": SIGMA_RANGE,
                "room_peak_ranges": ROOM_PEAK_RANGES,
            },
            "initialization": "exact pre-shift value function",
            "shift_source": "argmax absolute reward change",
            "hop_distance": "directed shortest path from state to shift source",
            "hop_bins": [
                {"label": label, "lower": lower, "upper": upper}
                for label, lower, upper in HOP_BINS
            ],
            "auc": "trapezoidal mean over call 0 through call 25",
            "aggregation": "mean states within hop bin, mean shifts within seed, Student-t 95% CI across seeds",
            "elapsed_seconds": elapsed,
        },
        "metric_definitions": {
            "xi_auc": "AUC of mean absolute residual xi per state",
            "tau_auc": "AUC of mean absolute topology signal tau per state",
            "beta_tau_auc": "AUC of mean absolute beta_t times tau per state",
            "topology_share": "beta_tau_auc / (xi_auc + beta_tau_auc)",
        },
        "summary": summary,
        "seed_results": seed_results,
    }
    output_stem.with_suffix(".json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    _write_csv(summary, output_stem.with_suffix(".csv"))
    _plot(summary, output_stem)
    print(f"results={output_stem.with_suffix('.json')}")
    print(f"elapsed_seconds={elapsed:.3f}")


if __name__ == "__main__":
    main()
