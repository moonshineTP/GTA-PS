"""RQ8: backup-time recovery, queue utility, and topology magnitude audit."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "misc"))
sys.path.insert(0, str(ROOT / "threads"))

from gta_ps.algorithms import DpHost, make_agent
from gta_ps.envs.fourrooms import FourRoomsTabular
from gta_ps.utils.metrics import value_error_l1
from plot_academic import apply_style, get_palette, get_utility_color
from rq8_eta_tuning import EPSILON, generate_sharp_reward_sequence, ndcg_at_k, queue_order
from runner import RunManager


DEFAULT_ROOM_SIZE = 5
GTA_KWARGS = {
    "alpha": 0.051719,
    "beta1": 0.159673,
    "beta2": 1.361556,
    "b": 2.849322,
    "arnoldi_k": 15,
}
RESIDUAL_FLOOR = 1e-4
METHODS = ("standard_ps", "gta_beta")
LABELS = {"standard_ps": "Standard-PS", "gta_beta": "GTA-PS"}


def _warm_start(agent: DpHost, value: np.ndarray) -> None:
    agent.V = value.copy()
    agent.module.V = value.copy()


def standard_queue_order(module: Any) -> np.ndarray:
    """Recover the current Standard-PS priority order without popping its heap."""
    xi = np.asarray(module.xi, dtype=float)
    states = np.flatnonzero(xi > 1e-6)
    return states[np.lexsort((states, -xi[states]))]


def counterfactual_backup_gains(
    P: np.ndarray,
    reward: np.ndarray,
    gamma: float,
    value: np.ndarray,
    value_star: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    """Return one-backup reductions in L1 value error for executable states."""
    value = np.asarray(value, dtype=float)
    value_star = np.asarray(value_star, dtype=float)
    gains = np.zeros(value.size, dtype=float)
    if candidates.size == 0:
        return gains
    expected = (P[:, candidates, :] * value[None, None, :]).sum(axis=2)
    targets = reward[candidates] + gamma * expected.max(axis=0)
    error_before = float(value_error_l1(value, value_star))
    error_after = error_before - np.abs(value[candidates] - value_star[candidates])
    error_after += np.abs(targets - value_star[candidates])
    gains[candidates] = np.maximum(error_before - error_after, 0.0)
    return gains


def prefix_auc(errors: Iterable[float]) -> float:
    values = np.asarray(tuple(errors), dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("errors must be a non-empty one-dimensional sequence")
    return float(np.mean(values))


def trace_snapshot(
    *,
    method: str,
    agent: DpHost,
    P: np.ndarray,
    reward: np.ndarray,
    value_star: np.ndarray,
    initial_error: float,
    seed: int,
    shift: int,
    backup: int,
    q: int,
) -> dict[str, Any]:
    """Measure recovery and the utility of the queue before the next backup."""
    module = agent.module
    raw_error = float(value_error_l1(agent.V, value_star))
    if method == "standard_ps":
        ranking = standard_queue_order(module)
        gains = counterfactual_backup_gains(
            P, reward, agent.gamma, agent.V, value_star, ranking
        )
        relevance = np.zeros(agent.n_states, dtype=float)
        relevance[ranking] = gains[ranking]
        return {
            "method": method,
            "seed": seed,
            "shift": shift,
            "backup": backup,
            "value_l1": raw_error,
            "tracking_error": raw_error / agent.n_states,
            "normalized_error": raw_error / max(initial_error, EPSILON),
            "queue_ndcg": ndcg_at_k(relevance, ranking, q),
            "blocked_frontier_count": float("nan"),
            "blocked_useful_count": float("nan"),
            "blocked_queued_count": float("nan"),
            "sensitivity_median": float("nan"),
            "sensitivity_q10": float("nan"),
            "sensitivity_q90": float("nan"),
            "executable_count": int(ranking.size),
        }

    ranking = queue_order(module)
    gains = counterfactual_backup_gains(
        P, reward, agent.gamma, agent.V, value_star, ranking
    )
    relevance = np.zeros(agent.n_states, dtype=float)
    relevance[ranking] = gains[ranking]
    top_q = np.asarray(sorted(module._top_q), dtype=int)
    blocked = top_q[np.asarray(module.xi[top_q]) <= RESIDUAL_FLOOR]
    queued = ranking[: min(q, ranking.size)]
    xi_max = max(float(np.max(np.abs(module.xi))), EPSILON)
    sensitivity = float(module.eta) * np.asarray(module.tau, dtype=float) / xi_max
    return {
        "method": method,
        "seed": seed,
        "shift": shift,
        "backup": backup,
        "value_l1": raw_error,
        "tracking_error": raw_error / agent.n_states,
        "normalized_error": raw_error / max(initial_error, EPSILON),
        "queue_ndcg": ndcg_at_k(relevance, ranking, q),
        "blocked_frontier_count": int(blocked.size),
        "blocked_useful_count": int(np.count_nonzero(gains[blocked] > EPSILON)),
        "blocked_queued_count": int(np.count_nonzero(np.isin(queued, blocked))),
        "sensitivity_median": float(np.median(sensitivity)),
        "sensitivity_q10": float(np.quantile(sensitivity, 0.10)),
        "sensitivity_q90": float(np.quantile(sensitivity, 0.90)),
        "executable_count": int(ranking.size),
    }


def run_seed(
    *,
    seed: int,
    room_size: int,
    gamma: float,
    n_shifts: int,
    budget: int,
    sharp_threshold: float,
    max_shift_attempts: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Run paired Standard-PS and GTA-PS recovery traces on one reward path."""
    env = FourRoomsTabular(
        room_size=room_size,
        corridor_len=1,
        k=1,
        gamma=gamma,
        seed=seed,
        n_peaks_per_room=1,
    )
    rewards, shifts = generate_sharp_reward_sequence(
        env, n_shifts, sharp_threshold, max_shift_attempts
    )
    q = min(2 * budget, env.n_states)
    eta_star = (1.0 - gamma) / gamma
    agents = {
        "standard_ps": make_agent(
            "standard_ps", P=env.P, r=rewards[0], gamma=gamma
        ),
        "gta_beta": make_agent(
            "gta_beta",
            P=env.P,
            r=rewards[0],
            gamma=gamma,
            q=q,
            eta=eta_star,
            **GTA_KWARGS,
        ),
    }
    pre_shift_value = env.value_iteration(rewards[0])
    for agent in agents.values():
        _warm_start(agent, pre_shift_value)

    traces: list[dict[str, Any]] = []
    for shift, reward in enumerate(rewards[1:]):
        value_star = env.value_iteration(reward)
        for method, agent in agents.items():
            previous_xi = (
                np.asarray(agent.module.xi, dtype=float).copy()
                if method == "gta_beta"
                else None
            )
            previous_tau = (
                np.asarray(agent.module.tau, dtype=float).copy()
                if method == "gta_beta"
                else None
            )
            agent.plan(K=0, r=reward)
            initial_error = float(value_error_l1(agent.V, value_star))
            path: list[float] = []
            for backup in range(budget + 1):
                row = trace_snapshot(
                    method=method,
                    agent=agent,
                    P=env.P,
                    reward=reward,
                    value_star=value_star,
                    initial_error=initial_error,
                    seed=seed,
                    shift=shift,
                    backup=backup,
                    q=q,
                )
                row["popped_state"] = None
                row["popped_blocked"] = None
                row["popped_gain"] = None
                if backup == 0 and method == "gta_beta":
                    delta_xi = np.asarray(agent.module.xi) - previous_xi
                    delta_tau = np.asarray(agent.module.tau) - previous_tau
                    row["delta_xi_inf"] = float(np.max(np.abs(delta_xi)))
                    row["delta_tau_l1"] = float(np.sum(np.abs(delta_tau)))
                    row["normalized_delta_tau_median"] = float(
                        agent.module.eta
                        * np.median(np.abs(delta_tau))
                        / max(row["delta_xi_inf"], EPSILON)
                    )
                else:
                    row["delta_xi_inf"] = None
                    row["delta_tau_l1"] = None
                    row["normalized_delta_tau_median"] = None
                path.append(float(row["normalized_error"]))
                row["prefix_error_auc"] = prefix_auc(path)
                if backup < budget:
                    value_before = agent.V.copy()
                    blocked = (
                        set(
                            np.asarray(
                                sorted(agent.module._top_q), dtype=int
                            )[np.asarray(agent.module.xi[sorted(agent.module._top_q)]) <= RESIDUAL_FLOOR].tolist()
                        )
                        if method == "gta_beta"
                        else set()
                    )
                    before_counts = agent.backup_counts.copy()
                    before = agent.completed_backups
                    agent.plan(K=1)
                    if agent.completed_backups - before > 1:
                        raise RuntimeError("a time point exceeded its one-backup budget")
                    popped = np.flatnonzero(agent.backup_counts > before_counts)
                    if popped.size == 1:
                        state = int(popped[0])
                        row["popped_state"] = state
                        row["popped_blocked"] = state in blocked
                        row["popped_gain"] = float(
                            counterfactual_backup_gains(
                                env.P,
                                reward,
                                gamma,
                                value_before,
                                value_star,
                                np.asarray([state]),
                            )[state]
                        )
                    elif popped.size > 1:
                        raise RuntimeError("one-backup step changed multiple states")
                traces.append(row)
    for item in shifts:
        item.update({"seed": seed, "eta_star": eta_star, "q": q})
    return traces, shifts, env.n_states


def _mean_ci(values: Iterable[float]) -> dict[str, float | list[float] | None]:
    array = np.asarray(tuple(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"mean": None, "ci95": None, "n": 0}
    mean = float(np.mean(array))
    if array.size == 1:
        return {"mean": mean, "ci95": [mean, mean], "n": 1}
    radius = float(student_t.ppf(0.975, array.size - 1) * np.std(array, ddof=1) / np.sqrt(array.size))
    return {"mean": mean, "ci95": [mean - radius, mean + radius], "n": int(array.size)}


def summarize_traces(rows: Iterable[dict[str, Any]], budget: int) -> dict[str, Any]:
    """Aggregate shifts within each seed before cross-seed confidence intervals."""
    grouped: dict[tuple[str, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["seed"], row["shift"], row["backup"])].append(row)
    per_seed: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for (method, seed, _shift, backup), values in grouped.items():
        per_seed[(method, seed, backup)].extend(values)

    metrics = (
        "tracking_error",
        "normalized_error",
        "prefix_error_auc",
        "queue_ndcg",
        "blocked_frontier_count",
        "blocked_useful_count",
        "blocked_queued_count",
        "sensitivity_median",
        "sensitivity_q10",
        "sensitivity_q90",
        "popped_blocked",
        "popped_gain",
    )
    curves: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for method in METHODS:
        curves[method] = {metric: [] for metric in metrics}
        for backup in range(budget + 1):
            seed_ids = sorted({key[1] for key in per_seed if key[0] == method and key[2] == backup})
            for metric in metrics:
                seed_values = []
                for seed in seed_ids:
                    rows_at_seed = per_seed[(method, seed, backup)]
                    metric_values = np.asarray(
                        [row.get(metric, np.nan) for row in rows_at_seed], dtype=float
                    )
                    finite = metric_values[np.isfinite(metric_values)]
                    seed_values.append(
                        float(np.mean(finite)) if finite.size else float("nan")
                    )
                curves[method][metric].append({"backup": backup, **_mean_ci(seed_values)})

    cumulative: list[dict[str, Any]] = []
    by_seed_shift: dict[tuple[str, int, int], float] = {}
    for method, seed, shift, backup in grouped:
        if backup == budget:
            path = grouped[(method, seed, shift, backup)]
            by_seed_shift[(method, seed, shift)] = float(
                np.mean(
                    [row.get("tracking_error", row["normalized_error"]) for row in path]
                )
            )
    seeds = sorted({key[1] for key in by_seed_shift if key[0] == "standard_ps"})
    shifts = sorted({key[2] for key in by_seed_shift})
    for shift in shifts:
        gaps = []
        for seed in seeds:
            standard = sum(by_seed_shift[("standard_ps", seed, prior)] for prior in range(shift + 1))
            gta = sum(by_seed_shift[("gta_beta", seed, prior)] for prior in range(shift + 1))
            gaps.append(standard - gta)
        cumulative.append({"shift": shift, **_mean_ci(gaps)})
    return {"curves": curves, "cumulative_tracking_gap": cumulative}


def _plot(summary: dict[str, Any], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    palette = get_palette("line", extra_keys=["standard_ps", "gta_beta"])
    panels = (
        ("normalized_error", "Normalized value error", "lower"),
        ("prefix_error_auc", "Prefix error AUC", "lower"),
        ("queue_ndcg", "Queue nDCG@q", "higher"),
        ("blocked_frontier_count", "Blocked top-q states", "count"),
        ("sensitivity_median", r"Median $\eta\tau/\|\xi\|_\infty$", "higher"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(10.6, 5.7))
    for axis, (metric, label, _direction) in zip(axes.flat[:5], panels):
        axis.set_facecolor(get_utility_color("bg_gray"))
        axis.grid(True, color=get_utility_color("grid"))
        for method in METHODS:
            values = summary["curves"][method][metric]
            x = np.asarray([item["backup"] for item in values])
            y = np.asarray([item["mean"] for item in values], dtype=float)
            ci = np.asarray([item["ci95"] for item in values], dtype=float)
            if not np.any(np.isfinite(y)):
                continue
            axis.plot(x, y, color=palette[method], label=LABELS[method], linewidth=1.5)
            axis.fill_between(x, ci[:, 0], ci[:, 1], color=palette[method], alpha=0.15)
        axis.set_xlabel("Cumulative backups")
        axis.set_ylabel(label)
    axes.flat[2].set_ylim(0.0, 1.05)
    axes.flat[0].legend(fontsize=7.5)
    gap_axis = axes.flat[5]
    gap_axis.set_facecolor(get_utility_color("bg_gray"))
    gap_axis.grid(True, color=get_utility_color("grid"))
    gaps = summary["cumulative_tracking_gap"]
    x = np.asarray([item["shift"] + 1 for item in gaps])
    y = np.asarray([item["mean"] for item in gaps], dtype=float)
    ci = np.asarray([item["ci95"] for item in gaps], dtype=float)
    gap_axis.axhline(0.0, color="black", linewidth=0.7)
    gap_axis.plot(x, y, color=palette["gta_beta"], linewidth=1.5)
    gap_axis.fill_between(x, ci[:, 0], ci[:, 1], color=palette["gta_beta"], alpha=0.15)
    gap_axis.set_xlabel("Shift event")
    gap_axis.set_ylabel("Cumulative tracking gap\n(Standard-PS minus GTA-PS)")
    figure.tight_layout()
    figure.savefig(output_path.with_suffix(".pdf"))
    figure.savefig(output_path.with_suffix(".png"), dpi=300)
    plt.close(figure)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--room-size", type=int, default=DEFAULT_ROOM_SIZE)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--n-shifts", type=int, default=8)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--sharp-threshold", type=float, default=0.75)
    parser.add_argument("--max-shift-attempts", type=int, default=100)
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "results" / "additional_evidence" / "rq8_time_ablation"
    )
    args = parser.parse_args()
    if args.seeds < 2 or args.workers < 1 or min(args.n_shifts, args.budget) < 1:
        raise ValueError("seeds must be at least 2; workers, shifts, and budget must be positive")
    if not 0.0 < args.gamma < 1.0:
        raise ValueError("gamma must lie in (0, 1)")

    started = time.perf_counter()
    seed_results = RunManager(n_workers=args.workers).run(
        run_seed,
        list(range(args.seeds)),
        {
            "room_size": args.room_size,
            "gamma": args.gamma,
            "n_shifts": args.n_shifts,
            "budget": args.budget,
            "sharp_threshold": args.sharp_threshold,
            "max_shift_attempts": args.max_shift_attempts,
        },
        desc="RQ8 backup-time seeds",
    )
    traces = [row for rows, _shifts, _states in seed_results for row in rows]
    shifts = [row for _rows, seed_shifts, _states in seed_results for row in seed_shifts]
    n_states = seed_results[0][2]
    summary = summarize_traces(traces, args.budget)
    output = {
        "protocol": {
            "host": "oracle_dp",
            "environment": "FourRoomsTabular",
            "n_states": n_states,
            "room_size": args.room_size,
            "gamma": args.gamma,
            "eta": (1.0 - args.gamma) / args.gamma,
            "eta_rule": "eta_star = (1 - gamma) / gamma",
            "q": min(2 * args.budget, n_states),
            "q_rule": "min(2K, |S|)",
            "budget": args.budget,
            "n_shifts": args.n_shifts,
            "seeds": args.seeds,
            "workers": args.workers,
            "methods": METHODS,
            "gta_parameters": GTA_KWARGS,
            "queue_relevance": "counterfactual one-backup reduction in L1 value error",
            "elapsed_seconds": time.perf_counter() - started,
        },
        "metric_definitions": {
            "prefix_error_auc": "mean normalized value error from backup 0 through k",
            "queue_ndcg": "nDCG@q of the actual queue with counterfactual one-backup gain relevance",
            "blocked_frontier_count": "top-q states with residual at most 1e-4",
            "blocked_useful_count": "blocked top-q states whose counterfactual gain is positive",
            "sensitivity_median": "median eta * tau / ||xi||_infinity",
            "cumulative_tracking_gap": "cumulative Standard-PS minus GTA-PS value error at backup K",
        },
        "summary": summary,
        "shift_metadata": shifts,
        "traces": traces,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / "rq8_time_ablation.json"
    output_path.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False), encoding="utf-8")
    _plot(summary, args.out_dir / "rq8_time_ablation")
    print(f"results={output_path}")


if __name__ == "__main__":
    main()
