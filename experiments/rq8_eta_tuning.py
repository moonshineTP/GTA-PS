"""RQ8 eta ablation under paired sharp FourRooms reward shifts."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "misc"))
sys.path.insert(0, str(ROOT / "threads"))

from gta_ps.algorithms import DpHost, make_agent
from gta_ps.envs.fourrooms import FourRoomsTabular
from gta_ps.utils.metrics import value_error_l1
from plot_academic import apply_style, get_palette, get_utility_color
from runner import RunManager


ETA_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
DEFAULT_ROOM_SIZE = 5
NOTATION = {
    "a": "compact-state capacity; identity tabular case equals the state count",
    "m": "frozen-priority batch size; fixed at 1",
    "q": "topology-frontier width; null means the full state frontier",
    "b": "GTA Boltzmann topology-policy temperature",
    "c": "eta multiplier eta / eta_star",
}
GTA_KWARGS = {
    "alpha": 0.051719,
    "beta1": 0.159673,
    "beta2": 1.361556,
    "b": 2.849322,
    "arnoldi_k": 15,
}
EPSILON = 1e-12


def resolve_eta_grid(
    gamma: float,
    multipliers: Sequence[float],
) -> tuple[float, tuple[float, ...]]:
    """Resolve a positive eta grid containing the canonical SR point."""
    gamma = float(gamma)
    values = tuple(float(value) for value in multipliers)
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must lie in (0, 1)")
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("eta multipliers must be positive")
    if len(set(values)) != len(values):
        raise ValueError("eta multipliers must be unique")
    if not any(np.isclose(value, 1.0) for value in values):
        raise ValueError("eta multipliers must include the canonical SR point 1.0")
    eta_star = (1.0 - gamma) / gamma
    return eta_star, tuple(value * eta_star for value in values)


def _reward_room(env: FourRoomsTabular, reward: np.ndarray) -> int:
    peak_state = int(np.argmax(reward))
    for room_index, states in enumerate(env._rooms):
        if peak_state in states:
            return room_index
    raise RuntimeError(f"reward peak state {peak_state} is outside the four rooms")


def _reward_distance(previous: np.ndarray, current: np.ndarray) -> float:
    denominator = float(np.sum(np.abs(previous)) + np.sum(np.abs(current)))
    return float(np.sum(np.abs(current - previous)) / max(denominator, EPSILON))


def generate_sharp_reward_sequence(
    env: FourRoomsTabular,
    n_shifts: int,
    threshold: float,
    max_attempts: int,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Generate paired reward shifts with a room change and fixed sharpness gate."""
    rewards = [env.r.copy()]
    metadata: list[dict[str, Any]] = []
    previous_room = _reward_room(env, rewards[0])
    for shift in range(n_shifts):
        for attempt in range(1, max_attempts + 1):
            candidate = env.shift_reward()
            room = _reward_room(env, candidate)
            distance = _reward_distance(rewards[-1], candidate)
            if room != previous_room and distance >= threshold:
                rewards.append(candidate.copy())
                metadata.append(
                    {
                        "shift": shift,
                        "attempts": attempt,
                        "previous_room": previous_room,
                        "current_room": room,
                        "reward_distance": distance,
                    }
                )
                previous_room = room
                break
        else:
            raise RuntimeError(
                f"failed to generate sharp reward shift {shift} after "
                f"{max_attempts} attempts"
            )
    return rewards, metadata


def _warm_start(agent: DpHost, value: np.ndarray) -> None:
    agent.V = value.copy()
    agent.module.V = value.copy()


def _distribution_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        f"{prefix}_l1": float(np.sum(np.abs(values))),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_q90": float(np.quantile(values, 0.90)),
        f"{prefix}_q99": float(np.quantile(values, 0.99)),
        f"{prefix}_max": float(np.max(values)),
    }


def queue_order(module: Any) -> np.ndarray:
    """Reconstruct the current GTA-PS queue order without mutating it."""
    xi = np.asarray(module.xi, dtype=float)
    tau = np.asarray(module.tau, dtype=float)
    beta = float(module.beta_t)
    state_ids = np.arange(xi.size)

    hot = state_ids[xi > 1e-4]
    hot_scores = np.maximum(xi[hot], xi[hot] + beta * tau[hot])
    hot_order = hot[np.lexsort((hot, -hot_scores))]

    if beta <= 0.0:
        return hot_order

    top_q = np.asarray(
        sorted(getattr(module, "_top_q", set(state_ids.tolist()))),
        dtype=int,
    )
    cold = top_q[xi[top_q] <= 1e-4]
    cold_scores = beta * tau[cold]
    eligible = cold_scores > 0.0
    cold = cold[eligible]
    cold_scores = cold_scores[eligible]
    cold_order = cold[np.lexsort((cold, -cold_scores))]
    return np.concatenate((hot_order, cold_order))


def ndcg_at_k(relevance: np.ndarray, ranking: np.ndarray, k: int) -> float:
    """Return linear-gain nDCG@k for a state ranking."""
    relevance = np.asarray(relevance, dtype=float)
    ranking = np.asarray(ranking, dtype=int)
    if relevance.ndim != 1:
        raise ValueError("relevance must be one-dimensional")
    if k < 1:
        raise ValueError("k must be positive")
    if ranking.size and (
        np.any(ranking < 0)
        or np.any(ranking >= relevance.size)
        or np.unique(ranking).size != ranking.size
    ):
        raise ValueError("ranking must contain unique valid state indices")

    cutoff = min(int(k), relevance.size)
    discounts = np.log2(np.arange(2, cutoff + 2, dtype=float))
    ideal = np.sort(relevance)[::-1][:cutoff]
    ideal_dcg = float(np.sum(ideal / discounts))
    if ideal_dcg <= EPSILON:
        return 1.0

    observed = relevance[ranking[:cutoff]]
    observed_discounts = discounts[: observed.size]
    return float(np.sum(observed / observed_discounts) / ideal_dcg)


def trace_snapshot(
    *,
    module: Any,
    value: np.ndarray,
    value_star: np.ndarray,
    initial_error: float,
    seed: int,
    shift: int,
    call: int,
    eta_multiplier: float,
    budget: int,
    shift_start_step: int = 0,
) -> dict[str, Any]:
    """Capture one validated scalar trace snapshot."""
    tau = np.asarray(module.tau, dtype=float)
    xi = np.asarray(module.xi, dtype=float)
    if np.any(tau < -1e-10):
        raise RuntimeError("tau contains a negative value beyond tolerance")

    expected_beta = float(
        module.beta1 * ((1.0 - float(module.slem)) + module.beta2)
    )
    if not np.isclose(float(module.beta_t), expected_beta, rtol=1e-8, atol=1e-10):
        raise RuntimeError("beta schedule does not match beta1 * (1 - SLEM + beta2)")
    if hasattr(module, "alpha_base") and not np.isclose(
        float(module.alpha_t), float(module.alpha_base), rtol=0.0, atol=1e-12
    ):
        raise RuntimeError("alpha schedule differs from the fixed alpha value")

    raw_error = float(value_error_l1(value, value_star))
    step_count = int(module._step_count)
    xi_l1 = float(np.sum(np.abs(xi)))
    tau_l1 = float(np.sum(np.abs(tau)))
    state_error = np.abs(np.asarray(value, dtype=float) - np.asarray(value_star, dtype=float))
    ndcg_k = min(int(budget), state_error.size)
    row: dict[str, Any] = {
        "seed": int(seed),
        "shift": int(shift),
        "call": int(call),
        "eta_multiplier": float(eta_multiplier),
        "eta": float(module.eta),
        "budget": int(budget),
        "cumulative_backups": step_count - int(shift_start_step),
        "total_backups": step_count,
        "value_l1": raw_error,
        "normalized_error": raw_error / max(float(initial_error), EPSILON),
        "alpha": float(module.alpha_t),
        "beta": float(module.beta_t),
        "slem": float(module.slem),
        "topology_gain": abs(float(module.beta_t)) * tau_l1 / max(xi_l1, EPSILON),
        "queue_ndcg": ndcg_at_k(state_error, queue_order(module), ndcg_k),
        "queue_ndcg_k": ndcg_k,
        "tau_refresh_count": int(module._tau_refresh_count),
        "tau_refresh_age": step_count - int(module._last_tau_refresh_step),
    }
    row.update(_distribution_stats(xi, "xi"))
    row.update(_distribution_stats(tau, "tau"))
    row["finite"] = bool(
        np.all(np.isfinite(np.fromiter(
            (float(value) for value in row.values() if isinstance(value, (int, float))),
            dtype=float,
        )))
    )
    if not row["finite"]:
        raise RuntimeError("trace snapshot contains a non-finite scalar")
    return row


def run_seed(
    *,
    seed: int,
    room_size: int,
    gamma: float,
    eta_multipliers: Sequence[float],
    n_shifts: int,
    n_calls: int,
    budget: int,
    sharp_threshold: float,
    max_shift_attempts: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Run all eta conditions on one paired reward sequence."""
    eta_star, eta_values = resolve_eta_grid(gamma, eta_multipliers)
    env = FourRoomsTabular(
        room_size=room_size,
        corridor_len=1,
        k=1,
        gamma=gamma,
        seed=seed,
        n_peaks_per_room=1,
    )
    rewards, shift_metadata = generate_sharp_reward_sequence(
        env,
        n_shifts=n_shifts,
        threshold=sharp_threshold,
        max_attempts=max_shift_attempts,
    )
    initial_reward = rewards[0]
    pre_shift_value = env.value_iteration(initial_reward)
    agents: dict[float, DpHost] = {}
    for multiplier, eta in zip(eta_multipliers, eta_values):
        agent = make_agent(
            "gta_beta",
            P=env.P,
            r=initial_reward,
            gamma=gamma,
            q=None,
            eta=eta,
            **GTA_KWARGS,
        )
        _warm_start(agent, pre_shift_value)
        agents[float(multiplier)] = agent

    traces: list[dict[str, Any]] = []
    for shift, reward in enumerate(rewards[1:]):
        value_star = env.value_iteration(reward)
        for multiplier, agent in agents.items():
            agent.plan(K=0, r=reward)
            initial_error = value_error_l1(agent.V, value_star)
            shift_start_step = int(agent.module._step_count)
            traces.append(
                trace_snapshot(
                    module=agent.module,
                    value=agent.V,
                    value_star=value_star,
                    initial_error=initial_error,
                    seed=seed,
                    shift=shift,
                    call=0,
                    eta_multiplier=multiplier,
                    budget=budget,
                    shift_start_step=shift_start_step,
                )
            )
            for call in range(1, n_calls + 1):
                before = int(agent.module._step_count)
                value = agent.plan(K=budget)
                completed = int(agent.module._step_count) - before
                if completed > budget:
                    raise RuntimeError("planning call exceeded its Bellman backup budget")
                traces.append(
                    trace_snapshot(
                        module=agent.module,
                        value=value,
                        value_star=value_star,
                        initial_error=initial_error,
                        seed=seed,
                        shift=shift,
                        call=call,
                        eta_multiplier=multiplier,
                        budget=budget,
                        shift_start_step=shift_start_step,
                    )
                )

    for item in shift_metadata:
        item["seed"] = seed
        item["eta_star"] = eta_star
    return traces, shift_metadata, env.n_states


def _path_metrics(path: Sequence[dict[str, Any]]) -> dict[str, float]:
    ordered = sorted(path, key=lambda row: int(row["call"]))
    calls = [int(row["call"]) for row in ordered]
    if calls != list(range(len(ordered))):
        raise ValueError("trace calls must form a contiguous sequence from zero")
    errors = np.asarray([float(row["normalized_error"]) for row in ordered])
    variation = float(np.sum(np.abs(np.diff(errors))) - abs(errors[-1] - errors[0]))
    reached = [
        float(row["cumulative_backups"])
        for row in ordered
        if float(row["normalized_error"]) <= 0.25
    ]
    queue_ndcg = np.asarray(
        [float(row["queue_ndcg"]) for row in ordered if "queue_ndcg" in row],
        dtype=float,
    )
    return {
        "auc": float(np.mean(errors)),
        "first_call": float(errors[min(1, len(errors) - 1)]),
        "end": float(errors[-1]),
        "roughness": max(0.0, variation),
        "threshold_backups": min(reached) if reached else float("nan"),
        "queue_ndcg": (
            float(np.mean(queue_ndcg)) if queue_ndcg.size else float("nan")
        ),
    }


def _seed_metrics(rows: Iterable[dict[str, Any]]) -> dict[float, dict[int, dict[str, float]]]:
    paths: dict[tuple[int, float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (int(row["seed"]), float(row["eta_multiplier"]), int(row["shift"]))
        paths[key].append(row)

    per_seed: dict[float, dict[int, dict[str, float]]] = defaultdict(dict)
    grouped: dict[tuple[int, float], list[dict[str, float]]] = defaultdict(list)
    for (seed, multiplier, _shift), path in paths.items():
        grouped[(seed, multiplier)].append(_path_metrics(path))
    for (seed, multiplier), metrics in grouped.items():
        seed_result: dict[str, float] = {}
        for name in (
            "auc",
            "first_call",
            "end",
            "roughness",
            "threshold_backups",
            "queue_ndcg",
        ):
            values = np.asarray([metric[name] for metric in metrics], dtype=float)
            seed_result[name] = (
                float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")
            )
        seed_result["auc_p90"] = float(np.quantile([metric["auc"] for metric in metrics], 0.90))
        per_seed[multiplier][seed] = seed_result
    return per_seed


def _bootstrap_mean_ci(
    values: np.ndarray,
    resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if values.size == 1 or resamples < 2:
        value = float(np.mean(values))
        return value, value
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def summarize_eta_metrics(
    rows: Iterable[dict[str, Any]],
    bootstrap_resamples: int = 2_000,
    rng_seed: int = 0,
) -> dict[float, dict[str, Any]]:
    """Aggregate nested shifts within seeds before cross-seed statistics."""
    per_seed = _seed_metrics(rows)
    rng = np.random.default_rng(rng_seed)
    summary: dict[float, dict[str, Any]] = {}
    for multiplier in sorted(per_seed):
        seed_results = per_seed[multiplier]
        item: dict[str, Any] = {"n": len(seed_results)}
        for metric in (
            "auc",
            "first_call",
            "end",
            "roughness",
            "auc_p90",
            "queue_ndcg",
        ):
            values = np.asarray(
                [seed_results[seed][metric] for seed in sorted(seed_results)],
                dtype=float,
            )
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                item[f"{metric}_mean"] = None
                item[f"{metric}_std"] = None
                item[f"{metric}_median"] = None
                item[f"{metric}_iqr"] = None
                item[f"{metric}_ci95"] = None
                continue
            item[f"{metric}_mean"] = float(np.mean(finite))
            item[f"{metric}_std"] = (
                float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
            )
            item[f"{metric}_median"] = float(np.median(finite))
            item[f"{metric}_iqr"] = [
                float(value) for value in np.quantile(finite, [0.25, 0.75])
            ]
            item[f"{metric}_ci95"] = list(
                _bootstrap_mean_ci(finite, bootstrap_resamples, rng)
            )
        threshold = np.asarray(
            [seed_results[seed]["threshold_backups"] for seed in sorted(seed_results)],
            dtype=float,
        )
        item["threshold_backups_mean"] = (
            float(np.nanmean(threshold))
            if np.any(np.isfinite(threshold))
            else None
        )
        summary[multiplier] = item
    return summary


def _holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def paired_auc_comparisons(
    rows: Iterable[dict[str, Any]],
    bootstrap_resamples: int,
    rng_seed: int = 1,
) -> list[dict[str, Any]]:
    """Compare every larger eta with the canonical condition by paired seed."""
    per_seed = _seed_metrics(rows)
    if 1.0 not in per_seed:
        raise ValueError("canonical eta multiplier 1.0 is missing")
    canonical = per_seed[1.0]
    rng = np.random.default_rng(rng_seed)
    comparisons: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    for multiplier in sorted(value for value in per_seed if value > 1.0):
        shared = sorted(set(canonical) & set(per_seed[multiplier]))
        differences = np.asarray(
            [
                per_seed[multiplier][seed]["auc"] - canonical[seed]["auc"]
                for seed in shared
            ],
            dtype=float,
        )
        if not shared:
            raise ValueError(f"no paired seeds for eta multiplier {multiplier}")
        if differences.size == 1 or bootstrap_resamples < 2:
            samples = differences.copy()
        else:
            indices = rng.integers(
                0,
                differences.size,
                size=(bootstrap_resamples, differences.size),
            )
            samples = differences[indices].mean(axis=1)
        ci = np.quantile(samples, [0.025, 0.975])
        denominator = samples.size + 1
        lower_tail = (int(np.count_nonzero(samples <= 0.0)) + 1) / denominator
        upper_tail = (int(np.count_nonzero(samples >= 0.0)) + 1) / denominator
        p_value = min(1.0, 2.0 * min(lower_tail, upper_tail))
        raw_p_values.append(p_value)
        comparisons.append(
            {
                "eta_multiplier": multiplier,
                "n": len(shared),
                "auc_difference_mean": float(np.mean(differences)),
                "auc_difference_ci95": [float(ci[0]), float(ci[1])],
                "p_value": p_value,
            }
        )
    adjusted = _holm_adjust(raw_p_values)
    for item, p_value in zip(comparisons, adjusted):
        item["holm_p_value"] = p_value
    return comparisons


def plot_eta_ablation(
    summary: dict[float, dict[str, Any]],
    output_path: Path,
    budget: int,
) -> tuple[Path, Path]:
    """Render planning error and queue quality on paired y-axes."""
    import matplotlib.pyplot as plt

    if 1.0 not in summary:
        raise ValueError("canonical eta multiplier 1.0 is required for the SR star")
    apply_style()
    palette = get_palette(
        "line",
        extra_keys=["planning_error", "queue_ndcg", "eta_sr"],
    )
    multipliers = np.asarray(sorted(summary), dtype=float)
    canonical_index = int(np.flatnonzero(np.isclose(multipliers, 1.0))[0])
    auc_means = np.asarray(
        [summary[value]["auc_mean"] for value in multipliers],
        dtype=float,
    )
    auc_intervals = np.asarray(
        [summary[value]["auc_ci95"] for value in multipliers],
        dtype=float,
    )
    ndcg_means = np.asarray(
        [summary[value]["queue_ndcg_mean"] for value in multipliers],
        dtype=float,
    )
    ndcg_intervals = np.asarray(
        [summary[value]["queue_ndcg_ci95"] for value in multipliers],
        dtype=float,
    )

    fig, left_axis = plt.subplots(figsize=(8, 4.6))
    right_axis = left_axis.twinx()
    left_axis.set_facecolor(get_utility_color("bg_gray"))
    left_axis.grid(True, color=get_utility_color("grid"))
    left_axis.tick_params(labelsize=8)
    right_axis.tick_params(labelsize=8)

    auc_handle = left_axis.errorbar(
        multipliers,
        auc_means,
        yerr=np.vstack(
            (auc_means - auc_intervals[:, 0], auc_intervals[:, 1] - auc_means)
        ),
        color=palette["planning_error"],
        marker="o",
        capsize=3,
        linewidth=1.5,
        markersize=5,
        label="Planning error AUC",
    )
    ndcg_handle = right_axis.errorbar(
        multipliers,
        ndcg_means,
        yerr=np.vstack(
            (
                ndcg_means - ndcg_intervals[:, 0],
                ndcg_intervals[:, 1] - ndcg_means,
            )
        ),
        color=palette["queue_ndcg"],
        marker="s",
        capsize=3,
        linewidth=1.5,
        markersize=4.8,
        label=rf"Queue nDCG@{budget}",
    )
    canonical_handle = left_axis.scatter(
        [1.0],
        [auc_means[canonical_index]],
        marker="*",
        s=155,
        color=palette["eta_sr"],
        edgecolor=get_utility_color("text_main"),
        linewidth=0.7,
        zorder=6,
        label=r"Canonical SR $\eta^\star$",
    )
    right_axis.scatter(
        [1.0],
        [ndcg_means[canonical_index]],
        marker="*",
        s=155,
        color=palette["eta_sr"],
        edgecolor=get_utility_color("text_main"),
        linewidth=0.7,
        zorder=6,
    )

    left_axis.set_xscale("log", base=2)
    left_axis.set_xticks(multipliers)
    left_axis.set_xticklabels([f"{value:g}" for value in multipliers])
    left_axis.set_xlabel(r"Diffusion ratio $\eta/\eta^\star$", fontsize=9)
    left_axis.set_ylabel(
        "Normalized error AUC (lower is better)",
        color=palette["planning_error"],
        fontsize=9,
    )
    right_axis.set_ylabel(
        rf"Queue nDCG@{budget} (higher is better)",
        color=palette["queue_ndcg"],
        fontsize=9,
    )
    left_axis.tick_params(axis="y", labelcolor=palette["planning_error"])
    right_axis.tick_params(axis="y", labelcolor=palette["queue_ndcg"])
    left_axis.set_title("Eta Tuning Tradeoff", fontsize=10)
    left_axis.legend(
        [auc_handle, ndcg_handle, canonical_handle],
        ["Planning error AUC", rf"Queue nDCG@{budget}", r"Canonical SR $\eta^\star$"],
        loc="upper left",
        fontsize=7.5,
    )
    fig.tight_layout()
    pdf_path = output_path.with_suffix(".pdf")
    png_path = output_path.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    return pdf_path, png_path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty trace")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _typed_trace(row: dict[str, Any]) -> dict[str, Any]:
    integer_fields = {
        "seed",
        "shift",
        "call",
        "budget",
        "cumulative_backups",
        "total_backups",
        "queue_ndcg_k",
        "tau_refresh_count",
        "tau_refresh_age",
    }
    typed: dict[str, Any] = {}
    for key, value in row.items():
        if not isinstance(value, str):
            typed[key] = value
        elif key == "finite":
            typed[key] = value.lower() == "true"
        elif key in integer_fields:
            typed[key] = int(value)
        else:
            typed[key] = float(value)
    return typed


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


def build_data_payload(
    *,
    protocol: dict[str, Any],
    shifts: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    summary: dict[float, dict[str, Any]] | dict[str, dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build strict JSON with raw, per-seed, and table-level RQ8 data."""
    typed_traces = [_typed_trace(row) for row in traces]
    normalized_summary = {
        float(multiplier): item for multiplier, item in summary.items()
    }
    per_seed = _seed_metrics(typed_traces)
    seed_rows = [
        {
            "eta_multiplier": multiplier,
            "seed": seed,
            **metrics,
        }
        for multiplier in sorted(per_seed)
        for seed, metrics in sorted(per_seed[multiplier].items())
    ]

    comparison_by_multiplier = {
        float(item["eta_multiplier"]): item for item in comparisons
    }
    table_rows: list[dict[str, Any]] = []
    for multiplier in sorted(normalized_summary):
        item = normalized_summary[multiplier]
        row: dict[str, Any] = {
            "eta_multiplier": multiplier,
            "eta": multiplier * float(protocol["eta_star"]),
            "n_seeds": item["n"],
        }
        for metric in (
            "auc",
            "first_call",
            "end",
            "roughness",
            "auc_p90",
            "queue_ndcg",
        ):
            row[f"{metric}_mean"] = item[f"{metric}_mean"]
            row[f"{metric}_std"] = item[f"{metric}_std"]
            interval = item[f"{metric}_ci95"]
            row[f"{metric}_ci95_low"] = interval[0] if interval else None
            row[f"{metric}_ci95_high"] = interval[1] if interval else None
        row["threshold_backups_mean"] = item["threshold_backups_mean"]
        comparison = comparison_by_multiplier.get(multiplier)
        row["auc_difference_vs_sr"] = (
            comparison["auc_difference_mean"] if comparison else None
        )
        row["auc_difference_vs_sr_ci95"] = (
            comparison["auc_difference_ci95"] if comparison else None
        )
        row["holm_p_value"] = comparison["holm_p_value"] if comparison else None
        table_rows.append(row)

    return _json_safe(
        {
            "schema_version": 1,
            "protocol": protocol,
            "metric_definitions": {
                "auc": {
                    "label": "Normalized planning-error AUC",
                    "direction": "lower_is_better",
                },
                "queue_ndcg": {
                    "label": f"Queue nDCG@{protocol['budget']}",
                    "direction": "higher_is_better",
                    "relevance": "absolute statewise oracle value error |V - V_star|",
                    "ranking": "residual queue followed by eligible topology-only states",
                },
            },
            "table_rows": table_rows,
            "paired_auc_comparisons": comparisons,
            "seed_rows": seed_rows,
            "shift_metadata": shifts,
            "traces": typed_traces,
        }
    )


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _json_summary(summary: dict[float, dict[str, Any]]) -> dict[str, Any]:
    return {f"{key:g}": value for key, value in summary.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--room-size", type=int, default=DEFAULT_ROOM_SIZE)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--eta-multipliers", type=float, nargs="+", default=ETA_MULTIPLIERS)
    parser.add_argument("--n-shifts", type=int, default=8)
    parser.add_argument("--n-calls", type=int, default=12)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--sharp-threshold", type=float, default=0.75)
    parser.add_argument("--max-shift-attempts", type=int, default=100)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "rebuttal" / "rq8_eta_tuning",
    )
    parser.add_argument(
        "--export-existing-dir",
        type=Path,
        help="Create the strict data JSON from an existing RQ8 CSV and summary.",
    )
    parser.add_argument(
        "--plot-existing-dir",
        type=Path,
        help="Regenerate the RQ8 plot from an existing summary JSON.",
    )
    args = parser.parse_args()

    if args.plot_existing_dir is not None:
        source_dir = args.plot_existing_dir
        saved = json.loads(
            (source_dir / "rq8_eta_tuning_summary.json").read_text(encoding="utf-8")
        )
        saved_summary = {
            float(multiplier): item
            for multiplier, item in saved["summary"].items()
        }
        pdf_path, png_path = plot_eta_ablation(
            saved_summary,
            source_dir / "rq8_eta_tuning",
            int(saved["protocol"]["budget"]),
        )
        print(f"plot_pdf={pdf_path}")
        print(f"plot_png={png_path}")
        return

    if args.export_existing_dir is not None:
        source_dir = args.export_existing_dir
        saved = json.loads(
            (source_dir / "rq8_eta_tuning_summary.json").read_text(encoding="utf-8")
        )
        traces = _read_csv(source_dir / "rq8_eta_tuning_traces.csv")
        payload = build_data_payload(
            protocol=saved["protocol"],
            shifts=saved["shift_metadata"],
            traces=traces,
            summary=saved["summary"],
            comparisons=saved["paired_auc_comparisons"],
        )
        data_path = source_dir / "rq8_eta_tuning_data.json"
        data_path.write_text(
            json.dumps(payload, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        print(f"data_json={data_path}")
        return

    if args.seeds < 2:
        raise ValueError("seeds must be at least 2")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if min(args.n_shifts, args.n_calls, args.budget) < 1:
        raise ValueError("n-shifts, n-calls, and budget must be positive")
    if not 0.0 <= args.sharp_threshold <= 1.0:
        raise ValueError("sharp-threshold must lie in [0, 1]")
    eta_multipliers = tuple(args.eta_multipliers)
    eta_star, eta_values = resolve_eta_grid(args.gamma, eta_multipliers)

    started = time.perf_counter()
    traces: list[dict[str, Any]] = []
    shifts: list[dict[str, Any]] = []
    n_states: int | None = None
    runner = RunManager(n_workers=args.workers)
    seed_results = runner.run(
        run_seed,
        list(range(args.seeds)),
        {
            "room_size": args.room_size,
            "gamma": args.gamma,
            "eta_multipliers": eta_multipliers,
            "n_shifts": args.n_shifts,
            "n_calls": args.n_calls,
            "budget": args.budget,
            "sharp_threshold": args.sharp_threshold,
            "max_shift_attempts": args.max_shift_attempts,
        },
        desc="RQ8 seeds",
    )
    for seed_traces, seed_shifts, seed_states in seed_results:
        traces.extend(seed_traces)
        shifts.extend(seed_shifts)
        n_states = seed_states

    summary = summarize_eta_metrics(
        traces,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    comparisons = paired_auc_comparisons(
        traces,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    elapsed = time.perf_counter() - started
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.out_dir / "rq8_eta_tuning_traces.csv"
    _write_csv(trace_path, traces)
    output = {
        "protocol": {
            "host": "oracle_dp",
            "environment": "FourRoomsTabular",
            "room_size": args.room_size,
            "outer_grid_size": 2 * args.room_size + 3,
            "n_states": n_states,
            "gamma": args.gamma,
            "eta_star": eta_star,
            "eta_multipliers": eta_multipliers,
            "eta_values": eta_values,
            "seeds": args.seeds,
            "workers": args.workers,
            "n_shifts": args.n_shifts,
            "n_calls": args.n_calls,
            "budget": args.budget,
            "sharp_threshold": args.sharp_threshold,
            "max_shift_attempts": args.max_shift_attempts,
            "bootstrap_resamples": args.bootstrap_resamples,
            "compression_capacity": n_states,
            "frozen_batch_size": 1,
            "gta_top_q": None,
            "notation": NOTATION,
            "gta_parameters": GTA_KWARGS,
            "elapsed_seconds": elapsed,
        },
        "shift_metadata": shifts,
        "summary": _json_summary(summary),
        "paired_auc_comparisons": comparisons,
    }
    summary_path = args.out_dir / "rq8_eta_tuning_summary.json"
    summary_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    data_path = args.out_dir / "rq8_eta_tuning_data.json"
    data_payload = build_data_payload(
        protocol=output["protocol"],
        shifts=shifts,
        traces=traces,
        summary=summary,
        comparisons=comparisons,
    )
    data_path.write_text(
        json.dumps(data_payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pdf_path, png_path = plot_eta_ablation(
        summary,
        args.out_dir / "rq8_eta_tuning",
        args.budget,
    )

    print(f"states={n_states}")
    print(f"elapsed_seconds={elapsed:.3f}")
    print(f"traces={trace_path}")
    print(f"summary={summary_path}")
    print(f"data_json={data_path}")
    print(f"plot_pdf={pdf_path}")
    print(f"plot_png={png_path}")


if __name__ == "__main__":
    main()
