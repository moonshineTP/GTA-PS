"""Wall-clock replanning across time-triggered GARNET reward shifts."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "threads"))
sys.path.insert(0, str(ROOT / "misc"))

from gta_ps.algorithms import make_agent
from gta_ps.envs.garnet import GarnetEnv
from gta_ps.utils.hyperparams import load_tuned_for_env
from plot_academic import LABELS, apply_style, get_palette, get_utility_color
from runner import RunManager

METHODS = ("standard_ps", "gta_beta")
DEFAULT_N_STATES = 200
DEFAULT_REWARD_SUPPORT = 20
DEFAULT_SGM_SHIFT_SECONDS = 10.0
CONVERGENCE_TOLERANCE = 1e-10
TRACE_POINTS_PER_PERIOD = 101


def _ci95(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(
        values.std(ddof=1)
        / np.sqrt(values.size)
        * student_t.ppf(0.975, df=values.size - 1)
    )


def shifted_geometric_mean(
    values: np.ndarray,
    *,
    shift: float = DEFAULT_SGM_SHIFT_SECONDS,
) -> float:
    """Return exp(mean(log(x + shift))) - shift."""
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.any(values < 0.0) or shift <= 0.0:
        raise ValueError("SGM values must be non-negative and shift must be positive")
    return float(np.exp(np.mean(np.log(values + shift))) - shift)


def time_weighted_error(times: np.ndarray, errors: np.ndarray) -> float:
    """Normalize the trapezoidal error integral by observed period duration."""
    times = np.asarray(times, dtype=float)
    errors = np.asarray(errors, dtype=float)
    if times.ndim != 1 or errors.shape != times.shape or times.size < 2:
        raise ValueError("times and errors must be aligned one-dimensional traces")
    if times[0] != 0.0 or np.any(np.diff(times) < 0.0) or times[-1] <= 0.0:
        raise ValueError("times must start at zero and increase to a positive duration")
    return float(np.trapezoid(errors, times) / times[-1])


def _backup_diagnostics(
    agent: Any,
    *,
    value: np.ndarray,
    P: np.ndarray,
    reward: np.ndarray,
    gamma: float,
    completed_before: int,
    counts_before: np.ndarray,
    tau_refreshes_before: int,
    slem_refreshes_before: int,
    real_updates_before: int,
    duration: float,
    requested_backups: int,
) -> dict[str, float | int]:
    completed = int(agent.completed_backups - completed_before)
    counts = np.asarray(agent.backup_counts, dtype=np.int64) - counts_before
    if completed > 0:
        probabilities = counts[counts > 0].astype(float) / completed
        effective_states = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
        max_share = float(counts.max() / completed)
    else:
        effective_states = 0.0
        max_share = 0.0

    expected = (P * value[None, None, :]).sum(axis=2)
    residual = np.abs(reward + gamma * expected.max(axis=0) - value)
    module = agent.module
    stored_residual = np.asarray(getattr(module, "xi", residual), dtype=float)
    beta_t = float(getattr(module, "beta_t", 0.0))
    tau = np.asarray(getattr(module, "tau", np.zeros_like(residual)), dtype=float)
    tau_refreshes = int(getattr(module, "_tau_refresh_count", 0))
    slem_refreshes = int(getattr(module, "_slem_refresh_count", 0))
    real_updates = int(
        getattr(agent, "completed_real_updates", 0) - real_updates_before
    )

    return {
        "completed_backups": completed,
        "completed_backups_per_second": float(completed / max(duration, 1e-12)),
        "backup_utilization": float(completed / max(requested_backups, 1)),
        "state_coverage_fraction": float(np.count_nonzero(counts) / counts.size),
        "effective_state_fraction": float(effective_states / counts.size),
        "max_state_backup_share": max_share,
        "bellman_residual_inf": float(residual.max()),
        "active_residual_fraction": float(np.mean(residual > 1e-4)),
        "residual_cache_linf": float(np.max(np.abs(stored_residual - residual))),
        "topology_signal_ratio": float(
            np.sum(np.abs(beta_t * tau)) / max(float(np.sum(residual)), 1e-12)
        ),
        "tau_refreshes": tau_refreshes - tau_refreshes_before,
        "slem_refreshes": slem_refreshes - slem_refreshes_before,
        "real_updates": real_updates,
        "real_updates_per_second": float(real_updates / max(duration, 1e-12)),
    }


def sparse_value_iteration(
    P: np.ndarray,
    reward: np.ndarray,
    gamma: float,
    *,
    branching: int,
    tolerance: float = 1e-4,
    max_iterations: int = 1000,
) -> np.ndarray:
    """Compute the oracle target without scanning zero transition entries."""
    successor_indices = np.argpartition(P, -branching, axis=2)[:, :, -branching:]
    successor_probabilities = np.take_along_axis(P, successor_indices, axis=2)
    value = np.zeros(P.shape[1], dtype=float)
    for _ in range(max_iterations):
        expected = np.sum(
            successor_probabilities * value[successor_indices],
            axis=2,
        )
        value_new = reward + gamma * expected.max(axis=0)
        if np.linalg.norm(value_new - value, ord=np.inf) < tolerance:
            return value_new
        value = value_new
    return value


def _make_problem(
    seed: int,
    *,
    n_states: int,
    n_actions: int,
    branching: int,
    n_nbr_rate: float,
    reward_support: int,
    gamma: float,
    n_shifts: int,
) -> tuple[GarnetEnv, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    env = GarnetEnv(
        n_states=n_states,
        n_actions=n_actions,
        branching=branching,
        reward_support=reward_support,
        gamma=gamma,
        n_nbr_rate=n_nbr_rate,
        reward_locality=True,
        seed=seed,
    )
    initial_reward = env.r.copy()
    rewards: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for _ in range(n_shifts):
        reward = env.shift_reward()
        rewards.append(reward)
        targets.append(
            sparse_value_iteration(
                env.P,
                reward,
                gamma,
                branching=branching,
            )
        )
    return env, initial_reward, rewards, targets


def _run_method(
    *,
    method: str,
    host: str,
    seed: int,
    env: GarnetEnv,
    initial_reward: np.ndarray,
    rewards: list[np.ndarray],
    targets: list[np.ndarray],
    gamma: float,
    period_seconds: float,
    min_calls_per_period: int,
    budget: int,
    eta_td: float,
    method_cfgs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    session_started = time.perf_counter()
    agent = make_agent(
        method,
        host=host,
        P=env.P,
        r=initial_reward,
        gamma=gamma,
        q=None,
        eta_td=eta_td,
        rng=np.random.default_rng(
            np.random.SeedSequence([seed, METHODS.index(method), 0xD1A])
        ),
        **dict(method_cfgs.get(method, {})),
    )
    initialized = time.perf_counter()
    periods: list[dict[str, Any]] = []
    interaction_rng = np.random.default_rng(
        np.random.SeedSequence([seed, METHODS.index(method), 0xEA1])
    )

    for shift_index, (reward, target) in enumerate(zip(rewards, targets)):
        period_started = time.perf_counter()
        value = agent.V if host == "dp" else agent.Q.max(axis=0)
        initial_error = float(np.mean(np.abs(value - target)))
        initial_error_inf = float(np.max(np.abs(value - target)))
        times = [0.0]
        errors_l1 = [initial_error]
        errors_linf = [initial_error_inf]

        if host == "dp":
            agent.plan(K=0, r=reward)
        else:
            agent.reward_update(reward)
        update_elapsed = time.perf_counter() - period_started
        value = agent.V if host == "dp" else agent.Q.max(axis=0)
        times.append(update_elapsed)
        errors_l1.append(float(np.mean(np.abs(value - target))))
        errors_linf.append(float(np.max(np.abs(value - target))))
        completed_before = int(agent.completed_backups)
        counts_before = np.asarray(agent.backup_counts, dtype=np.int64).copy()
        tau_refreshes_before = int(
            getattr(agent.module, "_tau_refresh_count", 0)
        )
        slem_refreshes_before = int(
            getattr(agent.module, "_slem_refresh_count", 0)
        )
        real_updates_before = int(
            getattr(agent, "completed_real_updates", 0)
        )
        completed_trace = [0, 0]

        calls = 0
        requested_backups = 0
        next_trace_time = period_seconds / (TRACE_POINTS_PER_PERIOD - 1)
        converged = errors_linf[-1] <= CONVERGENCE_TOLERANCE
        while time.perf_counter() - period_started < period_seconds:
            if host == "dp":
                value = agent.plan(K=budget)
            else:
                interaction_index = int(agent.completed_real_updates)
                state = interaction_index % env.n_states
                action = int(interaction_rng.integers(env.n_actions))
                successor = int(
                    interaction_rng.choice(
                        env.n_states,
                        p=env.P[action, state],
                    )
                )
                agent.real_step(
                    state,
                    action,
                    float(reward[state]),
                    successor,
                )
                agent.plan_steps(K=budget)
                value = agent.Q.max(axis=0)
            calls += 1
            requested_backups += budget
            elapsed = time.perf_counter() - period_started
            error_l1 = float(np.mean(np.abs(value - target)))
            error_linf = float(np.max(np.abs(value - target)))
            if elapsed >= next_trace_time:
                times.append(elapsed)
                errors_l1.append(error_l1)
                errors_linf.append(error_linf)
                completed_trace.append(
                    int(agent.completed_backups - completed_before)
                )
                next_trace_time += period_seconds / (TRACE_POINTS_PER_PERIOD - 1)
            if error_linf <= CONVERGENCE_TOLERANCE:
                converged = True
                remaining = period_seconds - elapsed
                if remaining > 0.0:
                    time.sleep(remaining)
                break

        duration = time.perf_counter() - period_started
        value = agent.V if host == "dp" else agent.Q.max(axis=0)
        final_error_l1 = float(np.mean(np.abs(value - target)))
        final_error_linf = float(np.max(np.abs(value - target)))
        if times[-1] < duration:
            times.append(duration)
            errors_l1.append(final_error_l1)
            errors_linf.append(final_error_linf)
            completed_trace.append(
                int(agent.completed_backups - completed_before)
            )
        diagnostics = _backup_diagnostics(
            agent,
            value=value,
            P=env.P,
            reward=reward,
            gamma=gamma,
            completed_before=completed_before,
            counts_before=counts_before,
            tau_refreshes_before=tau_refreshes_before,
            slem_refreshes_before=slem_refreshes_before,
            real_updates_before=real_updates_before,
            duration=duration,
            requested_backups=requested_backups,
        )
        if diagnostics["completed_backups"] < min_calls_per_period and not converged:
            raise RuntimeError(
                f"{method} completed {diagnostics['completed_backups']} backups "
                f"in {duration:.3f}s; need at least {min_calls_per_period}. "
                "Increase --period-seconds."
            )
        periods.append(
            {
                "shift": shift_index,
                "duration_seconds": duration,
                "overshoot_seconds": duration - period_seconds,
                "reward_update_seconds": update_elapsed,
                "completed_calls": calls,
                "requested_backups": requested_backups,
                "converged": converged,
                "recovery_auc": time_weighted_error(
                    np.asarray(times), np.asarray(errors_l1)
                ),
                "recovery_auc_linf": time_weighted_error(
                    np.asarray(times), np.asarray(errors_linf)
                ),
                "shift_end_l1": errors_l1[-1],
                "shift_end_linf": errors_linf[-1],
                "times_seconds": times,
                "l1_errors": errors_l1,
                "linf_errors": errors_linf,
                "completed_backups_trace": completed_trace,
                **diagnostics,
            }
        )

    session_finished = time.perf_counter()
    return {
        "initialization_seconds": initialized - session_started,
        "session_seconds": session_finished - session_started,
        "recovery_auc": float(np.mean([row["recovery_auc"] for row in periods])),
        "recovery_auc_linf": float(
            np.mean([row["recovery_auc_linf"] for row in periods])
        ),
        "shift_end_l1": float(np.mean([row["shift_end_l1"] for row in periods])),
        "shift_end_linf": float(
            np.mean([row["shift_end_linf"] for row in periods])
        ),
        "mean_calls_per_period": float(
            np.mean([row["completed_calls"] for row in periods])
        ),
        "mean_requested_backups_per_period": float(
            np.mean([row["requested_backups"] for row in periods])
        ),
        "mean_completed_backups_per_period": float(
            np.mean([row["completed_backups"] for row in periods])
        ),
        "mean_completed_backups_per_second": float(
            np.mean([row["completed_backups_per_second"] for row in periods])
        ),
        "mean_backup_utilization": float(
            np.mean([row["backup_utilization"] for row in periods])
        ),
        "mean_state_coverage_fraction": float(
            np.mean([row["state_coverage_fraction"] for row in periods])
        ),
        "mean_effective_state_fraction": float(
            np.mean([row["effective_state_fraction"] for row in periods])
        ),
        "mean_max_state_backup_share": float(
            np.mean([row["max_state_backup_share"] for row in periods])
        ),
        "mean_bellman_residual_inf": float(
            np.mean([row["bellman_residual_inf"] for row in periods])
        ),
        "mean_residual_cache_linf": float(
            np.mean([row["residual_cache_linf"] for row in periods])
        ),
        "mean_topology_signal_ratio": float(
            np.mean([row["topology_signal_ratio"] for row in periods])
        ),
        "mean_slem_refreshes_per_period": float(
            np.mean([row["slem_refreshes"] for row in periods])
        ),
        "mean_real_updates_per_period": float(
            np.mean([row["real_updates"] for row in periods])
        ),
        "mean_real_updates_per_second": float(
            np.mean([row["real_updates_per_second"] for row in periods])
        ),
        "mean_period_seconds": float(
            np.mean([row["duration_seconds"] for row in periods])
        ),
        "periods": periods,
    }


def _run_seed(
    seed: int,
    *,
    host: str,
    n_states: int,
    n_actions: int,
    branching: int,
    n_nbr_rate: float,
    reward_support: int,
    gamma: float,
    n_shifts: int,
    period_seconds: float,
    min_calls_per_period: int,
    budget: int,
    eta_td: float,
    method_cfgs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    env, initial_reward, rewards, targets = _make_problem(
        seed,
        n_states=n_states,
        n_actions=n_actions,
        branching=branching,
        n_nbr_rate=n_nbr_rate,
        reward_support=reward_support,
        gamma=gamma,
        n_shifts=n_shifts,
    )
    method_order = list(METHODS)
    np.random.default_rng(seed + 91_337).shuffle(method_order)
    records = {
        method: _run_method(
            method=method,
            host=host,
            seed=seed,
            env=env,
            initial_reward=initial_reward,
            rewards=rewards,
            targets=targets,
            gamma=gamma,
            period_seconds=period_seconds,
            min_calls_per_period=min_calls_per_period,
            budget=budget,
            eta_td=eta_td,
            method_cfgs=method_cfgs,
        )
        for method in method_order
    }
    return {"seed": seed, "n_states": env.n_states, "records": records}


def _metric_summary(values: np.ndarray) -> dict[str, float]:
    return {"mean": float(values.mean()), "ci95": _ci95(values)}


def _aggregate(
    raw: list[dict[str, Any]],
    *,
    sgm_shift_seconds: float,
    period_seconds: float,
    grid_points: int = 101,
) -> dict[str, Any]:
    time_grid = np.linspace(0.0, period_seconds, grid_points)
    summary: dict[str, Any] = {}
    scalar_metrics = (
        "initialization_seconds",
        "session_seconds",
        "recovery_auc",
        "recovery_auc_linf",
        "shift_end_l1",
        "shift_end_linf",
        "mean_calls_per_period",
        "mean_requested_backups_per_period",
        "mean_completed_backups_per_period",
        "mean_completed_backups_per_second",
        "mean_backup_utilization",
        "mean_state_coverage_fraction",
        "mean_effective_state_fraction",
        "mean_max_state_backup_share",
        "mean_bellman_residual_inf",
        "mean_residual_cache_linf",
        "mean_topology_signal_ratio",
        "mean_slem_refreshes_per_period",
        "mean_real_updates_per_period",
        "mean_real_updates_per_second",
        "mean_period_seconds",
    )
    for method in METHODS:
        method_summary: dict[str, Any] = {}
        for metric in scalar_metrics:
            values = np.asarray(
                [seed_row["records"][method][metric] for seed_row in raw],
                dtype=float,
            )
            method_summary[metric] = _metric_summary(values)
        session_values = np.asarray(
            [seed_row["records"][method]["session_seconds"] for seed_row in raw]
        )
        init_values = np.asarray(
            [
                seed_row["records"][method]["initialization_seconds"]
                for seed_row in raw
            ]
        )
        method_summary["session_sgm10_seconds"] = shifted_geometric_mean(
            session_values, shift=sgm_shift_seconds
        )
        method_summary["initialization_sgm10_seconds"] = shifted_geometric_mean(
            init_values, shift=sgm_shift_seconds
        )

        seed_curves_l1 = []
        seed_curves_linf = []
        for seed_row in raw:
            period_curves_l1 = []
            period_curves_linf = []
            for period in seed_row["records"][method]["periods"]:
                times = np.asarray(period["times_seconds"])
                period_curves_l1.append(
                    np.interp(time_grid, times, np.asarray(period["l1_errors"]))
                )
                period_curves_linf.append(
                    np.interp(time_grid, times, np.asarray(period["linf_errors"]))
                )
            seed_curves_l1.append(np.mean(period_curves_l1, axis=0))
            seed_curves_linf.append(np.mean(period_curves_linf, axis=0))
        seed_curves_array = np.asarray(seed_curves_l1)
        seed_curves_linf_array = np.asarray(seed_curves_linf)
        method_summary["time_curve"] = {
            "seconds": time_grid.tolist(),
            "mean_l1": seed_curves_array.mean(axis=0).tolist(),
            "ci95_l1": [
                _ci95(seed_curves_array[:, index])
                for index in range(seed_curves_array.shape[1])
            ],
            "mean_linf": seed_curves_linf_array.mean(axis=0).tolist(),
            "ci95_linf": [
                _ci95(seed_curves_linf_array[:, index])
                for index in range(seed_curves_linf_array.shape[1])
            ],
        }
        summary[method] = method_summary
    return summary


def _paired_summary(raw: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "initialization_seconds",
        "session_seconds",
        "recovery_auc",
        "recovery_auc_linf",
        "shift_end_l1",
        "shift_end_linf",
        "mean_calls_per_period",
        "mean_requested_backups_per_period",
        "mean_completed_backups_per_period",
        "mean_completed_backups_per_second",
        "mean_backup_utilization",
        "mean_state_coverage_fraction",
        "mean_effective_state_fraction",
        "mean_max_state_backup_share",
        "mean_bellman_residual_inf",
        "mean_residual_cache_linf",
        "mean_topology_signal_ratio",
        "mean_slem_refreshes_per_period",
        "mean_real_updates_per_period",
        "mean_real_updates_per_second",
    )
    paired = {}
    for metric in metrics:
        differences = np.asarray(
            [
                seed_row["records"]["gta_beta"][metric]
                - seed_row["records"]["standard_ps"][metric]
                for seed_row in raw
            ]
        )
        paired[metric] = {
            "mean_difference_gta_minus_standard": float(differences.mean()),
            "ci95": _ci95(differences),
        }
    return paired


def _plot(output: dict[str, Any], output_stem: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_style()
    colors = get_palette("line")
    grid_color = get_utility_color("grid")
    summary = output["summary"]
    protocol = output["protocol"]
    figure, axes = plt.subplots(1, 3, figsize=(15.6, 4.5))

    for method in METHODS:
        curve = summary[method]["time_curve"]
        seconds = np.asarray(curve["seconds"])
        mean_l1 = np.asarray(curve["mean_l1"])
        ci95_l1 = np.asarray(curve["ci95_l1"])
        mean_linf = np.asarray(curve["mean_linf"])
        ci95_linf = np.asarray(curve["ci95_linf"])
        axes[0].plot(seconds, mean_l1, color=colors[method], label=LABELS[method])
        axes[0].fill_between(
            seconds,
            mean_l1 - ci95_l1,
            mean_l1 + ci95_l1,
            color=colors[method],
            alpha=0.16,
        )
        axes[1].plot(seconds, mean_linf, color=colors[method], label=LABELS[method])
        axes[1].fill_between(
            seconds,
            mean_linf - ci95_linf,
            mean_linf + ci95_linf,
            color=colors[method],
            alpha=0.16,
        )
    axes[0].set_xlabel("Elapsed time after reward shift (s)")
    axes[0].set_ylabel(r"Mean $L_1$ value error (lower is better)")
    axes[0].grid(True, color=grid_color, alpha=0.3)
    axes[0].legend(loc="best")
    axes[1].set_xlabel("Elapsed time after reward shift (s)")
    axes[1].set_ylabel(r"Mean $L_\infty$ value error (lower is better)")
    axes[1].grid(True, color=grid_color, alpha=0.3)

    x = np.arange(len(METHODS))
    width = 0.34
    recovery_l1 = [summary[method]["recovery_auc"]["mean"] for method in METHODS]
    recovery_l1_ci = [summary[method]["recovery_auc"]["ci95"] for method in METHODS]
    recovery_linf = [
        summary[method]["recovery_auc_linf"]["mean"] for method in METHODS
    ]
    recovery_linf_ci = [
        summary[method]["recovery_auc_linf"]["ci95"] for method in METHODS
    ]
    axes[2].bar(
        x - width / 2,
        recovery_l1,
        width,
        yerr=recovery_l1_ci,
        capsize=4,
        color=[colors[method] for method in METHODS],
        label=r"$L_1$ AUC",
    )
    axes[2].bar(
        x + width / 2,
        recovery_linf,
        width,
        yerr=recovery_linf_ci,
        capsize=4,
        color=[colors[method] for method in METHODS],
        alpha=0.55,
        hatch="//",
        label=r"$L_\infty$ AUC",
    )
    axes[2].set_xticks(x, [LABELS[method] for method in METHODS])
    axes[2].set_ylabel("Recovery AUC (lower is better)")
    axes[2].grid(True, axis="y", color=grid_color, alpha=0.3)
    axes[2].legend(loc="lower right")
    diagnostics = []
    for method in METHODS:
        init_ms = summary[method]["initialization_seconds"]["mean"] * 1e3
        backups_per_second = summary[method]["mean_completed_backups_per_second"]["mean"]
        coverage = summary[method]["mean_state_coverage_fraction"]["mean"]
        real_updates = summary[method]["mean_real_updates_per_second"]["mean"]
        real_text = (
            f", {real_updates:.0f} real/s"
            if protocol.get("host") == "dyna"
            else ""
        )
        diagnostics.append(
            f"{LABELS[method]}: {backups_per_second:.0f} backups/s{real_text}, "
            f"coverage {coverage:.0%}, init {init_ms:.0f} ms"
        )
    axes[2].text(
        0.02,
        0.98,
        "\n".join(diagnostics),
        transform=axes[2].transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": grid_color, "alpha": 0.9},
    )

    figure.suptitle(
        "Wall-Clock Recovery Under Time-Triggered Reward Shifts\n"
        f"GARNET {protocol.get('host', 'dp').upper()} |S|={protocol['n_states']}, "
        f"{protocol['period_seconds']:.1f}s periods, "
        f"planning chunk={protocol['backup_budget_per_call']} backups",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wall-clock recovery under time-triggered GARNET shifts."
    )
    parser.add_argument("--n-states", type=int, default=DEFAULT_N_STATES)
    parser.add_argument("--n-actions", type=int, default=5)
    parser.add_argument("--branching", type=int, default=3)
    parser.add_argument("--n-nbr-rate", type=float, default=0.10)
    parser.add_argument("--reward-support", type=int, default=DEFAULT_REWARD_SUPPORT)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--host", choices=("dp", "dyna"), default="dp")
    parser.add_argument("--eta-td", type=float, default=0.1)
    parser.add_argument("--n-shifts", type=int, default=8)
    parser.add_argument("--period-seconds", type=float, default=10.0)
    parser.add_argument(
        "--min-backups-per-period",
        dest="min_calls_per_period",
        type=int,
        default=100,
    )
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument(
        "--slem-recompute-every",
        type=int,
        help="Override the GTA SLEM/Arnoldi refresh period in completed backups.",
    )
    parser.add_argument("--sgm-shift-seconds", type=float, default=10.0)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "rebuttal" / "wall_clock_multishift_time",
    )
    parser.add_argument(
        "--plot-from",
        type=Path,
        help="Regenerate the plot from an existing JSON artifact.",
    )
    args = parser.parse_args()

    if args.plot_from is not None:
        output = json.loads(args.plot_from.read_text(encoding="utf-8"))
        args.out_dir.mkdir(parents=True, exist_ok=True)
        _plot(output, args.out_dir / "wall_clock_multishift_time")
        return
    if (
        min(
            args.n_states,
            args.n_actions,
            args.branching,
            args.n_nbr_rate,
            args.reward_support,
            args.n_shifts,
            args.period_seconds,
            args.eta_td,
            args.min_calls_per_period,
            args.budget,
            args.sgm_shift_seconds,
            args.seeds,
            args.workers,
        )
        <= 0
    ):
        raise ValueError("sizes, times, budgets, seeds, and workers must be positive")
    if args.slem_recompute_every is not None and args.slem_recompute_every <= 0:
        raise ValueError("--slem-recompute-every must be positive")

    reference_env = GarnetEnv(
        n_states=args.n_states,
        n_actions=args.n_actions,
        branching=args.branching,
        reward_support=args.reward_support,
        gamma=args.gamma,
        n_nbr_rate=args.n_nbr_rate,
        reward_locality=True,
        seed=0,
    )
    if args.n_states == DEFAULT_N_STATES and reference_env.n_states != 200:
        raise RuntimeError("default GARNET geometry must contain 200 states")

    tuned = load_tuned_for_env("garnet_s200_dense")
    gta_config = dict(tuned.get("gta_beta") or {})
    if args.slem_recompute_every is not None:
        gta_config["slem_recompute_every"] = args.slem_recompute_every
    method_cfgs = {"standard_ps": {}, "gta_beta": gta_config}
    worker_kwargs = {
        "host": args.host,
        "n_states": args.n_states,
        "n_actions": args.n_actions,
        "branching": args.branching,
        "n_nbr_rate": args.n_nbr_rate,
        "reward_support": args.reward_support,
        "gamma": args.gamma,
        "n_shifts": args.n_shifts,
        "period_seconds": args.period_seconds,
        "min_calls_per_period": args.min_calls_per_period,
        "budget": args.budget,
        "eta_td": args.eta_td,
        "method_cfgs": method_cfgs,
    }
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    raw = RunManager(n_workers=args.workers).run(
        _run_seed,
        seeds,
        worker_kwargs,
        desc="wall-clock-time-cadence",
    )
    summary = _aggregate(
        raw,
        sgm_shift_seconds=args.sgm_shift_seconds,
        period_seconds=args.period_seconds,
    )
    output = {
        "protocol": {
            "environment": "GARNET",
            "host": args.host,
            "dyna_protocol": (
                "oracle-topology warm start with timed uniform-state random-action "
                "real updates and sampled TD planning backups"
                if args.host == "dyna"
                else None
            ),
            "n_states": reference_env.n_states,
            "n_actions": args.n_actions,
            "branching": args.branching,
            "n_nbr_rate": args.n_nbr_rate,
            "reward_support": args.reward_support,
            "gamma": args.gamma,
            "n_shifts": args.n_shifts,
            "period_seconds": args.period_seconds,
            "minimum_completed_backups_per_period": args.min_calls_per_period,
            "backup_budget_per_call": args.budget,
            "sgm_shift_seconds": args.sgm_shift_seconds,
            "seeds": seeds,
            "workers": args.workers,
            "gta_frontier_cap": None,
            "method_configs": method_cfgs,
            "timed_region": (
                "agent initialization plus time-triggered reward updates, "
                "queue work, backups, topology refreshes, Arnoldi work, and solves"
            ),
            "excluded_from_timing": (
                "reward-sequence generation and exact value iteration targets"
            ),
        },
        "summary": summary,
        "paired": _paired_summary(raw),
        "raw": raw,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / "wall_clock_multishift_time.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    _plot(output, args.out_dir / "wall_clock_multishift_time")
    print(f"results={output_path}")


if __name__ == "__main__":
    main()
