from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScriptSpec:
    path: str
    label: str
    forwarded_flags: dict[str, str]
    fixed_args: tuple[str, ...] = ()


GROUPS: dict[str, tuple[ScriptSpec, ...]] = {
    "sanity": (
        ScriptSpec(
            path="misc/time_benchmark.py",
            label="Time benchmark",
            forwarded_flags={
                "workers": "--workers",
                "seeds": "--seeds",
            },
        ),
        ScriptSpec(
            path="misc/convergence_benchmark.py",
            label="Run convergence benchmark",
            forwarded_flags={
                "seeds": "--seeds",
                "K": "--K",
                "n_shifts": "--n-shifts",
            },
        ),
        ScriptSpec(
            path="misc/showcase.py",
            label="Showcase ablation (time-series + CI)",
            forwarded_flags={
                "workers": "--workers",
                "seeds": "--seeds",
                "K": "--K",
                "n_shifts": "--n-shifts",
                "n_steps_per_shift": "--n-steps-per-shift",
                "host": "--host",
            },
            fixed_args=("--mode", "ablation"),
        ),
        ScriptSpec(
            path="misc/showcase.py",
            label="Showcase mode (single agent GIF)",
            forwarded_flags={
                "K": "--K",
                "n_shifts": "--n-shifts",
                "n_steps_per_shift": "--n-steps-per-shift",
                "host": "--host",
                "n_frames": "--n-frames",
                "seeds": "--seeds",
            },
            fixed_args=("--mode", "showcase"),
        ),
    ),
    "rqs": (
        ScriptSpec(
            path="experiments/rq1_proactiveness.py",
            label="RQ1 proactiveness",
            forwarded_flags={
                "workers": "--workers",
                "seeds": "--seeds",
                "K": "--K",
            },
        ),
        ScriptSpec(
            path="experiments/rq1_dp_timeline.py",
            label="RQ1 DP timeline",
            forwarded_flags={
                "workers": "--workers",
                "seeds": "--seeds",
                "n_steps_per_shift": "--n-steps",
                "n_shifts": "--n-shifts",
            },
        ),
        ScriptSpec(
            path="experiments/rq1_dyna_timeline.py",
            label="RQ1 Dyna timeline",
            forwarded_flags={
                "workers": "--workers",
                "seeds": "--seeds",
                "n_steps_per_shift": "--n-steps",
                "n_shifts": "--n-shifts",
            },
        ),
        ScriptSpec(
            path="experiments/rq2_allocation.py",
            label="RQ2 allocation",
            forwarded_flags={
                "workers": "--workers",
                "seeds": "--seeds",
                "n_shifts": "--n-shifts",
            },
        ),
        ScriptSpec(
            path="experiments/rq2_beta_magnitude.py",
            label="RQ2 beta",
            forwarded_flags={
                "workers": "--workers",
                "seeds": "--seeds",
            },
        ),
        ScriptSpec(
            path="experiments/rq3_mixing_tradeoff.py",
            label="RQ3 tradeoff",
            forwarded_flags={
                "workers": "--workers",
                "seeds": "--seeds",
            },
        ),
        ScriptSpec(
            path="experiments/rq4_asymmetry.py",
            label="RQ4 asymmetry",
            forwarded_flags={
                "workers": "--workers",
                "seeds": "--seeds",
                "K": "--K",
                "n_steps_per_shift": "--n-steps-per-shift",
                "n_shifts": "--n-shifts",
            },
        ),
        ScriptSpec(
            path="experiments/rq5_bottlenecks.py",
            label="RQ5 bottlenecks",
            forwarded_flags={
                "workers": "--workers",
                "seeds": "--seeds",
                "K": "--K",
                "n_steps_per_shift": "--n-steps-per-shift",
                "n_shifts": "--n-shifts",
            },
        ),
        ScriptSpec(
            path="experiments/rq6_curl_divergence.py",
            label="RQ6 curl divergence",
            forwarded_flags={
                "workers": "--workers",
                "seeds": "--seeds",
                "K": "--K",
                "n_steps_per_shift": "--n-steps-per-shift",
                "n_shifts": "--n-shifts",
            },
        ),
    ),
    "ablations": (
        ScriptSpec(
            path="misc/mixing_threshold_ablation.py",
            label="Mixing threshold ablation",
            forwarded_flags={
                "workers": "--workers",
                "seeds": "--seeds",
                "K": "--K",
            },
        ),
        ScriptSpec(
            path="misc/shift_rate_ablation.py",
            label="Shift-rate ablation",
            forwarded_flags={
                "workers": "--workers",
                "seeds": "--seeds",
                "K": "--K",
                "n_shifts": "--n-shifts",
            },
        ),
        ScriptSpec(
            path="misc/gamma_ablation.py",
            label="Gamma ablation",
            forwarded_flags={
                "workers": "--workers",
                "seeds": "--seeds",
                "K": "--K",
            },
        ),
        ScriptSpec(
            path="misc/periodicity_ablation.py",
            label="Periodicity ablation",
            forwarded_flags={
                "workers": "--workers",
                "seeds": "--seeds",
                "K": "--K",
            },
        ),
        ScriptSpec(
            path="misc/reward_support_ablation.py",
            label="Reward-support ablation",
            forwarded_flags={
                "workers": "--workers",
                "seeds": "--seeds",
                "K": "--K",
                "n_steps_per_shift": "--n-steps-per-shift",
                "n_shifts": "--n-shifts",
            },
        ),
        ScriptSpec(
            path="misc/cumulative_ablation.py",
            label="Cumulative ablation",
            forwarded_flags={
                "workers": "--workers",
                "seeds": "--seeds",
                "K": "--K",
            },
        ),
    ),
}


def is_truthy_env(value: str | None) -> bool:
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GTA-PS experiment groups sequentially with a restricted forwarded flag surface.",
        allow_abbrev=False,
    )
    parser.add_argument("group", choices=tuple(GROUPS.keys()))
    parser.add_argument("--workers", help="Worker count to forward. Use 'auto' to keep script defaults where supported.")
    parser.add_argument("--seeds", type=int, help="Seed count to forward when supported.")
    parser.add_argument("--K", type=int, help="Planning budget to forward when supported.")
    parser.add_argument("--n-steps-per-shift", dest="n_steps_per_shift", type=int,
                        help="Steps-per-shift override to forward when supported.")
    parser.add_argument("--n-shifts", dest="n_shifts", type=int,
                        help="Shift-count override to forward when supported.")
    parser.add_argument("--n-frames", dest="n_frames", type=int,
                        help="Animation-frame override to forward when supported.")
    parser.add_argument("--host", type=str, default=None,
                        help="Host override to forward when supported.")
    parser.add_argument("--start-from", dest="start_from", type=int, default=1,
                        help="Start from a specific script index (1-based). Scripts before this index are skipped.")
    return parser.parse_args()


def coerce_workers_for_tuner(workers: str | None) -> str | None:
    if workers is None:
        return None
    if workers.lower() == "auto":
        return None
    try:
        int(workers)
    except ValueError as error:
        raise ValueError("misc/tuner.py only accepts an integer worker count or 'auto'.") from error
    return workers


def build_command(spec: ScriptSpec, args: argparse.Namespace) -> list[str]:
    # The script path should be relative to the directory containing this orchestrator (code_root).
    script_path = str(Path(spec.path))
    command = [sys.executable, script_path, *spec.fixed_args]
    for dest, cli_flag in spec.forwarded_flags.items():
        value = getattr(args, dest, None)
        if value is None:
            continue
        if dest == "workers" and spec.path == "misc/tuner.py":
            value = coerce_workers_for_tuner(value)
            if value is None:
                continue
        command.extend([cli_flag, str(value)])
    return command


def main() -> int:
    args = parse_args()
    # Use the parent directory of this script as the working directory for sub-scripts.
    # We keep it relative to the caller to avoid issues with absolute path resolution in some environments.
    code_root = Path(__file__).parent
    mpl_cache_dir = code_root / ".mplconfig"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    run_env = os.environ.copy()
    run_env.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))
    run_env.setdefault("TMPDIR", str(mpl_cache_dir))
    run_env.setdefault("TEMP", str(mpl_cache_dir))
    run_env.setdefault("TMP", str(mpl_cache_dir))
    dry_run = is_truthy_env(os.environ.get("GTA_PS_ORCH_DRY_RUN"))
    scripts = GROUPS[args.group]

    print(f"Running group '{args.group}' from {code_root}")
    if dry_run:
        print("Dry run enabled via GTA_PS_ORCH_DRY_RUN; commands will be printed only.")

    for index, spec in enumerate(scripts, start=1):
        if index < args.start_from:
            print(f"[{index}/{len(scripts)}] {spec.label} - skipped (start-from={args.start_from})")
            continue
        command = build_command(spec, args)
        print(f"[{index}/{len(scripts)}] {spec.label}")
        print(f"    Raw script command: {subprocess.list2cmdline(command)}")
        if dry_run:
            continue
        completed = subprocess.run(command, cwd=code_root, env=run_env)
        if completed.returncode != 0:
            print(f"Stopping group '{args.group}' after failure in {spec.path}.", file=sys.stderr)
            return completed.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
