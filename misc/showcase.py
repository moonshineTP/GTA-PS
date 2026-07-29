"""
misc/showcase.py

Visual comparison: Standard-PS vs GTA-PS^beta on FourRooms.

Modes
-----
--mode ablation
    Parallel multi-seed run. Saves a PNG with cumulative reward, cumulative
    L1 error, and coverage for both agents, plus a JSON summary.

--mode showcase
    Single-seed animation saved as GIF via an ffmpeg MP4 intermediate.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.artist import Artist
from tqdm import tqdm

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "configs").exists():
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs").exists() and (parent / "src").exists():
            ROOT = parent
            break
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "threads"))
sys.path.insert(0, str(ROOT / "misc"))

from plot_academic import apply_style, get_palette
from runner import RunManager
from showcase_support import (
    K,
    LABELS,
    N_SHIFTS,
    VARIANTS,
    AblationOptions,
    FourRoomsRenderer,
    IndependentAgentSession,
    ShowcaseOptions,
    build_method_cfgs,
    fourrooms_env_kwargs_from_config,
    load_config,
    run_seed_ablation,
    simulate_showcase_frames,
)

_build_method_cfgs = build_method_cfgs
_fourrooms_env_kwargs_from_config = fourrooms_env_kwargs_from_config
_load_config = load_config
_run_seed_ablation = run_seed_ablation


def _find_executable(name: str) -> str | None:
    extensions = [""]
    if os.name == "nt":
        extensions.extend(os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD").split(os.pathsep))
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not directory:
            continue
        for extension in extensions:
            for suffix in (extension.lower(), extension.upper()):
                candidate = Path(directory) / f"{name}{suffix}"
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    return str(candidate)
    return None


def _require_ffmpeg() -> tuple[str, str]:
    ffmpeg = _find_executable("ffmpeg")
    ffprobe = _find_executable("ffprobe")
    if ffmpeg is None or ffprobe is None:
        missing = "ffmpeg" if ffmpeg is None else "ffprobe"
        raise RuntimeError(f"{missing} is required for showcase GIF generation")
    return ffmpeg, ffprobe


def _save_animation_with_ffmpeg(
    anim: animation.FuncAnimation,
    fig: plt.Figure,
    out_path: Path,
    *,
    frames: int,
    fps: int,
) -> None:
    ffmpeg, ffprobe = _require_ffmpeg()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fourrooms_showcase_", dir=out_path.parent) as tmp_dir:
        tmp_video = Path(tmp_dir) / "fourrooms_showcase.mp4"
        writer = animation.FFMpegWriter(
            fps=fps,
            codec="mpeg4",
            bitrate=900,
            extra_args=["-q:v", "5", "-threads", "2"],
        )
        with tqdm(total=frames, desc="showcase-mp4", unit="frame", dynamic_ncols=True, ascii=True) as pbar:
            last_frame = -1

            def _on_save_progress(frame_idx: int, _total: int) -> None:
                nonlocal last_frame
                if frame_idx > last_frame:
                    pbar.update(frame_idx - last_frame)
                    last_frame = frame_idx

            anim.save(str(tmp_video), writer=writer, progress_callback=_on_save_progress)

        palette = Path(tmp_dir) / "palette.png"
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-v",
                "warning",
                "-i",
                str(tmp_video),
                "-vf",
                "fps=8,palettegen=max_colors=96:stats_mode=diff",
                "-frames:v",
                "1",
                str(palette),
            ],
            check=True,
        )
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-v",
                "warning",
                "-i",
                str(tmp_video),
                "-i",
                str(palette),
                "-lavfi",
                "fps=8[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
                str(out_path),
            ],
            check=True,
        )

    probe = subprocess.run(
        [
            ffprobe,
            "-v",
            "quiet",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames",
            "-of",
            "csv=p=0",
            str(out_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Showcase GIF saved to {out_path} ({size_mb:.1f} MiB; {probe.stdout.strip()})")


def _aggregate_with_ci(series: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    from scipy import stats as sp_stats

    array = np.asarray(series, dtype=np.float64)
    mean = array.mean(axis=0)
    if len(series) < 2:
        return mean, np.zeros_like(mean)
    ci95 = sp_stats.sem(array, axis=0) * sp_stats.t.ppf(0.975, df=len(series) - 1)
    return mean, ci95


def _plot_band(
    ax,
    *,
    x: np.ndarray,
    mean: np.ndarray,
    ci95: np.ndarray,
    color: str,
    title: str,
    ylabel: str,
    xlabel: str | None,
    show_zero_line: bool = False,
) -> None:
    ax.plot(x, mean, color=color, lw=2)
    ax.fill_between(x, mean - ci95, mean + ci95, color=color, alpha=0.18)
    if show_zero_line:
        ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, ls="--", alpha=0.3)
    if xlabel is not None:
        ax.set_xlabel(xlabel)


def _build_ablation_statistics(raw: list[dict]) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    stats = {metric: {} for metric in ("return_history", "cumul_reward", "cumul_l1", "coverage")}
    for metric in stats:
        for variant in VARIANTS:
            stats[metric][variant] = _aggregate_with_ci([result[metric][variant] for result in raw])
    return stats


def _print_ablation_summary(stats: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]) -> None:
    print(f"\n{'Variant':<20}  {'Mean Return':>14}  {'Mean Coverage':>16}")
    print("-" * 62)
    for variant in VARIANTS:
        mean_return, _ = stats["return_history"][variant]
        mean_coverage, _ = stats["coverage"][variant]
        print(f"  {LABELS[variant]:<20}  {mean_return[-1]:>14.4f}  {mean_coverage[-1]:>16.1f}")


def _save_ablation_figure(
    *,
    stats: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    options: AblationOptions,
    out_path: Path,
) -> None:
    apply_style()
    colors = {variant: get_palette("line")[variant] for variant in VARIANTS}
    fig = plt.figure(figsize=(18, 10), layout="constrained")
    grid = gridspec.GridSpec(2, 3, figure=fig)
    episodes = np.arange(1, options.n_shifts + 1)
    cumulative_episodes = np.arange(0, options.n_shifts + 1)

    metric_specs = [
        ("cumul_reward", cumulative_episodes, "Cumulative reward", "Episode $N$", False),
        ("cumul_l1", cumulative_episodes, "Cumulative L1 error", "Episode $N$", True),
        ("coverage", episodes, "Unique states", "Episode", False),
    ]
    for col, (metric, x_axis, ylabel, xlabel, show_zero_line) in enumerate(metric_specs):
        for row, variant in enumerate(VARIANTS):
            mean, ci95 = stats[metric][variant]
            axis = fig.add_subplot(grid[row, col])
            _plot_band(
                axis,
                x=x_axis,
                mean=mean,
                ci95=ci95,
                color=colors[variant],
                title=f"{LABELS[variant]}\n{ylabel}",
                ylabel=ylabel,
                xlabel=xlabel if row == 1 else None,
                show_zero_line=show_zero_line,
            )

    fig.suptitle(
        rf"FourRooms: Standard-PS vs GTA-PS$^{{\beta}}$ - "
        rf"$K={options.K}$, $N_{{\mathrm{{shifts}}}}={options.n_shifts}$, "
        rf"$N_{{\mathrm{{seeds}}}}={options.seed_count}$, host={options.host}",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_ablation_json(
    *,
    stats: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    options: AblationOptions,
    out_path: Path,
) -> None:
    payload = {
        "config": {
            "mode": "ablation",
            "host": options.host,
            "K": options.K,
            "n_shifts": options.n_shifts,
            "seeds": options.seed_count,
            "T_shift": options.shift_steps,
            "env_kwargs": options.env_kwargs,
        },
        "returns": {
            variant: {"mean": stats["return_history"][variant][0].tolist(), "ci95": stats["return_history"][variant][1].tolist()}
            for variant in VARIANTS
        },
        "coverage": {
            variant: {"mean": stats["coverage"][variant][0].tolist(), "ci95": stats["coverage"][variant][1].tolist()}
            for variant in VARIANTS
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_ablation(args) -> None:
    options = AblationOptions.from_args(args)
    print("=" * 62)
    print("  FourRooms ablation: Standard-PS vs GTA-PS^beta")
    print("=" * 62)
    print(f"  seeds={options.seed_count}  n_shifts={options.n_shifts}  K={options.K}")
    print(f"  T_shift={options.shift_steps}  host={options.host}  workers={options.workers}")
    print("-" * 62)

    runner = RunManager(n_workers=options.workers)
    raw = runner.run(
        fn=_run_seed_ablation,
        seeds=list(range(options.seed_count)),
        fn_kwargs=dict(
            K=options.K,
            n_shifts=options.n_shifts,
            T_shift=options.shift_steps,
            host=options.host,
            tuned_cfg=options.tuned_cfg,
            env_kwargs=options.env_kwargs,
        ),
        desc="fourrooms-ablation",
    )

    stats = _build_ablation_statistics(raw)
    _print_ablation_summary(stats)
    figure_path = ROOT / "plots" / "fourrooms_ablation.png"
    _save_ablation_figure(stats=stats, options=options, out_path=figure_path)
    print(f"\nAblation plot saved to {figure_path}")

    json_path = ROOT / "results" / "misc" / "showcase" / "ablation.json"
    _write_ablation_json(stats=stats, options=options, out_path=json_path)
    print(f"Ablation results saved to {json_path}")


def _build_showcase_axes(fig: plt.Figure) -> dict[str, plt.Axes]:
    grid = gridspec.GridSpec(2, 3, figure=fig)
    return {
        "env_standard_ps": fig.add_subplot(grid[0, 0]),
        "env_gta_beta": fig.add_subplot(grid[1, 0]),
        "value_standard_ps": fig.add_subplot(grid[0, 1]),
        "value_gta_beta": fig.add_subplot(grid[1, 1]),
        "queue_standard_ps": fig.add_subplot(grid[0, 2]),
        "queue_gta_beta": fig.add_subplot(grid[1, 2]),
    }


def _render_showcase_frame(
    *,
    fig: plt.Figure,
    axes: dict[str, plt.Axes],
    renderers: dict[str, FourRoomsRenderer],
    frame_data: dict,
    options: ShowcaseOptions,
) -> list[Artist]:
    for axis in axes.values():
        axis.cla()

    sps_data = frame_data["sps"]
    gta_data = frame_data["gta"]
    renderers["standard_ps"].draw_interface(
        axes["env_standard_ps"],
        reward=sps_data["r_current"],
        agent_pos=sps_data["state"],
        agent_dir=sps_data["agent_dir"],
        title=f"{LABELS['standard_ps']} Environment",
    )
    renderers["gta_beta"].draw_interface(
        axes["env_gta_beta"],
        reward=gta_data["r_current"],
        agent_pos=gta_data["state"],
        agent_dir=gta_data["agent_dir"],
        title=f"{LABELS['gta_beta']} Environment",
    )
    renderers["standard_ps"].draw_metric(
        axes["value_standard_ps"],
        values=sps_data["V"],
        title=f"{LABELS['standard_ps']} V(s)",
        cmap="magma",
    )
    renderers["gta_beta"].draw_metric(
        axes["value_gta_beta"],
        values=gta_data["V"],
        title=f"{LABELS['gta_beta']} V(s)",
        cmap="magma",
    )
    renderers["standard_ps"].draw_metric(
        axes["queue_standard_ps"],
        values=sps_data["queue"],
        title=f"{LABELS['standard_ps']} Queue xi(s)",
        cmap="hot",
    )
    renderers["gta_beta"].draw_metric(
        axes["queue_gta_beta"],
        values=gta_data["queue"],
        title=f"{LABELS['gta_beta']} Combined Queue (xi+tau)(s)",
        cmap="hot",
    )
    fig.suptitle(
        rf"FourRooms - Step {frame_data['step']} | Episode {frame_data['shift']} | "
        rf"K={options.K} | host={options.host}",
        fontsize=11,
    )
    return []


def run_showcase(args) -> None:
    options = ShowcaseOptions.from_args(args)
    _require_ffmpeg()

    print("=" * 50)
    print("  FourRooms showcase animation")
    print("=" * 50)
    print(f"  seed={options.seed}  total_steps={options.total_steps}  n_frames={options.frame_count}")
    print(f"  K={options.K}  T_shift={options.shift_steps}  host={options.host}")
    print("-" * 50)
    print(f"Simulating {options.total_steps} steps...")

    with tqdm(total=options.total_steps, desc="showcase-sim", unit="step", dynamic_ncols=True, ascii=True) as pbar:
        sessions, captured_data = simulate_showcase_frames(options, on_step=lambda: pbar.update(1))

    apply_style()
    fig = plt.figure(figsize=(6, 2.7), layout="constrained")
    axes = _build_showcase_axes(fig)
    renderers = {
        "standard_ps": FourRoomsRenderer(sessions["standard_ps"].env),
        "gta_beta": FourRoomsRenderer(sessions["gta_beta"].env),
    }

    print(f"Animating {len(captured_data)} frames...")
    anim = animation.FuncAnimation(
        fig,
        lambda frame: _render_showcase_frame(
            fig=fig,
            axes=axes,
            renderers=renderers,
            frame_data=captured_data[frame],
            options=options,
        ),
        frames=len(captured_data),
        interval=120,
        blit=False,
        cache_frame_data=False,
    )
    out_path = ROOT / "plots" / "fourrooms_showcase_last400.gif"
    _save_animation_with_ffmpeg(anim, fig, out_path, frames=len(captured_data), fps=8)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FourRooms: Standard-PS vs GTA-PS^beta (2x3 showcase/ablation)")
    parser.add_argument("--config", default=None, help="YAML config path (FourRooms configs supported)")
    parser.add_argument(
        "--mode",
        choices=["ablation", "showcase"],
        default="ablation",
        help="'ablation': parallel multi-seed PNG; 'showcase': animated GIF",
    )
    parser.add_argument("--K", type=int, default=K)
    parser.add_argument("--n-steps-per-shift", type=int, default=None, help="Real steps between reward shifts")
    parser.add_argument("--shift-steps", dest="n_steps_per_shift", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--seeds", type=int, default=40, help="Seed count in ablation mode; fixed seed value in showcase mode")
    parser.add_argument("--n-shifts", type=int, default=N_SHIFTS, help="Reward-shift episodes")
    parser.add_argument("--n-frames", type=int, default=400, help="Animation frames in showcase mode; uses the last frames only")
    parser.add_argument("--host", choices=["dyna", "dp"], default="dyna", help="Host algorithm for both agents")
    parser.add_argument("--workers", default=5, help="Parallel workers for RunManager in ablation mode")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.mode == "ablation":
        run_ablation(args)
    else:
        run_showcase(args)


if __name__ == "__main__":
    main()
