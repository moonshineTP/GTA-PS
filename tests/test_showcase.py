import importlib
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("minigrid")
pytest.importorskip("scipy")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "misc"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "threads"))

showcase = importlib.import_module("showcase")
showcase_support = importlib.import_module("showcase_support")


class DummyTqdm:
    def __init__(self, *args, **kwargs):
        self.total = kwargs.get("total", args[0] if args else 0)
        self.count = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, step=1):
        self.count += step


def test_showcase_parser_supports_shift_steps_alias():
    args = showcase.build_parser().parse_args(["--mode", "showcase", "--shift-steps", "7", "--seeds", "9"])

    assert args.mode == "showcase"
    assert args.n_steps_per_shift == 7
    assert args.seeds == 9


def test_showcase_main_dispatches_selected_mode(monkeypatch):
    calls = []
    monkeypatch.setattr(showcase, "run_ablation", lambda args: calls.append(("ablation", args.mode, args.seeds)))
    monkeypatch.setattr(showcase, "run_showcase", lambda args: calls.append(("showcase", args.mode, args.seeds)))

    showcase.main(["--mode", "ablation", "--seeds", "2"])
    showcase.main(["--mode", "showcase", "--seeds", "7"])

    assert calls == [("ablation", "ablation", 2), ("showcase", "showcase", 7)]


def test_showcase_options_apply_mode_specific_env_overrides(tmp_path):
    cfg_path = tmp_path / "showcase.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "env: fourrooms",
                "room_size: 5",
                "corridor_len: 2",
                "k: 8",
                "n_peaks_per_room: 4",
                "sigma_range: [0.1, 0.2]",
            ]
        ),
        encoding="utf-8",
    )
    args = showcase.build_parser().parse_args(
        [
            "--mode",
            "showcase",
            "--config",
            str(cfg_path),
            "--n-shifts",
            "3",
            "--n-steps-per-shift",
            "11",
            "--n-frames",
            "7",
            "--seeds",
            "5",
        ]
    )

    options = showcase_support.ShowcaseOptions.from_args(args)

    assert options.seed == 5
    assert options.shift_steps == 11
    assert options.total_steps == 33
    assert options.frame_count == 7
    assert options.env_kwargs["room_size"] == 5
    assert options.env_kwargs["corridor_len"] == 2
    assert options.env_kwargs["k"] == showcase_support.SHOWCASE_ACTIVE_ROOMS
    assert options.env_kwargs["n_peaks_per_room"] == showcase_support.SHOWCASE_PEAKS_PER_ROOM
    assert options.env_kwargs["sigma_range"] == showcase_support.SHOWCASE_SIGMA_RANGE


def test_run_seed_ablation_returns_expected_payload_shape():
    result = showcase._run_seed_ablation(
        seed=0,
        K=1,
        n_shifts=2,
        T_shift=2,
        host="dyna",
        tuned_cfg=showcase._build_method_cfgs({}),
        env_kwargs={
            "gamma": showcase_support.GAMMA,
            "room_size": 3,
            "corridor_len": 1,
            "k": showcase_support.SHOWCASE_ACTIVE_ROOMS,
            "n_peaks_per_room": showcase_support.SHOWCASE_PEAKS_PER_ROOM,
            "sigma_range": showcase_support.SHOWCASE_SIGMA_RANGE,
        },
    )

    assert set(result) == {
        "return_history",
        "cumul_reward",
        "cumul_l1",
        "coverage",
        "final_V",
        "final_queue",
        "r_current",
    }
    assert len(result["return_history"]["standard_ps"]) == 2
    assert len(result["cumul_reward"]["gta_beta"]) == 3
    assert len(result["coverage"]["standard_ps"]) == 2
    assert len(result["final_V"]["standard_ps"]) == len(result["r_current"])
    assert len(result["final_queue"]["gta_beta"]) == len(result["r_current"])


def test_run_ablation_writes_summary_json(tmp_path, monkeypatch):
    recorded = {}

    class DummyRunManager:
        def __init__(self, n_workers):
            recorded["workers"] = n_workers

        def run(self, fn, seeds, fn_kwargs, desc):
            recorded["desc"] = desc
            recorded["seeds"] = list(seeds)
            return [fn(seed=seed, **fn_kwargs) for seed in seeds]

    monkeypatch.setattr(showcase, "RunManager", DummyRunManager)
    monkeypatch.setattr(showcase, "ROOT", tmp_path)
    monkeypatch.setattr(showcase, "apply_style", lambda: None)
    monkeypatch.setattr(
        showcase,
        "get_palette",
        lambda _name: {variant: f"C{index}" for index, variant in enumerate(showcase.VARIANTS)},
    )
    monkeypatch.setattr(showcase.plt, "savefig", lambda path, **kwargs: recorded.setdefault("figure_path", Path(path)))

    args = showcase.build_parser().parse_args(
        [
            "--mode",
            "ablation",
            "--seeds",
            "1",
            "--n-shifts",
            "1",
            "--n-steps-per-shift",
            "2",
            "--K",
            "1",
            "--workers",
            "auto",
        ]
    )
    showcase.run_ablation(args)

    payload = json.loads((tmp_path / "results" / "misc" / "showcase" / "ablation.json").read_text(encoding="utf-8"))
    assert recorded["workers"] == "auto"
    assert recorded["desc"] == "fourrooms-ablation"
    assert recorded["seeds"] == [0]
    assert recorded["figure_path"].name == "fourrooms_ablation.png"
    assert payload["config"]["mode"] == "ablation"
    assert payload["config"]["seeds"] == 1
    assert payload["config"]["T_shift"] == 2
    assert set(payload["returns"]) == set(showcase.VARIANTS)


def test_run_showcase_builds_animation_and_output_path(tmp_path, monkeypatch):
    recorded = {}

    class DummyAnimation:
        def __init__(self, fig, update, frames, interval, blit, cache_frame_data):
            self.fig = fig
            self.update = update
            self.frames = frames
            recorded["animation_frames"] = frames

    def fake_save(anim, fig, out_path, *, frames, fps):
        anim.update(0)
        recorded["out_path"] = out_path
        recorded["frames"] = frames
        recorded["fps"] = fps
        recorded["title"] = fig._suptitle.get_text()

    monkeypatch.setattr(showcase, "ROOT", tmp_path)
    monkeypatch.setattr(showcase, "apply_style", lambda: None)
    monkeypatch.setattr(showcase, "_require_ffmpeg", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(showcase, "_save_animation_with_ffmpeg", fake_save)
    monkeypatch.setattr(showcase.animation, "FuncAnimation", DummyAnimation)
    monkeypatch.setattr(showcase, "tqdm", DummyTqdm)

    args = showcase.build_parser().parse_args(
        [
            "--mode",
            "showcase",
            "--seeds",
            "3",
            "--n-shifts",
            "1",
            "--n-steps-per-shift",
            "2",
            "--n-frames",
            "1",
            "--K",
            "1",
        ]
    )
    showcase.run_showcase(args)

    assert recorded["animation_frames"] == 1
    assert recorded["frames"] == 1
    assert recorded["fps"] == 8
    assert recorded["out_path"].name == "fourrooms_showcase_last400.gif"
    assert "Step 2" in recorded["title"]
