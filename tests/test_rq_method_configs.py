"""
Regression tests for RQ4/RQ5 experiment method wiring.

These focus on:
- explicit mapping from legacy tuned keys to canonical forward/full GTA variants
- fail-fast behavior for ambiguous tuned blocks
- env-specific default config selection for no-config runs
"""
import importlib
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip("matplotlib")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

rq4 = importlib.import_module("rq4_asymmetry")
rq5 = importlib.import_module("rq5_bottlenecks")
rq1_dyna = importlib.import_module("rq1_dyna_timeline")
from gta_ps.algorithms import make_agent
from gta_ps.algorithms.empirical_model import SampleModel
from gta_ps.utils.hyperparams import ENV_DEFAULT_CONFIG, load_tuned_for_env

RQ_MODULES = [rq4, rq5]

def _legacy_cfg() -> dict:
    return {
        "default": {
            "alpha": 0.5,
            "beta": 1.0,
            "beta1": 1.0,
            "beta2": 0.1,
            "eta": 0.01,
            "b": 1.0,
        },
        "tuned": {
            "gta_base": {
                "alpha": 0.83,
                "beta": 0.25,
                "eta": 0.2,
                "b": 2.0,
            },
            "gta_beta": {
                "alpha": 0.67,
                "beta1": 0.7,
                "beta2": 0.9,
                "eta": 0.3,
                "b": 3.0,
            },
        },
    }


def _load_default_env_cfg(env_name: str) -> dict:
    cfg_path = ROOT / "configs" / ENV_DEFAULT_CONFIG[env_name]
    with cfg_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _expected_default_method_cfgs(env_name: str) -> dict:
    cfg = _load_default_env_cfg(env_name)
    tuned = load_tuned_for_env(env_name)
    assert tuned == (cfg.get("tuned") or {})

    base_params = {
        key: cfg.get(key)
        for key in ["alpha", "beta", "beta1", "beta2", "eta", "b"]
        if cfg.get(key) is not None
    }
    base_params.update(
        {
            key: value
            for key, value in (cfg.get("default") or {}).items()
            if value is not None
        }
    )
    gta_beta = {
        key: value
        for key, value in (tuned.get("gta_beta") or {}).items()
        if value is not None
    }
    expected_full = {**base_params, **gta_beta}
    expected_forward = dict(expected_full)
    expected_forward["alpha"] = 0.0
    return {
        "forward_gta": expected_forward,
        "full_gta": expected_full,
    }


def _expected_default_fourrooms_base_kwargs() -> dict:
    cfg = _load_default_env_cfg("fourrooms")
    return {
        "k": cfg["k_fourrooms"],
        "n_peaks_per_room": cfg["n_peaks_per_room"],
        "sigma_range": cfg["sigma_range"],
        "room_peak_ranges": cfg["room_peak_ranges"],
    }


def _write_rq4_fourrooms_cfg(path: Path) -> None:
    cfg = {
        "env": "fourrooms",
        "room_size": 3,
        "corridor_len": 5,
        "corridor_width": 1,
        "window_offset": 0,
        "k_fourrooms": 4,
        "n_peaks_per_room": 3,
        "sigma_range": [0.2, 0.4],
        "room_peak_ranges": [
            [0.7, 1.2],
            [0.4, 0.8],
            [0.2, 0.6],
            [0.1, 0.5],
        ],
        "default": {
            "alpha": 0.5,
            "beta": 1.0,
            "beta1": 1.0,
            "beta2": 0.1,
            "eta": 0.01,
            "b": 1.0,
        },
        "tuned": {
            "gta_beta": {
                "alpha": 0.73,
                "beta1": 0.55,
                "beta2": 0.77,
                "eta": 0.44,
                "b": 3.3,
            },
        },
    }
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")


@pytest.mark.parametrize("module", RQ_MODULES, ids=lambda m: m.__name__)
def test_rq_build_method_cfgs_maps_legacy_keys_to_canonical_variants(module):
    method_cfgs = module._build_method_cfgs(_legacy_cfg())

    assert set(method_cfgs) == {"forward_gta", "full_gta"}
    assert method_cfgs["forward_gta"]["alpha"] == pytest.approx(0.0)
    assert method_cfgs["forward_gta"]["beta1"] == pytest.approx(0.7)
    assert method_cfgs["forward_gta"]["beta2"] == pytest.approx(0.9)
    assert method_cfgs["full_gta"]["alpha"] == pytest.approx(0.67)
    assert method_cfgs["full_gta"]["beta1"] == pytest.approx(0.7)
    assert method_cfgs["full_gta"]["beta2"] == pytest.approx(0.9)

    variants = module._build_variants(method_cfgs)
    assert [variant["variant"] for variant in variants] == ["standard_ps", "forward_gta", "full_gta"]
    assert variants[1]["kwargs"]["alpha"] == pytest.approx(0.0)
    assert variants[2]["kwargs"]["alpha"] == pytest.approx(0.67)


@pytest.mark.parametrize("module", RQ_MODULES, ids=lambda m: m.__name__)
def test_rq_build_method_cfgs_rejects_nonexplicit_tuned_names(module):
    cfg = {
        "default": {"alpha": 0.5, "beta": 1.0, "beta1": 1.0, "beta2": 0.1, "eta": 0.01, "b": 1.0},
        "tuned": {
            "foo": {"beta": 0.2, "alpha": 0.9},
            "bar": {"beta1": 0.4, "beta2": 0.8, "alpha": 0.6},
        },
    }

    with pytest.raises(ValueError):
        module._build_method_cfgs(cfg)


def test_forward_gta_variant_uses_adaptive_beta_family():
    import numpy as np

    P = np.zeros((1, 2, 2), dtype=float)
    P[:, :, :] = np.eye(2)
    r = np.array([0.0, 1.0], dtype=float)

    agent = make_agent(
        "forward_gta",
        host="dp",
        P=P,
        r=r,
        gamma=0.95,
        beta1=0.7,
        beta2=0.9,
        alpha=0.8,
        eta=0.1,
        b=2.0,
    )

    assert agent.module.adaptive_beta is True
    assert agent.module.alpha_base == pytest.approx(0.0)
    assert agent.module.beta1 == pytest.approx(0.7)
    assert agent.module.beta2 == pytest.approx(0.9)


def test_rq1_dyna_timeline_uses_sample_model_for_dyna_agents():
    import numpy as np

    P = np.zeros((1, 2, 2), dtype=float)
    P[:, :, :] = np.eye(2)
    r = np.array([0.0, 1.0], dtype=float)

    agent = make_agent(
        "gta_base",
        host="dyna",
        P=P,
        r=r,
        gamma=0.95,
        model_type=rq1_dyna.DYNA_MODEL_TYPE,
        rng=np.random.default_rng(0),
    )

    assert rq1_dyna.DYNA_MODEL_TYPE == "sample"
    assert isinstance(agent.model, SampleModel)


@pytest.mark.parametrize("module", RQ_MODULES, ids=lambda m: m.__name__)
def test_rq_no_config_path_uses_env_specific_default_method_cfgs(module):
    garnet_method_cfgs, fourrooms_method_cfgs = module._resolve_env_method_cfgs(None)

    assert garnet_method_cfgs["forward_gta"]["alpha"] == pytest.approx(0.0)
    assert fourrooms_method_cfgs["forward_gta"]["alpha"] == pytest.approx(0.0)
    assert garnet_method_cfgs["forward_gta"] != fourrooms_method_cfgs["forward_gta"]
    assert garnet_method_cfgs["full_gta"] != fourrooms_method_cfgs["full_gta"]

    garnet_variants = module._build_variants(garnet_method_cfgs)
    fourrooms_variants = module._build_variants(fourrooms_method_cfgs)
    assert garnet_variants[1]["kwargs"] != fourrooms_variants[1]["kwargs"]
    assert garnet_variants[2]["kwargs"] != fourrooms_variants[2]["kwargs"]


@pytest.mark.parametrize("module", RQ_MODULES, ids=lambda m: m.__name__)
def test_rq_default_method_cfgs_follow_default_config_contract(module):
    garnet_method_cfgs, fourrooms_method_cfgs = module._resolve_env_method_cfgs(None)

    print(f"\n[{module.__name__}] resolved default method cfgs")
    print("  garnet:", garnet_method_cfgs)
    print("  fourrooms:", fourrooms_method_cfgs)

    for env_name, resolved in (
        ("garnet", garnet_method_cfgs),
        ("fourrooms", fourrooms_method_cfgs),
    ):
        expected = _expected_default_method_cfgs(env_name)
        assert resolved == expected
        assert resolved["forward_gta"]["alpha"] == pytest.approx(0.0)
        assert resolved["full_gta"]["alpha"] == pytest.approx(expected["full_gta"]["alpha"])
        assert resolved["forward_gta"]["beta"] == pytest.approx(resolved["full_gta"]["beta"])
        assert resolved["forward_gta"]["beta1"] == pytest.approx(resolved["full_gta"]["beta1"])
        assert resolved["forward_gta"]["beta2"] == pytest.approx(resolved["full_gta"]["beta2"])
        assert resolved["forward_gta"]["eta"] == pytest.approx(resolved["full_gta"]["eta"])
        assert resolved["forward_gta"]["b"] == pytest.approx(resolved["full_gta"]["b"])


def test_rq4_default_fourrooms_base_kwargs_follow_default_config_contract():
    assert rq4._resolve_fourrooms_base_kwargs(None) == _expected_default_fourrooms_base_kwargs()


def test_rq4_explicit_fourrooms_config_overrides_base_geometry(tmp_path):
    cfg_path = tmp_path / "rq4_fourrooms.yaml"
    _write_rq4_fourrooms_cfg(cfg_path)

    resolved = rq4._resolve_fourrooms_base_kwargs(str(cfg_path))

    assert resolved == {
        "room_size": 3,
        "corridor_len": 5,
        "corridor_width": 1,
        "window_offset": 0,
        "k": 4,
        "n_peaks_per_room": 3,
        "sigma_range": [0.2, 0.4],
        "room_peak_ranges": [
            [0.7, 1.2],
            [0.4, 0.8],
            [0.2, 0.6],
            [0.1, 0.5],
        ],
    }


def test_rq4_transport_modes_use_resolved_fourrooms_geometry(tmp_path):
    cfg_path = tmp_path / "rq4_fourrooms.yaml"
    _write_rq4_fourrooms_cfg(cfg_path)

    base_kwargs = rq4._resolve_fourrooms_base_kwargs(str(cfg_path))
    modes = rq4._resolve_fourrooms_modes(0.95, base_kwargs)
    diagnostics = rq4._fourrooms_mode_diagnostics(0.95, base_kwargs, modes)

    assert [label for label, _ in modes] == ["Symmetric", "One-Way", "Windy"]
    assert modes[1][1]["one_way_prob"] == pytest.approx(rq4.FOURROOMS_BELT_LEAK)
    assert modes[2][1]["drift"] == (0, 1)
    assert modes[2][1]["drift_strength"] > 0.0
    assert len(diagnostics["belt_cycle_positions"]) > 4
    assert diagnostics["One-Way"]["transport_l1_vs_symmetric"] > 0.0
    assert diagnostics["Windy"]["transport_l1_vs_symmetric"] > 0.0
    assert diagnostics["pairwise_transport_l1"]["Symmetric vs One-Way"] > 0.0
    assert diagnostics["pairwise_transport_l1"]["Symmetric vs Windy"] > 0.0


@pytest.mark.parametrize("module", RQ_MODULES, ids=lambda m: m.__name__)
def test_rq_explicit_config_path_shares_method_cfgs_across_envs(module, tmp_path):
    cfg_path = tmp_path / "methods.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "default:",
                "  alpha: 0.5",
                "  beta: 1.0",
                "  beta1: 1.0",
                "  beta2: 0.1",
                "  eta: 0.01",
                "  b: 1.0",
                "tuned:",
                "  gta_base:",
                "    alpha: 0.91",
                "    beta: 0.33",
                "    eta: 0.22",
                "    b: 2.2",
                "  gta_beta:",
                "    alpha: 0.73",
                "    beta1: 0.55",
                "    beta2: 0.77",
                "    eta: 0.44",
                "    b: 3.3",
            ]
        )
    )

    garnet_method_cfgs, fourrooms_method_cfgs = module._resolve_env_method_cfgs(str(cfg_path))

    assert garnet_method_cfgs == fourrooms_method_cfgs
    assert garnet_method_cfgs["forward_gta"]["alpha"] == pytest.approx(0.0)
    assert garnet_method_cfgs["full_gta"]["alpha"] == pytest.approx(0.73)
