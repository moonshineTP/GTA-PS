"""
Regression tests for RQ4/RQ5/RQ6 experiment method wiring.

These focus on:
- explicit mapping from legacy tuned keys to canonical forward/full GTA variants
- fail-fast behavior for ambiguous tuned blocks
- env-specific default config selection for no-config runs
"""
import importlib
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

rq4 = importlib.import_module("rq4_asymmetry")
rq5 = importlib.import_module("rq5_bottlenecks")
rq1_dyna = importlib.import_module("rq1_dyna_timeline")
from gta_ps.algorithms import make_agent
from gta_ps.algorithms.empirical_model import SampleModel

RQ_MODULES = [rq4, rq5, rq6]

EXPECTED_DEFAULT_METHOD_CFGS = {
    "garnet": {
        "forward_gta": {
            "alpha": 0.0,
            "beta1": 0.346922,
            "beta2": 0.269094,
            "eta": 0.015734,
            "b": 3.82617,
        },
        "full_gta": {
            "alpha": 0.455302,
            "beta1": 0.346922,
            "beta2": 0.269094,
            "eta": 0.015734,
            "b": 3.82617,
        },
    },
    "fourrooms": {
        "forward_gta": {
            "alpha": 0.0,
            "beta1": 1.281478,
            "beta2": 1.356273,
            "eta": 0.018831,
            "b": 2.801307,
        },
        "full_gta": {
            "alpha": 0.870918,
            "beta1": 1.281478,
            "beta2": 1.356273,
            "eta": 0.018831,
            "b": 2.801307,
        },
    },
}


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
def test_rq_default_method_cfgs_match_expected_yaml_values(module):
    garnet_method_cfgs, fourrooms_method_cfgs = module._resolve_env_method_cfgs(None)

    print(f"\n[{module.__name__}] resolved default method cfgs")
    print("  garnet:", garnet_method_cfgs)
    print("  fourrooms:", fourrooms_method_cfgs)

    for env_name, resolved in (
        ("garnet", garnet_method_cfgs),
        ("fourrooms", fourrooms_method_cfgs),
    ):
        expected = EXPECTED_DEFAULT_METHOD_CFGS[env_name]
        for variant, expected_values in expected.items():
            for key, value in expected_values.items():
                assert resolved[variant][key] == pytest.approx(value), (
                    f"{module.__name__} {env_name} {variant} {key} "
                    f"expected {value} but got {resolved[variant][key]}"
                )


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
