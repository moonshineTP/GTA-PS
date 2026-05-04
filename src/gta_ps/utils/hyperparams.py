"""
Hyperparameter configuration and Optuna search-space adapter for GTA-PS variants.

Usage
-----
from gta_ps.utils.hyperparams import GtaPSConfig, GtaPSSearchSpace, SPACES

space = SPACES["gta_beta"]
cfg   = space.suggest(trial)          # inside an Optuna objective
agent = GtaPS(env.P, env.r, **cfg.to_kwargs())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import optuna


# ---------------------------------------------------------------------------
# Reward-scale normalizer for the softmax temperature b
# ---------------------------------------------------------------------------

def compute_reward_scale(reward: np.ndarray, eps: float = 1e-8) -> float:
    """Return max(|r|) clamped to eps, for normalizing the softmax temperature b.

    The effective softmax argument is b * Q ≈ b * max|r| / (1 - γ).  Dividing
    the tuned b by compute_reward_scale(r) before passing it to GtaPS keeps
    that product in a numerically stable range regardless of reward magnitude.

    Parameters
    ----------
    reward : np.ndarray
        Current reward vector (or matrix) for the environment.
    eps : float
        Floor value to avoid division-by-zero on zero-reward MDPs.
    """
    return float(max(float(np.max(np.abs(reward))), eps))


# ---------------------------------------------------------------------------
# Config — flat hyperparameter record
# ---------------------------------------------------------------------------

@dataclass
class GtaPSConfig:
    """Complete, flat hyperparameter set for any GTA-PS variant.

    Passed to GtaPS via .to_kwargs(). Runtime-fixed args (P, r, gamma,
    arnoldi_k, slem_recompute_every) are caller-supplied and not stored here.
    """
    # topology weight schedule
    beta: float = 1.0           # fixed β_t when adaptive_beta=False
    beta1: float = 1.0          # GTA-PS^β amplitude
    beta2: float = 0.1          # GTA-PS^β offset threshold (linear schedule)
    # forward / backward mix
    alpha: float = 0.5          # fixed α_t
    # resolvent regularizer
    eta: Optional[float] = None  # None → (1-γ)/γ at construction time
    # topology-policy temperature (paper §3, Lemma 2) — must be finite
    b: float = 1.0              # softmax(b·Q) keeps P_t ergodic (ε-soft)
    # variant flags
    adaptive_beta: bool = False

    def to_kwargs(self) -> Dict[str, Any]:
        """Return GtaPS constructor kwargs.

        eta=None is omitted so the GtaPS default ((1-γ)/γ) applies.
        """
        d: Dict[str, Any] = {
            "beta": self.beta,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "alpha": self.alpha,
            "b": self.b,
            "adaptive_beta": self.adaptive_beta,
        }
        if self.eta is not None:
            d["eta"] = self.eta
        return d


# ---------------------------------------------------------------------------
# Search space — Optuna adapter
# ---------------------------------------------------------------------------

@dataclass
class GtaPSSearchSpace:
    """Declare which GTA-PS hyperparameters Optuna should tune and their bounds.

    Call .suggest(trial) inside an Optuna objective to draw a GtaPSConfig.
    Fixed overrides (values for non-tuned parameters) go in `fixed`.

    Example
    -------
    space = GtaPSSearchSpace(
        tune_beta1=True, tune_beta2=True, tune_b=True,
        fixed={"adaptive_beta": True},
    )
    cfg = space.suggest(trial)
    agent = GtaPS(env.P, env.r, gamma=gamma, **cfg.to_kwargs())
    """
    # --- which parameters to tune ---
    tune_beta: bool = False         # fixed β_t scalar (adaptive_beta=False variants)
    tune_beta1: bool = False
    tune_beta2: bool = False
    tune_alpha: bool = False
    tune_eta: bool = False
    tune_b: bool = False

    # --- search bounds (log=True applied where sensible) ---
    # beta is the fixed β_t scalar; allows negative (dampen) and positive (amplify).
    beta_bounds: tuple = (0.04, 0.4)   # linear scale — log not valid (can be negative)
    beta1_bounds: tuple = (0.04, 0.4)
    # beta2 is the additive offset in the unified linear scheduler:
    #   β_t = β₁ · ((1−SLEM) + β₂)
    # Negative values create a dead-zone (dampen); positive values add a floor.
    # Range (0.10, 0.5) is environment-agnostic: covers both slow-mixing
    # chains (FourRooms SLEM≈0.98 ⇒ 1−SLEM≈0.02) and fast-mixing ones (GARNET).
    beta2_bounds: tuple = (0.04, 0.4)   
    alpha_bounds: tuple = (0.35, 0.65)
    eta_bounds: tuple = (5e-4, 2e-2)
    b_bounds: tuple = (2.8, 4.8)  # interpretable as b̃ = b·max|r| ∈ [1, 20]

    # --- fixed overrides applied after Optuna suggestions ---
    fixed: Dict[str, Any] = field(default_factory=dict)

    def suggest(self, trial: optuna.Trial) -> GtaPSConfig:
        """Draw a GtaPSConfig from an Optuna trial."""
        kwargs: Dict[str, Any] = {}
        if self.tune_beta:
            kwargs["beta"] = trial.suggest_float("beta", *self.beta_bounds)
        if self.tune_beta1:
            kwargs["beta1"] = trial.suggest_float("beta1", *self.beta1_bounds, log=True)
        if self.tune_beta2:
            kwargs["beta2"] = trial.suggest_float("beta2", *self.beta2_bounds)
        if self.tune_alpha:
            kwargs["alpha"] = trial.suggest_float("alpha", *self.alpha_bounds)
        if self.tune_eta:
            kwargs["eta"] = trial.suggest_float("eta", *self.eta_bounds, log=True)
        if self.tune_b:
            kwargs["b"] = trial.suggest_float("b", *self.b_bounds, log=True)
        # fixed overrides win over Optuna suggestions
        kwargs.update(self.fixed)
        return GtaPSConfig(**kwargs)

    @property
    def param_names(self) -> list[str]:
        """Names of all tuned parameters (for display / logging)."""
        names = []
        for flag, name in [
            (self.tune_beta,      "beta"),
            (self.tune_beta1,     "beta1"),
            (self.tune_beta2,     "beta2"),
            (self.tune_alpha,     "alpha"),
            (self.tune_eta,       "eta"),
            (self.tune_b,         "b"),
        ]:
            if flag:
                names.append(name)
        return names


# ---------------------------------------------------------------------------
# Pre-defined spaces — one per GTA-PS variant
# ---------------------------------------------------------------------------

SPACES: Dict[str, GtaPSSearchSpace] = {
    "gta_base": GtaPSSearchSpace(
        tune_alpha=True,
        tune_beta=True,       # fixed β_t — was frozen at 1.0
        tune_eta=True,
        tune_b=True,
        fixed={"adaptive_beta": False},
    ),
    "gta_beta": GtaPSSearchSpace(
        tune_alpha=True,      # fixed α_t — was frozen at 0.5
        tune_beta1=True,
        tune_beta2=True,
        tune_eta=True,
        tune_b=True,
        fixed={"adaptive_beta": True},
    ),
}


# ---------------------------------------------------------------------------
# Environment → default config file  (canonical tuned-parameter source)
# ---------------------------------------------------------------------------

# Maps env_name (as used in script --envs args) to the YAML filename in configs/.
# "garnet" tunes on n_states=75; "garnet_s200" inherits the same params.
ENV_DEFAULT_CONFIG: Dict[str, str] = {
    "garnet":    "garnet_s75.yaml",
    "fourrooms": "fourrooms_k1.yaml",
    "reacher":   "reacher.yaml",
    "cargoal":   "cargoal.yaml",
}


def load_tuned_for_env(
    env_name: str,
    configs_dir: Any = None,       # Path | str | None
) -> Dict[str, Dict]:
    """Return {variant: kwargs_dict} from the tuned block of the default YAML config.

    Resolves configs_dir automatically when None (i.e. <package_root>/configs/).
    Returns an empty dict silently if the env has no entry in ENV_DEFAULT_CONFIG,
    the file does not exist, or PyYAML is not installed.
    """
    from pathlib import Path as _Path

    filename = ENV_DEFAULT_CONFIG.get(env_name)
    if filename is None:
        return {}

    if configs_dir is None:
        # src/gta_ps/utils/hyperparams.py → parents[3] == code/
        configs_dir = _Path(__file__).resolve().parents[3] / "configs"

    cfg_path = _Path(configs_dir) / filename
    if not cfg_path.exists():
        return {}

    try:
        import yaml
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("tuned") or {}
    except Exception:
        return {}
