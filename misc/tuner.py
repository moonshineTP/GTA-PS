"""
GTA-PS standalone Optuna tuner (misc/tuner.py).

GtaPSTuner
    Single-env Optuna study. Uses GtaPSSearchSpace to sample hyperparameters
    and measures GTA-PS error against the Standard PS baseline on the DP host.

For parallel grid-search tuning (used by rq1_proactiveness.py --tune),
see code/threads/tuner.py (TuningEngine + RunManager).
"""
import sys
import json
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Dynamic imports
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "threads"))

SRC = ROOT / "src" / "gta_ps"


def _direct_import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# gta_ps is a proper package on sys.path — use normal imports to avoid
# circular-import issues from _direct_import + relative imports.
from gta_ps.envs.garnet import GarnetEnv                          # noqa: E402
from gta_ps.envs.fourrooms import FourRoomsTabular                # noqa: E402
from gta_ps.algorithms import make_agent                                      # noqa: E402
from gta_ps.utils.hyperparams import (                            # noqa: E402
    GtaPSConfig, GtaPSSearchSpace, SPACES, compute_reward_scale,
)

# threads/runner.py is outside the gta_ps package — use _direct_import.
RunManager = _direct_import(
    "runner", ROOT / "threads" / "runner.py"
).RunManager


# ---------------------------------------------------------------------------
# YAML config helper
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    """Load a YAML experiment config. Requires pyyaml."""
    try:
        import yaml
    except ImportError:
        raise SystemExit("PyYAML required for --config: pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Config write-back helpers
# ---------------------------------------------------------------------------

# Maps tuner env_name → list of config filenames to update.
# garnet tunes on n_states=100 → garnet_s100.yaml is the primary target.
# fourrooms tunes on k=1 → fourrooms_k1.yaml is the primary target.
# Additional configs for the same env (s500, k4) inherit the same tuned params.
_ENV_CONFIG_MAP: Dict[str, List[str]] = {
    "garnet":    ["garnet_s100.yaml", "garnet_s500.yaml"],
    "fourrooms": ["fourrooms_k1.yaml", "fourrooms_k4.yaml"],
    "reacher":   ["reacher.yaml"],
    "cargoal":   ["cargoal.yaml"],
}


def _update_config_tuned(
    configs_dir: Path,
    env_name: str,
    variant: str,
    best_params: Dict,
) -> None:
    """Write best_params into the tuned.<variant> block of every matching config.

    Uses PyYAML for round-trip load/dump. YAML comments are not preserved
    (acceptable for machine-written sections).
    """
    try:
        import yaml
    except ImportError:
        print("  [warn] PyYAML not found — skipping config write-back. "
              "Install with: pip install pyyaml")
        return

    for filename in _ENV_CONFIG_MAP.get(env_name, []):
        cfg_path = configs_dir / filename
        if not cfg_path.exists():
            print(f"  [warn] config not found, skipping: {cfg_path}")
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        if "tuned" not in cfg or cfg["tuned"] is None:
            cfg["tuned"] = {}
        cfg["tuned"][variant] = best_params
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"  → wrote tuned.{variant} to {cfg_path.name}")


# ---------------------------------------------------------------------------
# Environment / agent helpers
# ---------------------------------------------------------------------------

def _env_kwargs_from_env(env) -> tuple:
    """Extract (env_name, flat_kwargs) from a built env instance.

    The returned flat_kwargs are the constructor keyword arguments needed to
    recreate the same env for a different seed, excluding `seed` itself.
    These kwargs are picklable and can be spread into worker fn_kwargs dicts.
    """
    if isinstance(env, GarnetEnv):
        return "garnet", dict(
            n_states=env.n_states,
            n_actions=env.n_actions,
            branching=env.branching,
            n_nbr_rate=round(env.n_nbr / env.n_states, 6),
            reward_support=env.reward_support,
        )
    if isinstance(env, FourRoomsTabular):
        return "fourrooms", dict(k=env.k)
    raise TypeError(f"Unsupported env type: {type(env).__name__!r}. "
                    f"Pass a GarnetEnv or FourRoomsTabular instance.")


def _make_env(
    env_name: str,
    gamma: float,
    seed: int,
    # garnet kwargs
    n_states: int = 100,
    n_actions: int = 5,
    branching: int = 3,
    n_nbr_rate: float = 0.15,
    reward_support: int = 4,
    # fourrooms kwargs
    k: int = 1,
):
    if env_name == "garnet":
        return GarnetEnv(
            n_states=n_states, n_actions=n_actions, branching=branching,
            n_nbr_rate=n_nbr_rate, reward_support=reward_support, gamma=gamma, seed=seed,
        )
    if env_name == "fourrooms":
        return FourRoomsTabular(k=k, gamma=gamma, seed=seed)
    raise ValueError(f"Unknown env: {env_name!r}")


def _eval_agent(agent, env, n_shifts: int, K: int, n_steps_per_shift: int = 100) -> float:
    # K = backups per host step; n_steps_per_shift = host steps between reward shifts.
    # Total backup budget per shift = K × n_steps_per_shift, matching the Dyna protocol.
    # DP host has oracle P/r so there are no model updates between host steps;
    # plan(K * n_steps_per_shift) is equivalent to calling plan(K) n_steps_per_shift times.
    budget = K * n_steps_per_shift
    errs = []
    for _ in range(n_shifts):
        r = env.shift_reward()
        V_star = env.value_iteration(r)
        V = agent.plan(K=budget, r=r)
        errs.append(float(np.max(np.abs(V - V_star))))
    return float(np.mean(errs))


# ---------------------------------------------------------------------------
# Top-level worker functions (must be module-level for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _run_baseline_seed(
    seed: int, env_name: str, gamma: float, n_shifts: int, K: int,
    n_steps_per_shift: int = 100,
    # env construction kwargs (forwarded from _env_kwargs_from_env)
    n_states: int = 100, n_actions: int = 5, branching: int = 3,
    n_nbr_rate: float = 0.15, reward_support: int = 4, k: int = 1,
) -> float:
    env = _make_env(env_name, gamma, seed=seed, n_states=n_states,
                    n_actions=n_actions, branching=branching, n_nbr_rate=n_nbr_rate,
                    reward_support=reward_support, k=k)
    agent = make_agent("standard_ps", P=env.P, r=env.r, gamma=gamma, q=20)
    return _eval_agent(agent, env, n_shifts, K, n_steps_per_shift)


def _run_trial_seed(
    seed: int, env_name: str, gamma: float, n_shifts: int, K: int,
    n_steps_per_shift: int = 100,
    # env construction kwargs (forwarded from _env_kwargs_from_env)
    n_states: int = 100, n_actions: int = 5, branching: int = 3,
    n_nbr_rate: float = 0.15, reward_support: int = 4, k: int = 1,
    variant: str = "gta_base",
    slem_init: float = 0.0,
    **cfg_kwargs,
) -> float:
    env = _make_env(env_name, gamma, seed=seed, n_states=n_states,
                    n_actions=n_actions, branching=branching, n_nbr_rate=n_nbr_rate,
                    reward_support=reward_support, k=k)

    # Bug 3 fix: normalise b using the FIRST SHIFTED reward, not the construction reward.
    if "b" in cfg_kwargs:
        r_first_shift = env.shift_reward()
        scale = compute_reward_scale(r_first_shift)
        cfg_kwargs = {**cfg_kwargs, "b": cfg_kwargs["b"] / scale}
        env = _make_env(env_name, gamma, seed=seed, n_states=n_states,
                        n_actions=n_actions, branching=branching, n_nbr_rate=n_nbr_rate,
                        reward_support=reward_support, k=k)

    agent = make_agent(variant, P=env.P, r=env.r, gamma=gamma, q=20, **cfg_kwargs)
    return _eval_agent(agent, env, n_shifts, K, n_steps_per_shift)


class GtaPSTuner:
    """Tune one GTA-PS variant on one environment via Optuna.

    Parameters
    ----------
    env       : pre-built env instance (seed=0).  GarnetEnv or FourRoomsTabular.
                All env parameters (n_states, n_actions, branching, n_nbr, k, …)
                are inferred from this object via _env_kwargs_from_env.
                Worker processes recreate the env per seed using those kwargs.
    space     : GtaPSSearchSpace describing which params to tune
    n_seeds   : random seeds per Optuna trial
    n_shifts  : reward shifts per seed
    K         : Bellman backup budget per host step
    n_workers : worker processes for seed-level parallelism (None → auto)
    variant   : GTA-PS variant key (e.g. "gta_base") — determines which search space to use
    """

    def __init__(
        self,
        env,
        space: GtaPSSearchSpace,
        n_seeds: int = 10,
        n_shifts: int = 10,
        K: int = 50,
        n_steps_per_shift: int = 100,
        n_workers: int | None = None,
        variant: str = "gta_base",
        seed_offset: int = 1000,
    ):
        self.env_name, self._env_kwargs = _env_kwargs_from_env(env)
        self.gamma  = env.gamma
        self.space  = space
        self.n_seeds  = n_seeds
        self.n_shifts = n_shifts
        self.K        = K
        self.n_steps_per_shift = n_steps_per_shift
        self.variant  = variant
        self.seed_offset = seed_offset
        self._runner  = RunManager(n_workers=n_workers)

        # SLEM from the provided env (already built at seed=0).
        from gta_ps.utils.spectral import slem_arnoldi
        _P_uni = np.einsum(
            'sa,ast->st',
            np.ones((env.n_states, env.n_actions)) / env.n_actions,
            env.P,
        )
        self._slem_init, _ = slem_arnoldi(_P_uni, k=min(15, env.n_states - 2))

        self.baseline_error = self._compute_baseline()

    def _compute_baseline(self) -> float:
        seeds = list(range(self.seed_offset, self.seed_offset + self.n_seeds))
        results = self._runner.run(
            _run_baseline_seed,
            seeds=seeds,
            fn_kwargs={
                "env_name": self.env_name,
                "gamma": self.gamma,
                "n_shifts": self.n_shifts,
                "K": self.K,
                "n_steps_per_shift": self.n_steps_per_shift,
                **self._env_kwargs,
            },
            desc="baseline",
        )
        return float(np.mean(results))

    def objective(self, trial: optuna.Trial) -> float:
        cfg = self.space.suggest(trial)
        fixed = {
            "env_name": self.env_name,
            "gamma": self.gamma,
            "n_shifts": self.n_shifts,
            "K": self.K,
            "n_steps_per_shift": self.n_steps_per_shift,
            "variant": self.variant,
            "slem_init": self._slem_init,
            **self._env_kwargs,
            **cfg.to_kwargs(),
        }
        seeds = list(range(self.seed_offset, self.seed_offset + self.n_seeds))
        jobs = [{"seed": s, **fixed} for s in seeds]
        # verbose=False: Optuna callback handles outer progress
        results = self._runner.run_jobs(
            _run_trial_seed, jobs,
            desc=f"trial-{trial.number}", verbose=False,
        )
        mean_err = float(np.mean(results))
        trial.set_user_attr("mean_inf_norm_error", mean_err)
        trial.set_user_attr("baseline_error", self.baseline_error)
        return mean_err

    @staticmethod
    def _progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Print one compact line per completed trial."""
        best = study.best_value
        print(
            f"  trial {trial.number:3d}  "
            f"err={trial.value:.4f}  "
            f"best={best:.4f}  "
            f"params={trial.params}"
        )

    def run(self, n_trials: int = 20) -> Dict:
        study = optuna.create_study(direction="minimize")
        study.optimize(
            self.objective,
            n_trials=n_trials,
            show_progress_bar=False,
            callbacks=[self._progress_callback],
        )
        best = study.best_trial
        best_params = {k: round(v, 6) for k, v in best.params.items()}
        best_err = round(best.value, 6)
        baseline = round(self.baseline_error, 6)
        improvement = round((baseline - best_err) / baseline * 100, 2)
        return {
            "env": self.env_name,
            "tuned_params": self.space.param_names,
            "baseline_error": baseline,
            "best_error": best_err,
            "improvement_pct": improvement,
            "best_params": best_params,
        }


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="GTA-PS Optuna hyperparameter tuner")
    parser.add_argument(
        "--envs", nargs="+", default=["fourrooms", "garnet"],
    )
    parser.add_argument(
        "--variants", nargs="+", default=list(SPACES.keys()),
        help="GTA-PS variant keys (default: all in SPACES)",
    )
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--seed-offset", type=int, default=10000)
    parser.add_argument("--n-shifts", type=int, default=8)
    parser.add_argument("--K", type=int, default=None,
                        help="Bellman backup budget per host step (default: budgets[0] from config, else 20)")
    parser.add_argument("--n-steps-per-shift", type=int, default=None,
                        help="Host agent steps between reward shifts (default: n_steps_per_shift from config, else 100)")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--n-workers", type=int, default=None,
                        help="parallel worker processes per trial (None → auto)")
    parser.add_argument("--out", default=str(ROOT / "results/misc/tuner/tune.json"))
    parser.add_argument(
        "--configs-dir", default=None,
        help="path to configs/ directory for write-back "
             "(default: configs/ relative to ROOT)",
    )
    parser.add_argument(
        "--config", default=None, metavar="YAML",
        help="Path to a YAML experiment config. Provides defaults for "
             "n_shifts, K (budgets[0]), gamma, n_actions, branching, reward_support.",
    )
    args = parser.parse_args()

    # ── load YAML config ─────────────────────────────────────────────────
    _cfg: dict = {}
    if args.config is not None:
        config_path = Path(args.config).resolve()
        _cfg = _load_config(str(config_path))
        # Apply YAML defaults where CLI left values at parser defaults
        _MAP = [("n_shifts", "n_shifts", 8), ("gamma", "gamma", 0.95),
                ("n_seeds",  "seeds",    30)]
        for dest, key, default in _MAP:
            if key in _cfg and getattr(args, dest) == default:
                setattr(args, dest, _cfg[key])

    # K: YAML budgets[0] → fallback 20
    if args.K is None:
        args.K = int(_cfg["budgets"][0]) if "budgets" in _cfg else 20

    # n_steps_per_shift: YAML → fallback 100 (matches rq2_allocation default)
    if args.n_steps_per_shift is None:
        args.n_steps_per_shift = int(_cfg.get("n_steps_per_shift", 100))

    _n_actions      = _cfg.get("n_actions",  10)
    _branching      = _cfg.get("branching",  3)
    _n_nbr          = _cfg.get("n_nbr_rate",  0.15)
    _reward_support = _cfg.get("reward_support", 4)
    _k_fourrooms    = _cfg.get("k_fourrooms",  1)

    configs_dir = (
        Path(args.configs_dir).resolve() if args.configs_dir
        else ROOT / "configs"
    )

    results: Dict = {}
    for env_name in args.envs:
        results[env_name] = {}
        # Build seed=0 env once per env_name; tuner extracts all params from it.
        _env0 = _make_env(
            env_name, args.gamma, seed=0,
            n_actions=_n_actions, branching=_branching, n_nbr_rate=_n_nbr,
            reward_support=_reward_support, k=_k_fourrooms,
        )
        for variant in args.variants:
            space = SPACES[variant]
            print(f"\n[{env_name}] {variant}  tuning: {space.param_names}")
            tuner = GtaPSTuner(
                env=_env0, space=space,
                n_seeds=args.n_seeds, n_shifts=args.n_shifts, K=args.K,
                n_steps_per_shift=args.n_steps_per_shift,
                n_workers=args.n_workers,
                variant=variant,
                seed_offset=args.seed_offset,
            )
            print(f" [DP + Standard PS] baseline: {tuner.baseline_error:.4f}"
                  f"  (K={args.K} × {args.n_steps_per_shift} steps/shift = {args.K * args.n_steps_per_shift} backups/shift)")
            result = tuner.run(n_trials=args.n_trials)
            results[env_name][variant] = result
            print(
                f"  Best {variant}: {result['best_error']:.4f}"
                f"  ({result['improvement_pct']:+.2f}% vs baseline)"
            )
            _update_config_tuned(
                configs_dir, env_name, variant, result["best_params"]
            )

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
