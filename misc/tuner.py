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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Dynamic imports
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "configs").exists():
    # Try reaching up to find the real project root (e.g. if compiled in artifacts/dist/)
    for p in Path(__file__).resolve().parents:
        if (p / "configs").exists() and (p / "src").exists():
            ROOT = p
            break
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
    ENV_DEFAULT_CONFIG, GtaPSSearchSpace, SPACES, compute_reward_scale,
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


def _resolve_config_for_env(
    env_name: str,
    configs_dir: Path,
    explicit_config: str | None,
) -> tuple[dict, Path | None]:
    if explicit_config is not None:
        cfg_path = Path(explicit_config).resolve()
        return _load_config(str(cfg_path)), cfg_path

    filename = ENV_DEFAULT_CONFIG.get(env_name)
    if filename is None:
        return {}, None

    cfg_path = configs_dir / filename
    if not cfg_path.exists():
        return {}, None
    return _load_config(str(cfg_path)), cfg_path


# ---------------------------------------------------------------------------
# Config write-back helpers
# ---------------------------------------------------------------------------

# Maps tuner env_name → list of config filenames to update.
# garnet tunes on n_states=75 → garnet_s75.yaml is the primary target.
# fourrooms tunes on k=1 → fourrooms_k1.yaml is the primary target.
# Additional configs for the same env (s200, k4) inherit the same tuned params.
_ENV_CONFIG_MAP: Dict[str, List[str]] = {
    "garnet":    ["garnet_s75.yaml", "garnet_s200.yaml"],
    "fourrooms": ["fourrooms_k1.yaml", "fourrooms_k4.yaml", "fourrooms_corridor.yaml", "fourrooms_k2_timeline.yaml"],
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

@dataclass(frozen=True)
class TunerRunConfig:
    env_name: str
    gamma: float
    seeds: int
    seed_offset: int
    n_shifts: int
    K: int
    n_steps_per_shift: int
    workers: int | str | None
    config_path: str | None
    env_kwargs: Dict[str, Any]

    @property
    def backups_per_shift(self) -> int:
        return self.K * self.n_steps_per_shift

    def seed_values(self) -> List[int]:
        return list(range(self.seed_offset, self.seed_offset + self.seeds))


@dataclass(frozen=True)
class AgentRunConfig:
    variant: str
    module_kwargs: Dict[str, Any]
    normalize_b_from_first_shift: bool = False


def _resolve_arg(value, cfg: dict, key: str, fallback):
    return value if value is not None else cfg.get(key, fallback)


def _resolve_env_kwargs(env_name: str, cfg: dict) -> Dict[str, Any]:
    if env_name == "garnet":
        return {
            "n_states": int(cfg.get("n_states", 75)),
            "n_actions": int(cfg.get("n_actions", 4)),
            "branching": int(cfg.get("branching", 3)),
            "n_nbr_rate": float(cfg.get("n_nbr_rate", 0.15)),
            "reward_support": int(cfg.get("reward_support", 4)),
        }
    if env_name == "fourrooms":
        return {"k": int(cfg.get("k_fourrooms", cfg.get("k", 1)))}
    raise ValueError(f"Unknown env: {env_name!r}")


def _build_run_config(args, cfg: dict, env_name: str, config_path: Path | None) -> TunerRunConfig:
    K = args.K
    if K is None:
        budgets = cfg.get("budgets")
        K = int(budgets[0]) if budgets else 20
    return TunerRunConfig(
        env_name=env_name,
        gamma=float(_resolve_arg(args.gamma, cfg, "gamma", 0.95)),
        seeds=int(args.seeds),
        seed_offset=int(args.seed_offset),
        n_shifts=int(_resolve_arg(args.n_shifts, cfg, "n_shifts", 6)),
        K=int(K),
        n_steps_per_shift=int(_resolve_arg(args.n_steps_per_shift, cfg, "n_steps_per_shift", 50)),
        workers=args.workers,
        config_path=str(config_path) if config_path is not None else None,
        env_kwargs=_resolve_env_kwargs(env_name, cfg),
    )


def _make_env(run_cfg: TunerRunConfig, seed: int):
    if run_cfg.env_name == "garnet":
        return GarnetEnv(
            gamma=run_cfg.gamma,
            seed=seed,
            **run_cfg.env_kwargs,
        )
    if run_cfg.env_name == "fourrooms":
        return FourRoomsTabular(
            k=run_cfg.env_kwargs["k"],
            gamma=run_cfg.gamma,
            seed=seed,
        )
    raise ValueError(f"Unknown env: {run_cfg.env_name!r}")


def _eval_agent(agent, env, run_cfg: TunerRunConfig) -> float:
    # K = backups per host step; n_steps_per_shift = host steps between reward shifts.
    # Total backup budget per shift = K × n_steps_per_shift, matching the Dyna protocol.
    # DP host has oracle P/r so there are no model updates between host steps;
    # plan(K * n_steps_per_shift) is equivalent to calling plan(K) n_steps_per_shift times.
    budget = run_cfg.backups_per_shift
    errs = []
    for _ in range(run_cfg.n_shifts):
        r = env.shift_reward()
        V_star = env.value_iteration(r)
        V = agent.plan(K=budget, r=r)
        errs.append(float(np.max(np.abs(V - V_star))))
    return float(np.mean(errs))


# ---------------------------------------------------------------------------
# Top-level worker function (must be module-level for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _run_seed(seed: int, run_cfg: TunerRunConfig, agent_cfg: AgentRunConfig) -> float:
    env = _make_env(run_cfg, seed=seed)
    module_kwargs = dict(agent_cfg.module_kwargs)

    # Normalise b using the first shifted reward, not the construction reward.
    if agent_cfg.normalize_b_from_first_shift:
        r_first_shift = env.shift_reward()
        scale = compute_reward_scale(r_first_shift)
        module_kwargs["b"] = module_kwargs["b"] / scale
        env = _make_env(run_cfg, seed=seed)

    agent = make_agent(
        agent_cfg.variant,
        P=env.P,
        r=env.r,
        gamma=run_cfg.gamma,
        q=20,
        **module_kwargs,
    )
    return _eval_agent(agent, env, run_cfg)


class GtaPSTuner:
    """Tune one GTA-PS variant on one environment via Optuna.

    Parameters
    ----------
    env       : pre-built env instance (seed=0).  GarnetEnv or FourRoomsTabular.
                All env parameters (n_states, n_actions, branching, n_nbr, k, …)
                are inferred from this object via _env_kwargs_from_env.
                Worker processes recreate the env per seed using those kwargs.
    space     : GtaPSSearchSpace describing which params to tune
    seeds   : random seeds per Optuna trial
    n_shifts  : reward shifts per seed
    K         : Bellman backup budget per host step
    workers : worker processes for seed-level parallelism (None → auto)
    variant   : GTA-PS variant key (e.g. "gta_base") — determines which search space to use
    """

    def __init__(
        self,
        run_cfg: TunerRunConfig,
        space: GtaPSSearchSpace,
        variant: str = "gta_base",
    ):
        self.run_cfg = run_cfg
        self.space  = space
        self.variant  = variant
        self._runner  = RunManager(n_workers=run_cfg.workers)
        self.baseline_error = self._compute_baseline()

    def _compute_baseline(self) -> float:
        jobs = [
            {
                "seed": seed,
                "run_cfg": self.run_cfg,
                "agent_cfg": AgentRunConfig(variant="standard_ps", module_kwargs={}),
            }
            for seed in self.run_cfg.seed_values()
        ]
        results = self._runner.run_jobs(
            _run_seed,
            jobs=jobs,
            desc="baseline",
        )
        return float(np.mean(results))

    def objective(self, trial: optuna.Trial) -> float:
        cfg = self.space.suggest(trial)
        module_kwargs = cfg.to_kwargs()
        agent_cfg = AgentRunConfig(
            variant=self.variant,
            module_kwargs=module_kwargs,
            normalize_b_from_first_shift="b" in module_kwargs,
        )
        jobs = [
            {
                "seed": seed,
                "run_cfg": self.run_cfg,
                "agent_cfg": agent_cfg,
            }
            for seed in self.run_cfg.seed_values()
        ]
        # verbose=False: Optuna callback handles outer progress
        results = self._runner.run_jobs(
            _run_seed, jobs,
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
            "env": self.run_cfg.env_name,
            "tuned_params": self.space.param_names,
            "baseline_error": baseline,
            "best_error": best_err,
            "improvement_pct": improvement,
            "best_params": best_params,
        }


def _print_run_stats(run_cfg: TunerRunConfig, variant: str, n_trials: int, tuned_params: List[str]) -> None:
    seed_last = run_cfg.seed_offset + run_cfg.seeds - 1
    print("  run stats:")
    print(f"    config: {run_cfg.config_path or '<none>'}")
    print(f"    variant: {variant}")
    print(f"    trials: {n_trials}")
    print(f"    seeds: {run_cfg.seed_offset}..{seed_last} ({run_cfg.seeds} total)")
    print(f"    K: {run_cfg.K}")
    print(f"    n_steps_per_shift: {run_cfg.n_steps_per_shift}")
    print(f"    n_shifts: {run_cfg.n_shifts}")
    print(f"    backups_per_shift: {run_cfg.backups_per_shift}")
    print(f"    gamma: {run_cfg.gamma}")
    print(f"    workers: {run_cfg.workers if run_cfg.workers is not None else 'auto'}")
    print(f"    env_kwargs: {run_cfg.env_kwargs}")
    print(f"    tuned_params: {tuned_params}")


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
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--seed-offset", type=int, default=100000)
    parser.add_argument("--n-shifts", type=int, default=4, help="reward shifts per seed (YAML n_shifts overrides)")
    parser.add_argument("--K", type=int, default=20, help="Bellman backup budget per host step (YAML budgets[0] overrides)")
    parser.add_argument("--n-steps-per-shift", type=int, default=50, help="steps per reward shift (YAML n_steps_per_shift overrides)")
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--workers", default=5)
    parser.add_argument("--out", default=str(ROOT / "results/misc/tuner/tune.json"))
    parser.add_argument("--configs-dir", default=None)
    parser.add_argument(
        "--config", default=None, metavar="YAML",
        help="Path to a YAML experiment config. Provides defaults for "
             "n_shifts, K (budgets[0]), gamma, n_actions, branching, reward_support.",
    )
    args = parser.parse_args()

    # ── load YAML config ─────────────────────────────────────────────────

    # K: YAML budgets[0] → fallback 20

    # n_steps_per_shift: YAML → fallback 50 (matches rq2_allocation default)

    configs_dir = (
        Path(args.configs_dir).resolve() if args.configs_dir
        else ROOT / "configs"
    )

    results: Dict = {}
    for env_name in args.envs:
        results[env_name] = {}
        env_cfg, env_cfg_path = _resolve_config_for_env(
            env_name=env_name,
            configs_dir=configs_dir,
            explicit_config=args.config,
        )
        run_cfg = _build_run_config(args, env_cfg, env_name, env_cfg_path)
        for variant in args.variants:
            space = SPACES[variant]
            print(f"\n[{env_name}] {variant}  tuning: {space.param_names}")
            _print_run_stats(run_cfg, variant, args.n_trials, space.param_names)
            tuner = GtaPSTuner(
                run_cfg=run_cfg,
                space=space,
                variant=variant,
            )
            print(f" [DP + Standard PS] baseline: {tuner.baseline_error:.4f}"
                  f"  (K={run_cfg.K} × {run_cfg.n_steps_per_shift} steps/shift = {run_cfg.backups_per_shift} backups/shift)")
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
