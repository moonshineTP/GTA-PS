"""
Hyperparameter tuning engine for GTA-PS.

TuningEngine composes a RunManager — it owns grid construction and result
aggregation, while all parallel execution is delegated to runner.run_jobs().
No executor logic lives here.

Seed isolation: always pass tuning seeds *disjoint* from evaluation seeds
to avoid selection bias. parallel_rq1.py enforces this via --tune-seed-offset.
"""
from __future__ import annotations

import itertools
from typing import Any, Callable

import numpy as np

from runner import RunManager


class TuningEngine:
    """Parallel grid search over a hyperparameter grid.

    Composes RunManager for execution. Owns only:
      - Grid expansion (itertools.product)
      - Job construction (merge config params + fixed kwargs + seed)
      - Per-config score aggregation

    Parameters
    ----------
    runner : RunManager
        Shared executor instance. The same RunManager can be used for both
        tuning and evaluation — no resource duplication.
    """

    def __init__(self, runner: RunManager):
        self.runner = runner

    def search(
        self,
        fn: Callable[..., float],
        param_grid: dict[str, list[Any]],
        fixed_kwargs: dict,
        seeds: list[int],
        lower_is_better: bool = True,
        desc: str = "tuning",
    ) -> tuple[dict, list[dict]]:
        """Run parallel grid search via the shared RunManager.

        Each (config, seed) pair becomes one job dict passed to runner.run_jobs().
        The config's parameter values are merged directly into the job kwargs, so
        fn must accept all param_grid keys alongside fixed_kwargs and `seed`.

        Parameters
        ----------
        fn : callable
            Module-level function (picklable). Signature::

                fn(seed: int, **param_grid_keys, **fixed_kwargs) -> float

        param_grid : dict
            Mapping param_name → list of candidate values.
            Example: {"beta1": [0.5, 1.0, 2.0], "beta2": [0.05, 0.1, 0.2]}
        fixed_kwargs : dict
            Kwargs forwarded unchanged to every fn call (env config, gamma, …).
        seeds : list[int]
            Tuning seeds — must be *disjoint* from evaluation seeds.
        lower_is_better : bool
            Sort direction for returned records.
        desc : str
            tqdm label forwarded to run_jobs.

        Returns
        -------
        best_params : dict
            Config with the best mean score across tuning seeds.
        records : list[dict]
            All configs sorted best-first; each has keys
            ``{"params", "mean_score", "seed_scores"}``.
        """
        configs: list[dict] = [
            dict(zip(param_grid.keys(), combo))
            for combo in itertools.product(*param_grid.values())
        ]

        # Build flat job list; track which config each job belongs to.
        # Priority (right wins in Python dict merge): fixed_kwargs < cfg < seed.
        # cfg must override fixed_kwargs so grid params are never silently shadowed.
        jobs: list[dict] = []
        job_config_idx: list[int] = []
        for ci, cfg in enumerate(configs):
            for s in seeds:
                jobs.append({**fixed_kwargs, **cfg, "seed": s})
                job_config_idx.append(ci)

        raw_scores = self.runner.run_jobs(fn, jobs, desc=desc)

        # Aggregate per config.
        raw: dict[int, list[float]] = {ci: [] for ci in range(len(configs))}
        for score, ci in zip(raw_scores, job_config_idx):
            raw[ci].append(float(score))

        records: list[dict] = [
            {
                "params": cfg,
                "mean_score": float(np.mean(raw[ci])),
                "seed_scores": raw[ci],
            }
            for ci, cfg in enumerate(configs)
        ]
        records.sort(key=lambda x: x["mean_score"], reverse=not lower_is_better)
        return records[0]["params"], records
