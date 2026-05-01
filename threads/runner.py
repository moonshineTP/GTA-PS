"""
Unified parallel executor for GTA-PS experiments.

RunManager is the single parallel execution primitive. All callers —
seed-level evaluation loops and the hyperparameter tuning engine alike —
dispatch work through run_jobs(). run() is a convenience wrapper for the
common case where the only varying argument per job is `seed`.

BLAS oversubscription: set OMP/MKL/OPENBLAS_NUM_THREADS=1 in the parent
process *before* importing numpy (see parallel_rq1.py).
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable

from tqdm import tqdm


class RunManager:
    """Unified parallel executor built on ProcessPoolExecutor.

    Core method: run_jobs() — accepts an arbitrary list of per-job kwarg dicts.
    Convenience method: run() — for the common case of varying only `seed`.

    Parameters
    ----------
    n_workers : int or None
        Worker count. Defaults to min(cpu_count, 5).
    """

    def __init__(self, n_workers: int | str | None = None):
        if n_workers == "auto":
            # auto mode: use most cores, but leave one for OS/coordination 
            # to reduce deadlock/starvation.
            self.n_workers = max((os.cpu_count() or 1) - 1, 1)
        elif isinstance(n_workers, str) and n_workers.isdigit():
            self.n_workers = int(n_workers)
        else:
            self.n_workers = n_workers or min(os.cpu_count() or 1, 5)

    def run_jobs(
        self,
        fn: Callable[..., Any],
        jobs: list[dict],
        desc: str = "jobs",
        verbose: bool = True,
    ) -> list[Any]:
        """Execute fn(**job) for each job in parallel and return results in job order.

        Parameters
        ----------
        fn : callable
            Must be defined at module top-level (pickling requirement under spawn).
            Called as fn(**jobs[i]) for each i.
        jobs : list[dict]
            Each dict is the complete set of kwargs for one fn call.
        desc : str
            tqdm progress bar label.
        verbose : bool
            If False, suppresses the tqdm progress bar (useful when an outer
            loop — e.g. Optuna — manages its own progress display).

        Returns
        -------
        list[Any]
            Results in the same order as `jobs`.
        """
        results: list[Any] = [None] * len(jobs)
        with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
            futures = {pool.submit(fn, **job): i for i, job in enumerate(jobs)}
            with tqdm(total=len(jobs), desc=desc, unit="job",
                      dynamic_ncols=True, ascii=True,
                      disable=not verbose) as pbar:
                for fut in as_completed(futures):
                    results[futures[fut]] = fut.result()
                    pbar.update(1)
        return results

    def run(
        self,
        fn: Callable[..., Any],
        seeds: list[int],
        fn_kwargs: dict,
        desc: str = "seeds",
    ) -> list[Any]:
        """Convenience wrapper: vary only `seed` across jobs.

        Equivalent to run_jobs with jobs = [{**fn_kwargs, "seed": s} for s in seeds].
        Results are returned in seed order.
        """
        # seed is rightmost so it always overrides any stale "seed" in fn_kwargs.
        jobs = [{**fn_kwargs, "seed": s} for s in seeds]
        return self.run_jobs(fn, jobs, desc=desc)
