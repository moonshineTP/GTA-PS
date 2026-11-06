# Supporting studies and utilities

This directory holds the ablations, benchmarks, tuners, and plotting helpers that surround the research-question suite in `experiments/`. Each study isolates one property of GTA-PS, writes a JSON result under `results/`, and renders a figure under `plots/`. Every script lists its options through `--help`.

## Ablations

| Script | Property tested | Output |
| --- | --- | --- |
| `mixing_threshold_ablation.py` | Planning budget `K` against mixing time: the topology advantage shrinks as `K` grows past the mixing horizon | `results/misc/mixing_threshold_ablation/`, `plots/mixing_threshold_ablation.png` |
| `shift_rate_ablation.py` | Advantage across the planning steps allowed per reward shift | `results/misc/shift_rate_ablation/`, `plots/shift_rate_ablation.png` |
| `cumulative_ablation.py` | Cumulative tracking error of Standard-PS and GTA-PS on a slow-mixing GARNET with a Dyna host | `results/misc/cumulative_tracking/`, `plots/cumulative_ablation.png` |
| `scheduler_ablation.py` | Linear and log `beta_t` schedulers against Standard-PS | `results/misc/scheduler_ablation/` |
| `beta_dynamics.py` | Traces of SLEM, `beta_t`, and the topology signal ratio across backups and shifts | `results/misc/beta_dynamics/` |
| `gamma_ablation.py` | Discount factor against the convergence trajectory | `results/gamma_ablation_garnet.json`, `plots/gamma_ablation_garnet.png` |
| `periodicity_ablation.py` | Periodic (bipartite) chains where the spectral gap closes | `results/periodicity_ablation.json`, `plots/periodicity_ablation.png` |
| `reward_support_ablation.py` | Reward density on GARNET | `results/reward_support_ablation.json`, `plots/reward_support_ablation.png` |

## Equal-time and solver studies

These protocols back the additional experiments summarized in `docs/additional-experiments.md`.

| Script | Purpose |
| --- | --- |
| `wall_clock_multishift.py` | Equal-elapsed-time replanning on GARNET with time-triggered reward shifts |
| `tune_wall_clock_garnet200.py` | Stability pass for GTA-PS settings before the GARNET-200 wall-clock runs |
| `solver_threshold_ablation.py` | LU-Woodbury against GMRES crossover on sparse, localized GARNETs |
| `xi_tau_hop_distance.py` | Residual and topology signal AUC by hop distance from the shifted reward |
| `arnoldi_cadence_ablation.py` | Aggregates and plots the GARNET-200 runs that vary the Arnoldi refresh period |

## Benchmarks and tuning

| Script | Purpose | Output |
| --- | --- | --- |
| `convergence_benchmark.py` | Backups needed to reach an error tolerance after each reward shift | `results/misc/convergence_benchmark/` |
| `time_benchmark.py` | Wall-clock time per backup and topology precomputation overhead | `results/misc/time_benchmark/` |
| `tuner.py` | Optuna search for GTA-PS hyperparameters, consumed by `load_tuned_for_env` | `results/misc/tuner/` |

## FourRooms showcase

`showcase.py` compares Dyna with Standard-PS and Dyna with GTA-PS on FourRooms and has two modes.

- `--mode ablation` runs seeds in parallel and saves a multi-panel figure with per-episode error, cumulative tracking error, and value and topology heatmaps to `plots/fourrooms_ablation.png`.
- `--mode showcase` records one seed as an animation at `plots/fourrooms_showcase.gif`, with one frame per real host step. The option `--n-frames` keeps only the last frames of the run.

`showcase_support.py` provides the shared rendering code, and `extract_showcase_frames.py` cuts still frames from the saved GIF.

```bash
python misc/showcase.py --mode ablation --seeds 30 --n-shifts 8 --K 40 --workers auto
python misc/showcase.py --mode showcase --n-shifts 20 --n-frames 200 --K 40
```

## Shared conventions

`plot_academic.py` defines the matplotlib style (`apply_style()`), the variant palette (`PALETTE`), and the display labels (`LABELS`). The statistical scripts share these conventions.

1. Seeds run in parallel worker processes, through `RunManager` in `threads/runner.py` or a process pool.
2. Confidence intervals are 95% Student-t intervals over seeds.
3. BLAS thread counts are pinned to one per worker through `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS`.
4. Each script writes JSON to `results/` and a figure to `plots/`.
