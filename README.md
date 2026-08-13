# GTA-PS

This repository contains the reference implementation and experiment suite for Graph Topology Augmentation for Prioritized Sweeping (GTA-PS). It includes tabular planning agents, environment adapters, research-question experiments, supporting ablations, and reproducibility utilities.

## Installation

Python 3.12 or newer is required. Create the project environment from the repository root.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e ".[dev,reacher]"
```

On Linux or macOS, replace `.venv\Scripts\python.exe` with `.venv/bin/python`.

The optional CarGoal environment uses an older Gymnasium stack and must remain in a separate environment. See the `cargoal` dependency group in `pyproject.toml` for its pinned packages.

## Reproduction entry points

The grouped runner executes scripts sequentially and stops at the first failure. A small verification pass is:

```powershell
.venv\Scripts\python.exe orchestrate_groups.py sanity --seeds 2 --workers 1
.venv\Scripts\python.exe orchestrate_groups.py rqs --seeds 2 --workers 1
.venv\Scripts\python.exe orchestrate_groups.py ablations --seeds 2 --workers 1
```

Windows launchers provide the same three groups with publication-scale defaults:

```cmd
run_sanity.cmd
run_rqs.cmd
run_ablations.cmd
```

Pass overrides directly to a launcher when a smaller or machine-specific run is needed. For example, `run_rqs.cmd --seeds 30 --workers 6`.

## Research-question system

The main grouped run covers RQ1-RQ5. Later studies have standalone entries because their protocols and output schemas differ.

| Study | Entry point | Purpose |
| --- | --- | --- |
| RQ1 | `experiments/rq1_*.py` | Convergence and adaptation after reward shifts |
| RQ2 | `experiments/rq2_beta_magnitude.py` | Effect of topology-confidence scaling |
| RQ3 | `experiments/rq3_alpha_tradeoff.py` | Accuracy and propagation tradeoff |
| RQ4 | `experiments/rq4_asymmetry.py` | Directed geometry and non-reversibility |
| RQ5 | `experiments/rq5_bottlenecks.py` | Computational scaling and bottlenecks |
| RQ6 | `experiments/rq6_curl_divergence.py` | Path-flow and cyclic geometry |
| RQ7 | `experiments/rq7_gps_batch_*.py` | Batch scheduling and abstraction comparisons |
| RQ8 | `experiments/rq8_eta_tuning.py` | Topology diffusion horizon |
| RQ9 | `experiments/rq9_time_ablation.py` | Equal-time planning behavior |

See `experiments/README.md` for the protocol map and common arguments.

## Miscellaneous runners

The `misc/` directory holds supporting studies and utilities. Use `run_sanity.cmd` for timing, convergence, and showcase checks. Use `run_ablations.cmd` for mixing, shift-rate, discount, periodicity, reward-support, and cumulative ablations. Other files in `misc/` are standalone diagnostics or artifact validators and expose their options through `--help`.

Additional experiment designs for diffusion-horizon tuning, solver crossover, and equal-time recovery are summarized in `docs/additional-experiments.md`.

## Outputs

Scripts write numerical artifacts under `results/` and figures under `plots/`. Existing result files are reference artifacts; use a new `--out-dir` when preserving them matters. Random seeds, worker counts, environment settings, and metric definitions are stored with the generated data where supported.

Run the test suite before comparing new results with the reference artifacts.

```powershell
.venv\Scripts\python.exe -m pytest -q
```
