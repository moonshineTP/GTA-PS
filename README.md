# GTA-PS

This repository is the official implementation of *Graph Topology Augmentation for Prioritized Sweeping in Non-stationary Reinforcement Learning*. GTA-PS enriches prioritized sweeping with spectral signals from the policy-induced transition graph, so planning recovers faster after reward shifts. The package provides tabular planning agents, environment adapters, the experiment suite behind each research question, and the saved artifacts that back the reported numbers.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/gta_ps/` | Agents (`algorithms/`), tabular environments (`envs/`), spectral and metric utilities (`utils/`) |
| `experiments/` | One entry point per research question, RQ1 to RQ9 |
| `misc/` | Ablations, benchmarks, tuners, the Arnoldi cadence aggregation, the FourRooms showcase, and shared plot styling |
| `configs/` | YAML settings for GARNET, FourRooms, Reacher, and CarGoal |
| `scripts/` | Validators that check saved reference artifacts |
| `tests/` | Unit and protocol tests for the package and the experiment runners |
| `docs/` | Protocol notes for the additional experiments |
| `results/`, `plots/` | Reference data and figures, indexed in `results/README.md` and `plots/README.md` |
| `threads/` | Parallel executor and hyperparameter tuning engine shared by the runners |

## Installation

Python 3.12 or newer is required. Create the environment from the repository root.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e ".[dev,reacher]"
```

On Linux or macOS, replace `.venv\Scripts\python.exe` with `.venv/bin/python`. The optional CarGoal environment depends on an older Gymnasium stack, so it lives in its own environment. The pinned packages sit in the `cargoal` group of `pyproject.toml`.

## Verifying the installation

Run the test suite first, since later comparisons with the reference artifacts rely on it.

```powershell
.venv\Scripts\python.exe -m pytest -q
```

A short end-to-end pass of each experiment group follows. The grouped runner executes scripts sequentially and stops at the first failure.

```powershell
.venv\Scripts\python.exe orchestrate_groups.py sanity --seeds 2 --workers 1
.venv\Scripts\python.exe orchestrate_groups.py rqs --seeds 2 --workers 1
.venv\Scripts\python.exe orchestrate_groups.py ablations --seeds 2 --workers 1
```

## Reproducing the experiments

Three Windows launchers run the same groups at publication scale (1,000 seeds, 10 workers by default). Extra arguments override those defaults, for example `run_rqs.cmd --seeds 30 --workers 6`.

```cmd
run_sanity.cmd
run_rqs.cmd
run_ablations.cmd
```

Studies with their own protocols run directly, and each script documents its options through `--help`.

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

The grouped `rqs` run covers RQ1 to RQ5, while RQ6 to RQ9 have standalone entries. Protocol details are in `experiments/README.md`, supporting studies are in `misc/README.md`, and the solver, diffusion-horizon, and equal-time studies are summarized in `docs/additional-experiments.md`.

## Outputs and reference artifacts

Scripts write numerical artifacts under `results/` and figures under `plots/`. Generated data stores its seeds, worker counts, environment settings, and metric definitions where the protocol supports it.

The committed files in `results/` and `plots/` are reference artifacts from final runs. Pass a new `--out-dir` during exploratory work to keep them intact. Smoke, pilot, and check outputs are excluded by `.gitignore`, so exploratory runs never enter version control by accident.
