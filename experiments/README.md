# GTA-PS Experiments (RQ1–RQ6)

This directory contains the core implementation of the GTA-PS Research Questions.

## Research Questions

### RQ1: Proactiveness & Convergence
- **Convergence Trajectory** (`rq1_proactiveness.py`): Measures the value error trajectory ($L_\infty$ norm) in stationary environments with no reward shifts.
- **Dynamic Timelines** (`rq1_dp_timeline.py`, `rq1_dyna_timeline.py`): Measures error across sequences of reward shifts in DP (oracle model) and Dyna (learned model) settings.

### RQ2: Topology Allocation & Schedules
- **Allocation Audit** (`rq2_allocation.py`): Evaluates how topology signals correlate with priority queue ordering.
- **Beta Schedule** (`rq2_beta.py`): Compares fixed $\beta$ vs. linear and adaptive $\beta_t$ schedules.

### RQ3: Convergence vs. Propagation Tradeoff (`rq3_mixing_tradeoff.py`)
- **Mission:** Audits the accuracy-coverage tradeoff controlled by the mixing weight $\alpha$.
- **Metric:** Top-1 oracle agreement vs. predecessor-side state coverage.

### RQ4: Geometry & Asymmetry Effects (`rq4_asymmetry.py`)
- **Mission:** Quantifies GTA-PS gain as environments become increasingly non-reversible (one-way doors, global drift).

### RQ5: Scalability & Bottlenecks (`rq5_bottlenecks.py`)
- **Mission:** Benchmarks the computational overhead of spectral calculations vs. planning speedup across state-space scales.

### RQ6: Curl & Divergence (`rq6_curl_divergence.py`)
- **Mission:** Compares a directed path-flow GARNET against a directed cyclic GARNET to isolate curl/divergence geometry.

---

## Common Arguments

| Flag | Description | Default |
| :--- | :--- | :--- |
| `--workers` | Number of parallel worker processes or **'auto'**. | auto |
| `--seeds` | Number of independent random seeds. | 20 |
| `--K` | Planning budget (backups) per host step. | Varies |
| `--env` | Environment ID (`garnet`, `fourrooms`, `reacher`, `cargoal`). | garnet |
| `--out-dir` | Directory for JSON results. | `results/` |
| `--plot-dir` | Directory for PNG plots. | `plots/` |

## Execution Example
```bash
# Run RQ1 proactiveness with 50 seeds
python experiments/rq1_proactiveness.py --seeds 50 --workers auto
```
