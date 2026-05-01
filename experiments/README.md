# GTA-PS Experiments (RQ1–RQ4)

This directory contains the core implementation of the GTA-PS Research Questions. All experiments utilize the **Dyna agent** host framework (planning over learned/count-based models) to ensure robustness under model estimation noise.

## Research Questions

### RQ1: Proactiveness & Convergence
- **Convergence Trajectory** (`rq1_proactiveness.py`): Measures the value error trajectory ($L_\infty$ norm) in stationary environments with no reward shifts.
- **Gamma Ablation** (`rq1_gamma_ablation.py`): Sweeps discount factor $\gamma \in [0.5, 0.98]$ to analyze its effect on convergence rate.
- **Dynamic Timelines** (`rq1_dp_timeline.py`, `rq1_dyna_timeline.py`): Measures error across sequences of reward shifts in DP and Dyna settings.

### RQ2: Topology Allocation (`rq2_allocation.py`)
- **Mission:** Evaluates the adaptive $\beta_t$ schedule against fixed allocations.
- **Metric:** Spearman rank correlation ($\rho$) between topology signal and priority queue pop order.

### RQ3: Convergence vs. Propagation Tradeoff (`rq3_tradeoff.py`)
- **Mission:** Audits the accuracy-coverage tradeoff controlled by the mixing weight $\alpha$.
- **Metric:** Top-1 oracle agreement vs. predecessor-side state coverage.

### RQ4: Geometry & Asymmetry Effects (`rq4_asymmetry_sweep.py`)
- **Mission:** Quantifies GTA-PS gain as environments become increasingly non-reversible.
- **Hindrances:** One-way doors, global drift (wind), and reversibility sweeps ($0.0 \to 1.0$).

---

## Common Arguments

| Flag | Description | Default |
| :--- | :--- | :--- |
| `--workers` | Number of parallel worker processes or **'auto'** (recommended). | auto |
| `--seeds` | Number of independent random seeds. | 10–30 |
| `--K` | Planning budget (backups) per host step. | Varies |
| `--out-dir` | Directory for JSON results. | `results/` |
| `--plot-dir` | Directory for PNG plots. | `plots/` |

## Execution Example
```bash
# Run the intensive asymmetry sweep for RQ4 using all available cores
python experiments/rq4_asymmetry_sweep.py --seeds 10 --workers auto
```
