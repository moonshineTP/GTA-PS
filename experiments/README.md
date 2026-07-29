# GTA-PS Experiments (RQ1–RQ6)

This directory contains the core implementation of the GTA-PS Research Questions.

## Research Questions

### RQ1: Proactiveness & Convergence
- **Convergence Trajectory** (`rq1_convergence.py`): Measures stationary value-error trajectories on GARNET and FourRooms, comparing planning budgets within the same figure.
- **Dynamic Timelines** (`rq1_dp_timeline.py`, `rq1_dyna_timeline.py`): Measures error across sequences of reward shifts in DP (oracle model) and Dyna (learned model) settings.

### RQ2: Topology Allocation & Schedules
- **Beta Audit** (`rq2_beta_magnitude.py`): Evaluates how topology signals correlate with priority queue ordering, varying over beta magnitudee.

### RQ3: Convergence vs. Propagation Tradeoff (`rq3_mixing_tradeoff.py`)
- **Mission:** Audits the accuracy-coverage tradeoff controlled by the mixing weight $\alpha$.
- **Metric:** Top-1 oracle agreement vs. predecessor-side state coverage.

### RQ4: Geometry & Asymmetry Effects (`rq4_asymmetry.py`)
- **Mission:** Quantifies GTA-PS gain as environments become increasingly non-reversible (one-way doors, global drift).

### RQ5: Scalability & Bottlenecks (`rq5_bottlenecks.py`)
- **Mission:** Benchmarks the computational overhead of spectral calculations vs. planning speedup across state-space scales.

### RQ6: Curl & Divergence (`rq6_curl_divergence.py`)
- **Mission:** Compares a directed path-flow GARNET against a directed cyclic GARNET to isolate curl/divergence geometry.

### RQ7: GPS Batch Audit (`rq7_gps_batch_topq.py`)
- **Mission:** Compares GPS H2 partition solves with simultaneous top-\(m\) Standard-PS and GTA-PS updates, then retains the Linear VaST-VI and GTA-PS-FA abstraction panel.
- **Default pass:** 30 seeds over six worker processes.

### RQ8: Eta Tuning
- **Status:** Implemented in `rq8_eta_tuning.py`; design in `code/docs/rebuttal/rq8-eta-tuning.md`.
- **Mission:** Tests shorter topology diffusion horizons after sharp reward shifts.
- **Metrics:** Normalized planning-error AUC and queue nDCG against oracle statewise value error.
- **Default pass:** 30 paired seeds over six worker processes.
- **Table data:** `results/rebuttal/rq8_eta_tuning/rq8_eta_tuning_data.json`.

RQ7 uses \(m\) for partition or batch size and \(q\) for topology-frontier
width. The GTA Boltzmann policy temperature remains \(b\). RQ8 uses
\(c=\eta/\eta^\star\).

### Solver threshold (`solver_threshold_ablation.py`)
- **Mission:** Uses a state-size ternary search on calibrated sparse, localized GARNETs to locate the LU-Woodbury to GMRES timing crossover under the wall-clock calibration's fixed rank (k=10).
- **Default pass:** 30 seeds over six worker processes, with seed-level 95% confidence intervals.

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
python experiments/rq1_convergence.py --seeds 50 --workers auto
```
