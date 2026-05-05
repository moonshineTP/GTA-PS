# GTA-PS Experiment Workspace

This repository contains the official implementation and experiment suite for **Graph Topology Augmentation for Prioritized Sweeping (GTA-PS)**, a model-based reinforcement learning algorithm that leverages spectral graph signals to accelerate replanning in nonstationary environments.

---

## 🚀 Overview

GTA-PS augments the standard priority signal (Bellman residual) with a topology signal derived from the directed Laplacian of the policy-induced Markov chain. This allows the agent to propagate value updates along the "proactive" frontier of reward diffusion, even when local residuals are small.

### Core Components
- `src/gta_ps/algorithms/gta_ps.py`: The primary algorithm implementing $\alpha$ and $\beta$ variants.
- `src/gta_ps/algorithms/dyna_ps.py`: Baseline Prioritized Sweeping (Moore & Atkeson, 1993).
- `src/gta_ps/utils/spectral.py`: High-performance spectral utilities (SLEM via Arnoldi, Resolvent solvers).

---

## 🛠 Installation

### Default Stack
Supports **GARNET**, **FourRooms**, **MiniGrid**, and **Reacher**.
Requires Python ≥ 3.12.

```bash
cd code/
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev,reacher]"
```

### CarGoal (Safety-Gymnasium)
The CarGoal environment requires a pinned older stack that conflicts with the default environment. **Use a separate virtual environment.**

```bash
python -m venv .venv-cargoal
# Activate the environment
pip install "safety-gymnasium==1.0.0" "gymnasium==0.26.3" \
            "mujoco==2.3.3" "gymnasium-robotics==1.2.2" "pygame==2.1.0"
pip install -e ".[dev]"
```

---

## 🧪 Running Experiments

### Smoke Tests
Verify your installation and core logic:
```bash
pytest tests/test_smoke.py -q
```

### Research Questions (RQs)
All primary experiments live in `experiments/`:
- **RQ1 (Proactiveness):** `rq1_proactiveness.py` (Convergence speed)
- **RQ2 (Allocation):** `rq2_allocation.py` (Adaptive schedule analysis)
- **RQ3 (Tradeoff):** `rq3_mixing_tradeoff.py` (Accuracy vs. Coverage)
- **RQ4 (Asymmetry):** `rq4_asymmetry.py` (Directed dynamics gain)

### Ablation Studies
Supporting audits live in `misc/`:
- **Mixing Threshold:** `mixing_threshold_ablation.py` (Phase behavior at $K \approx t_{\text{mix}}$)
- **Scheduler:** `scheduler_ablation.py` (Linear vs. Power vs. Log schedules)

### Benchmark Suite

The standard experiment configuration uses **1000 seeds** for statistical significance. Example with 8 workers:

```cmd
run_sanity.cmd --seeds 1000 --workers 8
```

Sample results from a 40-seed validation run:

**Time Complexity Benchmark** (40 reps + 1 warmup, 9 configs):
```
=== Time Complexity Benchmark (40 reps + 1 warmup, 9 configs) ===
----------------------------------------------------------------------
[FOURROOMS_S104] standard_ps  K=26    median=3.70ms  std=1.01ms  (40 reps)
[FOURROOMS_S104] gta_base     K=26    median=4.13ms  std=1.23ms  (40 reps)
[FOURROOMS_S104] gta_beta     K=26    median=4.61ms  std=1.10ms  (40 reps)
----------------------------------------------------------------------
[GARNET_S75    ] standard_ps  K=18    median=8.35ms  std=3.45ms  (40 reps)
[GARNET_S75    ] gta_base     K=18    median=9.18ms  std=4.04ms  (40 reps)
[GARNET_S75    ] gta_beta     K=18    median=6.91ms  std=3.90ms  (40 reps)
----------------------------------------------------------------------
[GARNET_S200   ] standard_ps  K=50    median=17.36ms  std=9.22ms  (40 reps)
[GARNET_S200   ] gta_base     K=50    median=17.54ms  std=8.94ms  (40 reps)
[GARNET_S200   ] gta_beta     K=50    median=15.90ms  std=9.83ms  (40 reps)
----------------------------------------------------------------------
```
Results saved to `results/misc/time_benchmark/time.json`.

**Convergence Benchmark** (budget_cap=2000, K=50):
```
[garnet]  eps=4.0  budget_cap=2000  K=50
  agent           mean_K    median_K     std_K   converged
  ------------  --------  ----------  --------  ----------
  standard_ps      239.2       183.0     229.6      100.0%
  gta_base         257.8       195.0     234.4      100.0%
  gta_beta         255.3       208.0     232.2      100.0%

[fourrooms]  eps=4.0  budget_cap=2000  K=50
  agent           mean_K    median_K     std_K   converged
  ------------  --------  ----------  --------  ----------
  standard_ps      716.3       698.5     507.2      100.0%
  gta_base         536.7       494.5     363.4      100.0%
  gta_beta         544.0       471.0     389.1      100.0%
```
Results saved to `results/misc/convergence_benchmark/convergence.json`.

---

## 🖥 Windows Orchestration

Grouped launchers are provided in the root for sequential execution:
```cmd
run_sanity.cmd    # Hyperparameter tuning and benchmarks
run_rqs.cmd       # Primary Research Question suite
run_ablations.cmd # Supporting theoretical validations
```

Example with custom parameters:
```cmd
run_rqs.cmd --workers 8 --seeds 100 --K 40
```

---

## ⚡ Native Compilation (Nuitka)

For large-scale runs (e.g., 1000+ seeds), use the **Nuitka** compilation pipeline in `compiled/` to generate high-performance native binaries (~2.6x speedup).

```cmd
# Compile all RQs
code\compiled\compile_rqs.cmd
# Run compiled RQs
code\compiled\run_compiled_rqs.cmd --seeds 1000 --workers 5
```

See `code/compiled/README.md` for more details.
