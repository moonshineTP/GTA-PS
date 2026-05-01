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
- **RQ3 (Tradeoff):** `rq3_tradeoff.py` (Accuracy vs. Coverage)
- **RQ4 (Asymmetry):** `rq4_asymmetry.py` (Directed dynamics gain)

### Ablation Studies
Supporting audits live in `misc/`:
- **Smoothness:** `smoothness_ablation.py` (Resolvent as low-pass filter)
- **Mixing Threshold:** `mixing_threshold_ablation.py` (Phase behavior at $K \approx t_{\text{mix}}$)
- **Scheduler:** `scheduler_ablation.py` (Linear vs. Power vs. Log schedules)

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
