# Solver threshold

## Question

At what GARNET state count does GMRES become faster than the LU plus
Woodbury rank-\(k\) topology refresh?

## Protocol

The experiment holds transition sparsity at three sampled successors per
state-action pair. It holds normalized locality at
`n_nbr_rate=0.10` and reverse-edge probability at `0.20`. Each evaluated
environment records realized policy-kernel density, mean out-degree, and
probability mass crossing the half-state cut.

Each trial changes \(k=10\) policy rows, then times
`topology_update_lazy`, including the subsequent \(\tau\) refresh. The two
conditions use the same GARNET instance and policy-row changes:

- LU plus Woodbury forces `solver_threshold >= |S|`.
- GMRES forces `solver_threshold = 0`.

A discrete ternary search minimizes
\(\left|\log(t_{\mathrm{LU+W}}/t_{\mathrm{GMRES}})\right|\) over the candidate
state sizes. The selected point is a machine-specific threshold candidate.
It does not establish an architecture-independent cutoff.

The plot reports seed-level mean runtime with a 95% Student-\(t\) confidence
interval. Each seed-level value averages three timing repeats.

## Artifacts

`code/misc/solver_threshold_ablation.py` writes JSON, PDF, and PNG artifacts.
The final table artifact is under
`code/results/rebuttal/solver_threshold_final_table_s100_5w_rank8/`. The plot uses the
shared Viridis line palette.

## Acceptance

- rank-\(k\) forward and transpose resolvents match fresh sparse LU solves;
- both timing lines use identical environments and update rows;
- the JSON records sparsity and cut calibration at every measured size;
- the plot marks the candidate crossover with a star.

## Local result

The final fixed-rank calibration used \(k=8\), 100 seeds, five worker
processes, and three timing repeats per seed. At 650 states, LU plus Woodbury
took \(46.97\pm0.81\) ms and GMRES took \(47.19\pm1.42\) ms. At 700
states, the corresponding times were \(54.02\pm0.94\) ms and
\(49.71\pm1.54\) ms. The intervals are 95% confidence intervals.

The local crossover candidate was therefore 650 states. The overlapping
intervals show that this is a practical switch point, not evidence of a sharp
timing boundary. The implementation cutoff remains a separate configuration
choice.

This value depends on the current machine, SciPy build, rank \(k=8\), and the
specified GARNET calibration. The JSON artifact retains the raw protocol and
all evaluated timings.
