# Additional experiments

These studies test operational properties that complement the main RQ suite. Each protocol has a standalone runner, an explicit output directory, and validation utilities for the saved artifacts.

## Diffusion-horizon tuning

`experiments/rq8_eta_tuning.py` varies the resolvent scale `eta / eta*` after controlled FourRooms reward shifts. It measures normalized error AUC, shift-end error, path variation, and queue nDCG using paired reward sequences across conditions.

The reference protocol uses 30 seeds, six workers, eight shifts, 12 planning calls per shift, and 50 backups per call. It sweeps `eta / eta*` over `0.25, 0.5, 1, 2, 4, 8`. The canonical point `eta = eta*` is a theoretical reference, not an assumed empirical optimum.

Reference artifacts are stored under `results/additional_evidence/rq8_eta_tuning/`.

## Solver crossover

`misc/solver_threshold_ablation.py` compares LU-Woodbury refreshes with GMRES on matched sparse GARNET instances. The experiment varies state count while holding transition sparsity, locality, update rank, and sampled policy-row changes fixed.

The reported crossover is machine-specific. It should be interpreted as a practical configuration measurement for the current SciPy build and hardware, not as an architecture-independent threshold. `scripts/validate_solver_threshold_final.py` checks the saved protocol and aggregates.

Reference artifacts are stored under `results/additional_evidence/solver_threshold_final_table_s100_5w_rank8/`.

## Equal-time recovery

`misc/wall_clock_multishift.py` compares Standard-PS and GTA-PS under equal elapsed periods on a sparse 200-state GARNET. Timing includes agent initialization, reward updates, queue work, Bellman backups, topology refreshes, Arnoldi work, and linear solves. Shared reward generation and exact targets remain outside the timed region.

The primary measurements are time-normalized recovery AUC, shift-end error, and completed backups per second. Method order is randomized within each paired seed. Results depend on the machine, numerical libraries, and concurrent load.

Reference artifacts are stored under:

- `results/additional_evidence/wall_clock_garnet200_r20_dp_final_s100_s6_t5_8w/`
- `results/additional_evidence/wall_clock_garnet200_r20_dyna_final_s100_s6_t5_8w/`

## Arnoldi cadence

`experiments/arnoldi_cadence_ablation.py` aggregates and plots the GARNET-200 runs that vary how often the SLEM estimate is refreshed. The per-period runs come from `misc/wall_clock_multishift.py`, and the reference artifacts are stored under `results/additional_evidence/arnoldi_cadence_30seed/`.

## Hop-distance signals

`misc/xi_tau_hop_distance.py` measures the residual signal and the topology signal AUC by hop distance from the shifted reward. Reference artifacts are stored under `results/additional_evidence/xi_tau_hop_distance_s100_5w/`.

## Validation

Run each generator with `--help` before changing its protocol. Artifact validators live in `scripts/` and fail when expected seeds, conditions, metrics, or companion plots are missing. Preserve existing reference outputs by selecting a new `--out-dir` during exploratory runs.
