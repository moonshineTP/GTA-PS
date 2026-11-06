# Results

This directory stores the numerical artifacts behind the reported experiments. Each file comes from a final run of the script named below, with seeds, protocol settings, and aggregates stored together in the JSON. Figures made from these files sit in `plots/`.

## Research-question artifacts

| Files | Producer |
| --- | --- |
| `rq1_*.json` | `experiments/rq1_convergence.py`, `rq1_dp_timeline.py`, `rq1_dyna_timeline.py` |
| `rq2_*.json` | `experiments/rq2_beta_magnitude.py` |
| `rq3_tradeoff_*.json` | `experiments/rq3_alpha_tradeoff.py` |
| `rq4_asymmetry.json` | `experiments/rq4_asymmetry.py` |
| `rq5_bottlenecks.json` | `experiments/rq5_bottlenecks.py` |

## Ablation artifacts

| Files | Producer |
| --- | --- |
| `gamma_ablation_garnet.json` | `misc/gamma_ablation.py` |
| `periodicity_ablation.json`, `ablation_aperiodicity.json` | `misc/periodicity_ablation.py` |
| `reward_support_ablation.json`, `ablation_reward_support.json` | `misc/reward_support_ablation.py` |
| `misc/<study>/` | One folder per script in `misc/`: `beta_dynamics`, `convergence_benchmark`, `cumulative_tracking`, `mixing_threshold_ablation`, `scheduler_ablation`, `shift_rate_ablation`, `showcase`, `time_benchmark`, `tuner` |

## Additional evidence

The folder `additional_evidence/` holds the reference runs for the extended studies. Folder names carry the protocol, and `final` marks the run kept as the reference.

| Folder | Producer | Content |
| --- | --- | --- |
| `rq8_eta_tuning/`, `rq8_eta_gamma/` | `experiments/rq8_eta_tuning.py` | Diffusion-horizon sweep, plus the discount-factor sweep at 0.90, 0.95, and 0.99 |
| `gps_batch_topq/`, `gps_batch_garnet75/` | `experiments/rq7_gps_batch_topq.py`, `rq7_gps_batch_garnet.py` | Batch scheduling and abstraction comparisons |
| `solver_threshold*/` | `misc/solver_threshold_ablation.py` | LU-Woodbury against GMRES crossover; the `final_table` folder is the reference table |
| `wall_clock_garnet200_r20_*` | `misc/wall_clock_multishift.py` | Equal-time recovery on GARNET-200 with DP and Dyna hosts; `final` folders are the reference runs, and `30seed`, `threads*`, and `tuning` folders support them |
| `wall_clock_multishift*/` | `misc/wall_clock_multishift.py` | Equal-time runs on GARNET-200 and GARNET-400 |
| `arnoldi_cadence_*` | `misc/arnoldi_cadence_ablation.py` | Arnoldi refresh cadence, with 30-seed, instrumented, and micro-benchmark variants |
| `xi_tau_hop_distance_s100_5w/` | `misc/xi_tau_hop_distance.py` | Signal AUC by hop distance from the shifted reward |

The equal-time ablation `experiments/rq9_time_ablation.py` has no retained reference run. Regenerate it with a fresh `--out-dir`.

## Working with these files

Run new experiments with a different `--out-dir` so reference artifacts stay intact. Smoke, pilot, swift, and check runs are excluded by `.gitignore`. The validators in `scripts/` check that a saved artifact holds its expected seeds, conditions, and metrics.
