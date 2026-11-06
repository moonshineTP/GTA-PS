# Plots

This directory stores the figures rendered from `results/`. PNG files are the working format, and PDF companions exist for selected figures.

| Figures | Source study |
| --- | --- |
| `rq1_dp_timeline_*`, `rq1_dyna_timeline_*` | RQ1 timelines with DP and Dyna hosts |
| `rq2_beta_ablation`, `rq3_alpha_tradeoff`, `rq4_asymmetry`, `rq5_bottlenecks` | RQ2 to RQ5 |
| `rq7_gps_batch_*` | RQ7 batch scheduling, including the `final_s100_c25_m1to16` panels |
| `rq8_eta_tuning` | RQ8 diffusion horizon |
| `solver_threshold_*` | LU-Woodbury against GMRES crossover |
| `wall_clock_garnet200_*` | Equal-time recovery on GARNET-200, where the `r20_*_final_*` pair is the reference run |
| `arnoldi_cadence_ablation` | Arnoldi refresh cadence |
| `cumulative_ablation`, `mixing_threshold_ablation`, `shift_rate_ablation` | Theory-facing ablations from `misc/` |
| `gamma_ablation_garnet`, `periodicity_ablation`, `reward_support_ablation` | Parameter ablations from `misc/` |
| `fourrooms_ablation`, `fourrooms_showcase*` | FourRooms comparison figure, animations, and still frames from `misc/showcase.py` |

Scripts write here by default. Pass `--plot-dir` to keep exploratory renders apart from these files.
