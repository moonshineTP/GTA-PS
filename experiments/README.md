# Experiment map

The experiment suite organizes the main evaluation questions as independent Python entry points. Grouped reproduction uses `orchestrate_groups.py`, while specialized protocols run directly.

| Study | Scripts | Measurement |
| --- | --- | --- |
| RQ1 | `rq1_convergence.py`, `rq1_dp_timeline.py`, `rq1_dyna_timeline.py` | Stationary convergence and reward-shift adaptation |
| RQ2 | `rq2_beta_magnitude.py` | Priority ordering under topology-confidence scaling |
| RQ3 | `rq3_alpha_tradeoff.py` | Oracle agreement and predecessor-side coverage |
| RQ4 | `rq4_asymmetry.py` | Performance under directed transition structure |
| RQ5 | `rq5_bottlenecks.py` | Spectral overhead and planning scalability |
| RQ6 | `rq6_curl_divergence.py` | Path-flow and cyclic transition geometry |
| RQ7 | `rq7_gps_batch_garnet.py`, `rq7_gps_batch_topq.py` | Batch scheduling and state abstraction |
| RQ8 | `rq8_eta_tuning.py` | Resolvent diffusion horizon after reward shifts |
| RQ9 | `rq9_time_ablation.py` | Planning quality under elapsed-time budgets |

RQ7 uses `m` for partition or batch size and `q` for topology-frontier width. The GTA policy temperature remains `b`. RQ8 reports the diffusion scale as `c = eta / eta*`.

## Running a study

Each script documents its accepted options through `--help`. A representative standalone run is:

```powershell
.venv\Scripts\python.exe experiments\rq1_convergence.py --seeds 30 --workers 6
```

Common options include `--seeds`, `--workers`, `--K`, `--out-dir`, and `--plot-dir`, although individual scripts expose only the parameters used by their protocol.

The grouped RQ entry runs RQ1-RQ5 sequentially:

```powershell
.venv\Scripts\python.exe orchestrate_groups.py rqs --seeds 30 --workers 6
```

Additional protocols and their reference artifacts are documented in `docs/additional-experiments.md`.
