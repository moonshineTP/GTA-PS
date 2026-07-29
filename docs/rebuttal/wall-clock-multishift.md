# Wall-clock shifts

## Questions

This ablation supports two responses.

- Jovy asks whether solver cost changes practical performance.
- wsJb asks whether fixed-\(K\) gains survive equal elapsed time.

## Protocol

The environment is a sparse 200-state GARNET with five actions and three
successors per state-action pair. Its neighborhood rate is 0.10. Each reward
shift activates 20 localized states.

Each paired seed uses the same transition kernel and six reward shifts. A
shift occurs after a 5-second wall-clock period. It does not occur after a
fixed number of calls or backups.

Each planning call requests \(K=10\) backups. Every GTA-PS period completes at
least 100 calls.

The timed region includes initialization, reward updates, queue work, backups,
topology refreshes, Arnoldi work, and linear solves. Reward generation and
exact value targets are excluded because both methods share them.

GTA-PS uses the tuned GARNET parameters and no top-\(q\) cap. Evaluation uses
100 paired seeds and eight worker processes. Method order is randomized within
each seed.

## Metrics

For period \(j\) with duration \(T_j\), the recovery metric is

\[
\operatorname{AUC}_j
=
\frac{1}{T_j}
\int_0^{T_j}
\frac{1}{|S|}
\lVert V_j(t)-V_j^\star\rVert_1\,dt.
\]

The reported recovery AUC averages this quantity over six shifts. Lower
values indicate faster recovery over elapsed time. Shift-end \(L_1\) measures
the last value error in each period.

Session runtime is aggregated with the 10-second shifted geometric mean:

\[
\operatorname{SGM10}(t_1,\ldots,t_N)
=
\exp\!\left(\frac{1}{N}\sum_{i=1}^N\log(t_i+10)\right)-10.
\]

The shift reduces the effect of large runtime outliers. It does not remove
initialization from the timed session.

## Result

| Host | Method | \(L_1\) AUC | \(L_\infty\) AUC | End \(L_1\) | End \(L_\infty\) | Backups/s |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| DP | Standard-PS | \(0.923\pm0.033\) | \(1.342\pm0.052\) | \(0.024\pm0.002\) | \(0.026\pm0.002\) | \(1771\pm37\) |
| DP | GTA-PS\(^\beta\) | \(1.840\pm0.063\) | \(2.783\pm0.106\) | \(0.345\pm0.018\) | \(0.395\pm0.021\) | \(785\pm13\) |
| Dyna | Standard-PS | \(12.178\pm0.097\) | \(19.313\pm0.155\) | \(12.144\pm0.097\) | \(19.096\pm0.158\) | \(1647\pm40\) |
| Dyna | GTA-PS\(^\beta\) | \(12.159\pm0.093\) | \(19.411\pm0.151\) | \(12.127\pm0.094\) | \(19.209\pm0.154\) | \(970\pm21\) |

Values are means with 95% Student-\(t\) intervals. Standard-PS recovers faster
in DP under equal elapsed periods. The Dyna errors are close, while GTA-PS
processes fewer backups per second in both hosts.

The experiment does not support wall-clock superiority in this regime. The
defensible claim remains backup efficiency under a fixed sequential-update
budget.

## Oracle target

Exact value targets are excluded from timing. The experiment exploits the
known three-successor structure when computing them, avoiding scans over zero
transition entries. A focused test checks agreement with the original dense
value-iteration implementation.

## Limits

Timing depends on the machine, SciPy build, and concurrent load. Eight worker
processes introduce shared CPU contention. The experiment covers one state
size, one GARNET family, and fixed hyperparameters.

The separate solver-threshold ablation places the local LU plus Woodbury to
GMRES crossover near 650 states. That internal crossover does not establish an
end-to-end GTA-PS crossover.

## Artifacts

- `code/misc/wall_clock_multishift.py`
- `code/results/rebuttal/wall_clock_garnet200_r20_dp_final_s100_s6_t5_8w/`
- `code/results/rebuttal/wall_clock_garnet200_r20_dyna_final_s100_s6_t5_8w/`
