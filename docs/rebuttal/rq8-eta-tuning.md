# RQ8 eta tuning

## Claim

RQ8 tests whether a shorter topology diffusion horizon improves replanning
stability after sharp reward relocation.

For the forward resolvent,

\[
G_\eta^{\mathrm{out}}
=((1+\eta)I-P_\pi)^{-1}
=\frac{1}{1+\eta}(I-\widetilde{\gamma}_\eta P_\pi)^{-1},
\qquad
\widetilde{\gamma}_\eta=\frac{1}{1+\eta}.
\]

The canonical point

\[
\eta^\star=\frac{1-\gamma}{\gamma}
\]

gives \(\widetilde{\gamma}_{\eta^\star}=\gamma\). For a fixed chain and
residual, a larger \(\eta\) shortens the forward diffusion horizon and reduces
its resolvent scale.

The directional hypothesis is:

\[
H_1:\quad
\exists c>1\ \text{such that}\ 
\operatorname{AUC}(c\eta^\star)
<
\operatorname{AUC}(\eta^\star),
\]

with lower tail error and no shift-end degradation. The null states that no
tested \(c>1\) satisfies these conditions.

## Notation

RQ8 uses \(m\) for simultaneous priority-batch size. It reserves \(q\) for
topology-frontier width and \(b\) for the GTA Boltzmann policy temperature.

RQ8 varies only \(c=\eta/\eta^\star\). It fixes \(m=1\), uses the full
frontier \(q=|\mathcal S|\), and applies no state compression. The latter is
the identity case \(a=|\mathcal S|=104\).

## Conditions

Use \(\gamma=0.95\), so \(\eta^\star=0.0526316\). Sweep the fixed ratios

\[
c=\frac{\eta}{\eta^\star}
\in\{0.25,0.5,1,2,4,8\}.
\]

This gives

\[
\eta\in
\{0.01316,0.02632,0.05263,0.10526,0.21053,0.42105\}.
\]

The \(c=1\) condition is the successor-representation reference point. Plot
it with a star marker. The star denotes a theoretical reference, not the
empirical optimum.

## Environment

Use `FourRoomsTabular` with these settings:

- `room_size=5`
- `corridor_len=1`
- `k=1`
- `gamma=0.95`
- `n_peaks_per_room=1`
- default transition dynamics

This setting contains 104 passable states. The complete four-room grid,
including its outer walls, is \(13\times13\).

Each accepted reward shift must satisfy both conditions:

\[
\operatorname{room}(\arg\max r_{j+1})
\ne
\operatorname{room}(\arg\max r_j),
\]

\[
d_j=
\frac{\lVert r_{j+1}-r_j\rVert_1}
{\lVert r_{j+1}\rVert_1+\lVert r_j\rVert_1+\epsilon}
\ge 0.75.
\]

Generate each seed's reward sequence once. Reuse it for every \(\eta\).
Reject sampling must stop after 100 attempts and report failure.

## Protocol

Use the oracle DP host and GTA-PS with `q=None` and frozen-priority batch size
\(m=1\). Hold these parameters fixed:

| Parameter | Value |
|---|---:|
| \(\alpha\) | 0.051719 |
| \(\beta_1\) | 0.159673 |
| \(\beta_2\) | 1.361556 |
| \(b\), policy temperature | 2.849322 |
| Arnoldi rank | 15 |

For each seed:

1. Compute the exact value function for the initial reward.
2. Warm-start every \(\eta\) condition from that value.
3. Apply eight accepted reward shifts.
4. Run 12 planning calls after each shift.
5. Allocate 50 Bellman backups to each call.
6. Record a snapshot before planning and after every call.

Use 30 paired seeds and six worker processes. Treat the seed as the
independent unit. Reward shifts remain nested observations within a seed.

## Metrics

Let \(V_{i,j,t}^{(\eta)}\) denote the value estimate for seed \(i\), shift
\(j\), and planning call \(t\). Define

\[
e_{i,j,t}^{(\eta)}
=
\frac{\lVert V_{i,j,t}^{(\eta)}-V_{i,j}^\star\rVert_1}
{\lVert V_{i,j,0}^{(\eta)}-V_{i,j}^\star\rVert_1+\epsilon}.
\]

The primary metric is normalized post-shift error area:

\[
\operatorname{AUC}_{i,j}^{(\eta)}
=
\frac{1}{13}\sum_{t=0}^{12}e_{i,j,t}^{(\eta)}.
\]

The queue metric compares the current GTA-PS queue with the oracle state
ranking. Define state relevance as

\[
g_{i,j,t}(s)=
\left|V_{i,j,t}^{(\eta)}(s)-V_{i,j}^\star(s)\right|.
\]

The queue order places residual states first by augmented priority. Eligible
topology-only states follow. At each snapshot, report linear-gain
\(\operatorname{nDCG}@K\), where \(K=50\) equals the next planning budget.
Higher nDCG means the queue allocates early backups closer to the largest
oracle value errors.

Secondary metrics are:

- first-call error \(e_{i,j,1}^{(\eta)}\);
- shift-end error \(e_{i,j,12}^{(\eta)}\);
- seed-level 90th percentile AUC;
- threshold backups required to reach \(e_t\le0.25\);
- excess path variation

\[
\mathcal{R}_{i,j}^{(\eta)}
=
\sum_{t=1}^{12}|e_t-e_{t-1}|
-|e_{12}-e_0|.
\]

\(\mathcal{R}\) measures error rebound during replanning. It equals zero for a
monotone path.

## Statistics

Average the 13 queue snapshots within each shift. Average the eight shifts
within each seed before inference. Report means, standard deviations,
medians, interquartile ranges, and 95% confidence intervals.

Compare every larger-\(\eta\) condition with \(\eta^\star\) using paired,
seed-cluster bootstrap differences. Use 10,000 resamples. Report bootstrap
confidence intervals and Holm-adjusted \(p\)-values for the three confirmatory
contrasts.

The evidence supports the claim only when a larger-\(\eta\) condition meets
all criteria:

1. Its 95% interval for paired AUC difference lies below zero.
2. Its Holm-adjusted \(p\)-value is below 0.05.
3. Its seed-level 90th percentile AUC is lower than the canonical point.
4. Its excess path variation is lower than the canonical point.
5. The upper 95% interval for shift-end error difference is below 0.05.

Report the result as inconclusive when these criteria disagree. Report the
null outcome when every larger-\(\eta\) condition loses on primary AUC.

## Plot

Create one full-width, twin-axis ablation plot:

- x-axis: \(\eta/\eta^\star\) on a base-two logarithmic scale;
- left ruler: mean normalized AUC, where lower is better;
- right ruler: mean queue \(\operatorname{nDCG}@50\), where higher is better;
- uncertainty: seed-level bootstrap 95% confidence intervals;
- canonical point: star marker at \(1\times\);
- titles and legends: identify each metric and the canonical SR point;
- palette: Viridis through `code/misc/plot_academic.py`.

Use point markers at every tested value. Do not draw a fitted optimum. Export
PDF and 300 DPI PNG.

## Budget

The full design contains 180 agent trajectories:

\[
6\ \eta\text{ values}\times30\ \text{seeds}=180.
\]

It contains 1,440 reward-shift episodes, 17,280 planning calls, 18,720
snapshots, and at most 864,000 Bellman backups.

Before the full run, time a pilot with one seed, all six \(\eta\) values, and
two shifts. Record wall time and peak memory. Estimate CPU-hours from the
measured pilot. Do not extrapolate from RQ7 timing.

## Risks

- The \(\eta\) sweep changes both diffusion reach and signal magnitude.
- Very small \(\eta\) may increase solver sensitivity.
- Very large \(\eta\) may suppress useful cross-room propagation.
- Reward rejection changes the shift distribution, so its rule must remain
  fixed before runs begin.
- A selected empirical optimum remains environment-specific.

## Implementation

`code/experiments/rq8_eta_tuning.py` implements the paired reward generator,
validated scalar traces, seed-level aggregation, paired bootstrap contrasts,
queue nDCG, and the two-panel Viridis ablation plot.

`rq8_eta_tuning_data.json` is the table and audit artifact. It stores metric
definitions, six table rows, 180 seed-level rows, 240 shift records, and all
18,720 raw snapshots. The exporter rejects non-standard `NaN` and infinity
tokens.

The full pass used 30 paired seeds and six workers on the 104-state
environment. The normalized error AUC was 0.5948 at \(\eta^\star\). It rose
to 0.6397, 0.7334, and 0.7844 at \(2\eta^\star\), \(4\eta^\star\), and
\(8\eta^\star\), respectively.

Queue nDCG peaked at 0.7171 for \(2\eta^\star\), compared with 0.7104 at
\(\eta^\star\). This ranking gain did not reduce planning error. Every tested
larger-\(\eta\) condition increased primary AUC with a positive paired 95%
interval and Holm-adjusted \(p<0.001\). The result rejects the proposed
larger-\(\eta\) performance claim under this protocol.

Artifacts are stored in
`code/results/rebuttal/rq8_eta_tuning/`.
