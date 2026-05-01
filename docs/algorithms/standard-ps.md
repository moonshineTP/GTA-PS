# Standard-PS Baseline

**File:** `src/gta_ps/algorithms/standard_ps.py`

`StandardPS` implements the classic prioritized sweeping algorithm of Moore & Atkeson (1993) for tabular Markov decision processes.  The priority of a state is the *Bellman residual* `ξ(s)`; the host performs Bellman backups until the highest residual falls below the threshold.

## Key Ingredients

- **PrioritizedSweepingModule** –  interface required by hosts.
- **BellmanMixin** – provides `apply_bellman_backup` and `compute_bellman_residual`.
- **StandardPS** – concrete module that maintains a min‑heap of states ordered by `ξ(s)`, updates residuals when rewards change, and propagates priority to predecessors.

## Interaction with Hosts

| Host | Relationship | Notes |
|------|--------------|-------|
| `DpHost` | Provides the oracle transition kernel and never updates the PS model. | `StandardPS` stays fixed throughout the episode. |
| `DynaHost` | After each real transition it calls `module.update_model(P_hat, r_hat)` and `module.update_value(s, V[s])` to keep the PS priorities consistent with the empirical model and value estimates. | The host supplies `P_hat` through the `frequency_model` utilities.

## Mixins

- `BellmanMixin` – stateless backup utilities used by both `StandardPS` and the hosts.
- `TdMixin` – provided for hosts that may perform TD updates; not used directly by `StandardPS`.

For deeper details, see the papers: Moore & Atkeson (1993) and Peng & Williams (1993).

---

See also: [[gta-ps.md]] and [[spectral.md]]
