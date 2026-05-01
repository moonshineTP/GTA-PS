# GTA-PS Algorithm

**File:** `src/gta_ps/algorithms/gta_ps.py`

`GtaPS` augments standard prioritized sweeping with *topology signals* derived from the directed Laplacian of the policy-induced Markov chain.  The priority of a state is

```
ξ(s) + β_t\big[(1-α_t)|Δf^{out}(s)| + α_t|Δf^{in}(s)|\big]
```

where `ξ(s)` is the Bellman residual, `Δf^{out/in}` are changes in the forward/backward resolvent potentials, and `α_t`, `β_t` are schedules driven by the spectral gap (SLEM).  With `β_t = 0` GTA‑PS reduces to `StandardPS`.

## Key Components

- **PrioritizedSweepingModule** – core interface.
- **BellmanMixin** / **TdMixin** – derived utilities.
- **GtaPS** – concrete class that maintains priority queues, topology policy `π^{PS}`, and solves linear systems (LU or GMRES) for resolvent potentials.

## Interaction with Hosts

| Host | Relationship | Notes |
|------|--------------|-------|
| `DpHost` | Uses `GtaPS` as its PS module; the oracle kernel is fixed. | `GtaPS` remains static after construction. |
| `DynaHost` | Calls `module.update_model(P_hat, r_hat)` and `module.update_value(s, V[s])` on each real step to sync the empirical model and values used for prefix propagation. |

## Supporting Utilities

`gta_ps.py` relies on `utils/spectral.py` for
- Laplacian construction (`in_laplacian`, `out_laplacian` → `P_pi`),
- SLEM estimation (`slem_arnoldi`), and
- Resolvent solver helpers (`resolvent_forward`, `resolvent_backward`).

These utilities are documented in [[spectral.md]].

---

See also: [[dyna-ps.md]]
