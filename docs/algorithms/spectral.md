# Spectral Utilities

**File:** `src/gta_ps/utils/spectral.py`

Utility functions that support the topology‑based scoring of `GtaPS`.

## Laplacians
- `out_laplacian(P)`:  `L^{out} = I - P` (row‑stochastic transition `P`).
- `in_laplacian(P)`:   `L^{in} = D^{in} - P^T` where `D^{in}` are column sums of `P`.

## Spectral Gap
- `slem_arnoldi(P, k=15) → float` – estimates the second largest modulus eigenvalue of a stochastic matrix `P`.  Used to drive the schedules `α_t` and `β_t` in `GtaPS`.

## Resolvent Potentials
- `resolvent_forward(P, r, eta, solver='auto')` – solves `(L^{out} + ηI)^{-1} r`.
- `resolvent_backward(P, r, eta, solver='auto')` – solves `(L^{in} + ηI)^{-1} r`.

These functions are called internally by `GtaPS` during the priority update.  They are also useful for diagnostics and analysis of a policy’s connectivity.

---

See also: [[gta-ps.md]]
