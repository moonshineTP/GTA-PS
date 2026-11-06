"""
Spectral utilities for GTA-PS.

All transition matrices P are (|S|, |S|) numpy arrays where P[s, :] is the
row probability distribution (row-stochastic).  Sparse CSR format is used
wherever |S| >= 200.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Tuple


# ---------------------------------------------------------------------------
# Laplacian constructors
# ---------------------------------------------------------------------------

def out_laplacian(P: np.ndarray) -> sp.csr_array:
    """L^out = I - P  (row-stochastic, right eigenvectors v_k)."""
    n = P.shape[0]
    if sp.issparse(P):
        return sp.eye(n, format="csr") - P
    return sp.eye(n, format="csr") - sp.csr_array(P)


def in_laplacian(P: np.ndarray) -> sp.csr_array:
    """L^in = D^in - P^T  (in-degree diagonal minus transpose)."""
    if sp.issparse(P):
        d_in = np.asarray(P.sum(axis=0)).ravel()
        Psp_T = sp.csr_array(P.transpose())
    else:
        Pdense = np.asarray(P)
        d_in = Pdense.sum(axis=0)
        Psp_T = sp.csr_array(Pdense.T)
    Psp_T.data[np.abs(Psp_T.data) < 1e-6] = 0.0
    Psp_T.eliminate_zeros()
    return sp.diags(d_in, format="csr") - Psp_T


def operator_split(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split P = S + A where S = (P + P^T)/2 (symmetric/dissipative)
    and A = (P - P^T)/2 (antisymmetric/curl).
    Returns dense arrays (S, A).
    """
    if sp.issparse(P):
        P = P.toarray()
    S = (P + P.T) / 2.0
    A = (P - P.T) / 2.0
    return S, A


# ---------------------------------------------------------------------------
# SLEM via Arnoldi / ARPACK
# ---------------------------------------------------------------------------

def slem_arnoldi(
    P: np.ndarray,
    k: int = 15,
    v0: "np.ndarray | None" = None,
) -> "tuple[float, np.ndarray | None]":
    """
    Estimate SLEM(P) = max{|lambda| : lambda in sigma(P), lambda != 1}
    using ARPACK (implicitly restarted Arnoldi, k Lanczos vectors).
    Works for non-symmetric / non-reversible chains.

    Parameters
    ----------
    v0 : ndarray or None
        Warm-start vector for ARPACK.  Pass the leading eigenvector returned
        from the previous call to reduce convergence iterations significantly.

    Returns
    -------
    slem : float
    v0_next : ndarray or None
        Leading eigenvector for warm-starting the next call.  None when the
        dense eigensolver is used (small n).
    """
    import scipy.linalg as sla

    n = P.shape[0]
    if sp.issparse(P):
        Pdense = P.toarray()
    else:
        Pdense = np.asarray(P)

    # For small n, ARPACK requires k+1 < n-1; use dense eig instead.
    if n <= k + 3:
        vals = sla.eig(Pdense, right=False)
        mods = np.abs(vals)
        mods = np.sort(mods)[::-1]
        for v in mods:
            if abs(v - 1.0) > 1e-6:
                return float(v), None
        # All eigenvalues ≈ 1 → near-periodic chain; SLEM is mods[1], not 0.
        return float(mods[1]), None

    Psp = sp.csr_array(Pdense)
    # Eliminate near-zero fill-ins that arise when P_pi is a softmax-weighted
    # mixture of sparse transitions.  Without this, every entry is technically
    # nonzero  and ARPACK runs dense matvecs
    # instead of sparse ones, inflating cost by ~n× for large b.
    Psp.data[np.abs(Psp.data) < 1e-6] = 0.0
    Psp.eliminate_zeros()
    k_eff = min(k, n - 2)
    try:
        vals, vecs = spla.eigs(
            Psp, k=k_eff + 1, which="LM",
            return_eigenvectors=True, v0=v0,
        )
    except Exception:
        vals, vecs = spla.eigs(
            Psp, k=k_eff + 1, which="LM", return_eigenvectors=True,
        )

    mods = np.abs(vals)
    order = np.argsort(mods)[::-1]
    mods = mods[order]
    vecs = vecs[:, order]
    v0_next = np.real(vecs[:, 0]).astype(float)

    for v in mods:
        if abs(v - 1.0) > 1e-6:
            return float(v), v0_next
    # All eigenvalues ≈ 1 → near-periodic chain; SLEM is mods[1], not 0.
    return float(mods[1]), v0_next


# ---------------------------------------------------------------------------
# Resolvent operators
# ---------------------------------------------------------------------------

def resolvent_forward(
    P: np.ndarray,
    r: np.ndarray,
    eta: float,
    solver: str = "auto",
) -> np.ndarray:
    """
    Compute f^out = (L^out + eta*I)^{-1} r = (I - P + eta*I)^{-1} r.

    At eta = eta* = (1-gamma)/gamma this equals gamma * V^{pi^GTA}.

    solver: 'lu' (exact, small |S|), 'gmres' (large sparse), 'auto'.
    """
    n = P.shape[0]
    if not sp.issparse(P):
        Psp = sp.csr_array(P)
    else:
        Psp = P
    A = sp.eye(n, format="csr") * (1.0 + eta) - Psp  # (1+eta)I - P

    if solver == "auto":
        solver = "lu" if n <= 300 else "gmres"

    if solver == "lu":
        lu = spla.splu(A.tocsc())
        return lu.solve(r)
    else:
        # Restarted GMRES with ILU(0) preconditioner
        try:
            ilu = spla.spilu(A.tocsc(), fill_factor=10, drop_tol=1e-4)
            M = spla.LinearOperator((n, n), ilu.solve)
        except RuntimeError:
            # Fallback to unpreconditioned if ILU encounters a singular pivot
            M = None

        f, info = spla.gmres(A, r, M=M, restart=20, maxiter=200, atol=1e-4)
        if info != 0:
            # fallback to direct
            lu = spla.splu(A.tocsc())
            f = lu.solve(r)
        return f


def resolvent_backward(
    P: np.ndarray,
    r: np.ndarray,
    eta: float,
    solver: str = "auto",
) -> np.ndarray:
    """
    Compute f^in = (L^in + eta*I)^{-1} r = (D^in - P^T + eta*I)^{-1} r.

    Captures backward attribution (left eigenvector direction).
    """
    n = P.shape[0]
    Lin = in_laplacian(P)
    A = Lin + sp.eye(n, format="csr") * eta

    if solver == "auto":
        solver = "lu" if n <= 300 else "gmres"

    if solver == "lu":
        lu = spla.splu(A.tocsc())
        return lu.solve(r)
    else:
        try:
            ilu = spla.spilu(A.tocsc(), fill_factor=10, drop_tol=1e-4)
            M = spla.LinearOperator((n, n), ilu.solve)
        except RuntimeError:
            # Fallback to unpreconditioned if ILU encounters a singular pivot
            M = None

        f, info = spla.gmres(A, r, M=M, restart=20, maxiter=200, atol=1e-4)
        if info != 0:
            lu = spla.splu(A.tocsc())
            f = lu.solve(r)
        return f
