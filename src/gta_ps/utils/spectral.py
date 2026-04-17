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
    """L^in = D^in - P^T  (left eigenvectors u_k)."""
    if sp.issparse(P):
        Pdense = P.toarray()
    else:
        Pdense = np.asarray(P)
    d_in = Pdense.sum(axis=0)          # column sums = in-degree
    n = Pdense.shape[0]
    D = sp.diags(d_in, format="csr")
    return D - sp.csr_array(Pdense.T)


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

def slem_arnoldi(P: np.ndarray, k: int = 15) -> float:
    """
    Estimate SLEM(P) = max{|lambda| : lambda in sigma(P), lambda != 1}
    using ARPACK (implicitly restarted Arnoldi, k Lanczos vectors).
    Works for non-symmetric / non-reversible chains.
    """
    n = P.shape[0]
    if not sp.issparse(P):
        P = sp.csr_array(P)
    k = min(k, n - 2)
    # Request k+1 eigenvalues; the dominant one is 1 (stationary)
    vals = spla.eigs(P, k=k + 1, which="LM", return_eigenvectors=False)
    mods = np.abs(vals)
    mods = np.sort(mods)[::-1]
    # Skip the leading 1.0 (up to numerical tolerance)
    for v in mods:
        if abs(v - 1.0) > 1e-8:
            return float(v)
    return float(mods[1])  # fallback


# ---------------------------------------------------------------------------
# Resolvent operators
# ---------------------------------------------------------------------------

def resolvent_forward(
    P: np.ndarray,
    r: np.ndarray,
    eps: float,
    solver: str = "auto",
) -> np.ndarray:
    """
    Compute f^out = (L^out + eps*I)^{-1} r = (I - P + eps*I)^{-1} r.

    At eps = eps* = (1-gamma)/gamma this equals gamma * V^{pi^GTA}.

    solver: 'lu' (exact, small |S|), 'gmres' (large sparse), 'auto'.
    """
    n = P.shape[0]
    if not sp.issparse(P):
        Psp = sp.csr_array(P)
    else:
        Psp = P
    A = sp.eye(n, format="csr") * (1.0 + eps) - Psp  # (1+eps)I - P

    if solver == "auto":
        solver = "lu" if n <= 300 else "gmres"

    if solver == "lu":
        lu = spla.splu(A.tocsc())
        return lu.solve(r)
    else:
        # Restarted GMRES with ILU(0) preconditioner
        ilu = spla.spilu(A.tocsc(), fill_factor=1.0)
        M = spla.LinearOperator((n, n), ilu.solve)
        f, info = spla.gmres(A, r, M=M, restart=20, maxiter=200, atol=1e-8)
        if info != 0:
            # fallback to direct
            lu = spla.splu(A.tocsc())
            f = lu.solve(r)
        return f


def resolvent_backward(
    P: np.ndarray,
    r: np.ndarray,
    eps: float,
    solver: str = "auto",
) -> np.ndarray:
    """
    Compute f^in = (L^in + eps*I)^{-1} r = (D^in - P^T + eps*I)^{-1} r.

    Captures backward attribution (left eigenvector direction).
    """
    n = P.shape[0]
    Lin = in_laplacian(P)
    A = Lin + sp.eye(n, format="csr") * eps

    if solver == "auto":
        solver = "lu" if n <= 300 else "gmres"

    if solver == "lu":
        lu = spla.splu(A.tocsc())
        return lu.solve(r)
    else:
        ilu = spla.spilu(A.tocsc(), fill_factor=1.0)
        M = spla.LinearOperator((n, n), ilu.solve)
        f, info = spla.gmres(A, r, M=M, restart=20, maxiter=200, atol=1e-8)
        if info != 0:
            lu = spla.splu(A.tocsc())
            f = lu.solve(r)
        return f
