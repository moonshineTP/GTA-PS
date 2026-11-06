import numpy as np
import scipy.sparse as sp
from gta_ps.utils.spectral import in_laplacian, out_laplacian

def test_out_laplacian():
    P = np.array([
        [0.1, 0.9],
        [0.8, 0.2]
    ])
    L_out = out_laplacian(P)
    expected = np.eye(2) - P
    np.testing.assert_allclose(L_out.toarray(), expected)

def test_in_laplacian():
    P = np.array([
        [0.1, 0.9],
        [0.8, 0.2]
    ])
    # The in-Laplacian is D^in - P^T, where D^in is the diagonal of column sums of P.
    L_in = in_laplacian(P)
    expected = np.diag(P.sum(axis=0)) - P.T
    np.testing.assert_allclose(L_in.toarray(), expected)

def test_in_laplacian_sparse():
    P = sp.csr_array(np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0]
    ]))
    L_in = in_laplacian(P)
    dense = P.toarray()
    expected = np.diag(dense.sum(axis=0)) - dense.T
    np.testing.assert_allclose(L_in.toarray(), expected)

def test_in_laplacian_is_a_graph_laplacian():
    # A non-doubly stochastic matrix, so D^in differs from the identity.
    P = np.array([
        [0.1, 0.7, 0.2],
        [0.5, 0.3, 0.2],
        [0.0, 0.9, 0.1]
    ])
    L_in = in_laplacian(P).toarray()

    # Every row sums to zero, so the constant vector spans the kernel.
    np.testing.assert_allclose(L_in.sum(axis=1), np.zeros(3), atol=1e-12)

    # The sparse and dense code paths agree.
    L_sparse = in_laplacian(sp.csr_array(P)).toarray()
    np.testing.assert_allclose(L_in, L_sparse)

    # D^in differs from I here, so the operator differs from I - P^T.
    assert not np.allclose(L_in, np.eye(3) - P.T)
