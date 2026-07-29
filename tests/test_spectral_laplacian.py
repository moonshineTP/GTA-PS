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
    # The true left-eigenvector compatible in-laplacian is I - P^T
    # This also guarantees that L_in^T = L_out.
    L_in = in_laplacian(P)
    expected = np.eye(2) - P.T
    np.testing.assert_allclose(L_in.toarray(), expected)

def test_in_laplacian_sparse():
    P = sp.csr_array(np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0]
    ]))
    L_in = in_laplacian(P)
    expected = np.eye(3) - P.toarray().T
    np.testing.assert_allclose(L_in.toarray(), expected)

def test_in_laplacian_preserves_left_eigenvectors():
    # Construct a non-doubly stochastic matrix
    P = np.array([
        [0.1, 0.7, 0.2],
        [0.5, 0.3, 0.2],
        [0.0, 0.9, 0.1]
    ])
    
    L_out = out_laplacian(P).toarray()
    L_in = in_laplacian(P).toarray()
    
    # Check that L_in is exactly the transpose of L_out
    np.testing.assert_allclose(L_in, L_out.T)
    
    # Let v be a left eigenvector of L_out: v^T L_out = lambda v^T
    # Then L_out^T v = lambda v
    # Since L_in = L_out^T, L_in v = lambda v
    # This verifies that right eigenvectors of L_in are left eigenvectors of L_out
    vals_out_left, vecs_out_left = np.linalg.eig(L_out.T)
    vals_in_right, vecs_in_right = np.linalg.eig(L_in)
    
    # Sort and compare eigenvalues
    np.testing.assert_allclose(np.sort(np.abs(vals_out_left)), np.sort(np.abs(vals_in_right)))
