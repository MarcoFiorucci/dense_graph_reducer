import numpy as np
import scipy.linalg as linalg

""" This function provides an approximation of the commute time measure introcuced by Von Luxburg et al.
    in Getting lost in space: Large sample analysis of the resistance distance """

def is_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)



def Laplacian_matrix(A):
    """Copied from
    http://networkx.lanl.gov/_modules/networkx/linalg/laplacianmatrix.html. Eg
    A = np.array([[0, 1, 0, 0, 1, 0],
                  [1, 0, 1, 0, 1, 0],
                  [0, 1, 0, 1, 0, 0],
                  [0, 0, 1, 0, 1, 1],
                  [1, 1, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0, 0]])
    Laplacian_matrix(A)
        array([[ 2., -1.,  0.,  0., -1.,  0.],
       [-1.,  3., -1.,  0., -1.,  0.],
       [ 0., -1.,  2., -1.,  0.,  0.],
       [ 0.,  0., -1.,  3., -1., -1.],
       [-1., -1.,  0., -1.,  3.,  0.],
       [ 0.,  0.,  0., -1.,  0.,  1.]])
"""
    assert is_symmetric(A)
    I = np.identity(A.shape[0])
    D = I * np.sum(A, axis=1)
    L = D - A
    return L


def get_commute_distance_using_Laplacian(S):
    """Original code copyright (C) Ulrike Von Luxburg, Python
    implementation by James McDermott."""

    n = S.shape[0]
    L = Laplacian_matrix(S)
    dinv = 1. / np.sum(S, 0)

    Linv = linalg.inv(L + np.ones(L.shape) / n) - np.ones(L.shape) / n

    Linv_diag = np.diag(Linv).reshape((n, 1))
    Rexact = Linv_diag * np.ones((1, n)) + np.ones((n, 1)) * Linv_diag.T - 2 * Linv

    # convert from a resistance distance to a commute time distance
    vol = np.sum(S)
    Rexact *= vol

    return Rexact


def Von_Luxburg_amplified_commute(A):
    """ Assumes A is symmetric.
    Original code copyright (C) Ulrike Von Luxburg, Python implementation by James McDermott."""

    R = get_commute_distance_using_Laplacian(A)

    n = A.shape[0]

    # compute commute time limit expression:
    d = np.sum(A, 1)
    Rlimit = np.tile((1. / d), (n, 1)).T + np.tile((1. / d), (n, 1))

    # compute correction term u_{ij}:
    tmp = np.tile(np.diag(A), (n, 1)).T / np.tile(d, (n, 1))
    tmp2 = np.tile(d, (n, 1)).T * np.tile(d, (n, 1))
    uAu = tmp + tmp.T - 2 * A / tmp2

    # compute amplified commute:
    tmp3 = R - Rlimit - uAu

    # enforce 0 diagonal:
    D = tmp3 - np.diag(np.diag(tmp3))

    return D


