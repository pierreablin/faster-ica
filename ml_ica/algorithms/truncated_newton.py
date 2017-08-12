"""
Python implementation of the Truncated Newton's method for ICA.
Reference for the algorithm without preconditioning:
Tillet, P. et al., "Infomax-ICA using Hessian-free optimization"
"""

# Authors: Pierre Ablin <pierre.ablin@inria.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Jean-Francois Cardoso <cardoso@iap.fr>
#
# License: BSD (3-clause)
from __future__ import print_function
from time import time
from itertools import product
import numpy as np
import scipy.sparse as sparse
from scipy import linalg
import scipy.sparse.linalg as slinalg

from ..tools import (loss, gradient, compute_h, regularize_h, solveh,
                     hessian_free, score, score_der, linesearch)


def full_hessian(Y, psidY):
    '''
    Computes the full hessian, in a sparse matrix. Very slow.
    '''
    N, T = Y.shape
    # log det part
    ind_ld = np.array(list(product(range(N), repeat=2)), dtype=int)
    I, J = ind_ld.T
    K = J
    L = I
    data = np.ones(N ** 2)
    row = J + N * I
    col = L + N * K
    H_ld = sparse.coo_matrix((data, (row, col))).tocsr()
    # density part
    values = np.zeros((N, N, N))

    for i in range(N):
        temp = psidY[i, :]
        values[i, :, :] = np.dot(temp[None, :] * Y, Y.T)
    values /= float(T)
    ind_sc = np.array(list(product(range(N), repeat=3)), dtype=int)
    I, J, L = ind_sc.T
    K = I
    data = values[I, J, L]
    row = J + N * I
    col = L + N * K
    H_d = sparse.coo_matrix((data, (row, col))).tocsr()
    return H_d + H_ld


def true_eigenvalue(Y, psidY):
    '''
    Computes the smallest eigenvalues of the true Hessian. Slow.
    '''
    H_full = full_hessian(Y, psidY)
    return slinalg.eigsh(H_full, k=1, which='SA')[0][0]


def conjugate_gradient(Y, psidY, h, G, lambda_reg, n_cg_it, tol):
    '''
    Uses the conjugate gradient method to compute the Newton direction H^-1 G.
    We take advantage of the Hessian free product, and precondition the
    algorithm with the hessian approximation h.
    '''
    x = np.zeros_like(G)
    r = G.copy()
    z = solveh(r, h)
    p = z
    rz = np.dot(r.ravel(), z.ravel())
    for i in range(n_cg_it):
        Ap = hessian_free(p, Y, psidY, lambda_reg)
        pAp = np.dot(p.ravel(), Ap.ravel())
        a = rz / pAp
        x += a * p
        r -= a * Ap
        r_norm = np.sqrt(np.dot(r.ravel(), r.ravel()))
        if r_norm / np.max(x) < tol:
            break
        z = solveh(r, h)
        rz_old = rz
        rz = np.dot(r.ravel(), z.ravel())
        b = rz / rz_old
        p = z + b * p
    return x


def truncated_ica(X, tol=1e-7, max_iter=100, l_fact=2., cg_tol=1e-2,
                  verbose=0, callback=None, cg_max=300):
    '''
    Main algorithm.
    The smallest eigenvalue of the Hessian is explecitly computed, but that
    duration is not taken into account.

    Parameters
    ----------
    X : array, shape (N, T)
        Matrix containing the signals that have to be unmixed. N is the
        number of signals, T is the number of samples. X has to be centered

    tol : float
        tolerance for the stopping criterion. Iterations stop when the norm
        of the gradient gets smaller than tol.

    max_iter : int
        Maximal number of iterations for the algorithm

    l_fact : float
        Used to regularize the full Hessian. Its eigen values are shiffted by
        l_fact * its smallest eigenvalue

    cg_tol : float
        Conjugate gradient stoping tolerance.

    verbose : 0, 1 or 2
        Verbose level. 0: No verbose. 1: One line verbose. 2: Detailed verbose

    cg_max : float
        Maximum number of inner conjugate gradient iterations
    Returns
    -------
    Y : array, shape (N, T)
        The estimated source matrix

    W : array, shape (N, N)
        The estimated unmixing matrix, such that Y = WX.
    '''
    N, T = X.shape
    Y = X.copy()
    W = np.eye(N)
    current_loss = loss(Y, W)
    t0 = time()
    timing = 0.
    t_cheats = 0.
    for n in range(max_iter):
        # Compute the score and its derivative
        psiY = score(Y)
        psidY = score_der(psiY)
        # Compute the gradient
        G = gradient(Y, psiY)
        # Stopping criterion
        gradient_norm = linalg.norm(G.ravel(), ord=np.inf)
        if callback is not None:
            callback(locals())
        if gradient_norm < tol:
            break
        # Compute the smallest eigenvalue of H, freezing time.
        t_h = time()
        l_min = true_eigenvalue(Y, psidY)
        t_cheat = time() - t_h
        t_cheats += t_cheat
        timing = time() - t0 - t_cheats
        # Regularisation constant
        l_reg = - l_fact * min(l_min, 0.)
        # Compute the approximation
        h = compute_h(Y, psidY)
        # Regularize it
        h = regularize_h(h, 1., 1)
        # Compute the direction by conjugate gradient
        direction = conjugate_gradient(Y, psidY, h, -G, l_reg, cg_max, cg_tol)
        # Do a line search in that direction
        success, Y_new, W_new, new_loss =\
            linesearch(Y, W, direction, current_loss)
        # If it fails, fall back to gradient
        if not success:
            direction = - G
            _, Y_new, W_new, new_loss =\
                linesearch(Y, W, direction, current_loss, 3)
        # Update
        Y = Y_new
        W = W_new
        current_loss = new_loss
        if verbose:
            info = 'iteration %d, gradient norm = %.4g' % (n, gradient_norm)
            ending = '\r' if verbose == 1 else '\n'
            print(info, end=ending)
    return Y, W


if __name__ == '__main__':
    N, T = 10, 1000
    rng = np.random.RandomState(1)
    S = rng.laplace(size=(N, T))
    A = rng.randn(N, N)
    X = np.dot(A, S)
    truncated_ica(X, verbose=True, max_iter=100)
