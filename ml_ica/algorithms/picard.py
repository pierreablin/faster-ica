# Authors: Pierre Ablin <pierre.ablin@inria.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Jean-Francois Cardoso <cardoso@iap.fr>
#
# License: BSD (3-clause)

from __future__ import print_function

from time import time

import numpy as np
from scipy import linalg

from ml_ica.tools import (gradient, compute_h, regularize_h, solveh,
                          score, score_der, linesearch)


def picard(X, max_iter=1000, tol=1e-7, mem_size=7, precon=2, lambda_min=0.01,
           ls_tries=10, verbose=0, callback=None):
    '''Runs Picard algorithm.

    The algorithm is detailed in::

        Pierre Ablin, Jean-Francois Cardoso, and Alexandre Gramfort
        Faster ICA by preconditioning with Hessian approximations
        ArXiv Preprint, June 2017
        https://arxiv.org/abs/1706.08171

    Parameters
    ----------
    X : array, shape (N, T)
        Matrix containing the signals that have to be unmixed. N is the
        number of signals, T is the number of samples. X has to be centered

    max_iter : int
        Maximal number of iterations for the algorithm

    tol : float
        tolerance for the stopping criterion. Iterations stop when the norm
        of the gradient gets smaller than tol.

    mem_size : int
        Size of L-BFGS's memory. Typical values for m are in the range 3-15

    precon : 1 or 2
        Chooses which Hessian approximation is used as preconditioner.
        1 -> H1
        2 -> H2
        H2 is more costly to compute but can greatly accelerate convergence
        (See the paper for details).

    lambda_min : float
        Constant used to regularize the Hessian approximations. The
        eigenvalues of the approximation that are below lambda_min are
        shifted to lambda_min.

    ls_tries : int
        Number of tries allowed for the backtracking line-search. When that
        number is exceeded, the direction is thrown away and the gradient
        is used instead.

    verbose : 0, 1 or 2
        Verbose level. 0: No verbose. 1: One line verbose. 2: Detailed verbose

    callback : None or function
        Optional function run at each iteration on all the local variables.


    Returns
    -------
    Y : array, shape (N, T)
        The estimated source matrix

    W : array, shape (N, N)
        The estimated unmixing matrix, such that Y = WX.
    '''
    # Init
    N, T = X.shape
    W = np.eye(N)
    Y = X.copy()
    s_list = []
    y_list = []
    r_list = []
    current_loss = None
    t0 = time()
    for n in range(max_iter):
        timing = time() - t0
        # Compute the score function
        psiY = score(Y)
        psidY = score_der(psiY)
        # Compute the relative gradient
        G = gradient(Y, psiY)
        # Stopping criterion
        gradient_norm = linalg.norm(G.ravel(), ord=np.inf)
        if gradient_norm < tol:
            break
        # Update the memory
        if n > 0:
            s_list.append(direction) # noqa
            y = G - G_old  # noqa
            y_list.append(y)
            r_list.append(1. / np.dot(direction.ravel(), y.ravel()))  # noqa
            if len(s_list) > mem_size:
                s_list.pop(0)
                y_list.pop(0)
                r_list.pop(0)
        G_old = G # noqa
        # Compute the Hessian approximation and regularize
        h = compute_h(Y, psidY, precon)
        h = regularize_h(h, lambda_min)
        # Find the L-BFGS direction
        direction = _l_bfgs_direction(G, h, s_list, y_list, r_list,
                                      precon, lambda_min)
        # Do a line_search in that direction:
        converged, new_Y, new_W, new_loss, alpha =\
            linesearch(Y, W, direction, current_loss, ls_tries)
        if not converged:
            direction = -G
            s_list, y_list, r_list = [], [], []
            _, new_Y, new_W, new_loss, alpha =\
                linesearch(Y, W, direction, current_loss, ls_tries)
        direction *= alpha
        Y = new_Y
        W = new_W
        current_loss = new_loss
        if verbose:
            info = 'iteration %d, gradient norm = %.4g' % (n, gradient_norm)
            ending = '\r' if verbose == 1 else '\n'
            print(info, end=ending)
        if callback is not None:
            callback(locals())
    return Y, W


def _l_bfgs_direction(G, h, s_list, y_list, r_list, precon, lambda_min):
    q = G.copy()
    a_list = []
    for s, y, r in zip(reversed(s_list), reversed(y_list), reversed(r_list)):
        alpha = r * np.sum(s * q)
        a_list.append(alpha)
        q -= alpha * y
    z = solveh(q, h)
    for s, y, r, alpha in zip(s_list, y_list, r_list, reversed(a_list)):
        beta = r * np.sum(y * z)
        z += (alpha - beta) * s
    return -z


if __name__ == '__main__':
    # Generate Laplace signals and a mixing matrix
    N, T = 3, 100000
    rng = np.random.RandomState(1)
    S = rng.laplace(size=(N, T))
    A = rng.randn(N, N)
    # Generate the mixture
    X = np.dot(A, S)
    Y, W = picard(X, verbose=True)
