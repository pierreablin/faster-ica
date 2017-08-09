"""
Python implementation of the simple quasi_newton ICA algorithm.
Reference:
M. Zibulevsky, "Blind source separation with relative newton method"
"""

# Authors: Pierre Ablin <pierre.ablin@inria.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Jean-Francois Cardoso <cardoso@iap.fr>
#
# License: BSD (3-clause)

from __future__ import print_function
import numpy as np
from time import time

from ..tools import (loss, gradient, compute_h, regularize_h, solveh,
                     score, score_der, linesearch)


def simple_quasi_newton_ica(X, maxiter=200, tol=1e-7, precon=2,
                            lambda_min=0.01, verbose=0, callback=None):
    '''
    Simple quasi-Newton algorithm.
    Highly inspired by:

    M. Zibulevsky, "Blind source separation with relative newton method"
    '''
    Y = X.copy()
    N, T = Y.shape
    W = np.eye(N)
    current_loss = loss(Y, W)
    t0 = time()
    for n in range(maxiter):
        timing = time() - t0
        # Compute the score and its derivative
        psiY = score(Y)
        psidY = score_der(psiY)
        # Compute gradient
        G = gradient(Y, psiY)
        # Stopping criterion
        gradient_norm = np.max(np.abs(G))
        if gradient_norm < tol:
            break
        # Compute the approximation
        H = compute_h(Y, psidY, precon)
        # Regularize H
        H = regularize_h(H, lambda_min)
        # Compute the descent direction
        direction = - solveh(G, H)
        # Do a line_search in that direction
        success, new_Y, new_W, new_loss =\
            linesearch(Y, W, direction, current_loss)
        # If the line search failed, fall back to the gradient
        if not success:
            direction = - G
            _, new_Y, new_W, new_loss =\
                linesearch(Y, W, direction, current_loss, 3)
        # Update
        Y = new_Y
        W = new_W
        current_loss = new_loss
        # Verbose and callback
        if callback is not None:
            callback(locals())
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
    simple_quasi_newton_ica(X, verbose=True)
