# Authors: Pierre Ablin <pierre.ablin@inria.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Jean-Francois Cardoso <cardoso@iap.fr>
#
# License: BSD (3-clause)

from __future__ import print_function
from copy import copy
import numpy as np
import numexpr as ne
from time import time


def picard(X, m=7, maxiter=1000, precon=1, tol=1e-7, lambda_min=0.01,
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

    m : int
        Size of L-BFGS's memory. Typical values for m are in the range 3-15

    maxiter : int
        Maximal number of iterations for the algorithm

    precon : 1 or 2
        Chooses which Hessian approximation is used as preconditioner.
        1 -> H1
        2 -> H2
        H2 is more costly to compute but can greatly accelerate convergence
        (See the paper for details).

    tol : float
        tolerance for the stopping criterion. Iterations stop when the norm
        of the gradient gets smaller than tol.

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
    Y = copy(X)
    s_list = []
    y_list = []
    r_list = []
    current_loss = None
    t0 = time()
    for n in range(maxiter):
        timing = time() - t0
        # Compute the score function
        psiY = ne.evaluate('tanh(Y / 2.)')
        psidY = ne.evaluate('(- psiY ** 2 + 1.) / 2.')  # noqa
        # Compute the relative gradient
        G = np.inner(psiY, Y) / float(T) - np.eye(N)
        # Stopping criterion
        gradient_norm = np.max(np.abs(G))
        if gradient_norm < tol:
            break
        # Update the memory
        if n > 0:
            s_list.append(direction) # noqa
            y = G - G_old  # noqa
            y_list.append(y)
            r_list.append(1. / (np.sum(direction * y)))  # noqa
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
                r_list.pop(0)
        G_old = G # noqa
        # Compute the Hessian approximation and regularize
        h = _hessian(Y, psidY, precon)
        h = _regularize(h, lambda_min)
        # Find the L-BFGS direction
        direction = _l_bfgs_direction(G, h, s_list, y_list, r_list,
                                      precon, lambda_min)
        # Do a line_search in that direction:
        converged, new_Y, new_W, new_loss, alpha =\
            _line_search(Y, W, direction, current_loss, ls_tries, verbose)
        if not converged:
            direction = -G
            s_list, y_list, r_list = [], [], []
            _, new_Y, new_W, new_loss, alpha =\
                _line_search(Y, W, direction, current_loss, ls_tries, 0)
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


def _loss(Y, W):
    '''
    Computes the loss function for Y, W
    '''
    T = Y.shape[1]
    log_det = np.linalg.slogdet(W)[1]
    logcoshY = np.sum(ne.evaluate('abs(Y) + 2. * log1p(exp(-abs(Y)))'))
    return - log_det + logcoshY / float(T)


def _line_search(Y, W, direction, current_loss, ls_tries, verbose):
    '''
    Performs a backtracking line search, starting from Y and W, in the
    direction direction. I
    '''
    N = Y.shape[0]
    projected_W = np.dot(direction, W)
    alpha = 1.
    if current_loss is None:
        current_loss = _loss(Y, W)
    for _ in range(ls_tries):
        Y_new = np.dot(np.eye(N) + alpha * direction, Y)
        W_new = W + alpha * projected_W
        new_loss = _loss(Y_new, W_new)
        if new_loss < current_loss:
            return True, Y_new, W_new, new_loss, alpha
        alpha /= 2.
    else:
        if verbose == 2:
            print('line search failed, falling back to gradient')
        return False, Y_new, W_new, new_loss, alpha


def _l_bfgs_direction(G, h, s_list, y_list, r_list, precon, lambda_min):
    q = copy(G)
    a_list = []
    for s, y, r in zip(reversed(s_list), reversed(y_list), reversed(r_list)):
        alpha = r * np.sum(s * q)
        a_list.append(alpha)
        q -= alpha * y
    z = _solve_hessian(q, h)
    for s, y, r, alpha in zip(s_list, y_list, r_list, reversed(a_list)):
        beta = r * np.sum(y * z)
        z += (alpha - beta) * s
    return -z


def _hessian(Y, psidY, precon):
    '''
    Computes the Hessian approximation
    '''
    T = Y.shape[1]
    # Build the diagonal of the Hessian, a.
    Y_squared = Y ** 2
    if precon == 2:
        h = np.inner(psidY, Y_squared) / float(T)
    elif precon == 1:
        sigma2 = np.mean(Y_squared, axis=1)
        psidY_mean = np.mean(psidY, axis=1)
        h = psidY_mean[:, None] * sigma2[None, :]
        diagonal_term = np.mean(Y_squared * psidY) + 1.
        h[np.diag_indices_from(h)] = diagonal_term
    else:
        raise ValueError('precon should be 1 or 2')
    return h


def _regularize(h, lambda_min):
    '''
    Regularizes h with the level lambda_min
    '''
    # Compute the eigenvalues of the Hessian
    eigenvalues = 0.5 * (h + h.T - np.sqrt((h - h.T) ** 2 + 4.))
    # Regularize
    problematic_locs = eigenvalues < lambda_min
    np.fill_diagonal(problematic_locs, False)
    i_pb, j_pb = np.where(problematic_locs)
    h[i_pb, j_pb] += lambda_min - eigenvalues[i_pb, j_pb]
    return h


def _solve_hessian(G, h):
    '''
    Returns the solution of hX = G
    '''
    return (G * h.T - G.T) / (h * h.T - 1.)


if __name__ == '__main__':
    # Generate Laplace signals and a mixing matrix
    N, T = 3, 100000
    S = np.random.laplace(size=(N, T))
    A = np.random.randn(N, N)
    # Generate the mixture
    X = np.dot(A, S)
    Y, W = picard(X, verbose=True)
