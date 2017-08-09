"""
Python translation of the relative trust-region ICA algorithm.
Reference:
H. Choi and S. Choi
"A relative trust-region algorithm for independent component analysis"
"""

# Authors: Pierre Ablin <pierre.ablin@inria.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Jean-Francois Cardoso <cardoso@iap.fr>
#
# License: BSD (3-clause)

from __future__ import print_function
import numpy as np
from time import time

from ..tools import loss, gradient, compute_h, score, score_der


def invert_z(h, G):
    '''
    Computation of h^-1 G, as the authors do.
    '''
    Tr = h + h.T
    dt = h * h.T - 1.
    Tr /= 2.
    sq = np.sqrt(Tr ** 2 - dt)
    lam1 = Tr - sq
    lam2 = Tr + sq
    ss = lam1 - h
    norms = np.sqrt(1. + ss ** 2)
    c1 = (G + ss * G.T) / np.abs(lam1)
    c2 = (ss * G - G.T) / np.abs(lam2)
    X = c1 - ss * c2
    X /= norms
    diag_ind = np.diag_indices_from(X)
    X[diag_ind] = G[diag_ind] / (1. + h[diag_ind])
    return X


def inner(A, B):
    '''
    Frobenius matrix inner product
    '''
    return np.dot(A.ravel(), B.ravel())


def norm(A):
    '''
    Frobenius matrix norm
    '''
    return np.sqrt(inner(A, A))


def trust_region_ica(X, maxiter=200, tol=1e-7, lambda_min=0.01, verbose=0,
                     callback=None):
    '''
    Trust region method for ICA.
    Python translation of the code from:

    H. Choi and S. Choi
    "A relative trust-region algorithm for independent component analysis"
    '''
    eps = 1e-15
    N, T = X.shape
    W = np.eye(N)
    RTT = 0.1
    Dmax = 10.
    delta = 1.
    Y = X.copy()
    objective = loss(Y, W)
    t0 = time()
    for n in range(maxiter):
        timing = time() - t0
        psiY = score(Y)
        psidY = score_der(psiY)
        G = gradient(Y, psiY)
        gradient_norm = np.max(np.abs(G))
        if gradient_norm < tol or delta < 1e-10:
            break
        H2 = compute_h(Y, psidY) - np.eye(N)
        direction = -invert_z(H2, G)
        d_inv = H2 * direction + direction.T
        predredN = -inner(direction, G + 0.5 * d_inv)
        gHg = inner(G, G * H2 + G.T)
        fullstep = False
        if predredN > 0:
            d_norm = norm(direction)
            if d_norm <= delta:
                step = 'fullstep'
                fullstep = True
            else:
                psd = - inner(G, G) / gHg * G
                psd_norm = norm(psd)
                if psd_norm >= delta:
                    direction = - delta / norm(G) * G
                    step = 'cauchy point'
                else:
                    pn_psd = direction - psd
                    a = inner(pn_psd, pn_psd)
                    b = 2. * inner(pn_psd, psd)
                    c = inner(psd, psd) - delta ** 2
                    t = (-b + np.sqrt(b ** 2 - 4. * a * c)) / 2. / a
                    direction = psd + t * pn_psd
                    step = 'dogleg'
        else:
            step = 'fail'
            if gHg > 5. * eps:
                psd = - inner(G, G) / gHg * G
                psd_norm = norm(psd)
                if psd_norm <= delta:
                    direction = psd
                else:
                    direction = delta / psd_norm * psd
            else:
                direction = -delta / norm(G) * G
        W_new = W + np.dot(direction, W)
        Y_new = np.dot(np.eye(N) + direction, Y)
        prev_obj = objective
        new_objective = loss(Y_new, W_new)
        actred = objective - new_objective
        if fullstep:
            predred = predredN
        else:
            predred = -inner(G, direction)
            predred -= 0.5 * inner(direction, direction * H2 + direction.T)
        RR = actred/(predred + eps)
        update = False
        if RR > RTT:
            update = True
        if RR <= 0.25:
            delta = 0.25 * norm(direction)
        if RR > 0.75 and norm(direction) > (1 - eps) * delta:
            delta = min(2. * delta, Dmax)
        if update:
            Y = Y_new
            W = W_new
            objective = new_objective
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
    trust_region_ica(X, verbose=True)
