import numpy as np
import numexpr as ne


def score(Y):
    '''
    Returns the score function evaluated for each sample
    '''
    return ne.evaluate('tanh(Y / 2)')


def score_der(psiY):
    '''
    Returns the derivative of the score
    '''
    return ne.evaluate('(- psiY ** 2 + 1.) / 2.')


def loss(Y, W):
    '''
    Computes the loss function for (Y, W)
    '''
    T = Y.shape[1]
    log_det = np.linalg.slogdet(W)[1]
    logcoshY = np.sum(ne.evaluate('abs(Y) + 2. * log1p(exp(-abs(Y)))'))
    return - log_det + logcoshY / float(T)


def gradient(Y, psiY):
    '''
    Returns the gradient at Y, using the score psiY
    '''
    N, T = Y.shape
    return np.inner(psiY, Y) / float(T) - np.eye(N)


def compute_h(Y, psidY, precon=2):
    '''
    Returns the diagonal coefficients of H 1/ H2 in a N x N matrix
    '''
    N, T = Y.shape
    if precon == 2:
        return np.inner(psidY, Y ** 2) / float(T)
    else:
        Y_squared = Y ** 2
        sigma2 = np.mean(Y_squared, axis=1)
        psidY_mean = np.mean(psidY, axis=1)
        h1 = psidY_mean[:, None] * sigma2[None, :]
        diagonal_term = np.mean(Y_squared * psidY)
        h1[np.diag_indices_from(h1)] = diagonal_term
        return h1


def regularize_h(h, lambda_min, mode=0):
    '''
    Regularizes the hessian approximation h using the constant lambda_min.
    Mode selects the regularization algorithm
    0 -> Shift each eigenvalue below lambda_min to lambda_min
    1 -> add lambda_min x Id to h
    '''
    if mode == 0:
        # Compute the eigenvalues of the Hessian
        eigenvalues = 0.5 * (h + h.T - np.sqrt((h-h.T) ** 2 + 4.))
        # Regularize
        problematic_locs = eigenvalues < lambda_min
        np.fill_diagonal(problematic_locs, False)
        i_pb, j_pb = np.where(problematic_locs)
        h[i_pb, j_pb] += lambda_min - eigenvalues[i_pb, j_pb]
    if mode == 1:
        h += lambda_min
    return h


def solveh(G, h):
    '''
    Returns H^-1 G
    '''
    return (G * h.T - G.T) / (h * h.T - 1.)


def hessian_free(M, Y, psidY, l_reg=0.):
    '''
    Computes the Hessian free product (H + l_reg * Id)M where H is the true
    Hessian, for a N x N matrix M.
    '''
    T = Y.shape[1]
    return l_reg * M + M.T + np.inner(psidY * np.dot(M, Y), Y) / float(T)
