import numpy as np
import numpy.linalg as linalg


def whitening(Y, mode='sph'):
    '''
    Whitens the data Y using sphering or pca
    '''
    R = np.dot(Y, Y.T) / Y.shape[1]
    U, D, _ = linalg.svd(R)
    if mode == 'pca':
        W = U.T / np.sqrt(D)[:, None]
        Z = np.dot(W, Y)
    elif mode == 'sph':
        W = np.dot(U, U.T / np.sqrt(D)[:, None])
        Z = np.dot(W, Y)
    return Z, W
