import numpy as np
from .derivatives import loss


def linesearch(Y, W, direction, initial_loss=None, n_ls_tries=5):
    '''
    Performs a simple backtracking linesearch in the direction "direction".
    Does n_ls_tries attempts before exiting.
    '''
    N = Y.shape[0]
    W_proj = np.dot(direction, W)
    step = 1.
    if initial_loss is None:
        initial_loss = loss(Y, W)
    for n in range(n_ls_tries):
        new_Y = np.dot(np.eye(N) + step * direction, Y)
        new_W = W + step * W_proj
        new_loss = loss(new_Y, new_W)
        if new_loss < initial_loss:
            success = True
            break
        step /= 2.
    else:
        success = False
    return success, new_Y, new_W, new_loss, step
