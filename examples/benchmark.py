import os

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from ml_ica.tools import whitening, callback
from ml_ica.algorithms import (picard, simple_quasi_newton_ica, truncated_ica,
                               trust_region_ica)


'''
Choose dataset
'''
# dataset = 'eeg'
dataset = 'fmri'

'''
Fetch dataset
'''
rng = np.random.RandomState(0)
filename = os.path.join(os.path.dirname(__file__), dataset + '.mat')
X = loadmat(filename)['X']

'''
Preprocess the signals
'''
X -= np.mean(X, axis=1, keepdims=True)
X, _ = whitening(X)
n_features, n_samples = X.shape

'''
Specify the tolerance and maximum number of iterations
'''
tol = 1e-7
maxiter = 250
# XXX : maxiter should be max_iter

'''
Run each algorithm on the dataset and plot the gradient curves
'''
plt.figure()
algorithm_list = [truncated_ica, trust_region_ica, simple_quasi_newton_ica,
                  picard]
algorithm_names = ['Truncated Newton ICA', 'Trust region ICA',
                   'Simple quasi-Newton ICA', 'Picard']

print('''

Running ica on %s dataset of size %d x %d...

''' % (dataset, n_features, n_samples))

for algorithm, name in zip(algorithm_list, algorithm_names):
    cb = callback(['timing', 'gradient_norm'])
    X_copy = X.copy()
    print('Running  %s ...' % name)
    algorithm(X_copy, verbose=1, callback=cb, tol=tol, maxiter=maxiter)
    gradients = cb['gradient_norm']
    times = cb['timing']
    print('Took %.2g s / %d iterations to reach a gradient norm of %.2g.' %
          (times[-1], len(times), max(gradients[-1], tol)))
    print('Average time per iteration: %.2g sec' % (times[-1] / len(times)))
    plt.semilogy(times, gradients, label=name)
    print('')

plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Gradient norm')
plt.show()
