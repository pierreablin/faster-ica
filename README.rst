Faster ICA by preconditioning with Hessian approximations
==========

This repository hosts several second order algorithms to solve maximum likelihood ICA.

The algorithms can be found in ``/algorithms``. They are:

* **Picard** (Preconditioned ICA for Real Data): A preconditioned L-BFGS algorithm. The fastest algorithm of the repository.
* Simple Quasi-Newton method.
* Relative Trust region method.
* Truncated Newton method.


These algorithms have all been rewritten in Python. They call the same gradient, Hessian and likelihood functions, which makes time comparison meaningful.

The algorithms come with a benchmark at ``example/benchmark.py``. This script runs each algorithm on the same real dataset (fMRI or EEG) and times it.


Dependencies
------------

These are the dependencies to run the algorithms:

* numpy (>=1.8)
* matplotlib (>=1.3)
* numexpr (>= 2.0)
* scipy (>=0.19)
Cite
----

If you use this code in your project, please cite `this paper <https://arxiv.org/abs/1706.08171>`_::

    Pierre Ablin, Jean-Francois Cardoso, and Alexandre Gramfort
    Faster ICA by preconditioning with Hessian approximations
