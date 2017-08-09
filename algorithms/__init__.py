"""
This module includes several second order maximum-likelihood ICA algorithms
used for comparison in:

        Pierre Ablin, Jean-Francois Cardoso, and Alexandre Gramfort
        Faster ICA by preconditioning with Hessian approximations
        https://arxiv.org/abs/1706.08171

Authors: Pierre Ablin <pierre.ablin@inria.fr>
         Alexandre Gramfort <alexandre.gramfort@inria.fr>
         Jean-Francois Cardoso <cardoso@iap.fr>

License: BSD (3-clause)
"""

from picard import picard
from simple_quasi_newton import simple_quasi_newton_ica
from trust_region import trust_region_ica
from truncated_newton import truncated_ica
