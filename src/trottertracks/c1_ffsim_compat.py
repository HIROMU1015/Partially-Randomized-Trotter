"""Task-local ffsim 0.0.83 bridge for SciPy 1.14.1 matrix logarithms.

Only the dedicated one-fit worker uses this context. It temporarily replaces
ffsim's batch-to-parameters helper in memory and restores it even on failure.
SciPy functions, installed files, and ffsim's fitter are never modified.
"""
from contextlib import contextmanager

VERSION = "ffsim0083_scipy1141_per_matrix_logm_v1"


def batch_logm(mats):
    """Apply the locked SciPy logm separately, retaining batch order/dtype.

The supported input is exactly ffsim's nonempty (n_mats, dim, dim) batch.
No projection, real cast, branch choice, or logm option is added.
    """
    import numpy as np
    import scipy.linalg
    mats = np.asarray(mats)
    if (mats.ndim != 3 or mats.shape[0] == 0 or mats.shape[1] == 0
            or mats.shape[1] != mats.shape[2]):
        raise ValueError("FFSIM_LOGM_EXPECTS_NONEMPTY_SQUARE_BATCH")
    return np.stack([scipy.linalg.logm(mat) for mat in mats])


@contextmanager
def per_matrix_logm():
    import ffsim.linalg.util as util
    original = util.unitaries_to_parameters

    def parameters(mats, real=False):
        return util.antihermitians_to_parameters(batch_logm(mats), real=real)

    util.unitaries_to_parameters = parameters
    try:
        yield
    finally:
        util.unitaries_to_parameters = original


def double_factorized(two_body_tensor, **kwargs):
    """One unchanged ffsim call, with only its logm shape dispatch bridged."""
    import ffsim
    with per_matrix_logm():
        return ffsim.linalg.double_factorized(two_body_tensor, **kwargs)
