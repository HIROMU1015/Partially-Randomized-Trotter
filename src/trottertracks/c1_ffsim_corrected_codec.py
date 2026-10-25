"""Dedicated-process ffsim 0.0.83 initialization adapter; no installed edits.

Reuse the proven saved XDF initialization in place of recomputing it, and pack
with real_schur_column_gauge_v1. All optimization mathematics stays in ffsim.
Temporary module bindings are process-local and restored on every exit. This
entry must not share a process with another concurrent ffsim fit.
"""
import importlib
from . import c1_real_orthogonal_codec as codec

VERSION = 'ffsim0083_saved_xdf_real_schur_initialization_v1'
SETTINGS = dict(tol=1e-8, optimize=True, cholesky=True, method='L-BFGS-B',
                options=None, callback=None, diag_coulomb_indices=None,
                return_optimize_result=True)


def fit(g, J, U, expected_x0, *, before_optimizer, after_optimizer):
    """One actual ffsim fitter invocation; hooks observe, never replace the loss.

The saved initial tensors must already have input/SOURCE identity and sector
qualification. Their exact packed x0 must equal that qualified snapshot before
the original optimizer can be entered. There is no legacy-codec fallback.
"""
    import numpy as np
    import scipy.optimize
    import ffsim
    import jax
    if not jax.config.jax_enable_x64:raise ValueError('JAX_X64_REQUIRED')
    if not callable(before_optimizer) or not callable(after_optimizer):
        raise ValueError('DIAGNOSTIC_HOOKS_REQUIRED')
    d=importlib.import_module('ffsim.linalg.double_factorized_decomposition')
    J,U,g=np.array(J,copy=True),np.array(U,copy=True),np.asarray(g)
    rank=len(J);seen=dict(initializer=0,pack=0,optimizer=0);state={}
    if (g.dtype!=np.float64 or g.shape!=(J.shape[-1],)*4 or not np.isfinite(g).all()):
        raise ValueError('REAL_FINITE_TARGET_TENSOR_REQUIRED')
    original_init=d._double_factorized_explicit_cholesky
    original_pack=d.df_tensors_to_params
    original_minimize=scipy.optimize.minimize

    def initializer(target,*,tol,max_vecs):
        seen['initializer']+=1
        if seen['initializer']!=1 or tol!=1e-8 or max_vecs!=rank or not np.array_equal(target,g):
            raise ValueError('SAVED_INITIALIZER_SCOPE')
        return J.copy(),U.copy()

    def pack(a,b,indices,*,real):
        seen['pack']+=1
        if seen['pack']!=1 or indices is not None or real is not True:
            raise ValueError('CORRECTED_CODEC_SCOPE')
        if not np.array_equal(a,J) or not np.array_equal(b,U):
            raise ValueError('SAVED_INITIALIZER_CHANGED')
        J2,U2,x,report,_=codec.roundtrip(a,b)
        if not np.array_equal(x,expected_x0):raise ValueError('QUALIFIED_X0_CHANGED')
        state.update(x0=x.copy(),J=J2,U=U2,codec=report)
        return x

    def minimize(fun,x0,**kwargs):
        seen['optimizer']+=1
        if (seen!={'initializer':1,'pack':1,'optimizer':1}
                or not np.array_equal(x0,state['x0'])
                or kwargs!=dict(method='L-BFGS-B',jac=True,callback=None,options=None)):
            raise ValueError('OPTIMIZER_INITIALIZATION_OR_SETTINGS_CHANGED')
        before_optimizer(fun,x0,state)
        # Recheck after the diagnostic hook, so it cannot silently alter x0.
        if not np.array_equal(x0,state['x0']):raise ValueError('X0_MUTATED_BY_HOOK')
        result=original_minimize(fun,x0,**kwargs)
        after_optimizer(fun,result)
        return result

    try:
        d._double_factorized_explicit_cholesky=initializer
        d.df_tensors_to_params=pack
        scipy.optimize.minimize=minimize
        result=ffsim.linalg.double_factorized(g,max_vecs=rank,**SETTINGS)
        if seen!={'initializer':1,'pack':1,'optimizer':1}:raise ValueError('FITTER_ROUTE_NOT_USED')
        return result
    finally:
        d._double_factorized_explicit_cholesky=original_init
        d.df_tensors_to_params=original_pack
        scipy.optimize.minimize=original_minimize
