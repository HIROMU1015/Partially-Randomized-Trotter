"""Operator-space diagnostics for saved P0 artifacts, with the original gate.

No propagation, state inference, normalizer, circuit, price or production repair.
The identity component is the normalized trace in the complete Fock space.
"""
import math
import numpy as np

TOL=1e-9


def require(value,message):
    if not value:raise ValueError(message)


def complex_value(value):
    z=complex(value)
    return dict(real=float(z.real),imag=float(z.imag),real_hex=float(z.real).hex(),imag_hex=float(z.imag).hex())


def decomposition(matrix):
    matrix=np.asarray(matrix)
    require(matrix.ndim==2 and matrix.shape[0]==matrix.shape[1] and np.isfinite(matrix).all(),'INVALID_OPERATOR_MATRIX')
    d=len(matrix); scalar=np.trace(matrix)/d
    traceless=matrix-scalar*np.eye(d)
    total=float(np.linalg.norm(matrix));identity=float(math.sqrt(d)*abs(scalar));nonidentity=float(np.linalg.norm(traceless))
    return dict(frobenius=total,identity_coefficient=complex_value(scalar),identity_frobenius=identity,
        nonidentity_frobenius=nonidentity,
        orthogonal_squared_norm_closure=abs(total**2-identity**2-nonidentity**2))


def error_metrics(approximation,target):
    target=np.asarray(target);approximation=np.asarray(approximation)
    require(target.shape==approximation.shape,'OPERATOR_DIMENSIONS_DIFFER')
    norm=float(np.linalg.norm(target)); out=decomposition(approximation-target)
    out.update(target_frobenius=norm,relative_frobenius=out['frobenius']/norm if norm else None,
        absolute_gate=TOL,within_original_absolute_gate=out['frobenius']<=TOL)
    return out


def saved_analysis(tail,nonidentity,jw_identity,saved_scalar):
    tail=np.asarray(tail); nonidentity=np.asarray(nonidentity);eye=np.eye(len(tail))
    return dict(target_tail=decomposition(tail),saved_Pauli_nonidentity=decomposition(nonidentity),
        using_JW_identity=error_metrics(nonidentity+jw_identity*eye,tail),
        using_saved_DF_scalar=error_metrics(nonidentity+saved_scalar*eye,tail),
        JW_identity=float(jw_identity),saved_DF_scalar=float(saved_scalar),
        scalar_difference=float(jw_identity-saved_scalar))


def attribute(default,bridge,shadow,target):
    """A vector telescope, not an additive budget of scalar error norms.

    bridge = no-filter JW on the default Fermion residual;
    shadow = no-filter Fermion + no-filter JW on the same ordered input.
    """
    default=np.asarray(default);bridge=np.asarray(bridge);shadow=np.asarray(shadow);target=np.asarray(target)
    jw=default-bridge;fermion=bridge-shadow;remainder=shadow-target;original=default-target
    closure=float(np.linalg.norm(original-(jw+fermion+remainder)))
    require(closure<=TOL,'ERROR_TELESCOPE_CLOSURE_FAILED')
    before=float(np.linalg.norm(original));after=float(np.linalg.norm(remainder))
    return dict(default=error_metrics(default,target),shadow=error_metrics(shadow,target),
        default_Fermion_then_no_filter_JW=error_metrics(bridge,target),
        JW_partial_sum_loss=decomposition(jw),pre_JW_Fermion_loss=decomposition(fermion),
        combined_threshold_loss=decomposition(default-shadow),
        vector_telescope_closure_frobenius=closure,
        unexplained_remainder_over_original=after/before if before else None,
        norm_contributions_are_not_additive=True,production_dictionary_adopted=False)
