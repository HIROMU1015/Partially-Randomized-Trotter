"""Explicit guarantee scopes and fixed-Rust diagnostics;no molecular analysis."""
from fractions import Fraction
import math

from . import c_pr_aware_df_zsxx_enclosure as z

VERSION='P0_guarantee_strategy_fixed_pair_v1'
SCOPE='C_PR_AWARE_DF_B1_FIXED_SAVED_CIRCUITS_V1'
FAMILY='artifacts/research_c_pr_aware_df_guarantee_strategy/2026-10-27'


def rz_capture(theta):
    """Actual Rust StandardGate matrix;not Python RZ.to_matrix semantics."""
    from flint import ctx
    from qiskit._accelerate.circuit import StandardGate
    z.runtime_model();theta=float(theta)
    if not math.isfinite(theta):raise ValueError('FINITE_RZ_ANGLE')
    M=StandardGate.RZGate._to_matrix([theta])
    with z._BALL_LOCK,ctx.workprec(z.PRECISION):
        residual=z._norm(z.gate_matrix('rz',theta)-z._matrix(M))
    return dict(status='VALIDATED_CAPTURED_RUST_RZ',theta_hex=theta.hex(),
        matrix_hex=[[[float(c.real).hex(),float(c.imag).hex()] for c in row] for row in M],
        operator_distance=residual,uniform_matrix_error_a=None,
        input_domain='one fixed binary64 angle',phase_fit=False)


def uniform_model_status(model):
    """Neither comments/test ULPs nor captured maxima are platform proofs."""
    if model is None:return dict(status='UNRESOLVED',uniform_a=None,uniform_zeta=None)
    if model.get('kind') in ('few_ULP','random_sampling_max_ULP','captured_max','ideal_template_image','held_output_box'):
        raise ValueError('NOT_A_UNIFORM_RUST_ERROR_MODEL')
    # No validated uniform platform model has been established in this task.
    raise ValueError('UNESTABLISHED_UNIFORM_MODEL')


def zero_failure_tail_limit(samples,alpha,*,iid,fresh_holdout,certified_decisions):
    """Exact binomial zero-failure upper at confidence1-alpha,under assumptions.

    Design formula only. No new trajectories or stored outcomes are analyzed.
    Certification interval straddles must count as failures/unresolved,not pass.
    """
    from flint import ctx
    if type(samples) is not int or samples<1:raise ValueError('POSITIVE_SAMPLES')
    a=z.rational(alpha)
    if not 0<a<1:raise ValueError('CONFIDENCE_RANGE')
    if not (iid is True and fresh_holdout is True and certified_decisions is True):
        return dict(status='UNRESOLVED',tail_probability_upper=None)
    with z._BALL_LOCK,ctx.workprec(z.PRECISION):
        bound=1-(z._real(a).log()/samples).exp()
        upper=z._fraction(bound.upper())
    return dict(status='CONDITIONAL_BINOMIAL_DESIGN_BOUND',tail_probability_upper_exact=str(upper),
        model='IID Bernoulli certificate-exceedance indicators;zero failures in fixed held-out n',
        not_uniform=True,PCG64_IID_verified=False,CA_achieved=None,shots=None,total_resources=None)


def finite_claim(checked_ids,execution_ids):
    """All executed occurrences must reference their own validated instance.

    Repeating a certified instance is covered as an instance,but changes a PR
    IID law if used instead of independent trajectories. No law proof follows.
    """
    missing=sorted(set(execution_ids)-set(checked_ids))
    return dict(status='UNRESOLVED_UNCHECKED_EXECUTIONS' if missing else 'FINITE_EXECUTION_SCOPE_ONLY',
        missing_ids=missing,whole_law_bound=None,IID_law_preserved=None,
        final_CA=None,guaranteed_shots=None,total_resources=None)
