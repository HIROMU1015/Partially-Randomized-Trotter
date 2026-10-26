"""Full input/output Frobenius pilot; no SVD or molecular work on import.

Uses existing residual norms and reference-norm guards. The small propagation
adapter retains norm growth of BOTH represented matrices and the shells.
Floating-point residuals are not rounding enclosures.
"""
import math
from . import c1_stage3_deterministic_accuracy as existing
from . import c1_subspace_synthesis_bound as bounds
from .c1_stage3_molecular import component_check

VERSION = 'full_fock_frobenius_both_norm_growth_v1'
TOLERANCES = dict(operator_component_abs=1e-10, unitarity_component_abs=1e-10)


def segment(raw, compiled, reference, label):
    # Existing numerical_segment is SVD-free. Its capped structural-unitary
    # surrogate is diagnostic only; propagate the DIRECT compiled/ref distance.
    report = existing.numerical_segment(raw, compiled, reference, TOLERANCES, label)
    direct = component_check(compiled, reference, 1e-10, label='compiled/ref:'+label)
    info = {name:bounds.reference_info(matrix) for name,matrix in
            [('raw',raw),('compiled',compiled),('reference',reference)]}
    epsilon = direct['frobenius']
    if not math.isfinite(epsilon) or epsilon < 0: raise ValueError('INVALID_FROBENIUS')
    return dict(**report, compiled_reference=direct,
        compiled_reference_frobenius=epsilon,
        operator_bound_pilot=math.nextafter(epsilon,math.inf) if epsilon else 0.,
        norm_upper_pilot=max(x['reference_norm_upper_pilot'] for x in info.values()),
        norm_guards=info, input_dimension=len(reference), output_dimension=len(reference),
        coverage='all full-Fock controlled input columns and all output components',
        evidence_level=bounds.LEVEL, floating_point_rounding_enclosed=False,
        rigorous_operator_bound=None, global_phase_fit=False)


def propagate(kernel, shells, *, s, m, saved_pf_error):
    if s not in (1,4) or type(m) is not int or not 0<=m<=12:
        raise ValueError('REGISTERED_ROUND_REQUIRED')
    if set(shells) != {'head','X','Y'}: raise ValueError('SHELL_COVERAGE')
    records=[kernel,*shells.values()]
    if (saved_pf_error is None or any(x.get('operator_bound_pilot') is None or
            x.get('norm_upper_pilot') is None for x in records)):
        return dict(s=s,m=m,status='UNRESOLVED_SYNTHESIS_OR_PF',
            complex_signal_bound_pilot=None,CA_achieved=None,shots=None,total_cost=None)
    values=[saved_pf_error]+[x['operator_bound_pilot'] for x in records]
    if any(not math.isfinite(x) or x<0 for x in values): raise ValueError('INVALID_ERROR')
    alpha=max(x['norm_upper_pilot'] for x in records)
    if not math.isfinite(alpha) or alpha<1: raise ValueError('INVALID_NORM_GUARD')
    n=s*(1<<m);growth=alpha**n
    # Normalize both operators by alpha. Existing unitary telescoping then
    # gives min(2,n*epsilon/alpha); rescale by alpha**n. No matrix is changed.
    scaled=bounds.repeat_bound(kernel['operator_bound_pilot']/alpha,n,
        reference_invariant=True,reference_norm_upper=1.)
    repeated=growth*scaled['operator_bound_pilot']
    # T C^n H - T0 W^n H0: telescope head/kernel/tail exactly once.
    # All represented segment norms <= alpha, so no hidden unitary assumption.
    axis={name:math.fsum([alpha*growth*shells['head']['operator_bound_pilot'],
        alpha**2*repeated,alpha*growth*shells[name]['operator_bound_pilot']]) for name in ('X','Y')}
    signal=2*alpha**2*growth*math.hypot(axis['X'],axis['Y'])
    combined=math.fsum([saved_pf_error,signal])
    if not all(math.isfinite(x) for x in [growth,repeated,signal,combined]):
        raise ValueError('PROPAGATION_OVERFLOW')
    phase=math.asin(combined) if combined<1 else None
    return dict(s=s,m=m,occurrences=n,status='POTENTIALLY_PHASE_ELIGIBLE_NUMERICAL'
        if phase is not None and phase<=math.pi/12 else 'INFEASIBLE_REGISTERED_BOUND',
        alpha=alpha,norm_growth_pilot=growth,kernel_repeat_bound_pilot=repeated,
        axis_state_distance_pilot=axis,complex_signal_bound_pilot=signal,
        saved_PF_operator_error=saved_pf_error,combined_complex_error_pilot=combined,
        phase_bias_bound_pilot=phase,registered_phase_room=math.pi/12,
        evidence_level=bounds.LEVEL,rigorous_signal_error_bound=None,
        floating_point_rounding_enclosed=False,CA_achieved=None,shots=None,total_cost=None)
