"""B2 reference-only error transport; passive import and preflight.

No molecular loader, reference engine, compiler or accuracy solver is added.
Local error bounds are scientific inputs, not inferred from agreement tests.
Arb helpers reuse the existing literal CAR arithmetic without its polar target.
"""
from copy import deepcopy
from fractions import Fraction as F
import importlib.metadata
from pathlib import Path

from . import c1_stage1_execution as io
from . import c_b1_b2_scope_integration as integration

BASE = '90a2a7baf07b98108111126504d202294a884af5'
FAMILY = 'artifacts/research_c_reference_numerical_uncertainty/2026-11-03'
CONTRACT = 'docs/research/research_c_reference_numerical_uncertainty_contract_unapproved_20261103.json'
ENTRY = 'scripts/prepare_research_c_reference_numerical_uncertainty.py'
LEVELS = ('observed_numerical_residual', 'precision_sensitivity',
          'numerical_uncertainty_estimate', 'validated_numerical_bound',
          'rigorous_enclosure', 'unknown')
TERMS = dict(reference_bridge=('reference_representation_bridge',),
             reference_uncertainty=('reference_evaluation_rounding',
                                    'own_ground_oracle_relation'))
OBLIGATIONS = ('reference_target_fixed', 'local_error_coverage_complete',
               'state_oracle_relation_bounded', 'raw_normalization_covered',
               'bound_arithmetic_outward')
BLOCKERS = (
    'Exact saved-bit CAR target versus reconstructed mathematical CAR/atom target must be bound explicitly',
    'Lambda/tau/atan/attenuation and literal nonunitary U corrections need a complete disjoint allocation',
    'Saved ground vector to own-DF ground oracle relation has no qualified state-error bound',
    'End-to-end precision/backend/call graph for those missing terms is not frozen',
)


def _q(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, F)):
        raise ValueError('EXACT_NONNEGATIVE_RATIONAL')
    value = F(value)
    if value < 0:
        raise ValueError('EXACT_NONNEGATIVE_RATIONAL')
    return value


def _up(value):
    """Outward dyadic rounding for the bound algebra; not an H4 precision policy."""
    value = _q(value)
    scale = 1 << 128
    return F(-(-value.numerator * scale // value.denominator), scale)


def product_error(left_error, right_error, left_norm, right_norm, local_error):
    """Norm bounds a,b refer to represented inputs; local_error covers fl(A B).

    ||fl(Ahat Bhat)-AB|| <= a eB + b eA + eA eB + eta.
    Unknown operation or operand uncertainty stays unknown, including eta.
    """
    if any(x is None for x in (left_error, right_error, left_norm, right_norm, local_error)):
        return None
    ea, eb, a, b, eta = map(_q, (left_error, right_error, left_norm, right_norm, local_error))
    return _up(a*eb + b*ea + ea*eb + eta)


def round_transport(kernel_error, both_kernel_norm_upper, squaring_errors,
                    *, state_norm_upper, state_error, expectation_errors):
    """13 raw-kernel rounds with distinct coherent and evaluation errors.

    alpha bounds BOTH K and exact represented Khat, without unitarity/clamp.
    C_m bounds Khat**(2**m)-K**(2**m); R_m bounds numerical power evaluation.
    C' <= 2 a C; R' <= (2 a+R)R+eta; a' <= a*a.
    The supplied raw kernels include phase/attenuation exactly once.
    """
    if len(squaring_errors) != 12 or len(expectation_errors) != 13:
        raise ValueError('TWELVE_SQUARES_THIRTEEN_EXPECTATIONS')
    for x in (kernel_error, both_kernel_norm_upper, state_norm_upper, state_error,
              *squaring_errors, *expectation_errors):
        if x is not None:
            _q(x)
    alpha = None if both_kernel_norm_upper is None else _up(max(F(1), _q(both_kernel_norm_upper)))
    coherent = None if kernel_error is None else _up(kernel_error)
    arithmetic = F(0)  # no power multiplication at m0; kernel error is separate
    s = None if state_norm_upper is None else _q(state_norm_upper)
    d = None if state_error is None else _q(state_error)
    rows = []
    for m in range(13):
        op = None if coherent is None or arithmetic is None else _up(coherent + arithmetic)
        terms = (op, alpha, s, d, expectation_errors[m])
        signal = (None if any(x is None for x in terms) else
                  _up(s*s*op + alpha*d*(2*s+d) + _q(expectation_errors[m])))
        rows.append(dict(m=m, coherent_occurrences=1 << m, numerical_squarings=m,
                         kernel_error_amplification=None if coherent is None else str(coherent),
                         power_evaluation_rounding=None if arithmetic is None else str(arithmetic),
                         raw_operator_error=None if op is None else str(op),
                         operator_norm_upper=None if alpha is None else str(alpha),
                         raw_complex_signal_error=None if signal is None else str(signal),
                         qualification='CONDITIONAL_ERROR_TRANSPORT_ONLY'))
        if m < 12:
            eta = squaring_errors[m]
            arithmetic = (None if alpha is None or arithmetic is None or eta is None else
                          _up((2*alpha+arithmetic)*arithmetic + _q(eta)))
            coherent = None if alpha is None or coherent is None else _up(2*alpha*coherent)
            alpha = None if alpha is None else _up(alpha*alpha)
    return rows


def weighted_events(events, operators, probabilities):
    """Existing signed-event algebra with exact rational weights and Arb.

    Fixed event angle bits are inputs. This helper does not validate that
    those bits represent mathematical atan(lambda*h/r), or unitary atoms.
    """
    from .c_pr_aware_df_fixed_car import Arithmetic
    from . import c_pr_aware_df_zsxx_enclosure as z
    from trotterlib.rte import event_unitary
    if not events or len(events) != len(probabilities):
        raise ValueError('EVENT_PROBABILITY_SCHEMA')
    p = list(map(_q, probabilities))
    if sum(p) != 1:
        raise ValueError('EXACT_NORMALIZED_PROBABILITIES')
    a = Arithmetic()
    out = a.zeros(len(next(iter(operators.values()))))
    for event, probability in zip(events, p, strict=True):
        if event.taylor_order != 0 or event.phase != 1 or event.product_component_ids:
            raise ValueError('FIXED_K0_EVENT_REQUIRED')
        out = out + z._real(probability)*event_unitary(event, operators, arithmetic=a)
    return out


def enclosed_rounds(raw_kernel, state):
    """Evaluate 13 literal-map signals, retaining Arb radii in every square.

    Reuses Arithmetic/Matrix, pack and outward Frobenius _norm. No midpoint
    recentering, SVD, state normalization, oracle assertion or <=2 clamp.
    Caller owns the existing Arb lock/workprec context and input semantics.
    This alone is not a bound against a mathematical CAR or ground oracle.
    """
    from .c_pr_aware_df_fixed_car import pack
    from . import c_pr_aware_df_zsxx_enclosure as z
    a = raw_kernel.ball
    if a.nrows() < 1 or a.nrows() != a.ncols() or state.nrows() != a.nrows() or state.ncols() != 1:
        raise ValueError('RAW_KERNEL_STATE_SHAPE')
    out = []
    for m in range(13):
        signal = state.conjugate().transpose()*a*state
        out.append(dict(m=m, signal_rectangle=pack(signal)[0],
                        norm=z._norm(a),
                        target='literal input raw kernel and state only',
                        own_ground_oracle_error=None))
        if m < 12:
            a = a*a
    return out


def evidence_record(quantity, values, *, identity, qualification, provenance,
                    unresolved=(), conditions=None, rounding_enclosed=False):
    """Thin wrapper accepted by the existing integration adapter.

    Conditions and error bounds are supplied evidence, never proof of their
    correctness. An observed value is kept under value, never under bound.
    """
    if identity != integration.identity():
        raise ValueError('CANDIDATE_SCIENTIFIC_SOURCE_MODEL_COMPILER')
    if quantity not in TERMS or qualification not in LEVELS or len(values) != 13 or not provenance:
        raise ValueError('B2_REFERENCE_EVIDENCE_SCHEMA')
    values = [None if x is None else str(_q(x)) for x in values]
    conditions = dict(conditions or {})
    qualified = qualification in ('validated_numerical_bound', 'rigorous_enclosure')
    if qualified and (unresolved or any(x is None for x in values) or
                      any(conditions.get(k) is not True for k in OBLIGATIONS) or
                      not conditions.get('basis')):
        raise ValueError('UNCLOSED_MODEL_OR_UNKNOWN_IS_NOT_A_BOUND')
    if qualification == 'rigorous_enclosure' and not rounding_enclosed:
        raise ValueError('ROUNDING_ENCLOSURE_REQUIRED')
    kind = ('rigorous_bound' if qualification == 'rigorous_enclosure' else
            'numerical_bound' if qualified else qualification)
    rows = [dict(m=m, quantity=quantity, norm='raw_complex_signal', value=x,
                 bound=x if qualified else None, kind=kind,
                 rounding_enclosed=bool(rounding_enclosed and qualified),
                 provenance=provenance,
                 **({'reference_uncertainty_included': False} if quantity == 'reference_bridge' else {}))
            for m, x in enumerate(values)]
    return dict(identity=deepcopy(identity), quantity=quantity, norm='raw_complex_signal',
                included_error_terms=list(TERMS[quantity]), space='sector36_reference',
                phase_sensitive=True, coherent_model='independent_IID_kernels',
                qualification='bound' if qualified else qualification,
                numerical_qualification=qualification, scope='model_reference', rows=rows,
                reference_target='saved_CAR_raw_finite_RTE',
                raw_conversion='||psi_hat||^2*operator_error + ||operator||*d*(2||psi_hat||+d) + expectation_rounding',
                error_model=conditions, unresolved=list(unresolved), provenance=provenance,
                proofs_verified_by_schema=False, actual_CA_achieved=None,
                PCG64_joint_stream_qualified=False, mandatory_stop=True)


def fixed():
    return dict(scope='C_REFERENCE_NUMERICAL_UNCERTAINTY_B2_V1', base_commit=BASE,
                status='NOT_READY', executable=False,
                candidate_sha256=integration.identity()['candidate'],
                stage_A_scientific_SOURCE=integration.STAGE_A_SOURCE,
                stage_A_result=integration.STAGE_A, model=deepcopy(integration.MODEL),
                reference_space='sector36', reference_target='saved_CAR_raw_finite_RTE',
                rounds=list(range(13)), preferred_method='A_precision_sensitivity_plus_B_local_error_transport',
                enclosure_reuse='C_optional_existing_literal_CAR_Arb192',
                science_precision_policy=None, call_limits=None, maximum_logical_units=None,
                entry=ENTRY, fresh_output=FAMILY+'/run01', retry=0, resume=0,
                partial_failure='stop dependent work;preserve partial output;no fallback or automatic retry',
                mandatory_stop=True, Stage_C='NOT_READY',
                reference_numerical_uncertainty=None, rigorous_bridge_bound=None,
                blockers=list(BLOCKERS), resource_caps=dict(CPU=None,RAM=None,workers=None,wall=None))


def preflight(root, path):
    """JSON, package metadata and byte identity only; no NPZ decode or numerics."""
    root = Path(root).resolve()
    c = io.read(path)
    if any(c.get(k) != v for k, v in fixed().items()):
        raise ValueError('FIXED_B2_NOT_READY_DESIGN')
    inherited = integration.preflight(root, root/integration.CONTRACT)
    if io.sha(root/c['source_manifest']) != c['source_manifest_sha256']:
        raise ValueError('B2_SOURCE_MANIFEST')
    source = io.read(root/c['source_manifest'])
    for name, want in source['files'].items():
        p = (root/name).resolve()
        if not p.is_relative_to(root) or io.sha(p) != want:
            raise ValueError('B2_SOURCE:'+name)
    for name, want in c['packages'].items():
        if importlib.metadata.version(name) != want:
            raise ValueError('B2_PACKAGE:'+name)
    if (root/c['fresh_output']).exists():
        raise ValueError('FRESH_NO_RETRY_RESUME')
    return dict(status='PASSIVE_TECHNICAL_CHECKS_PASS', execution_ready=False,
                method_status='NOT_READY', source_files=len(source['files']),
                inherited_source_files=inherited['source_files'],
                inherited_saved_inputs=inherited['saved_inputs'],
                scientific_calls=0, Stage_C_calls=0, maximum_logical_units=None,
                reference_numerical_uncertainty=None, rigorous_bridge_bound=None,
                actual_CA_achieved=None, mandatory_stop=True)


def run_science(*, execute_science=False):
    if not execute_science:
        raise PermissionError('SEPARATE_FIXED_SCIENTIFIC_PERMISSION_REQUIRED')
    raise NotImplementedError('NOT_READY_REFERENCE_TARGET_STATE_MODEL_AND_CALL_GRAPH')
