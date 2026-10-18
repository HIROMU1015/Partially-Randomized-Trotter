"""H4/H6 comparison primitives. No scientific imports or implicit real execution.

Counts refer to the pinned explicit decomposition, never compiled-cost bounds.
Import the existing counter by file to avoid the scientific package initializer.
"""
from __future__ import annotations
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

_spec = importlib.util.spec_from_file_location('_prtc_counter', Path(__file__).with_name('native_df_gate_model.py'))
counter = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = counter
_spec.loader.exec_module(counter)
NAMESPACE = 'PR_THEORY_COMPILED_H46_V1'
MODELS = ('A', 'B', 'NATIVE_MIXED')
METRICS = ('RZ', 'CX')
IDENTITY_FIELDS = ('system', 'input', 'hamiltonian', 'state', 'DF', 'fragment_order',
                   'preparation', 'basis_variant', 'prefix', 'q', 'r', 'K', 'formula',
                   'T', 'control', 'wrapper', 'scalar', 'P', 'compiler', 'profile',
                   'sampling_phase', 'law', 'seed_namespace')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def number(value, *, positive=False):
    if isinstance(value, bool) or value is None:
        raise ValueError('MISSING_OR_INVALID_COST')
    try:
        v = Fraction(str(value))
    except (ValueError, ZeroDivisionError, TypeError, OverflowError) as e:
        raise ValueError('NONFINITE_OR_INVALID_COST') from e
    if v < 0 or (positive and v == 0):
        raise ValueError('NONPOSITIVE_DENOMINATOR_OR_COST')
    return v


def identity_check(left, right, *, conditional=False):
    for name in IDENTITY_FIELDS:
        if name not in left or name not in right or left[name] is None or right[name] is None:
            raise ValueError('MISSING_IDENTITY:'+name)
        if left[name] != right[name]:
            raise ValueError('IDENTITY_MISMATCH:'+name)
    if left['system'] not in ('H4','H6') or left['control'] != 'symmetric_directional' or left['wrapper'] != 'full_hadamard':
        raise ValueError('UNSUPPORTED_CIRCUIT_SCOPE')
    if left['P'] != 0:
        raise ValueError('P_ZERO_REQUIRED')
    if left.get('numeric_envelope') != 'VERIFIED' or right.get('numeric_envelope') != 'VERIFIED':
        if not conditional or left.get('critical_metadata_bound') is not True or right.get('critical_metadata_bound') is not True:
            raise ValueError('FULL_ENVELOPE_OR_EXPLICIT_CONDITIONAL_POLICY_REQUIRED')
        return 'CONDITIONALLY_ALIGNED'
    return 'EXACTLY_ALIGNED'


def compare_row(row, *, conditional=False):
    """Pure numerical operation; CLI calls this only after new grant verification.

    Lower/upper counts are retained. No midpoint, tail-zero filling or fitting.
    A residual has no causal attribution without separate intervention evidence.
    """
    alignment = identity_check(row['theory_identity'], row['compiled_identity'], conditional=conditional)
    if row.get('model') not in MODELS or row.get('axis') not in ('cosine','sine') or row.get('metric') not in METRICS:
        raise ValueError('MODEL_AXIS_METRIC')
    if row.get('theory_axis') != row['axis'] or row.get('compiled_axis') != row['axis']:
        raise ValueError('AXIS_BINDING')
    if row.get('theory_metric') != row['metric'] or row.get('compiled_metric') != row['metric']:
        raise ValueError('METRIC_BINDING')
    if row.get('lowering_profile') != counter.PROFILE:
        raise ValueError('EXPLICIT_LOWERING_PROFILE_REQUIRED')
    kind = row.get('theory_kind')
    if kind == 'LAW_EXPECTATION':
        if row.get('all_positive_orders_included') is not True:
            raise ValueError('POSITIVE_RARE_ORDERS_MUST_NOT_BE_DROPPED')
        interpretation = 'EXPECTATION_VS_FINITE_SAMPLE_NOT_COMPILER_ONLY'
    elif kind == 'PAIRED_EVENT_SAMPLE':
        if not row.get('theory_event_digest') or row['theory_event_digest'] != row.get('compiled_event_digest'):
            raise ValueError('EVENT_SEQUENCE_MISMATCH')
        interpretation = 'PAIRED_SAME_EVENTS_STILL_DECOMPOSITION_SPECIFIC'
    elif kind == 'DETERMINISTIC':
        interpretation = 'DECOMPOSITION_VS_COMPILED'
    else:
        raise ValueError('THEORY_ESTIMAND_REQUIRED')
    if type(row.get('sample_size')) is not int or row['sample_size'] < 1:
        raise ValueError('SAMPLE_SIZE_REQUIRED')
    lo, hi = number(row.get('theory_lower'),positive=True), number(row.get('theory_upper'),positive=True)
    observed = number(row.get('compiled_mean'),positive=True)
    if lo > hi:
        raise ValueError('REVERSED_THEORY_INTERVAL')
    return dict(model=row['model'],axis=row['axis'],metric=row['metric'],alignment=alignment,
        rho_lower=str(observed/hi),rho_upper=str(observed/lo),
        residual_lower=str((observed-hi)/observed),residual_upper=str((observed-lo)/observed),
        interpretation=interpretation,attribution='UNATTRIBUTED',population_mean_certified=False,
        compiled_upper_bound_claim=False)


def unique_observations(rows):
    """Repeated models/endpoints/epsilons do not multiply a compiled observation."""
    seen = {}
    for row in rows:
        key = row.get('compiled_observation_id')
        if not key:
            raise ValueError('STABLE_COMPILED_OBSERVATION_ID_REQUIRED')
        value = (row['compiled_identity'],row['axis'],row['metric'],row['compiled_mean'],row['sample_size'])
        if key in seen and seen[key] != value:
            raise ValueError('CONFLICTING_REUSED_COMPILED_OBSERVATION')
        seen[key] = value
    return len(seen)


def paired_statistics(values):
    """Descriptive artificial/saved samples, not a population interval."""
    if not values or len({sample_id for sample_id,_ in values}) != len(values):
        raise ValueError('EMPTY_OR_DUPLICATE_SAMPLE')
    v = [float(number(x,positive=True)) for _,x in values]
    if any(not math.isfinite(x) for x in v):
        raise ValueError('NONFINITE_SAMPLE')
    return dict(n=len(v),mean=statistics.mean(v),standard_error=statistics.stdev(v)/math.sqrt(len(v)) if len(v)>1 else None,
                population_mean_certified=False)


def weighted_cost(pair, shots):
    if set(pair) != {'cosine','sine'} or set(shots) != {'cosine','sine'}:
        raise ValueError('PAIRED_AXES_REQUIRED')
    if any(type(n) is not int or n <= 0 for n in shots.values()):
        raise ValueError('PRESERVED_POSITIVE_INTEGER_SHOTS_REQUIRED')
    return sum(number(pair[a],positive=True)*shots[a] for a in pair)


def pf_pr_comparison(pr, pf):
    """The call's context must bind the same epsilon/bias/shot source and P."""
    for key in ('system','input','hamiltonian','state','DF','fragment_order','T','control','wrapper','P','compiler','profile',
                'epsilon','bias_policy','accounting_source'):
        if key not in pr or key not in pf or pr[key] is None or pr[key] != pf[key]:
            raise ValueError('PF_PR_CONTEXT_MISMATCH:'+key)
    # Preserve candidate-specific shot numbers; never force PR and PF to use the
    # same shots or rederive them from the cost model.
    return weighted_cost(pr['costs'],pr['shots'])/weighted_cost(pf['costs'],pf['shots'])


def native_expectation(*, model, bases, primitive_factory, T, q, ancilla, scalar_phase,
                       formula='2nd',components=None,orders=None,r=None,tau=None):
    """Counter integration. Call with real metadata only under separate approval.

    NATIVE_MIXED = prefix A, closed tail B. Never interpolate uniform totals.
    No full Qiskit object, molecule, state, event draw or transpile is created.
    """
    if model not in MODELS:
        raise ValueError('MODEL_REQUIRED')
    prefix_model = 'A' if model=='NATIVE_MIXED' else model
    if components is None:
        c = counter.deterministic_circuit(bases,primitive_factory,T,q,formula,
            'symmetric_directional',ancilla,scalar_phase,prefix_model)
        return {axis:counter.count(counter.hadamard_wrapper(c,ancilla,a)) for axis,a in [('cosine','X'),('sine','Y')]}
    if formula != '2nd':
        raise ValueError('PR_REQUIRES_PINNED_S2_POLICY')
    prefix = counter.partial_prefix_circuit(bases,primitive_factory,T,q,ancilla,
                'symmetric_directional',scalar_phase,prefix_model)
    tail = counter.expected_rte_occurrence(components,orders,r,tau,ancilla,
                'B' if model=='NATIVE_MIXED' else model)
    return {axis:counter.expected_partial_trajectory(tail,q,prefix,ancilla,a)
            for axis,a in [('cosine','X'),('sine','Y')]}


def verify_grant(plan, grant, *, source, input_sha):
    if (grant.get('status')!='APPROVED' or grant.get('approved') is not True or
        grant.get('runtime_authorization') is not True or plan.get('sealed') is not True):
        raise ValueError('NEW_APPROVED_H46_GRANT_REQUIRED')
    if source != plan.get('SOURCE') or len(source) != 40 or grant.get('SOURCE') != source:
        raise ValueError('SOURCE_BINDING')
    if not isinstance(input_sha,str) or len(input_sha)!=64 or any(x not in '0123456789abcdef' for x in input_sha):
        raise ValueError('INPUT_SHA256_REQUIRED')
    if plan.get('namespace') != NAMESPACE or grant.get('namespace') != NAMESPACE:
        raise ValueError('NEW_NAMESPACE_REQUIRED')
    if grant.get('plan_digest') != digest(plan) or plan.get('input_sha256') != input_sha or grant.get('input_sha256') != input_sha:
        raise ValueError('PLAN_INPUT_BINDING')
    if (grant.get('one_shot') is not True or grant.get('automatic_retry') is not False or
        plan.get('mandatory_stop') is not True or plan.get('next_stage_authorized') is not False):
        raise ValueError('ONE_SHOT_STOP_REQUIRED')
    if plan.get('operations') != ['COMPARE_SAVED_VALUES']:
        raise ValueError('NO_IMPLICIT_THEORY_BUILD_OR_COMPILE')
