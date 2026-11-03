"""Expectation-calibration design adapter; passive imports, no molecular runner.

Bounds below are conditional on the supplied support/enclosure/IID premises.
They do not establish those premises or authenticate scientific evidence.
Old all-word/actual-law qualification and old science artifacts are untouched.
"""
from copy import deepcopy
from decimal import Decimal, localcontext, ROUND_CEILING
from fractions import Fraction as F
from math import isqrt
from pathlib import Path

BASE = '3649b75af191916daf44abb91e14d34c43b83e8f'
FAMILY = 'artifacts/research_c_iid_grid_expected_bias/2026-11-04'
CONTRACT = 'docs/research/research_c_iid_grid_expected_bias_contract_unapproved_20261104.json'
SCOPE = 'C_PR_R32_FRESH_IID_GRID_EXPECTED_BIAS_DESIGN_V1'
ESTIMAND = 'abs(E_q[z_compiled_m]-E_q[z_literal_atom_m])'
TOTAL_ESTIMAND = 'abs(E_q[z_compiled_m]-z_saved_binary64_CAR_m)'
TERMS = ('native_to_q_atom', 'categorical_q_to_p', 'atom_to_mathematical_CAR',
         'mathematical_CAR_to_binary64_reference')


def rational(value):
    if isinstance(value, (float, bool)) or not isinstance(value, (str, int, F)):
        raise ValueError('EXACT_FINITE_SCALAR_REQUIRED')
    return F(value)


def nonnegative(value):
    x = rational(value)
    if x < 0:
        raise ValueError('NONNEGATIVE_BOUND_REQUIRED')
    return x


def probability(value):
    x = rational(value)
    if not 0 < x < 1:
        raise ValueError('PROBABILITY_RANGE')
    return x


def sqrt_upper(value, digits=60):
    """Exact integer square-root enclosure, independent of float/Decimal sqrt."""
    x = nonnegative(value)
    scale = 10**digits
    n = isqrt(x.numerator*scale*scale//x.denominator)
    if F(n*n, scale*scale) < x:
        n += 1
    return F(n, scale)


def log_inverse_upper(delta):
    """Upward division then correctly rounded Decimal.ln plus one successor.

    Unlike Decimal.power, ln is documented correctly rounded HALF_EVEN. The
    successor encloses that rounding; exact Fractions carry all later sums.
    """
    x = 1/probability(delta)
    with localcontext() as ctx:
        ctx.prec = 80
        ctx.rounding = ROUND_CEILING
        y = Decimal(x.numerator)/Decimal(x.denominator)
        return F(y.ln().next_plus())


def hoeffding_radius(width, samples, delta):
    if type(samples) is not int or samples < 1:
        raise ValueError('POSITIVE_FIXED_SAMPLE_COUNT')
    return nonnegative(width)*sqrt_upper(log_inverse_upper(delta)/(2*samples))


def planned_samples(width, tolerance, delta):
    e = nonnegative(tolerance)
    if not e:
        raise ValueError('POSITIVE_TOLERANCE')
    n = nonnegative(width)**2*log_inverse_upper(delta)/(2*e*e)
    return max(1, -(-n.numerator//n.denominator))


def grid_index(word, cdf):
    """No RNG: consume one supplied 53-bit integer against saved binary64 CDF.

    k/2**53 is exactly representable. This is NumPy searchsorted_right,
    including zero-mass bins and exact-boundary hits, not ideal p sampling.
    """
    from bisect import bisect_right
    if type(word) is not int or not 0 <= word < 1 << 53:
        raise ValueError('UNIFORM_53_BIT_WORD')
    x = tuple(F.from_float(v) for v in cdf)
    if not x or x[-1] != 1 or any(not 0 <= v <= 1 for v in x) or any(a > b for a, b in zip(x, x[1:])):
        raise ValueError('FIXED_BINARY64_CDF')
    return bisect_right(x, F(word, 1 << 53))


def sample_enclosure(native_pair, atom_bridge, *, word_id):
    """Triangle two *per-word* full512 legs, never B2's sector mean bound.

    Existing fixed_certificate can supply compiled/raw mathematical gate-pair
    enclosures. The new raw-to-literal-atom bridge must additionally enclose
    CAR construction/exponentials/target mismatch. This adapter does not
    implement or pretend that missing bridge is already available.
    """
    legs = (native_pair, atom_bridge)
    if any(r is None or r.get('upper_exact') is None for r in legs):
        return dict(kind='unknown', upper_exact=None, lower_exact=None)
    for r, leg in zip(legs, ('compiled_to_raw', 'raw_to_literal_atom')):
        if (r.get('scope') != 'per_word_full512' or r.get('word_id') != word_id
                or r.get('leg') != leg or r.get('kind') != 'rigorous_enclosure'
                or not r.get('provenance') or r.get('rounding_enclosed') is not True
                or r.get('phase_fit') is not False):
            raise ValueError('PER_WORD_FULL512_ENCLOSURE_REQUIRED')
    return dict(kind='rigorous_enclosure', lower_exact='0',
        upper_exact=str(sum((nonnegative(r['upper_exact']) for r in legs), F())),
        provenance=[r['provenance'] for r in legs],
        included_legs=['compiled_to_raw', 'raw_to_literal_atom'])


def sampling_gate(policy, count):
    from .c_iid_grid_bridge import MODEL
    if policy.get('model') != MODEL or policy.get('PCG64_joint_stream_qualified') is not False:
        raise ValueError('COUNTERFACTUAL_IID_MODEL_ONLY')
    if (policy.get('origin') != 'fresh_IID_grid_calibration' or policy.get('pool_old_samples') is not False
            or policy.get('adaptive') is not False or policy.get('fixed_samples') != count
            or policy.get('reuse') != 'calibration_only_not_measurement_shots'
            or policy.get('iid_assumed') is not True):
        raise ValueError('FIXED_FRESH_IID_CALIBRATION_REQUIRED')
    ids = policy.get('sample_ids', [])
    if len(ids) != count or len(set(ids)) != count:
        raise ValueError('DISTINCT_CALIBRATION_IDS')
    # Distinct IDs are bookkeeping, never a proof of stream independence.


def mean_upper(samples, *, support, delta, policy):
    """One-sided Hoeffding on true d_i; certified upper endpoints replace d_i.

    Unknown numerical errors remain null. No observed maximum defines support.
    The caller must separately qualify the deterministic population support.
    """
    values = tuple(samples)
    if not values:
        raise ValueError('NONEMPTY_FIXED_SAMPLE')
    sampling_gate(policy, len(values))
    probability(delta)
    if support is None or any(x.get('upper_exact') is None for x in values):
        return dict(status='UNRESOLVED', bound=None, reason='unknown support or per-sample error')
    if support.get('qualification') != 'deterministic_population_bound' or not support.get('provenance'):
        raise ValueError('OBSERVED_MAX_IS_NOT_SUPPORT')
    M = nonnegative(support['upper_exact'])
    upper = []
    for row in values:
        if row.get('kind') != 'rigorous_enclosure' or not row.get('provenance'):
            return dict(status='UNRESOLVED', bound=None, reason='observations are not enclosures')
        a, b = nonnegative(row['lower_exact']), nonnegative(row['upper_exact'])
        if a > b or b > M:
            raise ValueError('INVALID_RANGE_OR_ENCLOSURE')
        upper.append(b)
    u = min(M, sum(upper, F())/len(upper)+hoeffding_radius(M, len(upper), delta))
    return dict(status='CONDITIONAL_EXPECTATION_BOUND', bound=str(u), samples=len(upper),
        delta_cal=str(probability(delta)), support_upper=str(M),
        arithmetic='outward log/sqrt;exact rational sums;conditional on declared true enclosures',
        qualification='probabilistic_expectation', all_word_uniform=False,
        PCG64_joint_stream_qualified=False, actual_CA_achieved=None)


def paired_round_upper(samples, *, axis_support, delta_round, policy):
    """Signed paired X/Y differences; four tails share one round's delta.

    Each row encloses the TRUE paired difference. Pairing X/Y evaluations of
    one word list is a classical variance-reduction coupling, not shared
    measurement shots. No independence between the two coordinates is needed.
    """
    values = tuple(samples)
    if not values:
        raise ValueError('NONEMPTY_FIXED_SAMPLE')
    sampling_gate(policy, len(values))
    d = probability(delta_round)
    if axis_support is None or any(r.get(k) is None for r in values for k in ('X', 'Y')):
        return dict(status='UNRESOLVED', bound=None)
    if axis_support.get('qualification') != 'deterministic_population_bound' or not axis_support.get('provenance'):
        raise ValueError('OBSERVED_RANGE_IS_NOT_SUPPORT')
    a, b = rational(axis_support['lower_exact']), rational(axis_support['upper_exact'])
    if a > b:
        raise ValueError('INVALID_RANGE')
    rad = hoeffding_radius(b-a, len(values), d/4)
    intervals = {}
    for axis in ('X', 'Y'):
        endpoints = []
        for row in values:
            if row.get('kind') != 'rigorous_enclosure' or not row.get('provenance'):
                return dict(status='UNRESOLVED', bound=None)
            lo, hi = map(rational, row[axis])
            if not a <= lo <= hi <= b:
                raise ValueError('INVALID_RANGE_OR_ENCLOSURE')
            endpoints.append((lo, hi))
        lo = max(a, sum((v[0] for v in endpoints), F())/len(values)-rad)
        hi = min(b, sum((v[1] for v in endpoints), F())/len(values)+rad)
        intervals[axis] = (lo, hi)
    bound = sqrt_upper(sum((max(abs(a), abs(b))**2 for a, b in intervals.values()), F()))
    return dict(status='CONDITIONAL_EXPECTATION_BOUND', bound=str(bound),
        axis_intervals={k: list(map(str, v)) for k, v in intervals.items()},
        delta_cal=str(d), qualification='probabilistic_expectation', all_word_uniform=False,
        PCG64_joint_stream_qualified=False, actual_CA_achieved=None)


def propagate_expected(mu, *, norm_upper, shells, m):
    """Both-norm telescoping used by c1_h4_full_synthesis.propagate.

    Standalone outward scalar companion, since the old function explicitly
    returns a binary64 PILOT. For all segment norms <= a>=1 and n=2**m,
    E||K_n...K_1-L_n...L_1|| <= n*a**(n-1)*mu.
    Shell telescoping and |<v,Zv>-<u,Zu>| <= (||u||+||v||)||v-u||
    give the same conservative 2*sqrt(2)*a**(2*n+4)*min(2,n*mu/a)
    kernel term as the old implementation. Shell errors are counted once.
    No mean-unitary or per-word exact-unitarity premise is used.
    """
    if type(m) is not int or not 0 <= m <= 12 or set(shells) != {'head', 'X', 'Y'}:
        raise ValueError('REGISTERED_ROUND_AND_SHELLS')
    if mu is None or norm_upper is None or any(v is None for v in shells.values()):
        return dict(m=m, bound=None, status='UNRESOLVED')
    a = nonnegative(norm_upper)
    if a < 1:
        raise ValueError('BOTH_MAP_NORM_GROWTH_REQUIRED')
    n = 1 << m
    mu = nonnegative(mu)
    e = {k: nonnegative(v) for k, v in shells.items()}
    repeated = a**n*min(F(2), n*mu/a)
    axis = {k: a**(n+1)*e['head']+a*a*repeated+a**(n+1)*e[k] for k in ('X', 'Y')}
    bound = 2*a**(n+2)*sqrt_upper(axis['X']**2+axis['Y']**2)
    return dict(m=m, occurrences=n, bound=str(bound), status='CONDITIONAL_EXPECTATION_PROPAGATION',
        uniform_over_normalized_input_states=True, all_word_uniform=False,
        independent_IID_occurrences_required=True, same_word_repetition_covered=False,
        floating_point_pilot_promoted=False)


def failure_accounting(delta_cal):
    from .c_b1_b2_scope_integration import confidence_schema
    if delta_cal is None:
        return dict(status='CALIBRATION_CONFIDENCE_UNFIXED', unconditional_failure_upper=None)
    d = probability(delta_cal)
    conditional = confidence_schema(measurement_alpha='.01', calibration_delta=str(d),
        reported_alpha='.01', mode='conditional_on_calibration_event')
    union = min(F(1), F(1, 100)+d)
    return dict(calibration_success_lower=str(1-d), conditional_RPE_failure_upper='1/100',
        unconditional_failure_upper=str(union), conditional_policy=conditional,
        joint_reallocation_adopted=False, measurement_policy_changed=False)


def qualify(record, *, additional=()):
    """Separate expectation lane. A connected, disjoint chain must reach R_num.

    The synthesis-only primary estimand is NOT enough to call RPE arithmetic.
    Direct total transport may instead cover all four legs once. Public schema
    checks don't establish mathematical enclosure premises.
    """
    from .c_b1_b2_scope_integration import identity
    if record['identity'] != identity():
        raise ValueError('CANDIDATE_SOURCE_MODEL_COMPILER_IDENTITY')
    if (record.get('scope') != 'probabilistic_expectation' or record.get('quantity') not in (ESTIMAND, TOTAL_ESTIMAND)
            or record.get('norm') != 'raw_complex_signal' or record.get('all_word_uniform') is not False):
        raise ValueError('EXPECTATION_IS_NOT_ALL_WORD_BOUND')
    if ((record['quantity'] == ESTIMAND and record['included_error_terms'] != [TERMS[0]])
            or (record['quantity'] == TOTAL_ESTIMAND and record['included_error_terms'] != list(TERMS))):
        raise ValueError('ESTIMAND_ERROR_TERMS_OR_CHAIN_ORDER')
    all_records = [record, *additional]
    missing = []
    seen = []
    deltas = {}
    for i, r in enumerate(all_records):
        if r['identity'] != record['identity']:
            raise ValueError('TRANSPORT_IDENTITY')
        if r.get('norm') != 'raw_complex_signal':
            raise ValueError('RAW_COMPLEX_UNITS')
        terms = r['included_error_terms']
        if not terms or set(terms)-set(TERMS) or set(seen).intersection(terms) or len(set(terms)) != len(terms):
            raise ValueError('ERROR_DOUBLE_COUNT_OR_WRONG_TERM')
        seen += terms
        if [v['m'] for v in r['rows']] != list(range(13)):
            raise ValueError('ALL_THIRTEEN_ROUNDS')
        if r.get('qualification') != 'bound_from_qualified_enclosures' or r.get('state_coverage') != 'all_normalized_sector_inputs_embedded_full512':
            missing.append('qualified_enclosures_and_state_coverage:'+str(i))
        if not r.get('provenance') or r.get('phase_sensitive') is not True:
            missing.append('phase_shell_provenance:'+str(i))
        if i == 0:
            if r.get('shells_included') is not True:
                missing.append('native_shell_coverage')
            if r.get('fresh_IID_calibration') is not True or r.get('fixed_N_no_optional_stopping') is not True:
                raise ValueError('FRESH_FIXED_N')
            if r.get('coherent_model') != 'independent_IID_kernels':
                raise ValueError('INDEPENDENT_COHERENT_OCCURRENCES')
            for event in r['calibration_events']:
                event_id = event['id']; d = probability(event['delta'])
                if event_id in deltas and deltas[event_id] != d:
                    raise ValueError('CALIBRATION_EVENT_CONFLICT')
                deltas[event_id] = d
            if not deltas:
                raise ValueError('EXPLICIT_JOINT_CONFIDENCE')
            for row in r['rows']:
                events = row.get('calibration_event_ids', [])
                if not events or set(events)-set(deltas):
                    raise ValueError('ROUND_CALIBRATION_EVENT_COVERAGE')
        elif r.get('calibration_events'):
            raise ValueError('ADDITIONAL_RANDOM_CALIBRATION_NEEDS_JOINT_POLICY')
        for row in r['rows']:
            if row.get('bound') is None:
                missing.append('bound:'+str(i)+':m'+str(row['m']))
            else:
                nonnegative(row['bound'])
    if seen != list(TERMS):
        if set(seen) != set(TERMS):
            missing.append('complete_transport_to_binary64_CAR_reference')
        else:
            raise ValueError('CONNECTED_REFERENCE_CHAIN_ORDER')
    if any(x.get('bound') is None for r in all_records for x in r['rows']):
        rows = []
    else:
        rows = [dict(m=m, bound=str(sum((nonnegative(r['rows'][m]['bound'])
                    for r in all_records), F()))) for m in range(13)]
    d = sum(deltas.values(), F())
    probability(d)
    return dict(kind='IID_GRID_PROBABILISTIC_EXPECTATION_BOUND', identity=deepcopy(record['identity']),
        eligible=not missing, missing=missing,
        round_errors=rows if not missing else [], confidence=failure_accounting(d),
        all_word_uniform=False, actual_error_evidence=None, PCG64_joint_stream_qualified=False,
        PCG64_joint_stream_error=None, actual_CA_achieved=None, rigorous_CA=None)


def assess_if_adopted(target, packet, qualification, *, policy_adopted=False):
    if not policy_adopted or not qualification['eligible']:
        return dict(status='NOT_READY_NO_ACCURACY_CALLS', rounds=[], energy=None, actual_CA_achieved=None)
    if qualification['kind'] != 'IID_GRID_PROBABILISTIC_EXPECTATION_BOUND' or packet.get('error_evidence') is not None:
        raise ValueError('SEPARATE_PROBABILITY_QUALIFICATION')
    if packet.get('expectation_calibration_identity') != qualification['identity']:
        raise ValueError('PACKET_CANDIDATE_SOURCE_MODEL_BINDING')
    from . import c1_stage3_accuracy as accuracy
    from .c_pr_aware_df_stage_c import energy
    from .c_pr_aware_df_stage_c2 import upper_float
    plans = []
    if [r['m'] for r in packet['rounds']] != list(range(13)):
        raise ValueError('ROUND_COVERAGE')
    if [r['m'] for r in qualification['round_errors']] != list(range(13)):
        raise ValueError('QUALIFIED_ROUND_COVERAGE')
    for row, error in zip(packet['rounds'], qualification['round_errors']):
        ref = row['reference']
        ev = accuracy.RoundEvidence(m=row['m'], attenuation=ref['attenuation'],
            pf_operator_error=ref['pf_operator_error'], rte_corrected_operator_error=ref['rte_corrected_operator_error'],
            synthesis_signal_error=upper_float(rational(error['bound'])), finite_law_signal_error=0.,
            target_id=packet['target_id'], state_model='own_DF_ground_state_oracle', alias_branch_verified=True,
            iid_per_shot=True, evidence_kind='numerically_qualified_pilot', law_model='ideal_finite_RTE_law')
        plans.append(accuracy.assess_round(target, ev))
    e = energy(target, packet, plans)
    return dict(status='CONDITIONAL_ON_CALIBRATION_AND_IID_MODEL', rounds=plans, energy=e,
        confidence=qualification['confidence'], engine_reference_law='ideal_finite_RTE_law',
        raw_error_slot='complete compiled-to-binary64-CAR expectation transport;counted once',
        actual_CA_achieved=None, rigorous_CA=None, PCG64_joint_stream_qualified=False)


def run_science(*, execute_science=False):
    if not execute_science:
        raise PermissionError('SEPARATE_FIXED_SCIENTIFIC_EXECUTION_PERMISSION')
    raise NotImplementedError('NOT_READY_METHOD_N_DELTA_NUMERICAL_ENCLOSURE_AND_RANGE')


def preflight(root, contract):
    """Bytes/JSON only. Cannot decode arrays, sample, build, or call accuracy."""
    import hashlib
    import json
    root = Path(root).resolve()
    c = json.loads(Path(contract).read_text())
    if (c['scope'] != SCOPE or c['base_commit'] != BASE or c['executable'] is not False
            or c['maximum_logical_units'] is not None or c['retry'] != 0 or c['resume'] != 0
            or c['mandatory_stop'] is not True):
        raise ValueError('FIXED_NONEXECUTABLE_DESIGN')
    manifest = (root/c['source_manifest']).resolve()
    if (not manifest.is_relative_to(root)
            or hashlib.sha256(manifest.read_bytes()).hexdigest() != c['source_manifest_sha256']
            or json.loads(manifest.read_text())['files'] != c['source_files']):
        raise ValueError('SOURCE_MANIFEST')
    for group in (c['source_files'], c['saved_inputs']):
        for name, expected in group.items():
            p = (root/name).resolve()
            if not p.is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
                raise ValueError('SOURCE_INPUT_BYTES:'+name)
    if (root/c['fresh_output']).exists():
        raise ValueError('FRESH_NO_RETRY_RESUME')
    return dict(status='PASSIVE_NOT_READY_CHECKS_PASS', execution_ready=False, scientific_calls=0,
        arrays_decoded=0, samples_generated=0, stage_C_calls=0, maximum_logical_units=None,
        source_files=len(c['source_files']), saved_inputs=len(c['saved_inputs']), mandatory_stop=True)
