"""Approved saved-signal decomposition, noncancellation eligibility and shots.

Only the original stdlib accounting and envelope functions are used. No signal
evolution, phase rotation, arrays, cost prices, selection or ranking API.
"""
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math

import track_a_v1_contract as legacy
import d0_saved_value_reanalysis_v1 as accounting_source

IDS = tuple(f'B0I_p{p}_q{q}_2nd' for p in (5, 10, 15) for q in (1, 2, 4, 8))
REUSED = 'B0I_p15_q4_2nd'
EPSILONS = (.05, .01, .005, .001, .0005)
FACTORS = (1., 10.)
AXES = ('real', 'imag')
CAPS = dict(saved_component_construction=11, literal_component_copy=1,
            closure_and_envelope=12, endpoint_accounting=120,
            validation_candidate=12, validation_endpoint=120)
ZERO_CALLS = ('signal_evolution', 'phase_rotation', 'SCF', 'DF', 'state_solver',
              'sampling', 'compile', 'model_prediction', 'fit', 'weighted_cost',
              'PF_selection', 'ranking', 'PR_statistics', 'stage_B')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


class Permit:
    def __init__(self, contract, grant, journal=lambda value: None, *, synthetic=False):
        if synthetic:
            require(contract.get('kind') == 'ARTIFICIAL', 'ARTIFICIAL_PERMIT_ONLY')
        else:
            require(contract.get('scope') == 'H6_NONCANCELLATION_STAGE_A_12x10', 'STAGE_A_SCOPE')
            require(contract.get('candidate_ids') == list(IDS), 'EXACT12_SCOPE')
            require(contract.get('call_caps') == CAPS, 'EXACT_CALL_CAPS')
            require(grant.get('scientific_execution_authorized') is True, 'NEW_EXPLICIT_GRANT_REQUIRED')
            require(grant.get('SOURCE') == contract.get('SOURCE'), 'SOURCE_GRANT_BINDING')
            require(grant.get('allowed_scope') == contract['scope'] and grant.get('call_caps') == CAPS,
                    'GRANT_CALL_SCOPE')
            require(grant.get('retry') is False and grant.get('stage_B_authorized') is False,
                    'NO_RETRY_NO_STAGE_B')
        self.synthetic = synthetic
        self.journal = journal
        self.reserved = {k: 0 for k in CAPS}
        self.completed = {k: 0 for k in CAPS}

    def call(self, name, function, *args, **kwargs):
        require(name in CAPS and self.reserved[name] < CAPS[name], 'CALL_BUDGET:' + name)
        self.reserved[name] += 1
        self.journal(self.state())
        value = function(*args, **kwargs)
        self.completed[name] += 1
        self.journal(self.state())
        return value

    def state(self):
        return dict(reserved=dict(self.reserved), completed=dict(self.completed),
                    prohibited_calls={k: 0 for k in ZERO_CALLS})


def finite(value):
    require(type(value) in (float, int) and math.isfinite(value), 'FINITE_SAVED_VALUE_REQUIRED')
    return value


def parts_from_saved(record):
    z = lambda value: complex(finite(value['real']), finite(value['imag']))
    corrected = z(record['signals']['corrected'])
    truncated = z(record['signals']['exact_truncated'])
    reference = z(record['reference'])
    pack = lambda value: dict(real=value.real, imag=value.imag)
    return dict(discard=pack(truncated - reference), PF=pack(corrected - truncated))


def classified(accounting):
    if accounting['cost_acquisition_eligible']:
        return True
    if accounting['eligibility_under_declared_allowance'] == 'INELIGIBLE':
        return False
    return None


def candidate_audit(record, parts, envelope):
    """Distinct Fraction-based check; does not feed production values back."""
    for axis in AXES:
        a = Fraction(finite(record['signals']['exact_truncated'][axis]))
        b = Fraction(finite(record['reference'][axis]))
        c = Fraction(finite(record['signals']['corrected'][axis]))
        require(parts['discard'][axis] == float(a - b), 'AUDIT_SAVED_DISCARD_DIFFERENCE')
        require(parts['PF'][axis] == float(c - a), 'AUDIT_SAVED_PF_DIFFERENCE')
        require(record['error_decomposition']['total_signed'][axis] == float(c - b),
                'AUDIT_SAVED_TOTAL_DIFFERENCE')
        total = parts['discard'][axis] + parts['PF'][axis]
        target = record['error_decomposition']['total_signed'][axis]
        require(abs(total-target) <= max(1e-12 * max(abs(total), abs(target)), 1e-12),
                'AUDIT_CLOSURE')
        expected = float(abs(Fraction(parts['discard'][axis])) + abs(Fraction(parts['PF'][axis])))
        require(envelope[axis] == expected, 'AUDIT_ENVELOPE')
    return dict(status='PASS', fraction_saved_difference_and_envelope=True,
                closure_original_tolerance=True, input_values_not_replaced=True)


def endpoint_audit(record, envelope, epsilon, factor, result):
    """Check frozen binary64 order and integer ceil without another accounting call."""
    B = math.exp(record['log_B'] + factor * record['allowance']['log_B_margin'])
    statuses = []
    shots = []
    for axis in AXES:
        row = result['axes'][axis]
        b = envelope[axis]
        u = factor * record['allowance'][axis]
        # Explicit rounding at each subtraction preserves the legacy order.
        h = float(Fraction(float(Fraction(epsilon / math.sqrt(2)) - Fraction(b))) - Fraction(u))
        status = ('ELIGIBLE' if h > 0 else 'INELIGIBLE'
                  if max(0., b-u) >= epsilon / math.sqrt(2) else 'UNDETERMINED')
        require(row['headroom'] == h and row['allowance'] == u and row['status'] == status,
                'AUDIT_HEADROOM_OR_STATUS')
        statuses.append(status)
        if status == 'ELIGIBLE':
            log_n = math.log(2) + 2 * (record['log_B'] + factor * record['allowance']['log_B_margin']) - 2 * math.log(h) + math.log(math.log(2/.025))
            require(row['log_shot_bound'] == log_n and row['u_over_headroom'] == u/h,
                    'AUDIT_LOG_SHOT_OR_ALLOWANCE_RATIO')
            if log_n < math.log(2**52):
                bound = 2 * B**2 / h**2 * math.log(2/.025)
                require(row['shots'] == max(1, math.ceil(bound)), 'AUDIT_INTEGER_SHOT_ORDER')
                require(row['shots']-1 < bound <= row['shots'], 'AUDIT_CEIL_INTERVAL')
            else:
                require(row['shots'] is None and row['integer_status'] == 'LOG_DOMAIN_ONLY',
                        'AUDIT_LOG_ONLY_NOT_IMPUTED')
        else:
            require(row['shots'] is None, 'AUDIT_INELIGIBLE_AXIS_HAS_NO_SHOTS')
        shots.append(row['shots'])
    status = ('INELIGIBLE' if 'INELIGIBLE' in statuses else 'ELIGIBLE'
              if set(statuses) == {'ELIGIBLE'} else 'UNDETERMINED')
    total = sum(shots) if status == 'ELIGIBLE' and all(n is not None for n in shots) else None
    require(result['N_total'] == total and result['eligibility_under_declared_allowance'] == status,
            'AUDIT_ENDPOINT_STATUS_OR_TOTAL_SHOTS')
    require(result['alpha_axis'] == .025 and result['evidence_kind'] == 'EMPIRICAL'
            and result['certificate_verified_by_this_function'] is False, 'AUDIT_POLICY_CHANGED')
    return dict(status='PASS', original_order_headroom_and_integer_shots=True,
                no_cost_or_ranking_evaluated=True)


def analyse(records, recovery, permit, emit=lambda kind, value: None):
    require(isinstance(permit, Permit), 'VALIDATED_PERMIT_REQUIRED')
    ids = [r['cell']['id'] for r in records]
    require(len(ids) == len(set(ids)), 'DUPLICATE_CANDIDATE')
    require(set(ids) == set(IDS) if not permit.synthetic else bool(ids), 'EXACT12_INPUT')
    before = digest(records)
    components, endpoints, validations = [], [], []
    for record in records:
        cid = record['cell']['id']
        require(record['cell']['method'] == 'B0I' and record['cell']['formula'] == '2nd'
                and record['cell']['T'] == .8, 'FIXED_PHYSICAL_CONDITIONS')
        require('signed_parts' not in record['error_decomposition'], 'OLD_SCHEMA_CHANGED')
        if cid == REUSED:
            require(recovery['candidate_id'] == cid and recovery['T'] == .8, 'WRONG_RECOVERY_CANDIDATE')
            parts = permit.call('literal_component_copy', deepcopy, recovery['signed_parts_verbatim'])
            origin = 'REUSED_PUBLIC_LITERAL_COMPONENTS'
        else:
            parts = permit.call('saved_component_construction', parts_from_saved, record)
            origin = 'NEW_DIFFERENCES_OF_SAVED_SIGNALS'
        view = deepcopy(record)
        view['error_decomposition']['signed_parts'] = parts
        _, envelope = permit.call('closure_and_envelope', legacy.signal_envelopes, view)
        require(envelope is not None, 'NONCANCELLATION_ENVELOPE_UNRESOLVED')
        row = dict(candidate_id=cid, prefix=record['cell']['prefix'], q=record['cell']['q'],
                   T=.8, PF='2nd', component_origin=origin, signed_parts=parts,
                   total_signed_saved=deepcopy(record['error_decomposition']['total_signed']),
                   noncancellation_envelope=envelope, closure='PASS_ORIGINAL_ISCLOSE_1E_12',
                   input_record_sha256=digest(record), allowance=deepcopy(record['allowance']),
                   log_B=record['log_B'], full_H_reference='SAVED_UNROTATED', certified=False)
        components.append(row)
        emit('component', row)
        audit = permit.call('validation_candidate', candidate_audit, record, parts, envelope)
        validations.append(dict(candidate_id=cid, candidate_check=audit))
        view['error_decomposition']['total_signed'] = deepcopy(envelope)
        for epsilon in EPSILONS:
            for factor in FACTORS:
                result = permit.call('endpoint_accounting', accounting_source.accounting, view, epsilon, factor)
                eligible = classified(result)
                endpoint = dict(candidate_id=cid, prefix=record['cell']['prefix'], q=record['cell']['q'],
                    PF='2nd', T=.8, bias_mode='empirical_no_cancellation', epsilon=epsilon,
                    allowance_factor=factor, eligible=eligible,
                    status='ELIGIBLE' if eligible is True else 'INELIGIBLE' if eligible is False else 'UNRESOLVED',
                    accounting=result, saved_input_record_sha256=digest(record),
                    shots={a: result['axes'][ch]['shots'] for a,ch in (('cosine','real'),('sine','imag'))}
                    if eligible is True else None, N_total=result['N_total'],
                    previous_eligibility=None, new_analysis_not_old_record_correction=True)
                endpoints.append(endpoint)
                emit('endpoint', endpoint)
                require(eligible is not None, 'NUMERICALLY_UNRESOLVED_ENDPOINT:' + cid)
                checked = permit.call('validation_endpoint', endpoint_audit,
                                      record, envelope, epsilon, factor, result)
                validations.append(dict(candidate_id=cid, epsilon=epsilon, factor=factor,
                                        endpoint_check=checked))
    require(digest(records) == before, 'ORIGINAL_INPUT_RECORD_MUTATED')
    if not permit.synthetic:
        require(permit.completed == CAPS, 'EXACT_PRODUCTION_AND_VALIDATION_COUNTS')
    counts = Counter(r['status'] for r in endpoints)
    summary = dict(candidate_count=len(components), endpoint_count=len(endpoints),
        eligibility_counts={k:counts[k] for k in ('ELIGIBLE','INELIGIBLE','UNRESOLVED')},
        candidates=[dict(candidate_id=cid, counts=dict(Counter(r['status'] for r in endpoints
                         if r['candidate_id'] == cid))) for cid in ids],
        closure_passed=len(components), original_allowance_log_B_alpha_and_order_preserved=True,
        prior_unknown120_replaced_in_new_result_only=True, old_records_modified=False,
        stage_B_executed=False, scope30_cost_comparison_not_evaluated=True,
        empirical_only_no_rigorous_bias_certificate=True)
    return dict(components=components, endpoints=endpoints, validation=validations, summary=summary)
