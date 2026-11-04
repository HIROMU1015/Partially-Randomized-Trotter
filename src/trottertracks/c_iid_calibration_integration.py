"""Connect fresh records to existing compiler/certificate/statistics adapters.

No molecular runner, sampler, execution switch, retry, or Stage C entry is added.
Callbacks belong to an independently authorized caller; tests supply toy inputs.
The fixed A/B modules and their SOURCE/contract bytes remain unchanged.
"""
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json

from . import c_fresh_iid_calibration as fresh
from . import c_per_word_population_range as population
from . import c_totality_calibration as totality

BASE = '9098de49f1bd816dbaf810eb63815c841169236d'
A = 'ce3185513578a8329efe364c748b5ce2aacdda45'
SCOPE = 'C_IID_CALIBRATION_TECHNICAL_INTEGRATION_V1'
FAMILY = 'artifacts/research_c_iid_calibration_integration/2026-11-04/technical'
CONTRACT = 'docs/research/research_c_iid_calibration_connection_unapproved_20261104.json'


def calibration_binding(sample, context, occurrence):
    fresh.validate_sample(sample, context)
    if type(occurrence) is not int or not 0 <= occurrence < len(sample['kernels']):
        raise ValueError('COHERENT_OCCURRENCE')
    kernel = sample['kernels'][occurrence]
    return dict(sample_id=sample['sample_id'], source_id=sample['source_id'],
        candidate=sample['candidate'], law_id=sample['law_id'],
        occurrence=occurrence, word_id=kernel['word_id'])


def bind_operator(record, sample, context, occurrence, *, role, raw_id=None):
    """Record association, not a proof that a builder implements its word.

    A future runner must obtain the word from fresh.rte_events and the unchanged
    builder/P0. Digests include every gate and the exact global phase.
    """
    from . import c_pr_aware_df_fixed_certificate as fixed
    fixed._validate(record)
    if role not in ('raw', 'compiled') or (role == 'compiled') != (raw_id is not None):
        raise ValueError('RAW_COMPILED_PARENT_ID')
    return dict(record=deepcopy(record), record_id=fresh.digest(record), role=role,
        raw_id=raw_id, phase_hex=record['phase_hex'],
        phase_policy=context['identity']['phase_policy'], identity=deepcopy(context['identity']),
        context_id=context['context_id'], unit=sample['unit'], m=sample['m'], axis=sample['axis'],
        word=deepcopy(sample['kernels'][occurrence]['word']),
        calibration_binding=calibration_binding(sample, context, occurrence))


def _operator(value, sample, context, occurrence, role, raw_id=None):
    expected = bind_operator(value['record'], sample, context, occurrence, role=role, raw_id=raw_id)
    if value != expected:
        raise ValueError('OPERATOR_SOURCE_WORD_ANGLE_PHASE_OR_COMPILER_IDENTITY')
    if role == 'compiled' and any(g['name'] not in ('rz', 'sx', 'x', 'cx') for g in value['record']['gates']):
        raise ValueError('P0_NATIVE_GRAMMAR')
    return value['record']


def _reference(reference, sample, context, occurrence, target):
    kernel = sample['kernels'][occurrence]
    b = reference['binding']
    if (reference.get('status') != 'VALIDATED_LITERAL_REFERENCE_BLOCKS'
            or reference.get('target') != population.TARGET
            or b.get('target') != target or b.get('target_id') != fresh.digest(target)
            or target.get('target') != population.TARGET
            or target.get('candidate') != context['identity']['candidate']
            or b.get('word') != kernel['word'] or b.get('word_id') != kernel['word_id']
            or reference.get('precision_bits') != 192):
        raise ValueError('LITERAL_TARGET_CANDIDATE_WORD_ANGLE_OR_PRECISION')


def evaluate_word(raw, compiled, reference, *, sample, context, occurrence,
                  target, parallel=True, batch_size=8, partial=None):
    """C->R uses A's batch engine; R->D retains the literal CAR evaluator.

    Common-factor cancellation and cap2 are restricted to native unitary C/R.
    There is no cancellation or cap2 on R/D or the triangle C/D result.
    """
    from . import c_arb_column_batch as fast
    r = _operator(raw, sample, context, occurrence, 'raw')
    c = _operator(compiled, sample, context, occurrence, 'compiled', raw['record_id'])
    _reference(reference, sample, context, occurrence, target)
    if r['qubits'] != c['qubits']:
        raise ValueError('RAW_COMPILED_REGISTER')
    cr = fast.compare(c, r, parallel=parallel, batch_size=batch_size)
    if partial is not None: partial['compiled_raw'] = deepcopy(cr)
    rd = population.compare_literal(r, reference, parallel=parallel)
    if partial is not None: partial['raw_literal'] = deepcopy(rd)
    if rd['raw_id'] != cr['right_id'] or rd['target_id'] != reference['binding']['target_id']:
        raise ValueError('TRIANGLE_RAW_TARGET_IDENTITY')
    result = dict(status='VALIDATED_PER_WORD_COMPILED_LITERAL_OPERATOR',
        target=population.TARGET, binding=deepcopy(reference['binding']), kind='rigorous_enclosure',
        scope=rd['scope'], compiled_raw=cr, raw_literal=rd,
        raw_id=raw['record_id'], compiled_id=compiled['record_id'],
        upper_exact=str(F(cr['upper_exact']) + F(rd['upper_exact'])),
        lower_exact=str(max(F(), F(rd['lower_exact']) - F(cr['upper_exact']))),
        calibration_binding=calibration_binding(sample, context, occurrence),
        phase_fit=False, rounding_enclosed=True, precision_bits=192,
        all_word_uniform=False, all_q_words_compile_success_qualified=False,
        provenance='fresh same-word C/R batch + unchanged literal R/D; disjoint triangle legs',
        actual_CA_achieved=None, Stage_C_ready=False)
    check_certificate(result, raw, compiled, sample=sample, context=context,
        occurrence=occurrence, target=target)
    return result


def check_certificate(cert, raw, compiled, *, sample, context, occurrence, target):
    """Check persisted bindings/coverage without rerunning numerical evaluation."""
    r = _operator(raw, sample, context, occurrence, 'raw')
    c = _operator(compiled, sample, context, occurrence, 'compiled', raw['record_id'])
    cr, rd = cert['compiled_raw'], cert['raw_literal']
    b = cert['binding']; kernel = sample['kernels'][occurrence]; d = 1 << r['qubits']
    if (c['qubits'] != r['qubits']
            or cert.get('calibration_binding') != calibration_binding(sample, context, occurrence)
            or b.get('target') != target or b.get('target_id') != fresh.digest(target)
            or target.get('candidate') != sample['candidate'] or target.get('target') != population.TARGET
            or b.get('word') != kernel['word'] or b.get('word_id') != kernel['word_id']
            or cert.get('raw_id') != raw['record_id'] or cert.get('compiled_id') != compiled['record_id']
            or cr.get('left_id') != compiled['record_id'] or cr.get('right_id') != raw['record_id']
            or rd.get('raw_id') != raw['record_id'] or rd.get('word_id') != kernel['word_id']
            or rd.get('target_id') != b['target_id'] or rd.get('target') != population.TARGET):
        raise ValueError('CERTIFICATE_SAMPLE_SOURCE_LAW_WORD_PHASE_OR_TARGET')
    if (cert.get('status') != 'VALIDATED_PER_WORD_COMPILED_LITERAL_OPERATOR'
            or cr.get('status') != 'VALIDATED_BATCHED_NATIVE_PAIR'
            or rd.get('status') != 'VALIDATED_RAW_LITERAL_OPERATOR'
            or cert.get('kind') != cr.get('kind') or cert.get('kind') != rd.get('kind')
            or cert.get('kind') != 'rigorous_enclosure'
            or cert.get('phase_fit') is not False or cr.get('phase_fit') is not False
            or rd.get('phase_fit') is not False
            or any(x.get('rounding_enclosed') is not True or x.get('precision_bits') != 192
                   for x in (cert, cr, rd))
            or cr.get('original_full_operator_coverage') is not True
            or cr.get('nonunitary_literal_target_supported') is not False
            or rd.get('upper_method') != 'outward Frobenius of all certified column distances;no cap2'
            or cert.get('scope') != rd.get('scope')
            or (d == 512) != (cert.get('scope') == 'per_word_full512')):
        raise ValueError('CERTIFICATE_NUMERICAL_QUALIFICATION')
    for leg in (cr, rd):
        if (leg.get('dimension') != d or leg.get('columns_evaluated') != d
                or [x['column'] for x in leg.get('columns', [])] != list(range(d))):
            raise ValueError('FULL_INPUT_OUTPUT_COVERAGE')
        lo, hi = (F(leg[k]) for k in ('lower_exact', 'upper_exact'))
        if not 0 <= lo <= hi:
            raise ValueError('NUMERICAL_INTERVAL')
    if (F(cert['upper_exact']) != F(cr['upper_exact']) + F(rd['upper_exact'])
            or F(cert['lower_exact']) != max(F(), F(rd['lower_exact']) - F(cr['upper_exact']))):
        raise ValueError('TWO_LEG_TRIANGLE_NO_MISSING_ZERO_OR_CAP2')
    return True


def process_sample(path, sample, context, *, build_raw, compile_raw, make_reference,
                   target, certifier=evaluate_word, round_evaluator=None, parallel=True):
    """One attempt, persisted before callbacks. Failure closes the batch with STOP.

    No fixed-N continuation is introduced. Existing stop-on-failure callers are
    unchanged. A pending draw recovered from disk remains INCOMPLETE.
    """
    fresh.save_sample(path, sample, context)
    evidence = dict(occurrences=[], per_word_enclosures=[])
    status = 'SUCCESS'; stage = None; error = None; interrupted = None
    try:
        for i, kernel in enumerate(sample['kernels']):
            part = dict(occurrence=i, raw_status='UNKNOWN', compile_status='UNKNOWN')
            evidence['occurrences'].append(part)
            status = 'RAW_BUILD_FAILED'
            raw = build_raw(deepcopy(kernel), i)
            part['raw'] = deepcopy(raw)
            if raw is None:
                status = 'INCOMPLETE'; raise ValueError('MISSING_RAW_RESULT')
            _operator(raw, sample, context, i, 'raw')
            part['raw_status'] = 'SUCCESS'
            status = 'P0_COMPILE_FAILED'
            compiled = compile_raw(deepcopy(raw), i)
            part['compiled'] = deepcopy(compiled)
            if compiled is None:
                status = 'INCOMPLETE'; raise ValueError('MISSING_COMPILE_RESULT')
            part['compile_status'] = 'OUTPUT_RETURNED'
            status = 'NATIVE_INELIGIBLE'
            rec = _operator(compiled, sample, context, i, 'compiled', raw['record_id'])
            if (rec['qubits'] != raw['record']['qubits']
                    or any(g['name'] not in ('rz', 'sx', 'x', 'cx') for g in rec['gates'])):
                raise ValueError('P0_NATIVE_REGISTER_GRAMMAR')
            part['compile_status'] = 'SUCCESS'
            status = 'PER_WORD_CERTIFICATE_FAILED'
            reference = make_reference(deepcopy(kernel), i)
            if reference is None:
                status = 'INCOMPLETE'; raise ValueError('MISSING_LITERAL_REFERENCE')
            part['reference'] = deepcopy(reference)
            part['literal_binding'] = deepcopy(reference.get('binding'))
            cert = certifier(raw, compiled, reference, sample=sample, context=context,
                occurrence=i, target=target, parallel=parallel, partial=part)
            if cert is None:
                raise ValueError('MISSING_PER_WORD_CERTIFICATE')
            part['certificate_result'] = deepcopy(cert)
            check_certificate(cert, raw, compiled, sample=sample, context=context, occurrence=i, target=target)
            evidence['per_word_enclosures'].append(cert)
            for name in ('compiled_raw', 'raw_literal', 'certificate_result'): part.pop(name, None)
        if sample['unit'] == 'round' and round_evaluator is not None:
            status = 'NUMERICAL_UNEVALUABLE'
            evidence['round_enclosure'] = round_evaluator(deepcopy(sample), deepcopy(evidence))
            if evidence['round_enclosure'] is None:
                raise ValueError('MISSING_ROUND_NUMERICAL_RESULT')
        status = 'SUCCESS'
    except BaseException as exc:
        error = exc
        if not isinstance(exc, Exception):
            status = 'INCOMPLETE'; interrupted = exc
        elif isinstance(exc, ArithmeticError) and status == 'PER_WORD_CERTIFICATE_FAILED':
            status = 'NUMERICAL_UNEVALUABLE'
        stage = fresh.FAILURE_STAGES.get(status)
        if evidence['occurrences']:
            part = evidence['occurrences'][-1]
            if status == 'RAW_BUILD_FAILED': part['raw_status'] = 'FAILED'
            if status == 'P0_COMPILE_FAILED': part['compile_status'] = 'FAILED'
            if status == 'NATIVE_INELIGIBLE': part['compile_status'] = 'NATIVE_INELIGIBLE'
    row = fresh.outcome(sample, status=status, stage=stage, error=error, evidence=evidence)
    fresh.save_outcome(path, row)
    if status != 'SUCCESS': fresh.finish_batch(path)
    if interrupted is not None: raise interrupted
    return row


def calibration_payload(rows, *, context, target, **kwargs):
    """Qualify the new two-leg connection, then delegate B's unchanged gate."""
    rows = list(rows)
    for row in rows:
        fresh.validate_sample(row['sample'], context)
        certs = row.get('evidence', {}).get('per_word_enclosures', [])
        parts = row.get('evidence', {}).get('occurrences', [])
        for i, cert in enumerate(certs):
            if i >= len(parts): raise ValueError('CERTIFICATE_WITHOUT_NATIVE_INPUTS')
            if parts[i].get('raw_status') != 'SUCCESS' or parts[i].get('compile_status') != 'SUCCESS':
                raise ValueError('CERTIFICATE_WITHOUT_SUCCESSFUL_NATIVE_PIPELINE')
            check_certificate(cert, parts[i]['raw'], parts[i]['compiled'],
                sample=row['sample'], context=context, occurrence=i, target=target)
    return fresh.calibration_payload(rows, context=context, **kwargs)


def compiler_failure_summary(rows, *, context, planned_ids, delta_compile=None,
                             failure_policy=None, iid_assumed=False, event='pipeline'):
    """Translate B's statuses into A's trial outcomes, without conflating certs.

    Each coherent occurrence is one compiler trial. Certificate/numerical failures
    after native success do not become compiler failures. Unvisited trials stay
    UNKNOWN; a stop is never interpreted as a completed fixed-N experiment.
    """
    rows = list(rows); planned = list(planned_ids)
    ids = [r['sample']['sample_id'] for r in rows]
    if (not planned or len(set(planned)) != len(planned) or len(set(ids)) != len(ids)
            or ids != planned[:len(ids)]):
        raise ValueError('NO_DUPLICATE_DISCARD_REORDER_OR_REPLACEMENT')
    units = set(); trials = []
    for row in rows:
        s = row['sample']; fresh.validate_sample(s, context)
        if (row.get('status') not in fresh.STATUSES or row.get('retry', 0) != 0
                or row.get('resume', 0) != 0 or row.get('resampled', False) is not False):
            raise ValueError('OUTCOME_STATUS_NO_RETRY_RESUME_RESAMPLING')
        units.add((s['dataset_id'], s['unit'], s['m'], s['axis']))
        parts = row['evidence'].get('occurrences', [])
        if len(parts) > len(s['kernels']): raise ValueError('COHERENT_OCCURRENCE')
        for i, kernel in enumerate(s['kernels']):
            part = parts[i] if i < len(parts) else {}
            if part and part.get('occurrence') != i: raise ValueError('COHERENT_OCCURRENCE')
            raw, comp = part.get('raw_status'), part.get('compile_status')
            if raw == 'SUCCESS': _operator(part['raw'], s, context, i, 'raw')
            if comp == 'SUCCESS':
                _operator(part['compiled'], s, context, i, 'compiled', part['raw']['record_id'])
            outcome = ('RAW_BUILD_FAILURE' if raw == 'FAILED' else
                'COMPILE_EXCEPTION' if comp == 'FAILED' else
                'INVALID_NATIVE_OUTPUT' if comp == 'NATIVE_INELIGIBLE' else
                'SUCCESS' if raw == comp == 'SUCCESS' else 'UNKNOWN')
            trials.append(dict(draw_id=len(trials), sample_id=s['sample_id'], occurrence=i,
                word_id=kernel['word_id'], outcome=outcome, retry=0, resume=0, resampled=False))
    if len(units) > 1: raise ValueError('ONE_FIXED_COMPILER_STRATUM')
    result = dict(status='NOT_READY_UNADOPTED_FAILURE_POLICY', failure_upper=None, trials=trials,
        all_q_words_compile_success_qualified=False, scientific_policy_adopted=False, Stage_C_ready=False)
    if failure_policy not in ('stop_on_failure', 'record_all_fixed_N') or delta_compile is None or iid_assumed is not True:
        return result
    n = 1 if not rows or rows[0]['sample']['unit'] == 'kernel' else 1 << rows[0]['sample']['m']
    bound = totality.failure_rate(trials, planned_N=len(planned)*n, delta=delta_compile,
        event=event, experiment='fixed_N' if failure_policy == 'record_all_fixed_N' else 'stop_on_failure')
    return dict(result, **bound)


def preflight(root, contract=CONTRACT):
    """JSON/byte checks plus original A/B passive preflights only."""
    from . import c_batch_pair_pilot as pilot
    root = Path(root).resolve(); path = Path(contract)
    c = json.loads((path if path.is_absolute() else root/path).read_text())
    required = dict(scope=SCOPE, base_commit=BASE, A_commit=A, executable=False,
        method=None, N=None, delta_cal=None, delta_compile=None, failure_policy=None,
        maximum_logical_units=None, retry=0, resume=0, mandatory_stop=True)
    if any(type(c.get(k)) is not type(v) or c[k] != v for k, v in required.items()):
        raise ValueError('NONEXECUTABLE_UNADOPTED_CONNECTION_CONTRACT')
    def checked(name, want):
        p = (root/name).resolve()
        if not p.is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest() != want:
            raise ValueError('INTEGRATION_SOURCE_OR_CONTRACT_BYTES:'+name)
        return p
    source = json.loads(checked(c['source_manifest'], c['source_manifest_sha256']).read_text())
    if source.get('scope') != SCOPE or source.get('scientific_SOURCE') is not False:
        raise ValueError('INDEPENDENT_TECHNICAL_SOURCE')
    for name, want in {**source['files'], **c['readonly_contracts']}.items(): checked(name, want)
    checks = dict(B=fresh.preflight(root, root/fresh.CONTRACT),
        A=totality.preflight(root, root/totality.CONTRACT), pilot=pilot.preflight(root, root/pilot.CONTRACT))
    return dict(status='PASSIVE_INTEGRATION_PASS_SCIENCE_NOT_READY', checks=checks,
        scientific_calls=0, arrays_decoded=0, new_H4_build_compile=0, calibration_calls=0,
        Stage_C_calls=0, important_reviews=0, fixed_7_call_pilot_executed=False,
        all_q_words_compile_success_qualified=False, PCG64_joint_stream_qualified=False,
        actual_CA_achieved=None, execution_ready=False, mandatory_stop=True)
