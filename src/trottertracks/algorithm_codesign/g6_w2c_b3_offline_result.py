"""Read-only serial R1/F01 verification. Artificial inputs only in this release.

No scientific imports, discovery, sampling, authorization, retry or resume.
Memory is O(largest line + fixed 24-pair metadata), independent of trial count.
"""
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import re

BASE_COMMIT = '564409487a35e15d73f1d0f0c2611824b400ab02'
ARMS = ('ordinary', 'partial_return_tail', 'closed_P3_tail', 'full_return')
KINDS = ('ZERO_R', 'ZERO_C', 'EVENT')
ZERO = '0' * 64
RAW = 'INDEPENDENTLY_RECOMPUTED_FROM_RAW_BYTES'
HASHES = 'COMPARED_SAVED_EVENT_HASHES'
CLAIM = 'COMPARED_SUMMARY_CLAIMS'
UNKNOWN = 'INSUFFICIENT_INDEPENDENT_EVIDENCE'
HEX = re.compile(r'^[0-9a-f]{64}$')


class Invalid(ValueError):
    pass


class Unsupported(Invalid):
    pass


def require(ok, reason):
    if not ok:
        raise Invalid(reason)


def is_hash(value):
    return type(value) is str and HEX.fullmatch(value) is not None


def nonnegative(value):
    require(type(value) is int and value >= 0, 'NONNEGATIVE_INTEGER_REQUIRED')
    return value


def _unique(items):
    result = {}
    for key, value in items:
        require(key not in result, 'DUPLICATE_JSON_KEY')
        result[key] = value
    return result


def plain(text):
    def bad(value):
        raise Invalid('NONFINITE_JSON:' + value)
    return json.loads(text, object_pairs_hook=_unique, parse_constant=bad)


def pack(value):
    if type(value) is Fraction:
        return {'rat': [hex(value.numerator), hex(value.denominator)]}
    if type(value) is int:
        return {'int': hex(value)}
    if value is None or type(value) in (str, bool):
        return value
    if type(value) in (tuple, list):
        return [pack(v) for v in value]
    if type(value) is dict:
        # Match driver.exact's str-key normalization, including last-wins.
        normalized = {str(k): v for k, v in value.items()}
        return {k: pack(v) for k, v in sorted(normalized.items())}
    raise Invalid('NONEXACT_VALUE')


def canonical(value):
    return json.dumps(pack(value), sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def identity(value):
    return sha256(canonical(value).encode()).hexdigest()


def decode(text):
    """Driver hex codec only. Nested F3 evidence strings stay opaque strings."""
    def integer(value):
        require(type(value) is str and re.fullmatch(r'-?0x[0-9a-f]+', value), 'HEX_INT_TAG')
        n = int(value, 16)
        require(hex(n) == value, 'NONCANONICAL_HEX')
        return n
    def unpack(value):
        if type(value) is dict:
            if set(value) == {'int'}:
                return integer(value['int'])
            if set(value) == {'rat'}:
                a = value['rat']
                require(type(a) is list and len(a) == 2, 'RATIONAL_TAG')
                n, d = map(integer, a)
                require(d > 0, 'RATIONAL_DENOMINATOR')
                f = Fraction(n, d)
                require((f.numerator, f.denominator) == (n, d), 'NONCANONICAL_RATIONAL')
                return f
            return {k: unpack(v) for k, v in value.items()}
        if type(value) is list:
            return [unpack(v) for v in value]
        require(value is None or type(value) in (str, bool), 'UNTAGGED_NUMBER')
        return value
    value = unpack(plain(text))
    require(canonical(value) == text, 'NONCANONICAL_PAYLOAD')
    return value


@dataclass
class Row:
    payload: dict
    before: dict
    after: dict


class Reader:
    """Single-pass envelope iterator. SHA uses the original payload string."""
    def __init__(self, path):
        self.path = Path(path)
        self.stats = dict(rows=0, bytes=0, chain_sha256=ZERO, file_sha256=None,
                          incomplete_tail_bytes=0, peak_line_bytes=0)

    def __iter__(self):
        digest = sha256()
        previous = ZERO
        with self.path.open('rb') as f:
            for line in f:
                before = dict(self.stats, file_sha256=digest.hexdigest())
                digest.update(line)
                self.stats['bytes'] += len(line)
                self.stats['peak_line_bytes'] = max(self.stats['peak_line_bytes'], len(line))
                if not line.endswith(b'\n'):
                    self.stats['incomplete_tail_bytes'] = len(line)
                    break
                envelope = plain(line.decode('utf-8'))
                require(type(envelope) is dict and set(envelope) == {'previous', 'sha256', 'payload'},
                        'ENVELOPE_FIELDS')
                require(envelope['previous'] == previous and is_hash(envelope['sha256']) and
                        type(envelope['payload']) is str, 'CHAIN_PREVIOUS_OR_SHAPE')
                computed = sha256((previous + envelope['payload']).encode()).hexdigest()
                require(computed == envelope['sha256'], 'CHAIN_SHA256')
                value = decode(envelope['payload'])
                require(type(value) is dict, 'PAYLOAD_OBJECT_REQUIRED')
                previous = computed
                self.stats['rows'] += 1
                self.stats['chain_sha256'] = previous
                self.stats['file_sha256'] = digest.hexdigest()
                yield Row(value, before, dict(self.stats))
        self.stats['file_sha256'] = digest.hexdigest()


def file_digest(path, length=None):
    h = sha256()
    with Path(path).open('rb') as f:
        while length is None or length:
            b = f.read(1048576 if length is None else min(length, 1048576))
            if not b:
                require(length in (None, 0), 'PREFIX_BEYOND_FILE')
                break
            h.update(b)
            if length is not None:
                length -= len(b)
    return h.hexdigest()


def counts(value):
    require(type(value) is dict and set(value) <= set(KINDS), 'COUNT_KEYS')
    return {k: nonnegative(value.get(k, 0)) for k in KINDS}


def spec_check(spec):
    require(spec['phase'] in ('correctness', 'profile') and spec['pass_name'] in
            ('baseline', 'instrumented') and spec['arm'] in ARMS and spec['layer'] in ('A', 'B'), 'PAIR_IDENTITY')
    require(spec['phase'] != 'correctness' or spec['pass_name'] == 'baseline', 'CORRECTNESS_PASS')
    require(nonnegative(spec['N']) > 0, 'N_REQUIRED')
    return 16 if spec['phase'] == 'correctness' else spec['N']


def key(spec):
    return tuple(spec[x] for x in ('phase', 'pass_name', 'arm', 'layer'))


def trial(spec, axis, index):
    return '/'.join((*key(spec), axis, str(index)))


def ordinal(spec, descriptor):
    if descriptor is None:
        return -1
    axis, index = descriptor['axis'], descriptor['trial_index']
    n = spec_check(spec)
    require(axis in ('X', 'Y') and type(index) is int and 0 <= index < n and
            descriptor['trial_id'] == trial(spec, axis, index), 'FAILURE_TRIAL_ID')
    return (0 if axis == 'X' else n) + index


def load_failure(path):
    doc = plain(Path(path).read_text())
    if doc.get('kind') == 'LOG_BYTE_LIMIT':
        raise Unsupported('FAILURE_LOG_TRUNCATED')
    require(doc.get('kind') == 'TYPED_FAILURE', 'FAILURE_CONTROL_KIND')
    raw = doc.get('trial_failure_provenance')
    return decode(raw) if type(raw) is str else None


def check_failure(failure, spec, visible, path):
    if failure.get('failure_provenance_schema') != 'g6_w2c_b3_f01_failure_v1':
        raise Unsupported('F01_SCHEMA')
    require(failure.get('mandatory_STOP') is True and failure.get('retry') == 0 and
            failure.get('resume') is False and failure.get('is_random_zero') is False and
            failure.get('trace_scope') == 'COMPLETED_TRIALS_AFTER_NORMAL_EMIT_RETURN', 'FAILURE_STOP_SCOPE')
    attempt = failure['current_attempt']
    flags = ('result_acquired', 'event_present', 'record_constructed', 'candidate_ready',
             'stream_emit_started', 'stream_emit_returned', 'trial_completed')
    require(all(type(attempt.get(f)) is bool for f in flags), 'F01_FLAGS')
    for child, parent in (('event_present', 'result_acquired'), ('record_constructed', 'result_acquired'),
                          ('candidate_ready', 'record_constructed'), ('stream_emit_started', 'candidate_ready'),
                          ('stream_emit_returned', 'stream_emit_started'), ('trial_completed', 'stream_emit_returned')):
        require(not attempt[child] or attempt[parent], 'F01_STAGE_ORDER:' + child)
    require(attempt['event_present'] == (failure['inflight_event'] is not None), 'STALE_INFLIGHT_EVENT')
    if not attempt['result_acquired']:
        require(failure['inflight_event'] is None, 'RESULT_NOT_ACQUIRED')
    last = ordinal(spec, failure['last_completed_trial'])
    total = sum(counts(failure['completed_prefix']).values())
    require(total == last + 1 and total <= visible, 'F01_COMPLETED_PREFIX')
    if attempt['trial_id'] is None:
        require(attempt['trial_index'] is None and not any(attempt[f] for f in flags), 'F01_NONTRIAL_CONTEXT')
    else:
        current = ordinal(spec, attempt)
        require(current == (last if attempt['trial_completed'] else last + 1), 'INFLIGHT_NOT_NEXT_TRIAL')
    candidate = failure['candidate_trace_sha256']
    require(is_hash(failure['trace_sha256']) and
            (is_hash(candidate) if attempt['candidate_ready'] else candidate is None), 'F01_DIGEST_SHAPE')
    prefix = failure['stream_prefix_before_failure_record']
    if prefix.get('status') == 'UNAVAILABLE':
        return total, {'status': UNKNOWN, 'writer_snapshot': prefix}
    length = nonnegative(prefix['written_bytes_including_possible_partial_tail'])
    require(length <= Path(path).stat().st_size, 'WRITER_PREFIX_BEYOND_FILE')
    if prefix['file_hash_covers_written_bytes'] is True:
        require(file_digest(path, length) == prefix['written_bytes_sha256'], 'F01_WRITTEN_BYTES_SHA')
    else:
        require(prefix['written_bytes_sha256'] is None, 'UNAVAILABLE_FILE_HASH_NOT_NULL')
    rows = nonnegative(prefix['complete_rows'])
    physical_rows = 0
    committed_chain = ZERO
    physical_prefix = Reader(path)
    for row in physical_prefix:
        if row.after['bytes'] > length:
            break
        physical_rows += 1
        if physical_rows == rows:
            committed_chain = row.after['chain_sha256']
    require(rows <= physical_rows and physical_rows - rows <= 1 and
            prefix['last_complete_chain_sha256'] == committed_chain, 'F01_WRITER_ROW_CHAIN')
    if physical_rows != rows:
        require((prefix.get('failure') or {}).get('stage') in ('FILE_HASH', 'WRITE'), 'F01_WRITER_ROW_LAG')
    require(prefix['readback_verified'] is False and prefix['fsync_durability_verified'] is False,
            'UNSUPPORTED_DURABILITY_CLAIM')
    return total, {'status': RAW, 'writer_snapshot': prefix,
                   'verifier_readback_now': True, 'producer_readback_verified': False,
                   'producer_fsync_durability_verified': False}


def event_hash(payload):
    if 'event' in payload:
        return identity({'kind': payload['kind'], 'event': payload['event']})
    return payload['event_sha256']


def verify_pair(path, spec, summary=None, failure_path=None):
    """Independent pair; explicit paths only. Does not require a global receipt."""
    result = dict(pair=list(key(spec)), status='INSUFFICIENT_EVIDENCE', issues=[], evidence={},
                  committed_trials=0, visible_trial_rows=0, nontrial_rows=0)
    try:
        n = spec_check(spec)
        external_failure = load_failure(failure_path) if failure_path else None
        failure = external_failure
        control_without_provenance = bool(failure_path) and external_failure is None
        reader = Reader(path)
        failure_before = None
        seen_failure = False
        for row in reader:
            p = row.payload
            if p.get('kind') == 'TYPED_FAILURE':
                require(not seen_failure, 'DUPLICATE_FAILURE_RECORD')
                seen_failure = True
                if failure is not None:
                    # External control may contain secondary failure details absent from stream.
                    for field in ('current_attempt', 'completed_prefix', 'trace_sha256', 'last_completed_trial'):
                        require(failure[field] == p[field], 'FAILURE_CONTROL_BINDING')
                else:
                    failure = p
                failure_before = row.before
                result['nontrial_rows'] += 1
            elif 'trial_id' in p:
                require(not seen_failure, 'TRIAL_AFTER_FAILURE')
                result['visible_trial_rows'] += 1
            elif 'memory_sample' in p:
                require(not seen_failure, 'MEMORY_AFTER_FAILURE')
                result['nontrial_rows'] += 1
            else:
                raise Unsupported('UNKNOWN_PAIR_RECORD')
        result['file'] = reader.stats
        visible = result['visible_trial_rows']
        limit = visible
        if failure is not None:
            limit, evidence = check_failure(failure, spec, visible, path)
            result['failure'] = failure
            result['evidence']['failure_prefix'] = evidence
            writer = failure['stream_prefix_before_failure_record']
            if failure_before is not None and writer.get('status') != 'UNAVAILABLE':
                require(writer['complete_rows'] == failure_before['rows'] and
                        writer['last_complete_chain_sha256'] == failure_before['chain_sha256'] and
                        writer['written_bytes_including_possible_partial_tail'] == failure_before['bytes'],
                        'F01_STREAM_SNAPSHOT_BINDING')
            # A physically complete but uncommitted row is possible at flush/file-hash failure.
            extra = visible - limit
            require(extra in (0, 1), 'MULTIPLE_UNCOMMITTED_TRIALS')
            if extra:
                require(failure['current_attempt']['stream_emit_started'] and
                        not failure['current_attempt']['stream_emit_returned'] and
                        writer.get('failure', {}).get('stage') in ('FLUSH', 'FILE_HASH', 'WRITE'),
                        'UNCOMMITTED_ROW_WITHOUT_WRITE_FAILURE')
        axes = {a: dict(N_executed=0, counts=Counter(), digest=sha256(), prefix=sha256(),
                        full=True) for a in ('X', 'Y')}
        digest = sha256()
        full = True
        candidate_digest = None
        generator_cost = {k: [0, 0] for k in KINDS}
        second = Reader(path)
        index = 0
        for row in second:
            p = row.payload
            if 'trial_id' not in p:
                if 'memory_sample' in p:
                    axis = p['axis']
                    require(axis in axes and p['completed_trial'] == axes[axis]['N_executed'] and
                            p['completed_trial'] % 4096 == 0, 'MEMORY_ROW_POSITION')
                continue
            axis = 'X' if index < n else 'Y'
            i = index if axis == 'X' else index - n
            require(index < 2*n and p.get('axis') == axis and p['trial_id'] == trial(spec, axis, i),
                    'TRIAL_GAP_DUPLICATE_OR_IDENTITY')
            require(p.get('kind') in KINDS, 'TRIAL_KIND')
            cost = p['generator_exclusive_ns']
            require(type(cost) is list and len(cost) == 2, 'GENERATOR_COST_SHAPE')
            for v in cost:
                nonnegative(v)
            retained = spec['phase'] == 'correctness' or i < 16
            if retained:
                require(set(p) == {'trial_id', 'axis', 'kind', 'event', 'evidence', 'angle',
                                   'generator_exclusive_ns'}, 'FULL_RECORD_FIELDS')
                require((p['event'] is not None) == (p['kind'] == 'EVENT'), 'EVENT_KIND_BINDING')
                require((p['angle'] is not None) == (p['kind'] == 'EVENT' and spec['layer'] == 'B'),
                        'ANGLE_LAYER_BINDING')
                event_bytes = canonical({'kind': p['kind'], 'event': p['event']}).encode()
            else:
                require(set(p) == {'trial_id', 'axis', 'kind', 'event_sha256', 'evidence_sha256',
                                   'angle_sha256', 'generator_exclusive_ns', 'retention'} and
                        p['retention'] == 'HASHED_FULL_DOCUMENTS_NOT_RETAINED' and
                        all(is_hash(p[f]) for f in ('event_sha256', 'evidence_sha256', 'angle_sha256')),
                        'COMPACT_RECORD_FIELDS')
                event_bytes = None
            if index < limit:
                a = axes[axis]
                generator_cost[p['kind']] = [x+y for x, y in zip(generator_cost[p['kind']], cost)]
                a['counts'][p['kind']] += 1
                a['N_executed'] += 1
                if event_bytes is None:
                    full = a['full'] = False
                else:
                    digest.update(event_bytes)
                    a['digest'].update(event_bytes)
                    if i < 16:
                        a['prefix'].update(event_bytes)
            elif failure is not None and event_bytes is not None and full:
                c = digest.copy()
                c.update(event_bytes)
                candidate_digest = c.hexdigest()
            index += 1
        require(second.stats == reader.stats, 'INPUT_CHANGED_DURING_VERIFICATION')
        observed_counts = {k: sum(a['counts'][k] for a in axes.values()) for k in KINDS}
        result.update(committed_trials=limit, counts=observed_counts, generator_exclusive_ns_by_kind=generator_cost,
                      axes={a: dict(N_executed=v['N_executed'], counts=counts(dict(v['counts'])),
                                   event_trace_sha256=v['digest'].hexdigest() if v['full'] else None,
                                   first_correctness_prefix_sha256=v['prefix'].hexdigest()) for a, v in axes.items()},
                      trace_sha256=digest.hexdigest() if full else None)
        result['evidence'].update(envelope_chain_file=RAW, trial_identity_counts=RAW,
                                  full_trace=RAW if full else UNKNOWN,
                                  full_trace_limitation=None if full else
                                  'Compact event hashes cannot reconstruct concatenated canonical event bytes.')
        if failure is not None:
            require(observed_counts == counts(failure['completed_prefix']), 'F01_COUNT_BINDING')
            if full:
                require(digest.hexdigest() == failure['trace_sha256'], 'F01_TRACE_BINDING')
            if candidate_digest is not None:
                require(candidate_digest == failure['candidate_trace_sha256'], 'F01_CANDIDATE_BINDING')
            result['evidence']['candidate_trace'] = RAW if candidate_digest else UNKNOWN
        if summary is not None:
            require(key(summary) == key(spec), 'SUMMARY_PAIR_IDENTITY')
            for field, actual in (('output_rows', reader.stats['rows']), ('output_bytes', reader.stats['bytes']),
                                  ('output_chain_sha256', reader.stats['chain_sha256']),
                                  ('file_sha256', reader.stats['file_sha256'])):
                require(summary[field] == actual, 'SUMMARY_' + field.upper())
            require(counts(summary['counts']) == observed_counts, 'SUMMARY_COUNTS')
            require([v['axis'] for v in summary['axes']] == ['X', 'Y'], 'SUMMARY_AXES')
            for claim in summary['axes']:
                actual = result['axes'][claim['axis']]
                require(claim['N_executed'] == actual['N_executed'] and counts(claim['counts']) == actual['counts'] and
                        claim['first_correctness_prefix_sha256'] == actual['first_correctness_prefix_sha256'],
                        'SUMMARY_AXIS_COUNTS_PREFIX')
                require(is_hash(claim['event_trace_sha256']), 'AXIS_TRACE_SHAPE')
                if actual['event_trace_sha256'] is not None:
                    require(actual['event_trace_sha256'] == claim['event_trace_sha256'], 'AXIS_TRACE_BINDING')
            require(is_hash(summary['trace_sha256']), 'PAIR_TRACE_SHAPE')
            if full:
                require(digest.hexdigest() == summary['trace_sha256'], 'SUMMARY_TRACE_BINDING')
            result['evidence']['summary_binding'] = RAW
            result['evidence']['summary_full_trace'] = RAW if full else CLAIM
        if failure is not None or control_without_provenance:
            result['status'] = 'TECHNICAL_FAILURE'
            if control_without_provenance:
                result['evidence']['failure_prefix'] = UNKNOWN
        elif reader.stats['incomplete_tail_bytes'] or limit < 2*n:
            result['status'] = 'VERIFIED_PARTIAL_PREFIX'
        elif summary is None:
            result['status'] = 'INSUFFICIENT_EVIDENCE'
        else:
            if spec['phase'] == 'correctness':
                require(all(a['counts']['EVENT'] > 0 for a in axes.values()), 'CORRECTNESS_COVERAGE')
            result['status'] = 'VERIFIED_COMPLETE'
    except Unsupported as exc:
        result['status'] = 'UNSUPPORTED_SCHEMA'
        result['issues'].append(str(exc))
    except (Invalid, KeyError, TypeError, ValueError, UnicodeError, OSError, OverflowError) as exc:
        result['status'] = 'INTEGRITY_MISMATCH' if not isinstance(exc, FileNotFoundError) else 'INSUFFICIENT_EVIDENCE'
        result['issues'].append(str(exc))
    return result


def trial_hashes(path, axis):
    for row in Reader(path):
        p = row.payload
        if 'trial_id' in p and p['axis'] == axis:
            yield event_hash(p)


def cross_pair(left, right, axis, *, prefix_only=False):
    """Separate two-way streaming comparison, O(largest row), no trial set."""
    from itertools import zip_longest, islice
    a, b = trial_hashes(left, axis), trial_hashes(right, axis)
    if prefix_only:
        a, b = islice(a, 16), islice(b, 16)
    count = 0
    try:
        for x, y in zip_longest(a, b):
            require(x is not None and x == y, 'CROSS_PAIR_EVENT_HASH')
            count += 1
        require(count >= 16, 'CROSS_PAIR_PREFIX_INCOMPLETE')
        return dict(status='PASS', compared=count, evidence=HASHES, axis=axis)
    except (Invalid, KeyError, ValueError, OSError) as exc:
        return dict(status='FAIL', compared=count, reason=str(exc), axis=axis)


def safe_path(root, name):
    require(type(name) is str and name and not Path(name).is_absolute(), 'RELATIVE_INPUT_PATH')
    path = (root / name).resolve()
    require(path.is_relative_to(root), 'INPUT_PATH_ESCAPE')
    return path


def projection(candidate):
    return {k: candidate[k] for k in ('input', 'policy', 'cells', 'passes', 'correctness_trials_per_cell',
                                     'decisions', 'one_shot', 'call_budget', 'schedule_proposal', 'source_pins')}


def verify_bundle(manifest_path):
    """Explicit artificial manifest gates the current serial adapter; no discovery."""
    manifest_path = Path(manifest_path).resolve()
    m = plain(manifest_path.read_text())
    report = dict(schema='b3_offline_verification_v1', status='INSUFFICIENT_EVIDENCE',
                  artificial_only=True, pairs=[], cross_pairs=[], issues=[], mandatory_STOP=True)
    root = manifest_path.parent
    try:
        if m.get('schema') != 'b3_artificial_serial_inputs_v1':
            raise Unsupported('INPUT_MANIFEST_SCHEMA')
        require(m.get('artificial_only') is True and m.get('producer_commit') == BASE_COMMIT,
                'ARTIFICIAL_INPUT_AND_FIXED_PRODUCER_REQUIRED')
        require(m['contract_digest'] == identity(m['artificial_contract']), 'CONTRACT_DIGEST')
        expected_keys = [(phase, p, arm, layer) for phase, p in
                         (('correctness', 'baseline'), ('profile', 'baseline'), ('profile', 'instrumented'))
                         for arm in ARMS for layer in ('A', 'B')]
        require([key(s) for s in m['pairs']] == expected_keys, 'SERIAL_24_PAIR_SCHEDULE')
        require(len({s['path'] for s in m['pairs']}) == 24, 'DUPLICATE_PAIR_PATH')
        nmap = {a: m['profile_N'][a] for a in ARMS}
        for s in m['pairs']:
            require(s['N'] == nmap[s['arm']], 'MANIFEST_N_BINDING')
        if m.get('failure_prefix') is not None:
            parent_failure = plain(safe_path(root, m['failure_prefix']).read_text())
            require(parent_failure.get('kind') == 'TYPED_FAILURE' and parent_failure.get('mandatory_STOP') is True and
                    parent_failure.get('retry') == 0 and parent_failure.get('resume') is False, 'PARENT_FAILURE_STOP')
            completed = parent_failure['completed_pairs']
            require([key(c) for c in completed] == expected_keys[:len(completed)], 'FAILED_GLOBAL_PREFIX_ORDER')
            for i, s in enumerate(m['pairs']):
                path = safe_path(root, s['path'])
                if i < len(completed) or path.exists() or s.get('failure_control'):
                    report['pairs'].append(verify_pair(path, s, completed[i] if i < len(completed) else None,
                        safe_path(root, s['failure_control']) if s.get('failure_control') else None))
            require(not any(r['status'] == 'INTEGRITY_MISMATCH' for r in report['pairs']), 'FAILED_PREFIX_INTEGRITY')
            report.update(status='TECHNICAL_FAILURE', parent_failure=parent_failure,
                          receipt_present=False, cost_comparison_eligible=False)
            return report
        summary_reader = Reader(safe_path(root, m['summary']))
        docs = []
        for row in summary_reader:
            require(not docs, 'SUMMARY_MULTIPLE_ROWS')
            docs.append(row.payload)
        require(len(docs) == 1 and not summary_reader.stats['incomplete_tail_bytes'], 'SUMMARY_SINGLE_ROW')
        final = docs[0]
        if final.get('schema') != 'g6_w2c_b3_driver_result_v1':
            raise Unsupported('RESULT_SCHEMA')
        require(final['contract_digest'] == m['contract_digest'] and final['mandatory_STOP'] is True and
                final['population_cost_certificate'] is False and final['physical_RZ_certified'] is False,
                'RESULT_CONTRACT_SCOPE')
        require([key(s) for s in final['pairs']] == expected_keys, 'SUMMARY_24_PAIR_ORDER')
        receipt_path = safe_path(root, m['receipt'])
        receipt = plain(receipt_path.read_text()) if receipt_path.exists() else None
        if receipt is not None:
            require(receipt.get('summary_chain_sha256') == summary_reader.stats['chain_sha256'] and
                    receipt.get('mandatory_STOP') is True and receipt.get('receipt_self_write_cost_ns') is None,
                    'COMPLETION_RECEIPT_BINDING')
        admission = final['admission']
        require(admission.get('contract_digest') == m['contract_digest'] and admission.get('approved') is True and
                admission.get('fresh_claim') is True and admission.get('previously_consumed') is False and
                admission.get('retry') == 0 and admission.get('resume') is False and
                admission.get('scope') == 'B3_CORRECTNESS_AND_FIXED_FULL_N_TWO_PASS' and
                str(admission.get('claim_id', '')).startswith('B3_NEW_ARTIFICIAL'), 'ARTIFICIAL_ADMISSION_METADATA')
        for s, claim in zip(m['pairs'], final['pairs']):
            report['pairs'].append(verify_pair(safe_path(root, s['path']), s, claim,
                safe_path(root, s['failure_control']) if s.get('failure_control') else None))
        if not all(r['status'] == 'VERIFIED_COMPLETE' for r in report['pairs']):
            statuses = {r['status'] for r in report['pairs']}
            report['status'] = next(v for v in ('INTEGRITY_MISMATCH', 'UNSUPPORTED_SCHEMA', 'TECHNICAL_FAILURE',
                                   'VERIFIED_PARTIAL_PREFIX', 'INSUFFICIENT_EVIDENCE') if v in statuses)
        else:
            lookup = {key(s): (s, c) for s, c in zip(m['pairs'], final['pairs'])}
            for arm in ARMS:
                reference, claim = lookup[('correctness', 'baseline', arm, 'A')]
                for phase, p in (('correctness', 'baseline'), ('profile', 'baseline'), ('profile', 'instrumented')):
                    a, ca = lookup[(phase, p, arm, 'A')]
                    b, cb = lookup[(phase, p, arm, 'B')]
                    require(ca['trace_sha256'] == cb['trace_sha256'], 'A_B_SUMMARY_TRACE_CLAIMS')
                    if phase == 'profile':
                        require(ca['trace_sha256'] == lookup[('profile', 'baseline', arm, 'A')][1]['trace_sha256'],
                                'PASS_SUMMARY_TRACE_CLAIMS')
                    for axis in ('X', 'Y'):
                        for left, right, prefix in ((a, b, False), (reference, a, True), (reference, b, True)):
                            check = cross_pair(safe_path(root, left['path']), safe_path(root, right['path']),
                                               axis, prefix_only=prefix)
                            report['cross_pairs'].append(dict(check, left=left['path'], right=right['path']))
                        if phase == 'profile' and p == 'instrumented':
                            old = lookup[('profile', 'baseline', arm, 'A')][0]
                            report['cross_pairs'].append(cross_pair(safe_path(root, old['path']),
                                                                   safe_path(root, a['path']), axis))
            require(all(c['status'] == 'PASS' for c in report['cross_pairs']), 'CROSS_PAIR_MISMATCH')
            report['status'] = 'VERIFIED_COMPLETE' if receipt is not None else 'INSUFFICIENT_EVIDENCE'
        report['completed_calls_by_phase_pass'] = {phase + '/' + p: sum(v['committed_trials'] for v in report['pairs']
            if v['pair'][:2] == [phase, p]) for phase, p in (('correctness', 'baseline'), ('profile', 'baseline'), ('profile', 'instrumented'))}
        report.update(receipt_present=receipt is not None, summary_file=summary_reader.stats,
                      summary=final, completion_receipt=receipt, contract_digest=m['contract_digest'])
        if receipt is None:
            report['issues'].append('MISSING_COMPLETION_RECEIPT')
    except Unsupported as exc:
        report.update(status='UNSUPPORTED_SCHEMA', issues=[str(exc)])
    except (Invalid, KeyError, TypeError, ValueError, OSError) as exc:
        report.update(status='INSUFFICIENT_EVIDENCE' if isinstance(exc, FileNotFoundError)
                      else 'INTEGRITY_MISMATCH', issues=[str(exc)])
    return report


class PendingParallelAdapter:
    status = 'PENDING_SERVER_CONTRACT'

    def verify(self, *_args, **_kwargs):
        return {'status': self.status, 'implemented': False,
                'reason': 'No published adopted parallel schema; no inferred worker/scheduler semantics.'}
