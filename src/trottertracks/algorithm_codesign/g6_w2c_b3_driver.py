"""B3 preparation driver. Importing this module never imports scientific source.

Only an external, separately approved one-shot admission port may enter science.
The CLI is read-only preflight. Tests use synthetic endpoints and fake clocks.
No authority, claim, marker, grant, ledger or scientific result is issued here.
"""
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction as F
from hashlib import sha256
import importlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
from types import ModuleType

from . import g6_w2c_cost_preparation as audit

ARMS = audit.ARMS
AXES = ('X', 'Y')
LAYERS = ('A', 'B')
PASSES = ('baseline', 'instrumented')
INPUT = {'id': 'P5_general_order', 'p': ['1/5', '3/10', '1/2'], 'x': '5/7', 'm': 5}
POLICY = {'root_bits': 256, 'probability_bits': 160, 'eta': '1/1000000000000',
          'rho': '1/1000000000000', 'epsilon_axis': '1/200', 'primitive_error': '1/1000000',
          'alpha_axis': '1/320', 'alpha_familywise_16_axes': '1/20', 'sigma': 1}
# Saved values, not an evaluation of budget(). Provenance is checked by load_saved().
SAVED_N = dict(zip(ARMS, (1174526, 894664, 880312, 1014322)))
SCHEMA = 'g6_w2c_b3_driver_candidate_v1'
FROZEN_PINS = dict(zip(('g7_generator.py', 'return_aggregation.py', 'g7_provider.py',
                        'g6_f3_contract.py', 'g6_f3_precision.py', 'g6_f3_generator.py'), (
    'bb4177d26f1cee9721c2b083de01f4aa309f3ad39f22597cfae3b662881269fb',
    'f716d3c69fc1063e47b7618022662715f6b34a9c1f35cc3e7ed26813f5ac69f0',
    '17fff6d4d9fa9c9728091f6f01a4e6af1a389e8949c6e4a2ec6e3112f2fe3c62',
    '754c27f72f3356b3072ced4a00bef174b8922d2efedd0eff18d86a887fa598e1',
    'c2136326042fcf95240ecae41436cd953a1f857543ec9797e57f8fcbbe158706',
    '2ccce19571cdf33527abc92f049f35ef4e0d33d8a876132b2d1d70f6e63ad751')))


class Stop(RuntimeError):
    """Terminal technical/admission failure, never a random zero."""


def need(ok, reason):
    if not ok:
        raise Stop(reason)


def exact(value):
    """Normalize evidence maps without changing exact phase, order or weight."""
    if isinstance(value, dict):
        return {str(k): exact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(exact(v) for v in value)
    return value


def identity(value):
    return audit.fingerprint(exact(value))


def clock():
    return time.process_time_ns(), time.monotonic_ns()


def delta(before, after):
    need(len(before) == len(after) == 2 and
         all(type(v) is int for v in before + after) and
         all(b >= a for a, b in zip(before, after)), 'INVALID_CLOCK')
    return tuple(b-a for a, b in zip(before, after))


class Meter:
    """Bounded aggregate exclusive spans; nested children are subtracted once.

    Fine profiling and coarse spans are separate partitions, never added together.
    Measurements include observer effects; paired baseline is reported separately.
    """
    def __init__(self, read_clock):
        self.clock = read_clock
        self.stack, self.rows, self.stopped = [], {}, False

    @contextmanager
    def span(self, name):
        need(not self.stopped, 'METER_STOPPED')
        item = [name, self.clock(), [0, 0]]
        self.stack.append(item)
        try:
            yield item
        finally:
            try:
                need(self.stack and self.stack[-1] is item, 'SPAN_ORDER')
                elapsed = delta(item[1], self.clock())
                own = tuple(v-c for v, c in zip(elapsed, item[2]))
                need(min(own) >= 0, 'NEGATIVE_EXCLUSIVE_COST')
                self.stack.pop()
                if self.stack:
                    self.stack[-1][2] = [a+b for a, b in zip(self.stack[-1][2], elapsed)]
                row = self.rows.setdefault(item[0], {'calls': 0, 'cpu_ns': 0, 'wall_ns': 0})
                row['calls'] += 1
                row['cpu_ns'] += own[0];row['wall_ns'] += own[1]
                item.append(own)
            except BaseException:
                self.stopped = True
                raise

    def reconcile(self, total):
        need(not self.stack, 'OPEN_SPAN')
        sums = tuple(sum(row[k] for row in self.rows.values()) for k in ('cpu_ns', 'wall_ns'))
        need(all(v <= t for v, t in zip(sums, total)), 'DOUBLE_COUNTED_STAGE_COST')
        return {'exclusive_sum_ns': sums, 'outer_total_ns': total,
                'unattributed_and_observer_residual_ns': tuple(t-v for v, t in zip(sums, total)),
                'unknown_bit_operations': None, 'includes_observer_effect': True}


class FunctionObserver:
    """Own-thread Python call/return observer; never patches frozen file bytes.

    Names are filename:function, including nested functions. C arithmetic stays in
    its Python caller. Call counts and CPU/wall are not exact bit-operation counts.
    No per-call list is retained. A pre-existing profiler is never overwritten.
    """
    def __init__(self, paths, read_clock=clock):
        self.paths = {str(Path(p).resolve()) for p in paths}
        self.meter = Meter(read_clock)
        self.frames = []

    def hook(self, frame, event, arg):
        if event not in ('call', 'return') or frame.f_code.co_filename not in self.paths:
            return
        name = getattr(frame.f_code, 'co_qualname', frame.f_code.co_name)
        if isinstance(frame.f_locals.get('self'), (Meter, FunctionObserver)) or name in ('clock', 'delta', 'need'):
            return
        if event == 'call':
            cm = self.meter.span(Path(frame.f_code.co_filename).name + ':' +
                                 str(frame.f_code.co_firstlineno) + ':' + name)
            cm.__enter__();self.frames.append((frame, cm))
        else:
            need(self.frames and self.frames[-1][0] is frame, 'PROFILE_STACK_MISMATCH')
            self.frames.pop()[1].__exit__(None, None, None)

    def __enter__(self):
        need(sys.getprofile() is None, 'EXISTING_THREAD_PROFILER')
        sys.setprofile(self.hook)
        return self

    def __exit__(self, *args):
        sys.setprofile(None)
        need(not self.frames, 'INCOMPLETE_PROFILE_PREFIX')


class ChargedCache:
    """Explicit opt-in cache primitive. Production angle/law caches are disabled."""
    def __init__(self, policy_id, meter):
        self.policy_id, self.meter = policy_id, meter
        self.values, self.hits, self.misses = {}, 0, 0

    def get(self, policy_id, key, acquire):
        need(policy_id == self.policy_id, 'CACHE_POLICY_MISMATCH')
        with self.meter.span('cache.lookup'):
            hit = key in self.values
        if hit:
            self.hits += 1
            return self.values[key]
        self.misses += 1
        with self.meter.span('cache.acquire'):
            value = acquire()
        need(value is not None, 'UNKNOWN_CACHE_COST_OR_VALUE')
        with self.meter.span('cache.insert'):
            self.values[key] = value
        return value


class Stream:
    """Exclusive new output, incremental chain hash, bounded user-space buffer.

    A line is a hash envelope over exact canonical payload. Partial trailing lines
    after OS kill are a prefix, not valid completion. No retry/open-for-append.
    Output/log limits are contract limits, not CPU/RAM/worker/wall caps.
    """
    def __init__(self, path, meter, *, byte_limit, flush_rows):
        need(type(byte_limit) is int and byte_limit > 0 and
             type(flush_rows) is int and flush_rows > 0, 'OUTPUT_POLICY')
        self.path, self.meter = Path(path), meter
        self.limit, self.flush_rows = byte_limit, flush_rows
        self.bytes, self.rows, self.peak_line_bytes = 0, 0, 0
        self.previous = '0'*64
        self.file_hash = sha256()
        self.file_hash_covers_written_bytes = True
        self.failed = None
        self.closed = False
        self.file = self.path.open('xb', buffering=0)

    def emit(self, payload):
        need(not self.closed and self.failed is None, 'OUTPUT_STOPPED')
        stage = 'SERIALIZATION'
        try:
            with self.meter.span('save.serialize'):
                text = audit.canonical(exact(payload))
            with self.meter.span('save.hash'):
                stage = 'CHAIN_HASH'
                digest = sha256((self.previous + text).encode()).hexdigest()
                line = (json.dumps({'previous': self.previous, 'sha256': digest, 'payload': text},
                                   separators=(',', ':'))+'\n').encode()
            need(self.bytes+len(line) <= self.limit, 'RESULT_BYTE_LIMIT')
            stage = 'WRITE'
            with self.meter.span('save.write'):
                view = memoryview(line)
                while view:
                    written = self.file.write(view)
                    need(type(written) is int and written > 0, 'SHORT_WRITE')
                    self.bytes += written
                    with self.meter.span('save.file_hash'):
                        stage = 'FILE_HASH'
                        self.file_hash_covers_written_bytes = False
                        self.file_hash.update(view[:written])
                        self.file_hash_covers_written_bytes = True
                    stage = 'WRITE'
                    view = view[written:]
            self.rows += 1;self.previous = digest
            self.peak_line_bytes = max(self.peak_line_bytes, len(line))
            if self.rows % self.flush_rows == 0:
                stage = 'FLUSH'
                self.flush()
        except BaseException as exc:
            # Keep committed prefix plus failure descriptor even if disk is full.
            self.failed = {'type': type(exc).__name__, 'reason': str(exc), 'stage': stage,
                           'committed_rows': self.rows, 'written_bytes': self.bytes,
                           'last_complete_hash': self.previous,
                           'file_hash_covers_written_bytes': self.file_hash_covers_written_bytes,
                           'mandatory_STOP': True}
            raise

    def flush(self):
        with self.meter.span('save.flush'):
            self.file.flush()  # Unbuffered writes; no claim of fsync durability.

    def close(self):
        if not self.closed:
            try:
                self.flush()
            finally:
                with self.meter.span('save.finalize'):
                    self.file.close();self.closed = True


def check_pins(root, pins):
    return audit.check_source_files(root, pins)


def write_control_metadata(path, value, byte_limit):
    """Bound the two possible control/failure logs to half the shared log budget."""
    need(type(byte_limit) is int and byte_limit > 0, 'LOG_POLICY')
    data = (json.dumps(value, indent=2)+'\n').encode()
    complete = len(data) <= byte_limit
    if not complete:
        data = (json.dumps({'kind': 'LOG_BYTE_LIMIT', 'original_document_sha256': sha256(data).hexdigest(),
                            'original_bytes': len(data), 'mandatory_STOP': True})+'\n').encode()
    need(len(data) <= byte_limit, 'LOG_BUDGET_TOO_SMALL_FOR_FAILURE_DESCRIPTOR')
    with Path(path).open('xb') as f:f.write(data)
    return complete


def load_saved(raw_bytes, expected_sha):
    """Verbatim saved N and fixed input check; no budget/reference reexecution."""
    need(sha256(raw_bytes).hexdigest() == expected_sha, 'SAVED_RESULT_SHA_MISMATCH')
    raw = json.loads(raw_bytes)
    selected = [row for row in raw['rows'] if row['input'] == INPUT['id']]
    rows = {row['arm']: row for row in selected}
    need(len(selected) == 4 and set(rows) == set(ARMS), 'SAVED_ARM_SET')
    for arm, row in rows.items():
        need(all(row[k] == INPUT[k] for k in ('p', 'x', 'm')), 'SAVED_INPUT_MISMATCH')
        b = row['budget']
        need(type(b['N_per_axis']) is int and b['N_per_axis'] == SAVED_N[arm] and
             b['quantum_call_hard_cap_per_axis'] == SAVED_N[arm], 'SAVED_N_MISMATCH')
        need(b['alpha_axis'] == POLICY['alpha_axis'] and
             b['alpha_familywise_16_axes'] == POLICY['alpha_familywise_16_axes'] and
             b['uses_reference_B_new_or_m2'] is False, 'SAVED_BUDGET_POLICY')
    return rows


def schedule():
    return tuple({'arm': arm, 'layer': layer, 'axis': axis, 'N': SAVED_N[arm]}
                 for arm in ARMS for layer in LAYERS for axis in AXES)


def candidate_errors(candidate):
    """Static preflight, deliberately no imports of frozen scientific modules."""
    errors = []
    for name, expected in (('schema', SCHEMA), ('input', INPUT), ('policy', POLICY),
                           ('cells', list(schedule()))):
        if candidate.get(name) != expected:errors.append(name.upper()+'_MISMATCH')
    for name, digest in FROZEN_PINS.items():
        if candidate.get('source_pins', {}).get('src/trottertracks/algorithm_codesign/'+name) != digest:
            errors.append('FROZEN_SOURCE_PIN_MISMATCH:'+name)
    output = candidate.get('output_relative')
    if not (type(output) is str and output and not Path(output).is_absolute() and '..' not in Path(output).parts):
        errors.append('OUTPUT_NAMESPACE_REQUIRED')
    if candidate.get('adopted') is not True:errors.append('CONTRACT_NOT_ADOPTED')
    for key in ('angle', 'seed', 'schedule_and_output', 'source_release', 'external_admission'):
        if candidate.get('decisions', {}).get(key) is None:errors.append('UNADOPTED_'+key.upper())
    if candidate.get('contract_grants_execution') is not False:
        errors.append('CANDIDATE_CANNOT_GRANT_EXECUTION')
    return tuple(errors)


def validate_angle(policy):
    need(type(policy) is dict and set(policy) == {'tau', 'K_phi', 'J', 'mode'}, 'ANGLE_POLICY_REQUIRED')
    need(policy['mode'] == 'MATHEMATICAL_ENCLOSURE_ONLY_NO_PHYSICAL_CHANGE', 'ANGLE_ERROR_ALLOCATION_UNADOPTED')
    need(type(policy['tau']) is str and F(policy['tau']) > 0 and
         type(policy['K_phi']) is int and policy['K_phi'] > 0 and
         type(policy['J']) is int and policy['J'] > 0, 'ANGLE_PRECISION_REQUIRED')
    return {'angle_tau': F(policy['tau']), 'angle_root_bits': policy['K_phi'], 'angle_terms': policy['J']}


class Boundary:
    """One-use process-local guard plus external atomic one-shot port.

    This does not create/consume real claims in preparation. Port implementation,
    approved release and new claim ID must be supplied by later authorization.
    A boolean/edited candidate is never an approval credential.
    """
    def __init__(self):
        self.state = 'NEW'

    def enter(self, candidate, approval_port=None):
        need(self.state == 'NEW', 'STOP_NO_RETRY_OR_RESUME')
        self.state = 'STOPPED'  # Any admission failure is terminal for this object.
        errors = candidate_errors(candidate)
        need(not errors, ','.join(errors))
        validate_angle(candidate['decisions']['angle'])
        need(approval_port is not None and callable(getattr(approval_port, 'admit_once', None)),
             'SEPARATE_EXPLICIT_EXECUTION_APPROVAL_REQUIRED')
        receipt = approval_port.admit_once(identity(candidate))
        need(type(receipt) is dict and receipt.get('contract_digest') == identity(candidate) and
             receipt.get('approved') is True and receipt.get('fresh_claim') is True and
             receipt.get('previously_consumed') is False and receipt.get('retry') == 0 and
             receipt.get('resume') is False and type(receipt.get('claim_id')) is str and
             receipt['claim_id'].startswith('B3_NEW_') and receipt.get('scope') ==
             'B3_CORRECTNESS_AND_FIXED_FULL_N_TWO_PASS', 'INVALID_OR_OLD_ONE_SHOT_ADMISSION')
        self.state = 'ENTERED'
        return receipt

    def stop(self):
        self.state = 'STOPPED'


@dataclass(frozen=True)
class Outcome:
    kind: str
    event: object = None
    evidence: object = None

    def validate(self):
        need(self.kind in ('ZERO_R', 'ZERO_C', 'EVENT'), 'TYPED_FAILURE_IS_NOT_ZERO')
        need((self.event is not None) == (self.kind == 'EVENT'), 'OUTCOME_EVENT_MISMATCH')


class LegacyEndpoint:
    """Actual lazy G7/F3 connection, exercised only with mock modules in tests.

    Wrapping an owned object's packet/event records the branch/query arguments
    without extra draws, enumeration, sorting, roots or altered return values.
    Frozen files and other instances/processes are untouched.
    """
    def __init__(self, modules, arm, layer, angle, source_check, *, permit=None):
        real = any(isinstance(m, ModuleType) for m in modules.values())
        need((real and isinstance(permit, Boundary) and permit.state == 'ENTERED') or
             (not real and modules.get('synthetic_only') is True), 'ENDPOINT_EXECUTION_ADMISSION_REQUIRED')
        self.modules, self.arm, self.layer = modules, arm, layer
        self.stopped = False
        self.angle, self.source_check = angle, source_check
        self.packet_calls, self.event_args = 0, None
        params = {k: (F(v) if k in ('eta', 'rho') else v) for k, v in POLICY.items()
                  if k in ('root_bits', 'probability_bits', 'eta', 'rho')}
        p, x, m = tuple(map(F, INPUT['p'])), F(INPUT['x']), INPUT['m']
        self.session = None
        if arm == 'full_return' and layer == 'B':
            self.session = modules['f3'].prepare(p, x, m, sigma=POLICY['sigma'],
                                                **params, **validate_angle(angle))
            self.g = self.session._generator
        else:
            self.g = modules['g7'].make_generator(p, x, m, arm, **params)
        if arm == 'full_return':
            original = self.g.packet
            def packet(word):
                self.packet_calls += 1
                return original(word)
            self.g.packet = packet
        else:
            original = self.g.event
            def event(index, word, child):
                self.event_args = (index, tuple(word), child)
                return original(index, word, child)
            self.g.event = event
        self.preflight = None
        self.overlay = None
        if layer == 'B':
            if arm == 'full_return':
                self.preflight = self.session.preflight.digest
            else:
                self.preflight = identity(audit.canonical_snapshot(self.g, modules['g7']._law))
            self.overlay = audit.AuditParity(arm, {'input': INPUT, 'policy': POLICY, 'angle': angle},
                                             self.snapshot, source_check)
        self.initial_state = identity(self.snapshot())

    def snapshot(self):
        if self.arm == 'full_return':
            return self.modules['f3']._state(self.g)
        return audit.canonical_state(self.g)

    def shared_identity(self):
        return identity({'state': self.snapshot(), 'input': INPUT, 'policy': POLICY,
                         'arm': self.arm, 'layer': self.layer, 'angle': self.angle,
                         'source': self.source_check(), 'preflight': self.preflight})

    def sample(self, bits, trial_id):
        need(not self.stopped, 'ENDPOINT_STOP_NO_RETRY')
        try:return self._sample(bits, trial_id)
        except BaseException:
            self.stopped = True
            raise

    def _sample(self, bits, trial_id):
        self.packet_calls, self.event_args = 0, None
        captured = {}
        def operation():
            if self.session is not None:
                record = self.session.sample(bits, trial_id=trial_id)
                data = record.read();captured['f3'] = record
                if data['kind'] == 'EVENT':
                    kind, event = 'EVENT', data['event']
                else:
                    need(data['kind'] == 'ZERO' and data['reason'] in ('RAW_NONREDUCED', 'COIN_REJECT'),
                         'F3_UNEXPECTED_OUTCOME')
                    kind = 'ZERO_R' if data['reason'] == 'RAW_NONREDUCED' else 'ZERO_C'
                    event = None
            else:
                event = self.g.sample(bits)
                kind = ('EVENT' if event is not None else
                        'ZERO_R' if self.packet_calls == 0 else 'ZERO_C')
                need(event is not None or self.arm == 'full_return', 'CANONICAL_CANNOT_RANDOM_ZERO')
            captured.update(kind=kind, event=event)
            if self.layer == 'B' and event is not None:
                audit.event_witness(event, full=self.arm == 'full_return', degree_cap=INPUT['m'], labels=3)
            return exact({'kind': kind, 'event': event,
                          'f3': captured['f3'].document if 'f3' in captured else None})
        record = self.overlay.invoke(operation) if self.overlay is not None else operation()
        return Outcome(captured['kind'], captured['event'],
                       {'common': record.text if self.overlay is not None else None,
                        'f3': captured.get('f3')})

    def angle_for(self, outcome):
        need(not self.stopped, 'ENDPOINT_STOP_NO_RETRY')
        try:return self._angle_for(outcome)
        except BaseException:
            self.stopped = True
            raise

    def _angle_for(self, outcome):
        need(self.layer == 'B' and outcome.kind == 'EVENT', 'ANGLE_ONLY_B_ACCEPTED')
        if self.session is not None:
            # Retains all original F3 re-guard, event/packet requery and evidence.
            return self.session.angle_for_event(outcome.evidence['f3'])
        event = outcome.event
        def operation():
            need(self.event_args is not None, 'CANONICAL_GROUP_BINDING_MISSING')
            expected = self.g.event(*self.event_args)
            need(identity(expected) == identity(event), 'CANONICAL_EVENT_REQUERY_MISMATCH')
            angle = self.modules['precision'].enclose_atan(event['ratio'],
                tau=F(self.angle['tau']), root_bits=self.angle['K_phi'], terms=self.angle['J'],
                sigma=POLICY['sigma'], context={'event_id': identity(event), 'input': INPUT,
                                               'policy': POLICY, 'preflight': self.preflight})
            return angle.document
        return self.overlay.invoke(operation)


def seed_for(candidate, arm, axis):
    """Common A/B/pass prefix for each arm/axis, distinct X/Y and old diagnostics."""
    rule = candidate['decisions']['seed']
    need(arm in ARMS and axis in AXES, 'SEED_CELL_IDENTITY')
    need(type(rule) is dict and rule.get('algorithm') == 'FROZEN_G7_SHA256_COUNTER' and
         type(rule.get('namespace')) is str and rule['namespace'].startswith('B3_FULL_N_NEW_') and
         rule.get('IID_claim') is False, 'NEW_DESCRIPTIVE_SEED_RULE_REQUIRED')
    return rule['namespace'] + '/' + arm + '/' + axis


def memory_snapshot():
    """Own process only. Sampled RSS/HWM, no other process or worker inspection."""
    try:
        fields = {}
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith(('VmRSS:', 'VmHWM:')):
                k, v, unit = line.split();need(unit == 'kB', 'RSS_UNIT')
                fields[k[:-1]+'_bytes'] = int(v)*1024
        return fields
    except OSError:
        return {'VmRSS_bytes': None, 'VmHWM_bytes': None}


def run_pair(endpoint, bits_factory, cells, stream, meter, *, phase, pass_name,
             correctness_trials, memory_read=memory_snapshot):
    """Streaming execution engine with supplied endpoints. No RNG construction here.

    B accepts angle work per accepted event. A returns frozen ratios only.
    All completed/failed prefixes are retained; failure halts the whole pair.
    """
    need(len(cells) == 2 and tuple(c['axis'] for c in cells) == AXES and
         all((c['arm'], c['layer'], c['N']) ==
             (cells[0]['arm'], cells[0]['layer'], cells[0]['N']) for c in cells), 'PAIR_AXIS_N_BINDING')
    # Published as one reference only after emit returns. Candidate hash/counter
    # copies never mutate the last completed prefix on a failed attempt.
    committed = {'counts': Counter(), 'digest': sha256(), 'last_trial': None}
    axis_summaries = [];max_rss = None
    current_result = current_event = pending = None

    def context(cell=None, trial=None, stage='PAIR_IDENTITY'):
        return {'axis': cell['axis'] if cell else None, 'trial_index': trial,
                'trial_id': (f"{phase}/{pass_name}/{cell['arm']}/{cell['layer']}/{cell['axis']}/{trial}"
                             if cell is not None and trial is not None else None),
                'stage': stage, 'result_acquired': False, 'event_present': False,
                'record_constructed': False, 'candidate_ready': False,
                'stream_emit_started': False, 'stream_emit_returned': False}

    def stream_prefix():
        # Observed writes only: no reread or fsync, and a partial tail may exist.
        try:
            h = getattr(stream, 'file_hash', None)
            return {'complete_rows': getattr(stream, 'rows', None),
                    'written_bytes_including_possible_partial_tail': getattr(stream, 'bytes', None),
                    'last_complete_chain_sha256': getattr(stream, 'previous', None),
                    'written_bytes_sha256': (h.hexdigest() if h is not None and
                                              getattr(stream, 'file_hash_covers_written_bytes', True) else None),
                    'file_hash_covers_written_bytes': getattr(stream, 'file_hash_covers_written_bytes', None),
                    'failure': getattr(stream, 'failed', None),
                    'readback_verified': False, 'fsync_durability_verified': False}
        except BaseException as error:
            return {'status': 'UNAVAILABLE', 'type': type(error).__name__, 'reason': str(error)}

    current = context()
    try:
        with meter.span('setup.share_identity'):
            shared = endpoint.shared_identity()
        for cell in cells:
            current_result = current_event = pending = None
            current = context(cell, stage='AXIS_IDENTITY')
            committed = committed | {'axis_counts': Counter(), 'axis_digest': sha256(),
                                     'prefix_digest': sha256()}
            with meter.span('setup.axis_identity'):
                need(endpoint.shared_identity() == shared, 'X_Y_SETUP_SHARING_MISMATCH')
            current['stage'] = 'BITS_FACTORY'
            bits = bits_factory(cell['axis'])
            current['stage'] = 'TRIAL_BUDGET'
            n = correctness_trials if phase == 'correctness' else cell['N']
            need(type(n) is int and n > 0, 'INVALID_TRIAL_BUDGET')
            for trial in range(n):
                current_result = current_event = pending = None
                current = context(cell, trial, 'SAMPLING')
                trial_id = current['trial_id']
                with meter.span('online.generator_and_audit') as trial_span:
                    current_result = result = endpoint.sample(bits, trial_id)
                    current['result_acquired'] = True
                    current['stage'] = 'RESULT_ACCESS'
                    current_event = result.event
                    current['event_present'] = current_event is not None
                    current['stage'] = 'RESULT_VALIDATION'
                    result.validate()
                    trial_span[0] = 'online.'+result.kind+'.generator_and_audit'
                if phase == 'correctness' and result.kind == 'EVENT':
                    current['stage'] = 'EVENT_VALIDATION'
                    with meter.span('correctness.event_invariants'):
                        audit.event_witness(result.event, full=cell['arm'] == 'full_return',
                                            degree_cap=INPUT['m'], labels=3)
                current['stage'] = 'RECORD_CONSTRUCTION'
                payload = {'trial_id': trial_id, 'axis': cell['axis'], 'kind': result.kind,
                           'event': result.event, 'evidence': None, 'angle': None,
                           'generator_exclusive_ns': trial_span[3]}
                if result.kind == 'EVENT' and cell['layer'] == 'B':
                    current['stage'] = 'ANGLE_AND_REQUERY'
                    with meter.span('online.EVENT.angle_and_requery'):
                        a = endpoint.angle_for(result)
                        payload['angle'] = getattr(a, 'document', getattr(a, 'text', a))
                with meter.span('online.'+result.kind+'.evidence'):
                    current['stage'] = 'EVIDENCE_CONSTRUCTION'
                    if isinstance(result.evidence, dict):
                        payload['evidence'] = {k: getattr(v, 'document', v)
                                               for k, v in result.evidence.items()}
                    else:payload['evidence'] = result.evidence
                    current['record_constructed'] = True
                    event_text = audit.canonical(exact({'kind': result.kind, 'event': result.event})).encode()
                    current['stage'] = 'CANDIDATE_DIGEST'
                    pending = {'counts': committed['counts'].copy(),
                               'digest': committed['digest'].copy(),
                               'axis_counts': committed['axis_counts'].copy(),
                               'axis_digest': committed['axis_digest'].copy(),
                               'prefix_digest': committed['prefix_digest'].copy(),
                               'last_trial': {'axis': cell['axis'], 'trial_id': trial_id,
                                              'trial_index': trial}}
                    pending['axis_digest'].update(event_text);pending['digest'].update(event_text)
                    if trial < correctness_trials:pending['prefix_digest'].update(event_text)
                    pending['counts'][result.kind] += 1;pending['axis_counts'][result.kind] += 1
                    if phase == 'profile' and trial >= correctness_trials:
                        # All evidence is constructed/hashed; retain exact full
                        # documents for first 16/axis plus typed-failure context.
                        payload = {'trial_id': trial_id, 'axis': cell['axis'], 'kind': result.kind,
                                   'event_sha256': sha256(event_text).hexdigest(),
                                   'evidence_sha256': identity(payload['evidence']),
                                   'angle_sha256': identity(payload['angle']),
                                   'generator_exclusive_ns': trial_span[3],
                                   'retention': 'HASHED_FULL_DOCUMENTS_NOT_RETAINED'}
                    current['record_constructed'] = True
                    current['candidate_ready'] = True
                current['stage'] = 'STREAM_EMIT'
                current['stream_emit_started'] = True
                stream.emit(payload)
                current['stream_emit_returned'] = True
                committed = pending  # Single publication of counts, hashes and last trial.
                current['stage'] = 'TRIAL_COMPLETED'
                if (trial+1) % 4096 == 0:
                    current['stage'] = 'MEMORY_OBSERVATION'
                    with meter.span('observe.memory'):
                        sample = memory_read()
                        rss = sample.get('VmRSS_bytes')
                        if rss is not None:max_rss = max(max_rss or 0, rss)
                    stream.emit({'memory_sample': sample, 'completed_trial': trial+1, 'axis': cell['axis']})
            current_result = current_event = pending = None
            current = context(cell, stage='AXIS_COMPLETION')
            axis_summaries.append({'axis': cell['axis'], 'N_executed': n, 'counts': dict(committed['axis_counts']),
                                   'event_trace_sha256': committed['axis_digest'].hexdigest(),
                                   'first_correctness_prefix_sha256': committed['prefix_digest'].hexdigest(),
                                   'H_bits_supplied': getattr(bits, 'consumed_bits', None)})
            if phase == 'correctness':
                need(committed['axis_counts']['EVENT'] > 0, 'CORRECTNESS_ACCEPTED_PATH_NOT_OBSERVED_NO_EXTRA_TRIALS')
        current = context(stage='FINAL_IDENTITY')
        with meter.span('setup.final_identity'):
            need(endpoint.shared_identity() == shared, 'FINAL_STATE_CHANGED')
        return {'counts': dict(committed['counts']), 'axes': axis_summaries, 'trace_sha256': committed['digest'].hexdigest(),
                'shared_identity': shared, 'population_cost_certificate': False,
                'periodic_max_observed_RSS_bytes': max_rss,
                'RSS_sampling_is_not_continuous_peak': True}
    except BaseException as exc:
        completed = (current['trial_id'] is not None and committed['last_trial'] is not None and
                     committed['last_trial']['trial_id'] == current['trial_id'])
        failure = {'kind': 'TYPED_FAILURE', 'type': type(exc).__name__, 'reason': str(exc),
                   'failure_provenance_schema': 'g6_w2c_b3_f01_failure_v1',
                   'completed_prefix': dict(committed['counts']), 'completed_axes': axis_summaries,
                   'trace_sha256': committed['digest'].hexdigest(),
                   'trace_scope': 'COMPLETED_TRIALS_AFTER_NORMAL_EMIT_RETURN',
                   'last_completed_trial': committed['last_trial'],
                   'current_attempt': current | {'trial_completed': completed},
                   'inflight_event': current_event,
                   'candidate_trace_sha256': (pending['digest'].hexdigest()
                                               if current['candidate_ready'] and pending is not None else None),
                   'stream_prefix_before_failure_record': stream_prefix(), 'mandatory_STOP': True,
                   'is_random_zero': False, 'retry': 0, 'resume': False}
        if hasattr(exc, 'to_dict'):
            try:failure['frozen_failure_evidence'] = exc.to_dict()
            except BaseException as error:
                failure['frozen_failure_evidence_error'] = {'type': type(error).__name__, 'reason': str(error)}
        # Preserve the original exception and context even if saving failure fails.
        try:exc.b3_failure_provenance = failure
        except BaseException:pass
        try:stream.emit(failure)
        except BaseException as error:
            failure['failure_record_write_error'] = {'type': type(error).__name__, 'reason': str(error)}
            failure['stream_prefix_after_failure_record_attempt'] = stream_prefix()
        raise


def load_modules(root, pins):
    """Only after admission. Verify location and complete bytes before/after import."""
    check_pins(root, pins)
    package = __package__
    modules = {name: importlib.import_module(package+'.'+filename) for name, filename in
               (('g7', 'g7_generator'), ('f3', 'g6_f3_generator'), ('precision', 'g6_f3_precision'))}
    for module in modules.values():
        path = Path(module.__file__).resolve()
        need(path.is_relative_to(Path(root).resolve()), 'IMPORTED_SOURCE_LOCATION_MISMATCH')
    check_pins(root, pins)
    return modules


def isolated_pair(operation):
    """A fresh owned Linux child per arm/layer/pass, with one parent admission.

    No benchmark workers contend with one another. This protects per-process HWM
    and comparison conditions, without limiting available CPU/RAM/wall resources.
    Only our child is joined; no other process is inspected or manipulated.
    """
    need('fork' in multiprocessing.get_all_start_methods(), 'FRESH_LINUX_WORKER_REQUIRED')
    ctx = multiprocessing.get_context('fork')
    receive, send = ctx.Pipe(duplex=False)
    def worker():
        receive.close()
        try:send.send(('OK', operation()))
        except BaseException as exc:send.send(('FAIL', type(exc).__name__, str(exc)))
        finally:send.close()
    process = ctx.Process(target=worker)
    process.start();send.close()
    try:
        try:message = receive.recv()
        except EOFError as exc:raise Stop('OWN_WORKER_EXITED_WITHOUT_COMPLETION_PREFIX_ONLY') from exc
    finally:
        receive.close();process.join()
    need(message[0] == 'OK' and process.exitcode == 0, 'OWN_PAIR_WORKER_FAILED:'+str(message[1:]))
    return message[1]


def execute(candidate, *, root, output, approval_port=None, boundary=None):
    """Future approved correctness + paired full-N run; no CLI science shortcut.

    Output must be new. External port atomically admits this entire one-shot,
    including correctness; no per-cell retry. Current draft fails before imports.
    """
    guard = boundary or Boundary()
    target, all_results, stream = None, [], None
    try:
        receipt = guard.enter(candidate, approval_port)
        need(tuple(candidate['passes']) == PASSES and candidate['correctness_trials_per_cell'] == 16,
             'SCHEDULE_POLICY_MISMATCH')
        policy = candidate['decisions']['schedule_and_output']
        need(policy == candidate['schedule_proposal'], 'UNADOPTED_SCHEDULE_CHANGE')
        pins = candidate['source_pins']
        source_check = lambda: check_pins(root, pins)
        shared_meter = Meter(clock);shared_before = clock()
        # Verifies saved evidence before constructing any real generator.
        with shared_meter.span('shared.source_check'):
            source_check()
        evidence = candidate['saved_evidence']
        with shared_meter.span('shared.saved_budget_load'):
            rows = load_saved((Path(root)/evidence['path']).read_bytes(), evidence['sha256'])
        need(set(rows) == set(ARMS), 'MISSING_BUDGET_ROWS')
        with shared_meter.span('shared.module_load'):
            modules = load_modules(root, pins)
        target_path = Path(output).resolve()
        expected = (Path(root)/candidate['output_relative']).resolve()
        need(target_path == expected and expected.is_relative_to(Path(root).resolve()), 'OUTPUT_NAMESPACE_MISMATCH')
        need(not target_path.exists(), 'OUTPUT_EXISTS_NO_RETRY_OR_OVERWRITE')
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.mkdir(parents=False, exist_ok=False)
        target = target_path  # Failure handlers may write only after exclusive creation.
        shared_total = delta(shared_before, clock())
        traces, prefix_traces, written_bytes = {}, {}, 0
        # Paired workloads are run without competing benchmark workers. This is a
        # comparison schedule, not a CPU/RAM/worker cap. Resource policy is in docs.
        for phase, pass_name in (('correctness', 'baseline'), ('profile', 'baseline'),
                                 ('profile', 'instrumented')):
            for arm in ARMS:
                for layer in LAYERS:
                    def operation():
                        read_clock = clock
                        meter = Meter(read_clock);before = read_clock()
                        path = target/f'{phase}_{pass_name}_{arm}_{layer}.jsonl'
                        stream = Stream(path, meter, byte_limit=policy['result_bytes']-written_bytes,
                                        flush_rows=policy['flush_rows'])
                        with meter.span('observe.memory_baseline'):
                            memory_baseline = memory_snapshot()
                        observer = FunctionObserver([str(Path(root)/p) for p in pins
                                                     if p.endswith('.py')], read_clock)
                        context = observer if pass_name == 'instrumented' else _empty_context()
                        try:
                            with context:
                                with meter.span('setup.source_check'):
                                    source_check()
                                with meter.span('setup.constructor_and_preflight'):
                                    endpoint = LegacyEndpoint(modules, arm, layer,
                                        candidate['decisions']['angle'], source_check, permit=guard)
                                cells = [c for c in candidate['cells'] if c['arm'] == arm and c['layer'] == layer]
                                make_bits = lambda axis: modules['g7'].DeterministicBits(seed_for(candidate, arm, axis))
                                summary = run_pair(endpoint, make_bits, cells, stream, meter,
                                    phase=phase, pass_name=pass_name, correctness_trials=16)
                                with meter.span('observe.memory_final'):
                                    memory_final = memory_snapshot()
                                stream.close()
                            total = delta(before, read_clock())
                            summary.update(phase=phase, pass_name=pass_name, arm=arm, layer=layer,
                                coarse=meter.rows, reconciliation=meter.reconcile(total),
                                functions=observer.meter.rows if pass_name == 'instrumented' else None,
                                fine_reconciliation=observer.meter.reconcile(total) if pass_name == 'instrumented' else None,
                                output_bytes=stream.bytes, output_rows=stream.rows,
                                output_chain_sha256=stream.previous, file_sha256=stream.file_hash.hexdigest(),
                                peak_line_bytes=stream.peak_line_bytes,
                                memory=memory_final, memory_baseline=memory_baseline,
                                unknown_bit_operations=None,
                                cache={'angle_cache': 'DISABLED', 'law_cache': 'FROZEN_HELD_STATE_ONLY',
                                       'runtime_cache_bytes': 0, 'OS_file_cache': 'UNCONTROLLED_RECORDED_ENVIRONMENT'})
                            return summary
                        except BaseException as exc:
                            fail = {'kind': 'TYPED_FAILURE', 'type': type(exc).__name__, 'reason': str(exc),
                                    'coarse_prefix': meter.rows, 'stream_failure': stream.failed,
                                    'output_rows': stream.rows, 'output_bytes': stream.bytes,
                                    'last_complete_hash': stream.previous, 'mandatory_STOP': True,
                                    'cpu_wall_prefix_ns': delta(before, read_clock()),
                                    'control_log_self_write_cost_ns': None}
                            provenance = getattr(exc, 'b3_failure_provenance', None)
                            if provenance is not None:
                                try:fail['trial_failure_provenance'] = audit.canonical(exact(provenance))
                                except BaseException as error:
                                    fail['trial_failure_provenance_error'] = type(error).__name__+': '+str(error)
                            try:
                                write_control_metadata(path.with_suffix('.failure.json'), fail, policy['log_bytes']//2)
                            except BaseException:pass  # Never replace the original trial failure.
                            raise
                        finally:
                            if sys.exc_info()[0] is None:
                                stream.close()
                            else:
                                try:stream.close()
                                except BaseException:pass  # Cleanup must not hide the original cause.
                    with shared_meter.span('shared.worker_dispatch_and_receive'):
                        summary = isolated_pair(operation)
                    written_bytes += summary['output_bytes']
                    # Phase/order/weight/law trace must match A/B and paired passes.
                    key = (phase, arm)
                    need(key not in traces or traces[key] == summary['trace_sha256'], 'PAIRED_TRACE_MISMATCH')
                    traces[key] = summary['trace_sha256']
                    for axis in summary['axes']:
                        key = (arm, axis['axis'])
                        digest = axis['first_correctness_prefix_sha256']
                        need(key not in prefix_traces or prefix_traces[key] == digest,
                             'CORRECTNESS_PREFIX_TRACE_MISMATCH')
                        prefix_traces[key] = digest
                    all_results.append(summary)
        # Dispatch wall time contains child work. This shared parent coordinate
        # overlaps child wall and is never added to child process totals.
        shared_total = delta(shared_before, clock())
        final = {'schema': 'g6_w2c_b3_driver_result_v1', 'contract_digest': identity(candidate),
                 'admission': receipt, 'pairs': all_results, 'mandatory_STOP': True,
                 'population_cost_certificate': False, 'physical_RZ_certified': False,
                 'shared_setup': {'stages': shared_meter.rows,
                                  'reconciliation': shared_meter.reconcile(shared_total)},
                 'final_receipt_write_cost_ns': None,
                 'final_receipt_cost_status': 'SELF_MEASUREMENT_TRAILER_NOT_INCLUDED_IN_ARM_TOTALS'}
        # Final summary serialization/hash/write/flush are separate shared output.
        final_meter = Meter(clock);final_before = clock()
        final_stream = Stream(target/'summary.jsonl', final_meter,
                              byte_limit=policy['result_bytes']-written_bytes, flush_rows=1)
        try:final_stream.emit(final)
        finally:final_stream.close()
        final_total = delta(final_before, clock())
        receipt_data = {'summary_chain_sha256': final_stream.previous,
                        'shared_finalization_stages': final_meter.rows,
                        'reconciliation': final_meter.reconcile(final_total),
                        'receipt_self_write_cost_ns': None, 'mandatory_STOP': True}
        need(write_control_metadata(target/'completion_receipt.json', receipt_data, policy['log_bytes']//2),
             'COMPLETION_RECEIPT_TRUNCATED_STOP')
        return final
    except BaseException as exc:
        if target is not None and target.is_dir():
            failure = {'kind': 'TYPED_FAILURE', 'type': type(exc).__name__, 'reason': str(exc),
                       'completed_pairs': all_results,
                       'stream_failure': stream.failed if stream is not None else None,
                       'mandatory_STOP': True, 'retry': 0, 'resume': False}
            try:
                write_control_metadata(target/'failure_prefix.json', failure, policy['log_bytes']//2)
            except OSError:pass
        raise
    finally:
        guard.stop()


@contextmanager
def _empty_context():
    yield


def main(argv=None):
    """Read-only preflight only. No --run/--approve or authority-writing switch."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    args = parser.parse_args(argv)
    candidate = json.loads(args.candidate.read_text())
    errors = candidate_errors(candidate)
    print(json.dumps({'preflight_only': True, 'blockers': errors,
                      'real_science_started': False, 'mandatory_STOP': True}, indent=2))
    return 2 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
