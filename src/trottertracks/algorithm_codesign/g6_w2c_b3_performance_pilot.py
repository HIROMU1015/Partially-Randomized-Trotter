"""Unadopted artificial performance prototypes; no scientific execution entry.

Uses the published F01 run_pair with a deterministic, explicitly artificial
endpoint. Never imports G7/F3/ReturnKernel or creates production authorization.
"""
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from fractions import Fraction as F
from hashlib import sha256
import importlib.abc
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import resource
import sys
import time

sys.dont_write_bytecode = True
FORBIDDEN = ('g7_generator', 'g7_provider', 'g6_f3_contract', 'g6_f3_precision',
             'g6_f3_generator', 'return_aggregation', 'g9_p5', 'g9_native')
BLOCKED_IMPORTS = []


class ScienceImportBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.rsplit('.', 1)[-1].startswith(FORBIDDEN):
            BLOCKED_IMPORTS.append(fullname)
            raise RuntimeError('ARTIFICIAL_PILOT_FORBIDS_SCIENCE_IMPORT:' + fullname)


sys.meta_path.insert(0, ScienceImportBlocker())
from . import g6_w2c_b3_driver as D


def assert_no_science():
    loaded = [n for n in sys.modules if n.rsplit('.', 1)[-1].startswith(FORBIDDEN)]
    if loaded or BLOCKED_IMPORTS:
        raise RuntimeError(('FORBIDDEN_IMPORT', loaded, BLOCKED_IMPORTS))


def packed_once(value):
    """Fuse driver.exact and audit._pack; JSON sorts string keys exactly once."""
    if type(value) is F:
        return {'rat': [hex(value.numerator), hex(value.denominator)]}
    if type(value) is int:
        return {'int': hex(value)}
    if value is None or type(value) in (str, bool):
        return value
    if isinstance(value, dict):
        # str conversion and last-wins collisions match driver.exact.
        return {str(k): packed_once(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [packed_once(v) for v in value]
    raise D.audit.InvalidEvidence('nonexact or unsupported evidence type')


def canonical_fast(value, backend='python'):
    packed = packed_once(value)
    if backend == 'orjson':
        import orjson
        try:
            data = orjson.dumps(packed, option=orjson.OPT_SORT_KEYS)
            if data.isascii() and b'\x7f' not in data:
                return data.decode('ascii')
        except (TypeError, ValueError):
            pass  # Preserve ensure_ascii / surrogate behavior of the reference.
    return json.dumps(packed, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def atan_reference(z, terms):
    """Independent artificial-input copy of the frozen endpoint expression."""
    partial = sum(((-1)**j * z**(2*j+1)/(2*j+1) for j in range(terms)), F(0))
    nxt = (-1)**terms * z**(2*terms+1)/(2*terms+1)
    return partial + min(F(0), nxt), partial + max(F(0), nxt), partial, abs(nxt)


def atan_recurrence(z, terms):
    power, square, partial = z, -z*z, F(0)
    for j in range(terms):
        partial += power/(2*j+1)
        power *= square
    nxt = power/(2*terms+1)
    return partial + min(F(0), nxt), partial + max(F(0), nxt), partial, abs(nxt)


def atan_gmp(z, terms):
    import gmpy2
    q = gmpy2.mpq(z.numerator, z.denominator)
    power, square, partial = q, -q*q, gmpy2.mpq(0)
    for j in range(terms):
        partial += power/(2*j+1)
        power *= square
    nxt = power/(2*terms+1)
    values = partial + min(0, nxt), partial + max(0, nxt), partial, abs(nxt)
    return tuple(F(int(v.numerator), int(v.denominator)) for v in values)


class PrototypeStream(D.Stream):
    """Copied F01 emit with only canonical encoding replaced; no source patch.

    Nonzero buffering is a separately labelled R1-changing experiment. Its
    successful-write prefix refers to Python-buffer acceptance, unlike R1.
    """
    def __init__(self, *args, backend='python', buffer_bytes=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.backend = backend
        if buffer_bytes:
            self.file.close()
            self.file = self.path.open('r+b', buffering=buffer_bytes)

    def emit(self, payload):
        D.need(not self.closed and self.failed is None, 'OUTPUT_STOPPED')
        stage = 'SERIALIZATION'
        try:
            with self.meter.span('save.serialize'):
                text = canonical_fast(payload, self.backend)
            with self.meter.span('save.hash'):
                stage = 'CHAIN_HASH'
                digest = sha256((self.previous + text).encode()).hexdigest()
                line = (json.dumps({'previous': self.previous, 'sha256': digest, 'payload': text},
                                   separators=(',', ':'))+'\n').encode()
            D.need(self.bytes+len(line) <= self.limit, 'RESULT_BYTE_LIMIT')
            stage = 'WRITE'
            with self.meter.span('save.write'):
                view = memoryview(line)
                while view:
                    written = self.file.write(view)
                    D.need(type(written) is int and written > 0, 'SHORT_WRITE')
                    self.bytes += written
                    with self.meter.span('save.file_hash'):
                        stage = 'FILE_HASH'
                        self.file_hash_covers_written_bytes = False
                        self.file_hash.update(view[:written])
                        self.file_hash_covers_written_bytes = True
                    stage = 'WRITE'
                    view = view[written:]
            self.rows += 1
            self.previous = digest
            self.peak_line_bytes = max(self.peak_line_bytes, len(line))
            if self.rows % self.flush_rows == 0:
                stage = 'FLUSH'
                self.flush()
        except BaseException as exc:
            self.failed = {'type': type(exc).__name__, 'reason': str(exc), 'stage': stage,
                           'committed_rows': self.rows, 'written_bytes': self.bytes,
                           'last_complete_hash': self.previous,
                           'file_hash_covers_written_bytes': self.file_hash_covers_written_bytes,
                           'mandatory_STOP': True}
            raise


class ArtificialBits:
    """Synthetic counter namespace; no scientific RNG construction or import."""
    def __init__(self, seed):
        if not seed.startswith('ARTIFICIAL_B3_PERF/'):
            raise ValueError('ARTIFICIAL_NAMESPACE_REQUIRED')
        self.seed, self.counter, self.consumed_bits = seed, 0, 0

    def bits(self, n):
        if not 0 < n <= 256:
            raise ValueError('SYNTHETIC_BIT_WIDTH')
        value = int.from_bytes(sha256((self.seed + '/' + str(self.counter)).encode()).digest(), 'big')
        self.counter += 1
        self.consumed_bits += n
        return value >> (256-n)  # No bit pooling.


class Cancelled(RuntimeError):
    pass


class InjectedFailure(RuntimeError):
    pass


class Control:
    def __init__(self, ctx, barrier_parties=0):
        self.lock = ctx.Lock()
        self.failure_ns = ctx.Value('q', 0, lock=False)
        self.started = ctx.Value('q', 0, lock=False)
        self.barrier = ctx.Barrier(barrier_parties) if barrier_parties else None

    def start(self):
        with self.lock:
            if self.failure_ns.value:
                raise Cancelled('ARTIFICIAL_STOP_NO_NEW_TRIAL')
            now = time.monotonic_ns()
            self.started.value += 1
            return now

    def fail(self):
        with self.lock:
            if not self.failure_ns.value:
                self.failure_ns.value = time.monotonic_ns()
            return self.failure_ns.value


@dataclass(frozen=True)
class Task:
    key: str
    arm: str
    layer: str
    n: int
    replica: int = 0
    workload: str = 'mixed'
    variant: str = 'reference'
    pass_name: str = 'baseline'
    instrumented: bool = False
    buffer_bytes: int = 0
    failure_trial: int = -1
    track_starts: bool = False


class ArtificialEndpoint:
    def __init__(self, task, control):
        self.task, self.control = task, control
        self.started, self.completed = [], []
        self.first_sample = True
        self._state = {'ARTIFICIAL': True, 'arm': task.arm, 'replica': task.replica,
                       'fixed': [F(i, i+17) for i in range(1, 65)]}

    def shared_identity(self):
        return D.identity(self._state)

    def sample(self, bits, trial_id):
        if self.first_sample and self.control and self.control.barrier:
            self.control.barrier.wait(timeout=30)
        self.first_sample = False
        start = self.control.start() if self.control else time.monotonic_ns()
        if self.task.track_starts:
            self.started.append({'trial_id': trial_id, 'start_ns': start})
        if self.task.failure_trial >= 0 and trial_id.endswith('/X/'+str(self.task.failure_trial)):
            if self.control:
                self.control.fail()
            raise InjectedFailure('ARTIFICIAL_SAMPLE_FAILURE')
        choice = bits.bits(160)
        # Explicit CPU-bound work even on ZERO/reject paths; no sleep mock.
        x = choice
        rounds = {'light': 16, 'mixed': 96, 'heavy': 192}[self.task.workload]
        for j in range(rounds):
            x = ((x ^ (x >> 11)) * 6364136223846793005 + j) & ((1 << 192)-1)
        kind = ('ZERO_R', 'ZERO_C', 'EVENT', 'EVENT')[choice % 4]
        if kind != 'EVENT':
            return D.Outcome(kind, evidence={'ARTIFICIAL': True, 'token': x})
        width = {'light': 24, 'mixed': 96, 'heavy': 256}[self.task.workload]
        q = F((1 << width)+(x & 65535)+1, (1 << (width+2))+57)
        loops = {'light': 1, 'mixed': 6, 'heavy': 12}[self.task.workload]
        loops += D.ARMS.index(self.task.arm) * (0 if self.task.workload == 'light' else 2)
        r = q
        for j in range(loops):
            r = (r + F(j+1, j+31)) * F(j+7, j+9)
        # Synthetic packet/law sorting, reduction, exact weight and phase.
        law = sorted((F((x >> j) % 31+1, 97+j), j) for j in range(4 + D.ARMS.index(self.task.arm)*2))
        raw = tuple((choice >> j) % 3 for j in range(4))
        word = raw[::-1]
        event = {'raw_word': raw, 'word': word, 'circuit_time_word': word[::-1],
                 'phase_i_power': choice % 4, 'ratio': q, 'weight': r,
                 'proposal': F(2, 3), 'coefficient': r*F(2, 3), 'child': choice % 3,
                 'provider_calls': {0: 1, 1: 2}, 'ARTIFICIAL': True}
        evidence = {'ARTIFICIAL': True, 'law': law, 'packet': r, 'token': x}
        return D.Outcome('EVENT', event, evidence)

    def angle_for(self, result):
        q = result.event['ratio']
        terms = {'light': 4, 'mixed': 12, 'heavy': 32}[self.task.workload]
        function = {'reference': atan_reference, 'recurrence': atan_recurrence,
                    'gmp': atan_gmp, 'orjson': atan_gmp}[self.task.variant]
        # Artificial endpoint processing, NOT an F3 enclosure/certificate.
        values = function(q, terms)
        return {'ARTIFICIAL_NOT_AN_ENCLOSURE': True, 'endpoint_surrogate': values,
                'requery_surrogate': D.identity(result.event)}


def normalized_payload(payload):
    value = dict(payload)
    value.pop('generator_exclusive_ns', None)
    if 'memory_sample' in value:
        value['memory_sample'] = 'NONDETERMINISTIC_RSS_EXCLUDED'
    return value


class ObservedStream:
    def __init__(self, stream, endpoint):
        self.stream, self.endpoint = stream, endpoint

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def emit(self, payload):
        self.stream.emit(payload)
        if 'trial_id' in payload:
            if self.endpoint.task.track_starts:
                self.endpoint.completed.append({'trial_id': payload['trial_id'], 'end_ns': time.monotonic_ns()})


def readback(path):
    previous, count, h = '0'*64, 0, sha256()
    normalized_previous, normalized_hash = '0'*64, sha256()
    with Path(path).open('rb') as handle:
        for line in handle:
            h.update(line)
            row = json.loads(line)
            expected = sha256((previous+row['payload']).encode()).hexdigest()
            if row['previous'] != previous or row['sha256'] != expected:
                raise AssertionError('CHAIN_READBACK_MISMATCH')
            previous, count = expected, count+1
            payload = json.loads(row['payload'])
            if 'trial_id' in payload:
                text = json.dumps(normalized_payload(payload), sort_keys=True,
                                  separators=(',', ':'), ensure_ascii=True)
                normalized_digest = sha256((normalized_previous+text).encode()).hexdigest()
                normalized_line = (json.dumps({'previous': normalized_previous, 'sha256': normalized_digest,
                                              'payload': text}, separators=(',', ':'))+'\n').encode()
                normalized_hash.update(normalized_line)
                normalized_previous = normalized_digest
    return {'rows': count, 'last_chain': previous, 'file_sha256': h.hexdigest(),
            'bytes': Path(path).stat().st_size,
            'normalized_file_sha256': normalized_hash.hexdigest(),
            'normalized_chain_sha256': normalized_previous}


def perform_task(task, output, control=None):
    before = D.clock()
    baseline_memory = D.memory_snapshot()
    io_before = proc_io()
    endpoint = ArtificialEndpoint(task, control)
    meter = D.Meter(D.clock)
    path = Path(output)/(task.key+'.jsonl')
    options = dict(byte_limit=1 << 32, flush_rows=4096)
    stream = (D.Stream(path, meter, **options) if task.variant == 'reference' and not task.buffer_bytes else
              PrototypeStream(path, meter, backend='orjson' if task.variant == 'orjson' else 'python',
                              buffer_bytes=task.buffer_bytes, **options))
    observed = ObservedStream(stream, endpoint)
    cells = [{'arm': task.arm, 'layer': task.layer, 'axis': axis, 'N': task.n} for axis in D.AXES]
    def bits(axis):
        return ArtificialBits(f'ARTIFICIAL_B3_PERF/{task.replica}/{task.arm}/{axis}')
    observer = D.FunctionObserver([__file__]) if task.instrumented else None
    status, result, failure = 'OK', None, None
    setup_done = D.clock()
    try:
        with observer if observer else nullcontext():
            result = D.run_pair(endpoint, bits, cells, observed, meter, phase='profile',
                                pass_name=task.pass_name, correctness_trials=16)
    except BaseException as exc:
        status = 'CANCELLED' if isinstance(exc, Cancelled) else 'FAILED'
        failure = getattr(exc, 'b3_failure_provenance', {'reason': repr(exc)})
        if control and status == 'FAILED':
            control.fail()
    finally:
        stream.close()
    after = D.clock()
    elapsed = D.delta(before, after)
    assert_no_science()
    completed = sum(result['counts'].values()) if result else sum(failure['completed_prefix'].values())
    return {'task': asdict(task), 'status': status, 'result': result, 'failure': failure,
            'cpu_ns': elapsed[0], 'wall_ns': elapsed[1],
            'setup_cpu_ns': setup_done[0]-before[0], 'setup_wall_ns': setup_done[1]-before[1],
            'memory': D.memory_snapshot(), 'baseline_memory': baseline_memory, 'completed': completed,
            'proc_io_delta': {k: v-io_before.get(k, 0) for k, v in proc_io().items()},
            'bytes': stream.bytes, 'file_sha256': stream.file_hash.hexdigest(),
            'chain_sha256': stream.previous, 'rows': stream.rows,
            'stage_cost': meter.rows, 'stage_reconciliation': meter.reconcile(elapsed),
            'fine_cost': observer.meter.rows if observer else None,
            'started': endpoint.started, 'completed_times': endpoint.completed,
            'worker_pid': os.getpid(), 'end_ns': time.monotonic_ns(),
            'forbidden_imports': BLOCKED_IMPORTS.copy()}


def _child(task, output, control, send):
    try:
        send.send(perform_task(task, output, control))
    except BaseException as exc:
        control.fail()
        send.send({'task': asdict(task), 'status': 'WORKER_ERROR', 'error': repr(exc),
                   'completed': 0, 'bytes': 0})
    finally:
        send.close()


POOL_CONTROL = None


def _pool_init(control):
    global POOL_CONTROL
    POOL_CONTROL = control


def _pool_task(args):
    return perform_task(*args, control=POOL_CONTROL)


def scheduler(tasks, workers, output, reuse=False):
    """Owned fresh processes; atomic start gate; no retry/resume or kill-all.

    A failing worker stops NEW trial admission. Already admitted work may finish,
    so this is explicitly NOT the serial R1 failure contract. Reuse is separately
    measured and only used for success-path benchmarks.
    """
    if workers < 1 or len({t.key for t in tasks}) != len(tasks):
        raise ValueError('WORKER_COUNT_OR_OUTPUT_COLLISION')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'ARTIFICIAL_RUN.json').write_text(json.dumps({'artificial_only': True, 'retry': 0, 'resume': False}))
    ctx = mp.get_context('fork')
    control = Control(ctx, min(workers, len(tasks)) if any(t.failure_trial >= 0 for t in tasks) else 0)
    before = D.clock()
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    results, active, next_task = [], {}, 0
    launched = []
    memory_samples, aggregate_rss_peak = 0, 0
    def sample_owned_memory(pids):
        nonlocal memory_samples, aggregate_rss_peak
        total = 0
        for pid in [os.getpid(), *pids]:
            try:
                for line in Path(f'/proc/{pid}/status').read_text().splitlines():
                    if line.startswith('VmRSS:'):
                        total += int(line.split()[1])*1024
                        break
            except FileNotFoundError:
                pass
        memory_samples += 1
        aggregate_rss_peak = max(total, aggregate_rss_peak)
    if reuse:
        if any(t.failure_trial >= 0 for t in tasks):
            raise ValueError('FAILURE_TEST_REQUIRES_FRESH_SCHEDULER')
        with ctx.Pool(workers, initializer=_pool_init, initargs=(control,)) as pool:
            pending = pool.map_async(_pool_task, [(t, output) for t in tasks], chunksize=1)
            while not pending.ready():
                sample_owned_memory([p.pid for p in pool._pool])
                pending.wait(.05)
            results = pending.get()
        launched = [t.key for t in tasks]
    else:
        while active or (next_task < len(tasks) and not control.failure_ns.value):
            while len(active) < workers and next_task < len(tasks) and not control.failure_ns.value:
                task = tasks[next_task]
                next_task += 1
                receive, send = ctx.Pipe(duplex=False)
                child = ctx.Process(target=_child, args=(task, output, control, send))
                launch_ns = time.monotonic_ns()
                child.start()
                send.close()
                active[receive] = (child, task, launch_ns)
                launched.append(task.key)
            if not active:
                break
            sample_owned_memory([p.pid for p, _, _ in active.values()])
            for receive in wait(list(active), timeout=.1):
                child, task, launch_ns = active.pop(receive)
                try:
                    row = receive.recv()
                except EOFError:
                    control.fail()
                    row = {'task': asdict(task), 'status': 'WORKER_LOST', 'completed': 0, 'bytes': 0}
                receive.close()
                child.join()
                row['exitcode'] = child.exitcode
                row['dispatch_to_collection_wall_ns'] = time.monotonic_ns()-launch_ns
                if child.exitcode or row['status'] not in ('OK', 'CANCELLED'):
                    control.fail()
                results.append(row)
    after = D.clock()
    child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu = (child_after.ru_utime+child_after.ru_stime-child_before.ru_utime-child_before.ru_stime)
    wall = (after[1]-before[1])/1e9
    parent_cpu = (after[0]-before[0])/1e9
    failed_at = control.failure_ns.value
    starts = [s for r in results for s in r.get('started', [])]
    finishes = [s for r in results for s in r.get('completed_times', [])]
    for row in results:
        path = output/(row['task']['key']+'.jsonl')
        if path.exists() and row['status'] != 'WORKER_LOST':
            row['readback'] = readback(path)  # Outside timed region; read-after-write warm-cache audit.
            if row['readback']['file_sha256'] != row.get('file_sha256'):
                raise AssertionError('READBACK_FILE_HASH')
            row['normalized_file_sha256'] = row['readback']['normalized_file_sha256']
            row['normalized_chain_sha256'] = row['readback']['normalized_chain_sha256']
    memory = [r.get('memory', {}).get('VmHWM_bytes', 0) or 0 for r in results]
    completed = sum(r['completed'] for r in results)
    row = {'artificial_only': True, 'workers': workers, 'reuse': reuse, 'wall_s': wall,
           'worker_total_cpu_s': child_cpu, 'parent_cpu_s': parent_cpu,
           'total_cpu_s': child_cpu+parent_cpu, 'throughput': completed/wall,
           'completed': completed, 'bytes': sum(r['bytes'] for r in results),
           'peak_worker_hwm_bytes': max(memory, default=0),
           'sum_worker_hwm_bytes_upper_estimate': sum(sorted(memory, reverse=True)[:workers]),
           'aggregate_memory_is_sum_of_hwm_not_measured_simultaneous_peak': True,
           'sampled_parent_and_workers_rss_peak_bytes': aggregate_rss_peak,
           'rss_sample_count': memory_samples, 'rss_is_not_unique_physical_memory': True,
           'workers_results': sorted(results, key=lambda r: r['task']['key']),
           'launched': launched, 'not_launched': [t.key for t in tasks if t.key not in launched],
           'failure_ns': failed_at, 'starts_after_failure': sum(s['start_ns'] > failed_at for s in starts) if failed_at else 0,
           'completed_after_failure': sum(s['end_ns'] > failed_at for s in finishes) if failed_at else 0,
           'stop_detection_to_last_worker_collection_s': max(0, (after[1]-failed_at)/1e9) if failed_at else None,
           'worker_start_logs_complete': all(t.track_starts for t in tasks),
           'forbidden_imports': BLOCKED_IMPORTS.copy(), 'retry': 0, 'resume': False}
    (output/'summary.json').write_text(json.dumps(row, indent=2, default=json_default))
    return row


def json_default(value):
    if type(value) is F:
        return {'rat': [hex(value.numerator), hex(value.denominator)]}
    raise TypeError(type(value).__name__)


def proc_io():
    try:
        return {k: int(v) for k, v in (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}
    except OSError:
        return {}


def _buffer_crash_child(path, buffer_bytes, send):
    meter = D.Meter(D.clock)
    stream = PrototypeStream(path, meter, byte_limit=1 << 24, flush_rows=4096,
                             backend='orjson', buffer_bytes=buffer_bytes)
    for i in range(2):
        stream.emit({'ARTIFICIAL': True, 'trial_id': str(i), 'ratio': F(7, 31)})
    send.send({'accepted_rows': stream.rows, 'accepted_bytes': stream.bytes,
               'accepted_file_sha256': stream.file_hash.hexdigest()})
    send.close()
    os._exit(17)  # Intentional crash of this one owned artificial child only.


def buffer_crash_probe(output, buffer_bytes):
    ctx = mp.get_context('fork')
    recv, send = ctx.Pipe(False)
    process = ctx.Process(target=_buffer_crash_child, args=(output, buffer_bytes, send))
    process.start()
    send.close()
    accepted = recv.recv()
    recv.close()
    process.join()
    return dict(accepted, artificial_only=True, buffer_bytes=buffer_bytes,
                exitcode=process.exitcode, retained=readback(output))


def tasks_for(count=8, n=128, workload='mixed', variant='reference', pass_overlap=False):
    result = []
    for i in range(count):
        arm, layer = D.ARMS[(i % 8)//2], D.LAYERS[i % 2]
        p = 'instrumented' if pass_overlap and i >= 8 else 'baseline'
        replica = 0 if pass_overlap else i//8
        result.append(Task(f'ARTIFICIAL_pair_{i:03d}', arm, layer, n, replica, workload, variant,
                           pass_name=p, instrumented=p == 'instrumented'))
    return result
