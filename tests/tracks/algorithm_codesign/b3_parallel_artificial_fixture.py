"""Private artificial host/authority/endpoint adapters; not production code."""
from contextlib import ExitStack
from dataclasses import asdict
from fractions import Fraction as F
from hashlib import sha256
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
from unittest.mock import patch

import test_g6_w2c_b3_admission as Q
import test_g6_w2c_b3_driver as T
import test_g6_w2c_b3_host_integration as I
from trottertracks.algorithm_codesign import g6_w2c_b3_parallel_candidate as P

A, D, H = I.A, I.D, I.H
ROOT = Q.ROOT
REAL_CLOCK = lambda: (time.process_time_ns(), time.monotonic_ns())


class Bits:
    """Same counter/digest discard rule, exclusively an ARTIFICIAL namespace."""
    def __init__(self, arm, axis):
        self.seed = ('ARTIFICIAL_R1_PARALLEL/'+arm+'/'+axis).encode()
        self.counter = self.consumed_bits = 0

    def bits(self, n):
        value, available = 0, 0
        while available < n:
            raw = sha256(self.seed+self.counter.to_bytes(8, 'big')).digest()
            self.counter += 1
            value = (value << 256) | int.from_bytes(raw, 'big')
            available += 256
        self.consumed_bits += n
        return value >> (available-n)


class BadResult:
    @property
    def event(self):
        raise RuntimeError('ARTIFICIAL_EVENT_ACCESS_FAILURE')


class Endpoint:
    def __init__(self, task, settings):
        self.task, self.settings, self.angle_calls = task, settings, 0
        self.value = {'artificial': True, 'state': [F(i+1, i+31) for i in range(32)]}

    def shared_identity(self):
        return D.identity(self.value)

    def fault(self, stage, trial_id=None):
        for pair, phase, where, trial in self.settings.faults:
            if pair == self.task['pair_index'] and phase == self.task['phase_index'] and where == stage:
                if trial_id is None or trial_id.endswith('/X/'+str(trial)):
                    return True
        return False

    def sample(self, bits, trial_id):
        if self.fault('exit', trial_id):
            os._exit(88)
        if self.fault('sample', trial_id):
            raise RuntimeError('ARTIFICIAL_SAMPLE_BEFORE_RESULT')
        if self.fault('event', trial_id):
            return BadResult()
        token = bits.bits(160)
        loops = {'light': 8, 'mixed': 96, 'heavy': 192}[self.settings.workload]
        x = token
        for j in range(loops):
            x = (x*6364136223846793005+j) & ((1 << 192)-1)
        if self.fault('zero'):
            return D.Outcome('ZERO_R', evidence={'artificial': True})
        kind = 'EVENT' if self.fault('after_event',trial_id) else ('ZERO_R', 'ZERO_C', 'EVENT', 'EVENT')[token % 4]
        if kind != 'EVENT':
            return D.Outcome(kind, evidence={'artificial': True, 'token': x})
        width = {'light': 24, 'mixed': 96, 'heavy': 256}[self.settings.workload]
        ratio = F((1 << width)+(x & 8191)+1, (1 << (width+2))+57)
        weight = ratio
        for j in range(1 if self.settings.workload == 'light' else 8):
            weight = (weight+F(j+1, j+19))*F(j+7, j+11)
        event = T.event(canonical=self.task['arm'] != 'full_return')
        event.update(ratio=ratio, weight=weight, coefficient=weight*event['proposal'])
        if self.fault('after_event',trial_id):
            event['phase_i_power'] = 99
        self.last_trial = trial_id
        evidence = {'artificial': True, 'law': sorted((F((x >> j) % 31+1, 97+j), j) for j in range(8)),
                    'evidence': weight}
        return D.Outcome('EVENT', event, evidence)

    def angle_for(self, result):
        self.angle_calls += 1
        if self.fault('angle'):
            raise RuntimeError('ARTIFICIAL_ANGLE_REQUERY_FAILURE')
        terms = {'light': 4, 'mixed': 12, 'heavy': 32}[self.settings.workload]
        z = result.event['ratio']
        partial = sum(((-1)**j*z**(2*j+1)/(2*j+1) for j in range(terms)), F(0))
        nxt = (-1)**terms*z**(2*terms+1)/(2*terms+1)
        return {'artificial_not_a_real_enclosure': True, 'partial': partial,
                'remainder': abs(nxt), 'event_requery_surrogate': D.identity(result.event)}


def endpoint_factory(task, settings):
    return Endpoint(task, settings)


def bits_factory(arm, axis):
    return Bits(arm, axis)


def fixture(root, settings):
    c, ledger, auth = Q.fixture(root)
    path = root/A.HOST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((ROOT/A.HOST).read_bytes())
    c['schema'] = A.HOST_SCHEMA
    c['source_pins'][A.HOST] = sha256(path.read_bytes()).hexdigest()
    c['source_roles']['host'] = [A.HOST]
    c['candidate']['source_pins'] = c['source_pins']
    c['entrypoint'] = dict(path=A.HOST, sha256=c['source_pins'][A.HOST], api='execute_approved_v1')
    policy = {'settings': asdict(settings), 'source_sha256': sha256(Path(P.__file__).read_bytes()).hexdigest(),
              'fixture_sha256': sha256(Path(__file__).read_bytes()).hexdigest(), 'artificial_only': True}
    path = root/'parallel-policy.json'
    path.write_text(json.dumps(policy, sort_keys=True))
    c['input_pins']['parallel-policy.json'] = sha256(path.read_bytes()).hexdigest()
    auth = Q.authorize(c, root, ledger)
    return c, ledger, auth


def invoke(root, c, ledger, auth, settings, admission_fault=lambda point: None,
           watch=None, original=False, synchronized_failure=False):
    ports = []
    def port_factory(contract, path, store, authority):
        port = I.ArtificialBoundaryBridge(contract, path, store, authority, fault=admission_fault)
        ports.append(port)
        return port
    def operation(candidate, **kwargs):
        return P.execute_artificial_bundle(candidate, settings=settings,
            endpoint_factory=endpoint_factory, bits_factory=bits_factory, watch=watch, **kwargs)
    before, child_before = REAL_CLOCK(), resource.getrusage(resource.RUSAGE_CHILDREN)
    io_before = P.own_io()
    try:
        with ExitStack() as stack:
            if synchronized_failure:
                # Fault-only rendezvous: force a known number of entered calls.
                cap = settings.inflight_cap if settings.mode == 'P2' else settings.workers
                barrier = mp.get_context('fork').Barrier(cap)
                original_sample = Endpoint.sample
                def rendezvous(endpoint, bits, trial_id):
                    if endpoint.task['phase_index'] == 0 and trial_id.endswith('/X/0'):
                        rank = barrier.wait(timeout=15)
                        if rank == 0 or synchronized_failure == 'all':
                            raise RuntimeError('ARTIFICIAL_SYNCHRONIZED_FAILURE')
                        until, accumulator = time.monotonic()+.1, 0
                        while time.monotonic() < until:
                            accumulator = (accumulator*33+17) & ((1 << 128)-1)
                    return original_sample(endpoint, bits, trial_id)
                stack.enter_context(patch.object(Endpoint, 'sample', rendezvous))
            stack.enter_context(patch.object(H, '__file__', str(root/A.HOST)))
            stack.enter_context(patch.object(A, 'ProductionAdmission', port_factory))
            if not original:
                stack.enter_context(patch.object(D, 'execute', operation))
            else:
                # Unmodified execute/isolated_pair/Stream; only scientific dependencies mocked.
                real_pair = D.run_pair
                original_observer = D.FunctionObserver
                original_isolated = D.isolated_pair
                def isolated(operation):
                    def measured():
                        io = P.own_io()
                        row = operation()
                        row['artificial_io'] = {k:v-io.get(k,0) for k,v in P.own_io().items()}
                        return row
                    return original_isolated(measured)
                class ArtificialObserver(original_observer):
                    def __init__(self, paths, read_clock=D.clock):
                        super().__init__([*paths, D.__file__, __file__], read_clock)
                def pair(endpoint, bits, cells, stream, meter, **kw):
                    toy = [dict(c, N=settings.profile_n) for c in cells]
                    return real_pair(endpoint, bits, toy, stream, meter, **kw)
                def make_endpoint(mods, arm, layer, *args, **kwargs):
                    return Endpoint(dict(arm=arm, layer=layer, pair_index=D.ARMS.index(arm)*2+D.LAYERS.index(layer),
                                         phase_index=-1), settings)
                class FakeBits:
                    def __init__(self, seed):
                        self.inner = Bits(*seed.split('/')[-2:])
                    @property
                    def consumed_bits(self):
                        return self.inner.consumed_bits
                    def bits(self, n):
                        return self.inner.bits(n)
                stack.enter_context(patch.object(D, 'run_pair', pair))
                stack.enter_context(patch.object(D, 'load_saved', lambda *args: dict.fromkeys(D.ARMS, {})))
                stack.enter_context(patch.object(D, 'load_modules', lambda *args: {'g7': T.NS(DeterministicBits=FakeBits)}))
                stack.enter_context(patch.object(D, 'LegacyEndpoint', make_endpoint))
                stack.enter_context(patch.object(D, 'FunctionObserver', ArtificialObserver))
                stack.enter_context(patch.object(D, 'isolated_pair', isolated))
            result = H.execute_approved(c, root=root, ledger=ledger, authority=auth)
    except BaseException as exc:
        result = getattr(exc, 'report', None)
        raise
    finally:
        after, child_after = REAL_CLOCK(), resource.getrusage(resource.RUSAGE_CHILDREN)
        outer = dict(parent_cpu_ns=after[0]-before[0], wall_ns=after[1]-before[1],
                     worker_cpu_ns=round((child_after.ru_utime+child_after.ru_stime-child_before.ru_utime-child_before.ru_stime)*1e9),
                     io={k:v-io_before.get(k,0) for k,v in P.own_io().items()})
        for p in ports:
            p.artificial_outer_measurement = outer
        if isinstance(result, dict):
            result['host_outer'] = outer
            result['fake_claim_status'] = ledger.row(c['scope'])[4]
    result['host_outer'] = outer
    result['fake_claim_status'] = ledger.row(c['scope'])[4]
    return result


def run(settings, root=None, original=False):
    owned = tempfile.TemporaryDirectory(prefix='b3-r1-artificial-') if root is None else None
    root = Path(owned.name) if owned else Path(root)
    try:
        c, ledger, auth = fixture(root, settings)
        return invoke(root, c, ledger, auth, settings, original=original)
    finally:
        if owned:
            owned.cleanup()


def read_stream(path):
    h, previous, rows, partial = sha256(), '0'*64, [], b''
    for line in Path(path).read_bytes().splitlines(keepends=True):
        h.update(line)
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            partial = line
            break
        assert row['previous'] == previous
        previous = sha256((previous+row['payload']).encode()).hexdigest()
        assert row['sha256'] == previous
        payload = json.loads(row['payload'])
        if 'generator_exclusive_ns' in payload:
            del payload['generator_exclusive_ns']
        rows.append(payload)
    return {'rows': rows, 'partial_bytes': len(partial), 'file_sha': h.hexdigest(), 'chain': previous}
