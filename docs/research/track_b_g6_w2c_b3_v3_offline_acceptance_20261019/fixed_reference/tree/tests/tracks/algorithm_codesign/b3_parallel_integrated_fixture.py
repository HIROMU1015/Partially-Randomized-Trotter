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

from trottertracks.algorithm_codesign import g6_w2c_b3_parallel_scheduler as P
from trottertracks.algorithm_codesign import g6_w2c_b3_parallel_admission as V
from trottertracks.algorithm_codesign import g6_w2c_b3_parallel_host as H

A, D = V.A, P.D
ROOT = Q.ROOT
REAL_CLOCK = lambda: (time.process_time_ns(), time.monotonic_ns())
PARENT_CANARY = None


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
        if PARENT_CANARY is not None:
            raise AssertionError('PARENT_MEMORY_INHERITED')
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



class ArtificialFactory:
    artificial = True
    def __init__(self, runtime_manifest): self.runtime_manifest = runtime_manifest
    def __call__(self, task, settings): return Endpoint(task,settings)


def fixture(root, settings):
    r1,ledger,_ = Q.fixture(root)
    p=root/A.HOST;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/A.HOST).read_bytes())
    r1['schema']=A.HOST_SCHEMA;r1['source_pins'][A.HOST]=sha256(p.read_bytes()).hexdigest()
    r1['candidate']['source_pins']=r1['source_pins'];r1['source_roles']['host']=[A.HOST]
    r1['entrypoint']=dict(path=A.HOST,sha256=r1['source_pins'][A.HOST],api='execute_approved_v1')
    pins=dict(r1['source_pins'])
    for name in V.ROLES.values():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/name).read_bytes())
        pins[name]=sha256(path.read_bytes()).hexdigest()
    candidate=json.loads(json.dumps(r1['candidate']));candidate['source_pins']=pins
    c=dict(schema=V.SCHEMA,contract_grants_execution=False,content_adopted=False,run_ready=False,
           science_release=None,scientific_adoption=None,retry=0,resume=False,mandatory_STOP=True,
           r1_reference=r1,source_pins=pins,input_pins=dict(r1['input_pins']),candidate=candidate,
           scope=r1['scope'],claim_id=r1['claim_id'],settings=asdict(settings),
           observer='cached_all_call_return',transport='coalesced_reserve_enter',
           output_schema='b3_parallel_result_v3',
           measurement_condition='sequential_pairs' if settings.mode=='S' else 'concurrent_pairs',
           runtime_manifest=V.manifest(ROOT,pins))
    c['schema_pins']=c['runtime_manifest']['schema_pins']
    for name,pin in c['schema_pins'].items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/name).read_bytes())
        c['input_pins'][name]=pin
    fixture_path=root/'artificial-fixture.py';fixture_path.write_bytes(Path(__file__).read_bytes())
    c['input_pins']['artificial-fixture.py']=sha256(fixture_path.read_bytes()).hexdigest()
    c['runtime_manifest']['artificial_fixture']=dict(path='tests/tracks/algorithm_codesign/b3_parallel_integrated_fixture.py',
        sha256=c['input_pins']['artificial-fixture.py'])
    return c,ledger,authorize(c,root,ledger)


def authorize(c,root,ledger):
    binding=V.validate(c,root,True)
    statement=root/'parallel-artificial-statement.txt';statement.write_text('ARTIFICIAL ONLY; no science approval.\n')
    path=root/'parallel-artificial-authorization.json'
    data=A.encoded(dict(schema='b3_independent_approval_v1',kind='ARTIFICIAL_NO_SCIENCE',binding=binding,
        statement_reference=str(statement),statement_sha256=A.digest(statement.read_bytes())))
    path.write_bytes(data)
    anchor=A.TrustAnchor('ARTIFICIAL_NO_SCIENCE',D.identity(c),binding['release_commit'],str(path),A.digest(data),
        str(statement),A.digest(statement.read_bytes()),str(ledger.directory),ledger.store_id)
    return A.PinnedApproval(anchor)


def invoke(root,c,ledger,auth,settings=None,**kwargs):
    before,children,io=D.clock(),resource.getrusage(resource.RUSAGE_CHILDREN),P.own_io()
    result=None
    try:
        result=H.execute_artificial_v3(c,root=root,ledger=ledger,authority=auth,
              endpoint_factory=ArtificialFactory(c['runtime_manifest']),bits_factory=bits_factory,**kwargs)
        return result
    except BaseException as exc:
        result=getattr(exc,'report',None)
        raise
    finally:
        after,ch=D.clock(),resource.getrusage(resource.RUSAGE_CHILDREN)
        if isinstance(result,dict):
            result['host_outer']=dict(parent_cpu_ns=after[0]-before[0],wall_ns=after[1]-before[1],
                worker_cpu_ns=round((ch.ru_utime+ch.ru_stime-children.ru_utime-children.ru_stime)*1e9),
                io={k:v-io.get(k,0)for k,v in P.own_io().items()})
            result['fake_claim_status']=ledger.row(c['scope'])[4]


def run(settings,root=None):
    owned=tempfile.TemporaryDirectory(prefix='b3-r1-artificial-')if root is None else None
    root=Path(owned.name)if owned else Path(root)
    try:
        c,l,a=fixture(root,settings);return invoke(root,c,l,a,settings)
    finally:
        if owned:owned.cleanup()

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
