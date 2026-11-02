"""Limited pilot preparation V1. Production is closed in this SOURCE.

The artificial API exercises the unchanged v3 atomic engine and worker protocol.
An outer candidate digest is included in the fake approval's formal payload.
Ordered instrumentation is a NEW evidence contract, never retroactive evidence.
"""
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import sys
import time

from . import g6_w2c_b3_parallel_admission as A
from . import g6_w2c_b3_parallel_scheduler as P
from . import g6_w2c_b3_parallel_host as H

D = P.D
SCHEMA = 'b3_limited_pilot_preparation_v1'
DOC = 'docs/research/track_b_g6_w2c_b3_limited_pilot_readiness_20261019/'
SOURCE = 'src/trottertracks/algorithm_codesign/g6_w2c_b3_pilot_candidate.py'
VERIFIER = 'src/trottertracks/algorithm_codesign/g6_w2c_b3_pilot_offline.py'
SCHEMA_PATH = DOC+'pilot.schema.json'


def production_entry(*args, **kwargs):
    raise A.A.Denied('PILOT_SOURCE_NOT_ADOPTED_NO_SCIENCE_ENTRY')


class ProductionFactory:
    def __call__(self, *args, **kwargs):
        return production_entry()


class ProductionBits(ProductionFactory):
    pass


def candidate(root, profile_n=256):
    """Read scientific SOURCE as bytes only; no scientific imports or execution."""
    root = Path(root).resolve()
    # Frozen input/science roles are the exact reference published with v3.
    ref = root/'docs/research/track_b_g6_w2c_b3_parallel_fused_r1_technical_integration_20261019'
    reference_path=ref/'measurements/benchmark/light-PF-0-contract.json'
    old = json.loads(reference_path.read_text())
    pins = dict(old['source_pins'])
    for p,h in pins.items():
        A.require(A.A.digest((root/p).read_bytes())==h,'FROZEN_SOURCE_MISMATCH:'+p)
    for p in (SOURCE,VERIFIER,SCHEMA_PATH):
        pins[p] = A.A.digest((root/p).read_bytes())
    return dict(schema=SCHEMA, status='UNADOPTED_EXECUTION_CLOSED', artificial_only=True,
        contract_grants_execution=False, content_adopted=False, run_ready=False,
        science_release=None, scientific_adoption=None, retry=0, resume=False, mandatory_STOP=True,
        scope='PROPOSED_LIMITED_G7_PILOT_V1_NO_AUTHORITY', claim_id=None,
        runtime_root=str(root), python=A.python_identity(), source_pins=pins,
        input_pins={old['candidate']['saved_evidence']['path']:old['candidate']['saved_evidence']['sha256'],
            str(reference_path.relative_to(root)):A.A.digest(reference_path.read_bytes()),
            **old['schema_pins']},
        scientific_reference={k:old['candidate'][k] for k in
            ('input','policy','original_source_commits','saved_evidence','saved_rows','cells',
             'decisions','acceptance','call_budget','one_shot')},
        fixed=dict(target='P5_general_order',p=['1/5','3/10','1/2'],x='5/7',m=5,sigma=1,
            arms=['ordinary','partial_return_tail','closed_P3_tail','full_return'],layers=['A','B'],axes=['X','Y'],
            angle=dict(tau='1/1000000',K_phi=32,J=32,math_only=True), IID_claim=False),
        proposal=dict(profile_n=profile_n, correctness_per_axis_pair=16, modes=['S','P1'],
            workers=8,start_method='spawn',phase_barriers=True,
            profile_passes=['baseline','instrumented'], correctness_calls=512,
            online_calls=64*profile_n,total_calls=512+64*profile_n,
            repetition=1, retry=0, resume=False, codec='fused',observer='cached_all_call_return',
            result_bytes=1<<36,log_bytes=1<<28,trace='ordered_parent_decisions_v1',
            seed_namespace='PROPOSED_B3_LIMITED_G7_PILOT_20261019_V1',
            cpu_inference='DESCRIPTIVE_FIXED_PREFIX_NO_IID_CI',science_authorization=None))


def validate(c):
    A.require(c['schema']==SCHEMA and c['status']=='UNADOPTED_EXECUTION_CLOSED','PILOT_SCHEMA')
    A.require(Path(__file__).resolve()==Path(c['runtime_root'])/SOURCE,'PILOT_LOADED_SOURCE_LOCATION')
    A.require(c==candidate(c['runtime_root'],c['proposal']['profile_n']),'PILOT_BINDING_MISMATCH')
    for path,h in c['input_pins'].items():
        A.require(A.A.digest((Path(c['runtime_root'])/path).read_bytes())==h,'PILOT_INPUT_MISMATCH')
    A.require(type(c['proposal']['profile_n']) is int and c['proposal']['profile_n'] in (64,256,1024),
              'UNADOPTED_PILOT_SIZE_CANDIDATE')
    A.require(not any(n.rsplit('.',1)[-1] in P.FORBIDDEN for n in sys.modules),'SCIENCE_ALREADY_IMPORTED')
    return D.identity(c)


class ArtificialAdmission(A.ArtificialAdmission):
    def __init__(self, c, *args, **kwargs):
        self.pilot = deepcopy(c)
        super().__init__(*args,**kwargs)

    def _check(self,digest):
        binding = validate(self.pilot)
        A.require(self.contract.get('pilot_candidate_binding')==binding,'PILOT_INNER_BINDING')
        return super()._check(digest)


class Trace:
    """Bounded by the SAME global log budget. No recursive self-accounting."""
    def __init__(self,path,budget,run_id):
        self.file=Path(path).open('xb',buffering=0)
        self.budget,self.run_id=budget,run_id
        self.reserve,self.wrote=budget.reserve,budget.wrote
        self.seq,self.previous,self.bytes=0,'0'*64,0

    def emit(self,op,**fields):
        payload=dict(seq=self.seq,run_id=self.run_id,op=op,**fields)
        raw=P.safe_json(payload)
        digest=sha256(self.previous.encode()+raw).hexdigest()
        data=P.safe_json(dict(payload=raw.decode(),previous=self.previous,sha256=digest))+b'\n'
        self.reserve('log',len(data))
        view=memoryview(data)
        while view:
            n=self.file.write(view)
            D.need(type(n)is int and n>0,'TRACE_SHORT_WRITE')
            self.wrote('log',n);self.bytes+=n;view=view[n:]
        self.seq+=1;self.previous=digest


class Coordinator(P.Coordinator):
    def __init__(self,settings,target,tasks,planned,run_id):
        super().__init__(settings,target,tasks,planned)
        try:
            self.trace=Trace(Path(target)/'operations.jsonl',self.budget,run_id)
            for channel,n in list(self.budget.reserved.items()):
                self.trace.emit('WRITE_RESERVE',channel=channel,amount=n)
            for channel,n in list(self.budget.observed.items()):
                # Initial statefile / PLAN are known writes; trace bytes are separate.
                if channel=='log':n-=self.trace.bytes
                if n:self.trace.emit('WRITE_ACK',channel=channel,amount=n)
            original_set=self.states.set
            def set_state(index,name):
                previous=P.STATES[self.states.mapping[index]]
                original_set(index,name)
                op={'START_RESERVED':'RESERVE','IN_FLIGHT':'ENTER','COMPLETED':'COMPLETE',
                    'FAILED':'FAIL','CANCELLED_BEFORE_START':'CANCEL',
                    'TERMINATED_WITH_UNKNOWN_RESULT':'UNKNOWN'}.get(name)
                if op:
                    if op=='FAIL' and previous!='IN_FLIGHT':op='UNKNOWN'
                    task=next(t for t in self.tasks if t['offset']<=index<t['offset']+2*t['n'])
                    local=index-task['offset'];axis='X'if local<task['n']else'Y';i=local%task['n']
                    self.trace.emit(op,task_key=task['key'],
                        trial_id=f"{task['phase']}/{task['pass_name']}/{task['arm']}/{task['layer']}/{axis}/{i}",
                        observed_source_state=name)
            self.states.set=set_state
            reserve,wrote=self.budget.reserve,self.budget.wrote
            def record_reserve(channel,n):
                reserve(channel,n);self.trace.emit('WRITE_RESERVE',channel=channel,amount=n)
            def record_wrote(channel,n):
                wrote(channel,n);self.trace.emit('WRITE_ACK',channel=channel,amount=n)
            self.budget.reserve,self.budget.wrote=record_reserve,record_wrote
        except BaseException:
            self.states.close();self.journal.file.close()
            if hasattr(self,'trace'):self.trace.file.close()
            raise

    def log(self,row):
        if hasattr(self,'trace') and row.get('event') in ('PHASE_BEGIN','PHASE_END'):
            self.trace.emit(row['event'],phase=row['phase'])
        return super().log(row)

    def stop(self,detail):
        if self.stop_ns is None and hasattr(self,'trace'):
            self.stopped.value=1
            try:self.trace.emit('STOP',parent_arrival=self.arrival+1,detail=repr(detail))
            except BaseException:pass
        return super().stop(detail)


def execute_artificial(c, inner, *,root,ledger,authority,endpoint_factory,bits_factory,
                       admission_fault=lambda point:None,watch=None):
    """Private fake authority only. No production dispatch or unseal parameter."""
    before=D.clock();ch=resource.getrusage(resource.RUSAGE_CHILDREN);io=P.own_io()
    control=None;result=None
    with H._execution_context(inner):
        port=ArtificialAdmission(c,inner,Path(root),ledger,authority,fault=admission_fault)
        try:
            with port:
                A.require(endpoint_factory.artificial is True,'ARTIFICIAL_ENDPOINT_REQUIRED')
                expected=Path(inner['runtime_manifest']['runtime_root'])/inner['runtime_manifest']['artificial_fixture']['path']
                A.require(Path(endpoint_factory.__call__.__code__.co_filename).resolve()==expected and
                    Path(bits_factory.__code__.co_filename).resolve()==expected and
                    A.A.digest(expected.read_bytes())==inner['input_pins']['artificial-fixture.py'],
                    'ARTIFICIAL_FACTORY_SOURCE_BINDING')
                digest=D.identity(inner['candidate'])
                port._check(digest);port.admit_artificial(digest);port._check(digest)
                output=A.A.exact_path(root,inner['candidate']['output_relative'])
                output.mkdir(parents=True,exist_ok=False)
                settings=P.Settings(**inner['settings']);settings.check()
                tasks,planned=P.plan_for(settings.profile_n)
                A.require(planned<=c['proposal']['total_calls'],'PILOT_GLOBAL_CALL_CAP')
                control=Coordinator(settings,output,tasks,planned,D.identity(inner))
                for phase in range(3):
                    if control.stopped.value:break
                    control.run_phase([t for t in tasks if t['phase_index']==phase],root,
                        inner['source_pins'],endpoint_factory,bits_factory,watch)
                    P.validate_success(control.results)
                    validate(c);A.check_runtime(root,inner['runtime_manifest'])
                result=control.report()
                if not result['complete']:raise P.BundleFailure(result)
                # Close trace before summary: the summary binds a complete decision log.
                trace_meta=dict(sha256=A.A.digest((output/'operations.jsonl').read_bytes()),
                    chain=control.trace.previous,rows=control.trace.seq,bytes=control.trace.bytes)
                summary=dict(schema='b3_pilot_artificial_summary_v1',pilot_binding=validate(c),
                    inner_binding=D.identity(inner),trace=trace_meta,result=result)
                stream=D.Stream(output/'pilot_summary.jsonl',D.Meter(D.clock),byte_limit=settings.result_bytes,flush_rows=1)
                stream.file=P.ReservedFile(stream.file,lambda n:control.budget.reserve('result',n),
                    lambda n:control.budget.wrote('result',n))
                try:stream.emit(summary)
                finally:stream.close()
                receipt=dict(schema='b3_pilot_artificial_receipt_v1',artificial_only=True,
                    pilot_binding=validate(c),inner_binding=D.identity(inner),
                    summary_sha256=stream.file_hash.hexdigest(),summary_chain=stream.previous,
                    trace_prefix=trace_meta,production_permission=False)
                control.log(dict(event='PILOT_COMPLETION_RECEIPT',**receipt))
                f=P.ReservedFile((output/'pilot_receipt.json').open('xb',buffering=0),
                    lambda n:control.budget.reserve('result',n),lambda n:control.budget.wrote('result',n))
                try:
                    data=A.A.encoded(receipt);view=memoryview(data)
                    while view:
                        n=f.write(view);D.need(type(n)is int and n>0,'RECEIPT_SHORT_WRITE');view=view[n:]
                    f.flush()
                finally:f.close()
        except BaseException as exc:
            if control is not None:
                control.stop(dict(parent_failure=repr(exc)))
                if control.active:
                    control.run_phase([],root,inner['source_pins'],endpoint_factory,bits_factory,watch,drain_only=True)
                result=control.report()
                result['primary_failure']=repr(exc)
            else:raise
        finally:
            if control is not None:
                control.states.close();control.journal.file.close();control.trace.file.close()
    after=D.clock();endch=resource.getrusage(resource.RUSAGE_CHILDREN)
    if result is not None:
        result['output']=dict(limits=control.budget.limit,reserved=dict(control.budget.reserved),
            observed=dict(control.budget.observed))
        result['host_outer']=dict(parent_cpu_ns=after[0]-before[0],wall_ns=after[1]-before[1],
            worker_cpu_ns=round((endch.ru_utime+endch.ru_stime-ch.ru_utime-ch.ru_stime)*1e9),
            io={k:v-io.get(k,0)for k,v in P.own_io().items()})
        result['fake_claim_status']=ledger.row(inner['scope'])[4]
        result['pilot_binding']=validate(c)
        result['operation_trace_sha256']=A.A.digest((output/'operations.jsonl').read_bytes())
    return result
