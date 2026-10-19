"""Concrete API-object -> exact-document -> Adapter -> worker connection.

Current construction accepts only built-in synthetic symbols. The dormant
resolver is a separate AST-only SOURCE artifact, with no activation API here.
"""
from collections import Counter
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path
from hashlib import sha256
from . import contract as K, synthetic as X
from .bits import ObservedBits,ArtificialBits,validate_trace
from .monitoring import Meter
from .guard import install,require,observation
from ..g6_w2c_b3_live_preflight_v1 import spec as S
from ..g6_w2c_b3_real_adapter_v1 import codec as C, result as R
from .. import g6_w2c_b3_parallel_driver as D


def from_runtime(value):
    """No trust in an object's read(): verify its exact original document/digest."""
    require(type(value.document) is str and type(value.digest) is str,'RUNTIME_EVIDENCE_FIELDS')
    result=C.Document(value.document,value.digest);result.read();return result


def source_checkpoint(c):
    root=Path(c['runtime_root'])
    for p,h in c['source_pins'].items():
        require(sha256((root/p).read_bytes()).hexdigest()==h,'SOURCE_CHECKPOINT:'+p)
    return c['source_release_candidate']


class ArtificialFactory:
    artificial=True
    def __init__(self,c,m):
        self.c,self.m=deepcopy(c),deepcopy(m);self.seal=S.digest([c,m])
        self.runtime_manifest=deepcopy(c['worker_runtime'])
    def __call__(self,task,settings):
        install();require(type(self) is ArtificialFactory and self.artificial is True,'BUILTIN_FACTORY_ONLY')
        require(S.digest([self.c,self.m])==self.seal,'FACTORY_CONTEXT_CHANGED')
        K.preflight(self.c,self.m);require(task in self.m['tasks'],'WORKER_TASK')
        return Endpoint(self.c,self.m,task,settings,symbols=X.Symbols())


class Endpoint:
    def __init__(self,c,m,task,settings,*,symbols):
        install();require(type(symbols) is X.Symbols,'NO_REAL_OR_CALLER_SUPPLIED_SYMBOLS')
        self.c,self.m,self.task,self.settings=c,m,deepcopy(task),settings
        self.symbols=symbols;self.arm,self.layer=task['arm'],task['layer']
        self.full=self.arm=='full_return';self.meter=Meter();self.calls=Counter()
        self.stopped=False;self.session=None;self.overlay=None;self.current_phase='setup'
        self.event_args=None;self.requery=None;self.last=None;self.record=None
        self.pending_document=None;self.pending_trace=None
        p,x,m0=tuple(map(F,S.INPUT['p'])),F(S.INPUT['x']),S.INPUT['m']
        precision={k:(F(v) if k in ('eta','rho') else v) for k,v in S.PRECISION.items()
                   if k in ('eta','rho','root_bits','probability_bits')}
        self.angle=dict(angle_tau=F(S.ANGLE['tau']),angle_root_bits=S.ANGLE['K_phi'],angle_terms=S.ANGLE['J'])
        self._count('primary_constructions',1)
        with self.meter.span('constructor_setup'):
            if self.full and self.layer=='B':
                self._count('F3_prepare',1)
                self.session=symbols.f3.prepare(p,x,m0,sigma=1,**precision,**self.angle)
                self.g=self.session._generator
                self.preflight=from_runtime(self.session.preflight).digest
            else:
                self._count('G7_make_generator',1)
                self.g=symbols.g7.make_generator(p,x,m0,self.arm,**precision)
                self.preflight='G7_UNMONITORED_LAYER_A'
            require((tuple(self.g.p),self.g.x,self.g.m,self.g.arm)==(p,x,m0,self.arm),'CONSTRUCTOR_INPUT_BINDING')
            require((self.g.K,self.g.H,self.g.eta,self.g.rho)==(256,160,F(S.PRECISION['eta']),F(S.PRECISION['rho'])),
                    'CONSTRUCTOR_PRECISION_BINDING')
            if m['scenario']=='coverage' and self.full:
                self.g.artificial_acceptance=F(0)
            self._install_observers()
            if self.layer=='B':
                if not self.full:
                    self._count('canonical_snapshot',1)
                    self.preflight=D.identity(symbols.audit.canonical_snapshot(self.g,symbols.g7._law))
                self._count('AuditParity_construction',1)
                self.overlay=symbols.audit.AuditParity(self.arm,
                    dict(input=S.INPUT,policy=S.PRECISION,angle=S.ANGLE),self.snapshot,
                    lambda:source_checkpoint(self.c))
        self.initial_identity=self.shared_identity()

    def _count(self,name,limit):
        require(self.calls[name]<limit,'BOUNDARY_BUDGET_EXCEEDED:'+name)
        self.calls[name]+=1

    def _install_observers(self):
        original=self.g.event
        def event(*args):
            if self.current_phase=='angle':self._count('B_event_requery',32)
            elif not self.full:self.event_args=tuple(args)
            value=original(*args)
            if self.current_phase=='angle':self.requery=(tuple(args),deepcopy(value))
            return value
        self.g.event=event
        if self.full:
            original_packet=self.g.packet
            def packet(word):
                name='Full_angle_packet' if self.current_phase=='angle' else 'Full_sample_packet'
                self._count(name,32)
                return original_packet(word)
            self.g.packet=packet

    def snapshot(self):
        return self.symbols.f3._state(self.g) if self.full else self.symbols.audit.canonical_state(self.g)

    def shared_identity(self):
        return D.identity(dict(source=self.c['source_release_candidate'],manifest=S.digest(self.m),
            arm=self.arm,layer=self.layer,state=self.snapshot(),input=S.INPUT,precision=S.PRECISION,
            angle=S.ANGLE,preflight=self.preflight))

    def context(self,tid):
        if self.session is not None:
            status=self.session.status()
            return dict(trial_id=tid,input_id=self.session.request.input_id,
                policy_id=self.session.request.policy_id,source_id=status['source_id'],
                preflight_id=status['preflight_id'],sigma=1)
        return dict(trial_id=tid,input_id=S.digest(S.INPUT),policy_id=S.digest([S.PRECISION,S.ANGLE]),
            source_id=self.c['source_release_candidate'],preflight_id=self.preflight,sigma=1)

    def _fail(self,exc,stage,tid=None):
        self.stopped=True
        if isinstance(exc,R.AdapterFailure):return exc
        original=None
        if hasattr(exc,'evidence'):
            original=from_runtime(exc.evidence).transport()
        audit_failure=None
        if self.overlay is not None and self.overlay.failure is not None:
            seal=self.overlay.failure
            audit_failure=C.DriverDocument(seal.text,seal.digest).transport()
        return R.AdapterFailure('ARTIFICIAL_SCIENTIFIC_MISMATCH' if 'SCIENTIFIC_MISMATCH' in str(exc)
            else 'LIVE_WRAPPER_TECHNICAL_FAILURE',str(exc),stage,
            details=dict(trial_id=tid,original_scientific_failure=original,guard=observation(),
                original_driver_audit_failure=audit_failure,
                acquired_original_result=self.pending_document.transport() if self.pending_document is not None else None,
                observed_bit_consumption=self.pending_trace.consumption() if self.pending_trace is not None else None,
                counters=dict(self.calls),prefix_only=True))

    def sample(self,bits,trial_id):
        install();require(not self.stopped,'WRAPPER_TERMINAL_STOP')
        try:
            self.pending_document=None;self.pending_trace=None
            self._count('scheduled_sampling',32)
            require(type(bits) is ArtificialBits,'BUILTIN_ARTIFICIAL_BITS_ONLY')
            phase,pa,arm,layer,axis,index=trial_id.split('/')
            require((phase,pa,arm,layer)==('correctness','baseline',self.arm,self.layer)
                and axis in ('X','Y') and 0<=int(index)<16,'TRIAL_BINDING')
            scenario=self.m['scenario'];mode=self.task['condition']
            if scenario=='worker_exit' and mode=='S' and self.task['pair_index']==1:
                import os
                os._exit(87)
            self.current_phase='sample';self.requery=None
            trace=ObservedBits(bits,self.g,self.symbols.dyadic_index,full=self.full)
            self.pending_trace=trace
            packet_before=self.calls['Full_sample_packet']
            captured={}
            def operation():
                if self.session is not None:
                    runtime_record=self.session.sample(trace,trial_id=trial_id)
                    doc=from_runtime(runtime_record);data=doc.read()
                    event=data.get('event');captured['runtime_record']=runtime_record
                else:
                    event=self.g.sample(trace)
                    raw=tuple(trace.raw) if self.full else event['raw_word']
                    branch='ACCEPTED' if event is not None else 'RAW_NONREDUCED' if R.reduced(raw)!=raw else 'COIN_REJECT'
                    doc=C.Document.create(dict(schema='g7_observation_candidate_v1',**self.context(trial_id),
                        arm=self.arm,branch=branch,event=event,
                        trace=trace.f3_trace() if self.full else dict(draws=list(trace.draws)),
                        packet_calls=self.calls['Full_sample_packet']-packet_before,
                        group_index=self.event_args[0] if not self.full else None))
                # Publish acquired bytes into failure context before any later
                # invariant/audit/adapter step can fail. This is not completion.
                self.pending_document=doc
                if scenario=='scientific_mismatch' and mode=='S' and self.task['pair_index']==1:
                    raise ValueError('ARTIFICIAL_SCIENTIFIC_MISMATCH_EVENT_INVARIANT')
                if self.layer=='B' and event is not None:
                    self._count('nested_B_event_witness',32)
                    self.symbols.audit.event_witness(event,full=self.full,degree_cap=5,labels=3)
                captured.update(doc=doc,event=event)
                return doc.document
            audit=None
            with self.meter.span('sample_audit'):
                if self.overlay is not None:
                    self._count('B_sample_audit',32);before=self.overlay.completed
                    seal=self.overlay.invoke(operation)
                    audit=R.canonical_audit(C.DriverDocument(seal.text,seal.digest),arm=self.arm,
                        identity=self.overlay.identity.digest,state_id=self.overlay.state_id,completed_calls_before=before)
                else:operation()
            self.record=captured['doc']
            self.runtime_record=captured.get('runtime_record')
            with self.meter.span('codec_adapter'):
                if self.session is not None:a=R.adapt_f3(self.record,expected=self.context(trial_id))
                else:a=R.adapt_g7(captured['event'],observation=self.record,expected=self.context(trial_id),arm=self.arm)
            consumption=trace.consumption();validate_trace(consumption,self.arm,axis)
            require(len(trace.draws)<=(7 if self.full else 6),'PER_SAMPLE_DRAW_BOUND')
            self.calls['bit_draw_requests']+=len(trace.draws)
            require(self.calls['bit_draw_requests']<=32*(7 if self.full else 6),'PAIR_DRAW_BUDGET')
            gid=self.m['run_id']+'/'+mode+'/'+trial_id
            evidence=dict(schema='b3_live_object_bridge_artificial_v1',artificial_only=True,
                runtime_origin='BUILTIN_API_SHAPED_OBJECTS_NOT_G7_LAW',adapted=a,consumption=consumption,
                audit=audit,connection=dict(global_trial_id=gid,global_offset=self.task['offset']+(0 if axis=='X' else 16)+int(index),
                    source_binding=self.c['source_release_candidate'],manifest_binding=S.digest(self.m),
                    runtime_binding=self.m['runtime_binding'],worker_pid=observation()['pid'],
                    task_key=self.task['key'],route=S.route(self.arm,self.layer),session=self.initial_identity),
                boundary_counts=dict(self.calls),guard=observation(),monitoring=self.meter.snapshot(),
                live_API_compatibility='NOT_ESTABLISHED')
            self.last=D.Outcome(a['driver_kind'],a['event'],evidence);self.last.validate();return self.last
        except BaseException as exc:
            failure=self._fail(exc,'release.sample',trial_id)
            if failure is exc:raise
            raise failure from exc

    def angle_for(self,outcome):
        install();require(not self.stopped,'WRAPPER_TERMINAL_STOP')
        try:
            require(outcome is self.last and self.layer=='B' and outcome.kind=='EVENT','ANGLE_OUTCOME_BINDING')
            mode=self.task['condition'];scenario=self.m['scenario']
            if scenario==mode+'_angle' and self.task['pair_index']==1:
                raise ValueError('ARTIFICIAL_ANGLE_FAILURE')
            self.current_phase='angle';self._count('B_angle',32);a=outcome.evidence['adapted']
            with self.meter.span('angle_requery'):
                if self.session is not None:
                    self._count('Full_angle',32)
                    angle=from_runtime(self.session.angle_for_event(self._runtime_record()))
                    require(self.requery is not None,'FULL_REQUERY_NOT_OBSERVED')
                    args,event=self.requery;query=dict(word=tuple(args[0]),child=args[1])
                    result=R.angle_binding(a,requery_input=query,requery_event=event,angle_record=angle,synthetic=True)
                else:
                    self._count('canonical_angle',32);captured={};before=self.overlay.completed
                    def operation():
                        require(self.event_args is not None,'CANONICAL_GROUP_PROVENANCE_REQUIRED')
                        event=self.g.event(*self.event_args)
                        require(C.fingerprint(event)==C.fingerprint(outcome.event),'CANONICAL_REQUERY_MISMATCH')
                        runtime=self.symbols.precision.enclose_atan(event['ratio'],tau=F(S.ANGLE['tau']),
                            root_bits=32,terms=32,sigma=1,context=dict(event_id=D.identity(event),
                                input=S.INPUT,policy=S.PRECISION,preflight=self.preflight))
                        captured.update(event=event,angle=from_runtime(runtime));return captured['angle'].document
                    seal=self.overlay.invoke(operation)
                    outcome.evidence['angle_audit']=R.canonical_audit(C.DriverDocument(seal.text,seal.digest),arm=self.arm,
                        identity=self.overlay.identity.digest,state_id=self.overlay.state_id,completed_calls_before=before)
                    query=dict(group_index=self.event_args[0],raw_word=tuple(self.event_args[1]),
                               word=outcome.event['word'],child=self.event_args[2])
                    result=R.canonical_angle_binding(a,requery_input=query,requery_event=captured['event'],
                        angle_record=captured['angle'],input_payload=S.INPUT,policy_payload=S.PRECISION,synthetic=True)
            outcome.evidence['boundary_counts']=dict(self.calls)
            outcome.evidence['monitoring']=self.meter.snapshot()
            return result
        except BaseException as exc:
            failure=self._fail(exc,'release.angle',self.last.evidence['adapted']['trial_id'] if self.last else None)
            if failure is exc:raise
            raise failure from exc

    def _runtime_record(self):
        # The F3 API requires its original Evidence object. Retain that object;
        # reconstructing a codec Document would violate the exact runtime type.
        return self.runtime_record
