"""S -> P1 candidate: ONE coordinator, claim, call ledger and byte budget.

Derived from fixed Bundle Bridge V1 with versioned API-port injection.
The only executable API binds the fixed artificial API port factories. Production has no
implementation, dispatch table, unlock flag, supplied factory or authority API.
"""
from collections import Counter
from dataclasses import replace
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json,os,resource,sys,time
from .. import g6_w2c_b3_pilot_storage_v2 as R
from . import contract as K
from . import adapter as F
from ..g6_w2c_b3_bundle_store_v1 import FakeStore,Denied,private
P,D,A=R.P,R.D,R.A


def production_entry(*args,**kwargs):raise Denied('BUNDLE_CANDIDATE_HAS_NO_SCIENCE_DISPATCH')
class ProductionFactory:
    def __call__(self,*args,**kwargs):return production_entry()
class ProductionBits(ProductionFactory):pass


def validate(c,m):
    F.PORT.install_guard()
    if Path(__file__).resolve()!=Path(c['runtime_root'])/K.PKG/'bridge.py':raise Denied('LOADED_INTEGRATION_BRIDGE')
    try:return K.guard(c,m)
    except F.PORT.Rejected as exc:raise Denied("INTEGRATION_BINDING:"+str(exc)) from exc


class Fault:
    """One owned I/O fault; observations are captured before throwing."""
    def __init__(self,spec=None):self.spec=spec or {};self.hits=0;self.control=None;self.observation=None
    def __call__(self,point,**context):
        if self.hits or point!=self.spec.get('point'):return
        ctl=self.control
        if self.spec.get('condition') and (ctl is None or ctl.condition!=self.spec['condition']):return
        if any(context.get(k)!=v for k,v in self.spec.get('match',{}).items()):return
        self.hits+=1
        if ctl is not None:
            self.observation=dict(point=point,condition=ctl.condition,reserved=ctl.reservations,started=ctl.started,cursors=dict(ctl.next_indices),tickets=[dict(v)for v in ctl.current.values()],states=dict(ctl.states.counts))
        if self.spec.get('action')=='partial':
            ctl.trace.file=Partial(ctl.trace.file);return
        raise OSError('ARTIFICIAL_STORAGE_FAULT:'+point)

class Partial:
    def __init__(self,file):self.file=file;self.first=True
    def write(self,data):
        if not self.first:raise OSError('ARTIFICIAL_PARTIAL_WRITE')
        self.first=False;return self.file.write(data[:11])
    def __getattr__(self,n):return getattr(self.file,n)


class Coordinator(R.Coordinator):
    def __init__(self,*args,manifest,**kwargs):
        self.manifest=manifest;self.condition='SHARED';self.post_close_credits=[];self.closed=False
        super().__init__(*args,**kwargs)
    def _record(self,op,**fields):
        key=fields.get('task_key');condition=key.split('__',1)[0]if key else self.condition
        if key:
            fields['global_trial_id']=self.manifest['run_id']+'/'+condition+'/'+fields['trial_id']
            fields['global_offset']=next(t['offset']for t in self.tasks if t['key']==key)+self._local_index(key,fields['trial_id'])
        return super()._record(op,condition=condition,**fields)
    def _local_index(self,key,tid):
        t=next(t for t in self.tasks if t['key']==key);a,i=tid.rsplit('/',2)[-2:];return (0 if a=='X'else t['n'])+int(i)
    def run_phase(self,tasks,*args,drain_only=False,**kwargs):
        try:return super().run_phase(tasks,*args,drain_only=drain_only,**kwargs)
        except BaseException:
            if not drain_only:
                self._record('PHASE_ABORT',phase=tasks[0]['phase_index'])
                self.log(dict(event='PHASE_ABORT',phase=tasks[0]['phase_index']))
            raise
    def condition_event(self,event,**fields):
        self._record(event,**fields);self.log(dict(event=event,condition=self.condition,**fields))
    def post_notify(self,op,**fields):self.post_close_credits.append(dict(op=op,**fields))


def _bytes(value):return P.safe_json(value)+b'\n'

def _save(out,name,value,ctl,fault,*,slot=None,pre_reserved=False):
    """All retained artifacts, including final witness slots, consume global credit."""
    before=D.clock();fault(name+'.hash');data=_bytes(value)
    if slot is not None:
        D.need(len(data)<=slot,'FIXED_FINAL_SLOT_TOO_SMALL');data+=b' '*(slot-len(data))
    if not pre_reserved:ctl.budget.reserve('result',len(data))
    fault(name+'.open');f=(out/name).open('xb',buffering=0)
    try:
        view=memoryview(data)
        while view:
            fault(name+'.write');n=f.write(view);D.need(type(n)is int and 0<n<=len(view),'SHORT_DOCUMENT_WRITE')
            ctl.budget.wrote('result',n);view=view[n:];fault(name+'.after_write',remaining=len(view))
        fault(name+'.flush');f.flush()
    finally:
        try:fault(name+'.close')
        finally:f.close()
    ctl.parent_save_cost[name]=D.delta(before,D.clock())
    return sha256(data).hexdigest()


def _condition(ctl,mode,c,m,root,factory,fault):
    ctl.condition=mode
    faults=tuple(tuple(x) for x in m['condition_faults'].get(mode,[]))
    ctl.settings=replace(ctl.settings,mode=mode,faults=faults,parent_fault=m['parent_fault']if mode==m['modes'][0]else'',sync_failure=m['sync_failure']if mode=='P1'else'')
    ctl.rendezvous=ctl.ctx.Barrier(8)if ctl.settings.sync_failure else None
    start=D.clock();ch=resource.getrusage(resource.RUSAGE_CHILDREN);b0=deepcopy(dict(reserved=dict(ctl.budget.reserved),observed=dict(ctl.budget.observed)));r0=ctl.reservations;s0=ctl.started
    ctl.condition_event('CONDITION_BEGIN')
    try:
        for phase in range(0 if mode=='S'else 3,3 if mode=='S'else 6):
            if ctl.stopped.value:break
            ctl.run_phase([t for t in ctl.tasks if t['phase_index']==phase],c['runtime_root'],c['source_pins'],factory,F.bits_factory,None)
            rows=[dict(r,task=dict(r['task'],phase_index=r['task']['phase_index']%3))for r in ctl.results if r['task']['condition']==mode]
            P.validate_success(rows);validate(c,m)
    except BaseException as exc:
        ctl.stop(dict(condition=mode,parent_failure=repr(exc)))
        if ctl.active:ctl.run_phase([],c['runtime_root'],c['source_pins'],factory,F.bits_factory,None,drain_only=True)
    selected=[t for t in ctl.tasks if t['condition']==mode];rows=[r for r in ctl.results if r['task']['condition']==mode]
    complete=not ctl.stopped.value and len(rows)==24 and all(r['status']=='OK'for r in rows)and sum(ctl.states.mapping[t['offset']:t['offset']+2*t['n']].count(4)for t in selected)==256+32*m['profile_n']
    ctl.condition_event('CONDITION_END',complete=complete)
    after=D.clock();end=resource.getrusage(resource.RUSAGE_CHILDREN)
    status=dict(condition=mode,complete=complete,reserved=ctl.reservations-r0,started=ctl.started-s0,completed=sum(ctl.states.mapping[t['offset']:t['offset']+2*t['n']].count(4)for t in selected),planned=256+32*m['profile_n'],
        parent_cpu_ns=after[0]-start[0],worker_cpu_ns=round((end.ru_utime+end.ru_stime-ch.ru_utime-ch.ru_stime)*1e9),wall_ns=after[1]-start[1],budget_before=b0,budget_after=dict(reserved=dict(ctl.budget.reserved),observed=dict(ctl.budget.observed)),
        shared_setup_or_finalization_allocated=False)
    if complete:_save(ctl.target,mode+'_receipt.json',dict(schema='b3_condition_receipt_v1',run_id=m['run_id'],manifest_binding=K.digest(m),**status),ctl,fault)
    return status


def execute_artificial(c,m,*,root,store,storage_fault=None):
    """Pinned artificial API-port injection; ONE global fake claim per bundle."""
    binding=validate(c,m);root=private(root)
    if type(store)is not FakeStore or store.root!=root:raise Denied('BUILTIN_FAKE_STORE_ONLY')
    fault=storage_fault if storage_fault is not None else Fault()
    if type(fault)is not Fault:raise Denied('BUILTIN_FAULT_ONLY')
    before=D.clock();children=resource.getrusage(resource.RUSAGE_CHILDREN);io=P.own_io();ctl=None;store_errors=[]
    try:store.claim(m,fault)
    except BaseException:
        # Commit-ACK unknown still spends the scope. Never open a second claim.
        try:store.stop(m,fault)
        except BaseException:pass
        raise
    out=root/'output';out.mkdir(exist_ok=False)
    settings=P.Settings(mode='S',profile_n=m['profile_n'],codec='fused',workload='light',call_limit=m['call_limit'],result_bytes=m['byte_limits']['result'],log_bytes=m['byte_limits']['log'])
    runtime=A.manifest(c['runtime_root'],c['source_pins']);runtime['artificial_fixture']=dict(path=K.PKG+'adapter.py',sha256=c['source_pins'][K.PKG+'adapter.py'])
    factory=F.ArtificialFactory(runtime,dict(artificial_only=True,run_id=m['run_id'],tasks=m['tasks'],variant=m['port_variant'],port_faults=m['port_faults'],candidate_binding=K.digest(c),manifest_binding=binding,runtime_binding=K.digest(runtime),endpoint_binding=c['endpoint_binding']))
    conditions=[];receipt=None;marker=None;late_errors=[]
    try:
        ctl=Coordinator(settings,out,m['tasks'],m['planned'],m['run_id'],fault,manifest=m);fault.control=ctl;ctl.parent_save_cost={}
        # Reserve final evidence before calls. No free diagnostic/finalization tail.
        ctl.budget.reserve('result',K.FINAL_SLOT);ctl.budget.reserve('result',K.DELIVERY_SLOT);ctl.budget.reserve('result',K.OBSERVATION_SLOT)
        _save(out,'candidate.json',c,ctl,fault);_save(out,'manifest.json',m,ctl,fault)
        for mode in m['modes']:
            if ctl.stopped.value:break
            status=_condition(ctl,mode,c,m,c['runtime_root'],factory,fault);conditions.append(status)
            if not status['complete']:break
        ctl.condition='SHARED';validate(c,m)
        logical=ctl.report()
        _save(out,'summary.json',dict(schema='b3_bundle_summary_v1',port_binding=m['endpoint_binding'],run_id=m['run_id'],manifest_binding=binding,candidate_binding=K.digest(c),conditions=conditions,planned=m['planned'],reserved=logical['reserved'],started=logical['started_observed'],states=logical['states'],trace_prefix=dict(ctl.trace.prefix)),ctl,fault)
        _save(out,'diagnostic.json',dict(schema='b3_bundle_diagnostic_v1',complete=False,final_acceptance=False,run_id=m['run_id'],reserved=ctl.reservations,states=logical['states']),ctl,fault)
    except BaseException as exc:
        if ctl is None:
            try:store.stop(m,fault)
            except BaseException:pass
            raise
        ctl.stop(dict(parent_failure=repr(exc)))
        if ctl.active:ctl.run_phase([],c['runtime_root'],c['source_pins'],factory,F.bits_factory,None,drain_only=True)
    finally:
        try:store.stop(m,fault)
        except BaseException as exc:store_errors.append(repr(exc));ctl.stop(dict(admission_stop_failure=repr(exc)))if ctl else None
        if ctl:
            ctl.condition='SHARED'
            try:_save(out,'FAKE_ONLY_snapshot.json',store.snapshot(m),ctl,fault)
            except BaseException as exc:ctl.stop(dict(snapshot_failure=repr(exc)))
            R.close_storage(ctl);ctl.closed=True;ctl.budget.notify=ctl.post_notify
    report=ctl.report();base_complete=report['complete'] and m['modes']==['S','P1'] and len(conditions)==2 and all(x['complete']for x in conditions)and not store_errors
    # A condition projection is a useful artificial check, never a bundle success.
    if base_complete:
        try:
            receipt=dict(schema='b3_global_bundle_receipt_v1',port_binding=m['endpoint_binding'],run_id=m['run_id'],manifest_binding=binding,candidate_binding=K.digest(c),planned=m['planned'],reserved=ctl.reservations,completed=report['states']['COMPLETED'],conditions=['S','P1'],summary_sha256=sha256((out/'summary.json').read_bytes()).hexdigest(),state_sha256=sha256((out/'trial_states.bin').read_bytes()).hexdigest(),trace_prefix=dict(ctl.trace.prefix),storage_closed_before_receipt=True,claim_status=store.snapshot(m)['claim'][3],production_permission=False)
            receipt_sha=_save(out,'receipt.json',receipt,ctl,fault)
            marker=dict(schema='b3_bundle_post_close_marker_v1',run_id=m['run_id'],manifest_binding=binding,receipt_sha256=receipt_sha,receipt_closed_before_marker=True)
            _save(out,'marker.json',marker,ctl,fault)
        except BaseException as exc:late_errors.append(repr(exc));base_complete=False
    after=D.clock();ch=resource.getrusage(resource.RUSAGE_CHILDREN)
    host=dict(parent_cpu_ns=after[0]-before[0],worker_cpu_ns=round((ch.ru_utime+ch.ru_stime-children.ru_utime-children.ru_stime)*1e9),wall_ns=after[1]-before[1],io={k:v-io.get(k,0)for k,v in P.own_io().items()})
    snapshot=store.snapshot(m)
    files={p.name:dict(bytes=p.stat().st_size,sha256=sha256(p.read_bytes()).hexdigest()) for p in out.iterdir()if p.is_file()}
    # Fixed-size final/delivery slots avoid recursive byte-size/credit claims.
    # These two ACKs are explicitly projected until the external delivery exists.
    projected=dict(ctl.budget.observed);projected['result']=projected.get('result',0)+K.FINAL_SLOT+K.DELIVERY_SLOT+K.OBSERVATION_SLOT
    budget=dict(limits=ctl.budget.limit,reserved=dict(ctl.budget.reserved),observed=projected,unknown_credit={k:ctl.budget.reserved[k]-projected.get(k,0)for k in ctl.budget.limit})
    complete=base_complete and all(x==0 for x in budget['unknown_credit'].values())
    report.update(schema='b3_global_bundle_result_v1',port_binding=m['endpoint_binding'],complete=complete,manifest_binding=binding,candidate_binding=K.digest(c),run_id=m['run_id'],source_release_candidate=c['source_release_candidate'],conditions=conditions,output=budget,host_outer=host,
        host_boundary='before claim through post-close receipt/marker; fixed final/delivery persistence is separately observed',
        condition_parent_cpu_ns=sum(x['parent_cpu_ns']for x in conditions),condition_worker_cpu_ns=sum(x['worker_cpu_ns']for x in conditions),
        fake_claim=snapshot,admission_stop_errors=store_errors,post_close_errors=late_errors,post_close_credits=list(ctl.post_close_credits),
        physical_files_before_final=files,final_slot_bytes=K.FINAL_SLOT,delivery_slot_bytes=K.DELIVERY_SLOT,external_observation_slot_bytes=K.OBSERVATION_SLOT,projected_self_ACK_requires_external_delivery=True,
        parent_save_diagnostic=ctl.parent_save_cost,fault_observation=fault.observation,fault_hits=fault.hits,
        production_permission=False,science_imports=[n for n in sys.modules if n.rsplit('.',1)[-1]in P.FORBIDDEN])
    final_before=D.clock();final_ok=True
    try:final_sha=_save(out,'final.json',report,ctl,fault,slot=K.FINAL_SLOT,pre_reserved=True)
    except BaseException as exc:final_ok=False;final_sha=sha256((out/'final.json').read_bytes()).hexdigest()if(out/'final.json').exists()else None;late_errors.append(repr(exc))
    final_after=D.clock()
    delivery=dict(schema='b3_bundle_external_delivery_v1',run_id=m['run_id'],manifest_binding=binding,final_sha256=final_sha,final_write_flush_close_observed=final_ok,
        final_persistence_cpu_wall_ns=D.delta(final_before,final_after),outer_parent_cpu_ns=final_after[0]-before[0],outer_wall_ns=final_after[1]-before[1],
        worker_cpu_ns=host['worker_cpu_ns'],post_final_errors=late_errors,production_permission=False,
        own_close_is_not_self_certified=True,fsync_durability_verified=False)
    try:_save(out,'delivery.json',delivery,ctl,fault,slot=K.DELIVERY_SLOT,pre_reserved=True)
    except BaseException:raise
    return dict(complete=complete and final_ok,report=report,delivery=delivery,output=str(out),fault_hits=fault.hits)
