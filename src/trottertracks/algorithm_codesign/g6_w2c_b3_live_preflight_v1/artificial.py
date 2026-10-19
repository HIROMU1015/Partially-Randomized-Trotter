"""Correctness-only artificial adapter -> existing spawn/F01/Storage V2.

The historical scheduler, codec, result adapter, worker protocol and storage are
reused unchanged. This wrapper supplies sixteen correctness pair tasks only.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from hashlib import sha256
import json
from .guard import install, require, observation
from . import contract as K, spec as S
from .interface import ConnectionIdentity, validate_connection, coverage
from ..g6_w2c_b3_real_adapter_v1 import adapter as OLD, bits as B, codec as C, offline as O, fixture as X
from ..g6_w2c_b3_real_adapter_v1.bridge import Coordinator, Fault, _save
from .. import g6_w2c_b3_pilot_storage_v2 as V2
from ..g6_w2c_b3_bundle_store_v1 import FakeStore, private
from .. import g6_w2c_b3_offline_result as V


def bits_factory(arm, axis):
    install()
    return B.ArtificialBits(arm,axis)


class ArtificialFactory:
    artificial = True

    def __init__(self, c, m):
        self.c, self.m = deepcopy(c), deepcopy(m)
        self.runtime_manifest = deepcopy(c['worker_runtime'])
        self.seal = S.digest([self.c,self.m])

    def __call__(self, task, settings):
        install()
        require(type(self) is ArtificialFactory and self.artificial is True, 'BUILTIN_FACTORY_ONLY')
        require(S.digest([self.c,self.m]) == self.seal, 'FACTORY_MUTATION')
        K.preflight(self.c,self.m,destination=self.m['storage_destination'])
        require(task in self.m['tasks'], 'UNAUTHORIZED_TASK')
        context = dict(artificial_only=True, run_id=self.m['run_id'],tasks=self.m['tasks'],
            port_faults=self.m['artificial_faults'], candidate_binding=S.digest(self.c),
            manifest_binding=S.digest(self.m),runtime_binding=S.digest(self.runtime_manifest),
            endpoint_binding=dict(real_dispatch=False,adapter=__name__+'.ArtificialFactory'))
        return Endpoint(task,settings,context,self.c,self.m)


class Endpoint:
    def __init__(self,task,settings,context,c,m):
        install()
        self.base = OLD.Endpoint(task,settings,context)
        self.c,self.m,self.task = c,m,task

    def shared_identity(self):
        return self.base.shared_identity()

    def sample(self,bits,trial_id):
        install()
        outcome = self.base.sample(bits,trial_id)
        phase,pass_name,arm,layer,axis,index = trial_id.split('/')
        ident = ConnectionIdentity(self.c['source_release_candidate'],S.digest(S.INPUT),
            S.digest(S.PRECISION),S.digest(self.c['python']),self.m['run_id'],self.m['global_claim'],
            self.task['condition'],arm,layer,axis,phase,pass_name,int(index),self.shared_identity())
        outcome.evidence['live_preflight'] = dict(connection_binding=ident.binding(),
            global_trial_id=ident.global_id(),route=S.route(arm,layer),guard=observation(),
            artificial_only=True,live_compatibility='NOT_ESTABLISHED')
        return outcome

    def angle_for(self,outcome):
        install()
        return self.base.angle_for(outcome)


def run(c,m,root,store,*,storage_fault=None):
    """Only a new owned FAKE_ONLY store is accepted. No factory injection API."""
    install()
    root = private(root)
    K.preflight(c,m,destination=root/'output')
    require(type(store) is FakeStore and store.root == root, 'BUILTIN_FAKE_STORE_ONLY')
    require(storage_fault is None or type(storage_fault) is Fault, 'BUILTIN_ARTIFICIAL_FAULT_ONLY')
    fault = storage_fault if storage_fault is not None else Fault()
    try:
        store.claim(m,fault)
    except BaseException:
        try: store.stop(m)
        except BaseException: pass
        raise
    out=root/'output'
    out.mkdir(exist_ok=False)
    settings=V2.P.Settings(mode='S',profile_n=16,codec='fused',workload='light',call_limit=512,
                         result_bytes=m['budgets']['result_bytes'],log_bytes=m['budgets']['log_bytes'])
    ctl=Coordinator(settings,out,m['tasks'],512,m['run_id'],fault,manifest=m)
    fault.control=ctl
    ctl.parent_save_cost={}
    factory=ArtificialFactory(c,m)
    conditions=[]
    try:
        # Reserve fixed final/caller slots in the SAME global result budget.
        for size in (4<<20,4096,4096): ctl.budget.reserve('result',size)
        _save(out,'candidate.json',c,ctl,fault)
        _save(out,'manifest.json',m,ctl,fault)
        for mode in ('S','P1'):
            if ctl.stopped.value: break
            ctl.condition=mode
            ctl.settings=replace(ctl.settings,mode=mode)
            ctl.condition_event('CONDITION_BEGIN')
            selected=[t for t in m['tasks'] if t['condition']==mode]
            ctl.run_phase(selected,c['runtime_root'],c['source_pins'],factory,bits_factory,None)
            rows=[r for r in ctl.results if r['task']['condition']==mode]
            okay=not ctl.stopped.value and len(rows)==8 and all(r['status']=='OK' for r in rows)
            conditions.append(dict(mode=mode,complete=okay,pairs=len(rows)))
            ctl.condition_event('CONDITION_END',complete=okay)
            if not okay: break
            K.preflight(c,m,destination=out)
        ctl.condition='SHARED'
    except BaseException as exc:
        ctl.stop(dict(parent_failure=repr(exc)))
        if ctl.active:
            ctl.run_phase([],c['runtime_root'],c['source_pins'],factory,bits_factory,None,drain_only=True)
    finally:
        store.stop(m)
        _save(out,'FAKE_ONLY_snapshot.json',store.snapshot(m),ctl,fault)
        V2.close_storage(ctl)
        ctl.closed=True
        ctl.budget.notify=ctl.post_notify
    report=ctl.report()
    report.update(schema='b3_correctness_only_artificial_result_v1',conditions=conditions,
        candidate_binding=S.digest(c),manifest_binding=S.digest(m),guard=observation(),
        live_compatibility='NOT_ESTABLISHED',separate_reviewer=False)
    try:
        if report['complete']:
            report['receipt_sha256']=_save(out,'receipt.json',dict(schema='ARTIFICIAL_CORRECTNESS_RECEIPT',completed=512,
                candidate_binding=S.digest(c),manifest_binding=S.digest(m),
                source_release_candidate=c['source_release_candidate'],claim=store.snapshot(m),
                state_sha256=sha256((out/'trial_states.bin').read_bytes()).hexdigest(),
                trace_sha256=sha256((out/'operations.jsonl').read_bytes()).hexdigest(),
                storage_closed=True,production_permission=False),ctl,fault)
        _save(out,'final.json',report,ctl,fault,slot=4<<20,pre_reserved=True)
        _save(out,'delivery.json',dict(final_sha256=sha256((out/'final.json').read_bytes()).hexdigest(),
            artificial_only=True),ctl,fault,slot=4096,pre_reserved=True)
        _save(out,'caller.json',dict(delivery_sha256=sha256((out/'delivery.json').read_bytes()).hexdigest(),
            observed=True,artificial_only=True),ctl,fault,slot=4096,pre_reserved=True)
    except BaseException as exc:
        # Evidence failure is terminal and cannot be repaired into success.
        report['complete']=False
        report['finalization_failure']=repr(exc)
    return report


def inspect(root,c,m):
    """Read-only checker for the NEW task projection; reuses exact adapter validators.

    Same-author check, not a separate independent reviewer. Storage V2 itself is
    unchanged and its fault/mutation regressions are retained separately.
    """
    root=Path(root);out=root/'output'
    witnesses=['final.json','delivery.json','caller.json']
    if any(not (out/n).is_file() for n in witnesses):
        return dict(status='INCOMPLETE_EVIDENCE',missing=[n for n in witnesses if not (out/n).is_file()])
    report=json.loads((out/'final.json').read_text())
    delivery=json.loads((out/'delivery.json').read_text())
    caller=json.loads((out/'caller.json').read_text())
    require(delivery['final_sha256']==sha256((out/'final.json').read_bytes()).hexdigest(), 'FINAL_DELIVERY')
    require(caller['observed'] is True and caller['delivery_sha256']==sha256((out/'delivery.json').read_bytes()).hexdigest(),'CALLER_DELIVERY')
    require(report['candidate_binding']==S.digest(c) and report['manifest_binding']==S.digest(m),'REPORT_BINDING')
    if not report['complete']:
        stopped=False;entries_after_stop=0
        for line in (out/'operations.jsonl').read_bytes().splitlines():
            row=json.loads(json.loads(line)['payload'])
            if row['op']=='STOP': stopped=True
            if row['op']=='ENTER' and stopped: entries_after_stop+=1
        require(stopped and entries_after_stop==0,'FAILURE_STOP_TRACE')
        require(not (out/'receipt.json').exists(),'NO_FAILURE_RECEIPT')
        return dict(status='TECHNICAL_STOP',completed=report['states']['COMPLETED'],
                    conditions=report['conditions'],receipt=False,failures=report['failures'],
                    ENTER_after_STOP=entries_after_stop)
    receipt=json.loads((out/'receipt.json').read_text())
    require(receipt['completed']==512 and receipt['storage_closed'] is True,'RECEIPT')
    require(sha256((out/'receipt.json').read_bytes()).hexdigest()==report['receipt_sha256'],'RECEIPT_DIGEST')
    require(receipt['candidate_binding']==S.digest(c) and receipt['manifest_binding']==S.digest(m),'RECEIPT_BINDING')
    require(receipt['claim']['claim'][3:]==['STOPPED',1],'SPENT_CLAIM_STOPPED')
    for name,key in [('trial_states.bin','state_sha256'),('operations.jsonl','trace_sha256')]:
        require(sha256((out/name).read_bytes()).hexdigest()==receipt[key],'RECEIPT_FILE:'+name)
    require((out/'trial_states.bin').read_bytes()==bytes([4])*512,'ALL_COMPLETE')
    joins={};payloads={};guards={};cells=[]
    for task in m['tasks']:
        path=out/(task['key']+'.jsonl');previous='0'*64
        counts={x:[0,0] for x in ('X','Y')}
        counters={x:0 for x in ('X','Y')}
        for line in path.read_bytes().splitlines():
            row=json.loads(line);require(row['previous']==previous,'ROW_CHAIN')
            previous=sha256((previous+row['payload']).encode()).hexdigest()
            require(row['sha256']==previous,'ROW_HASH')
            p=V.decode(row['payload'])
            tid=p['trial_id'];parts=tid.split('/');axis,index=parts[-2],int(parts[-1]);ev=p['evidence']
            adapted,consumption,angle,counters[axis]=O.reference(task['arm'],axis,counters[axis],tid,task['layer'])
            require(C.driver_canonical(ev['adapted'])==C.driver_canonical(adapted),'SAVED_ADAPTER_PROJECTION')
            require(C.driver_canonical(ev['consumption'])==C.driver_canonical(consumption),'SAVED_CONSUMPTION')
            require(C.driver_canonical(p.get('angle'))==C.driver_canonical(angle),'SAVED_ANGLE')
            audit=None
            if task['layer']=='B':
                arm=task['arm'];identity=C.fingerprint(dict(arm=arm,source=X.SOURCE))
                state=C.fingerprint(dict(artificial=True,arm=arm,layer='B'))
                seal=C.DriverDocument.create(dict(arm=arm,identity=identity,state_id=state,
                    payload=adapted['codec']['f3']['document'],completed_calls_before=index+(16 if axis=='Y' else 0),confidence_certified=False))
                audit=O.R.canonical_audit(seal,arm=arm,identity=identity,state_id=state,completed_calls_before=index+(16 if axis=='Y' else 0))
                require(C.driver_canonical(ev.get('audit'))==C.driver_canonical(audit),'SAVED_AUDIT')
            check=validate_connection(adapted,consumption,trial_id=tid,
                arm=task['arm'],layer=task['layer'],axis=axis,angle=angle,audit=audit)
            gid=m['run_id']+'/'+task['condition']+'/'+tid
            require(ev['dispatch']['global_trial_id']==gid==ev['live_preflight']['global_trial_id'],'GLOBAL_ID')
            require(ev['live_preflight']['route']==S.route(task['arm'],task['layer']),'ROUTE')
            ident=ConnectionIdentity(c['source_release_candidate'],S.digest(S.INPUT),S.digest(S.PRECISION),
                S.digest(c['python']),m['run_id'],m['global_claim'],task['condition'],task['arm'],task['layer'],
                axis,'correctness','baseline',index,
                V2.D.identity(dict(schema='ARTIFICIAL_ADAPTER_V1',task=task,
                    endpoint=dict(real_dispatch=False,adapter=__name__+'.ArtificialFactory'),
                    contract=S.digest(c),manifest=S.digest(m))))
            require(ev['live_preflight']['connection_binding']==ident.binding(),'CONNECTION_BINDING')
            g=ev['live_preflight']['guard'];require(not g['core_loaded'] and not g['blocked_attempts'],'WORKER_CORE_LOAD')
            guards[g['pid']]=g
            offset=task['offset']+(0 if axis=='X' else 16)+index
            require(offset not in joins,'DUPLICATE_TRIAL');joins[offset]=gid
            counts[axis][0]+=1;counts[axis][1]+=check['classification']=='ACCEPTED'
            # A/B and S/P1 compare event and bit prefixes; contexts deliberately differ.
            key=(task['arm'],axis,index)
            value=C.fingerprint(dict(event=ev['adapted']['event'],consumption=ev['consumption']))
            require(key not in payloads or payloads[key]==value,'AB_SP1_PREFIX')
            payloads[key]=value
        for axis,(completed,accepted) in counts.items():
            cells.append(dict(cell=[task['condition'],task['arm'],task['layer'],axis],completed=completed,accepted=accepted))
    entered=[];completed=[];stopped=False
    previous='0'*64
    for seq,line in enumerate((out/'operations.jsonl').read_bytes().splitlines()):
        envelope=json.loads(line)
        row=json.loads(envelope['payload'])
        require(envelope['previous']==previous and row['seq']==seq and row['run_id']==m['run_id'],'TRACE_CHAIN')
        previous=sha256(previous.encode()+envelope['payload'].encode()).hexdigest()
        require(envelope['sha256']==previous,'TRACE_DIGEST')
        op=row.get('op')
        if op=='STOP': stopped=True
        if op=='ENTER':
            require(not stopped,'ENTER_AFTER_STOP');entered.append(row['global_offset'])
        if op=='COMPLETE': completed.append(row['global_offset'])
    require(sorted(entered)==list(range(512)) and sorted(completed)==list(range(512)),'TRACE_JOIN')
    require(report['total_cpu_ns']==report['parent_cpu_ns']+report['worker_cpu_ns'],'CPU_ADD_ONCE')
    return dict(status='VERIFIED_ARTIFICIAL_CORRECTNESS_ONLY',trials=len(joins),full_documents=len(joins),
        guards=list(guards.values()),cells=cells,coverage_candidate=coverage(cells),
        core_loaded=[],live_compatibility='NOT_ESTABLISHED',receipt_caller_join=True,
        total_cpu_ns=report['total_cpu_ns'],parent_cpu_ns=report['parent_cpu_ns'],
        worker_cpu_ns=report['worker_cpu_ns'],output=report['output'])
