"""Versioned pilot orchestration over the unchanged Global Bundle/Storage V2."""
from pathlib import Path
from dataclasses import replace
from hashlib import sha256
import time,resource,json,os
from . import contract as K,spec as S,metadata as M
from .guard import install,require,observation
from .run_control import SimpleStore,mock_approval,private
from .execution import Coordinator
from .wrapper import ArtificialFactory
from .bits import ArtificialBits
from ..g6_w2c_b3_real_adapter_v1.bridge import Fault,_save
from .. import g6_w2c_b3_pilot_storage_v2 as V2


def bits_factory(arm,axis):install();return ArtificialBits(arm,axis)
def children():
    r=resource.getrusage(resource.RUSAGE_CHILDREN);return round((r.ru_utime+r.ru_stime)*1e9)
def resources():
    return dict(monotonic_ns=time.monotonic_ns(),loadavg=os.getloadavg(),affinity=sorted(os.sched_getaffinity(0)),
        meminfo=Path('/proc/meminfo').read_text(),pressure={n:(Path('/proc/pressure')/n).read_text() for n in ('cpu','memory','io')},
        own_status=Path('/proc/self/status').read_text(),own_io=Path('/proc/self/io').read_text())


def preserve_finalization_failure(out,m,store,ctl,report,exc):
    """Best effort within a dedicated pre-reserved slot; never refund a claim."""
    store.stop(m,unknown=True)
    report.update(complete=False,classification='INCOMPLETE_EVIDENCE',finalization_failure=repr(exc))
    marker=dict(schema='b3_pilot_finalization_failure_v1',mandatory_STOP=True,retry=0,resume=False,
        claim=m['global_claim'],manifest_binding=S.digest(m),source=m.get('measured_SOURCE'),
        classification='INCOMPLETE_EVIDENCE',failure=repr(exc)[:1500],claim_spent=True,execution_state='UNKNOWN',
        execution_metadata=M.context(m),unverified_success_receipt_must_not_be_used=True)
    try:report['emergency_STOP_marker_sha256']=_save(out,'STOP_failure_marker.json',marker,ctl,Fault(),slot=4096,pre_reserved=True)
    except BaseException as failure:report['emergency_marker_failure']=repr(failure)
    return report


def _run(c,m,root,store,permit,outer_begin):
    live=permit is not None;execution_meta=M.context(m,live=live);root=private(root)
    if live:K.preflight_runtime(c,m,permit)
    else:K.preflight(c,m)
    require(type(store) is SimpleStore and store.root==root,'OWNED_STORE')
    try:
        if live:store.claim_live(c,m,permit)
        else:store.claim(m,mock_approval(m))
    except BaseException:
        try:store.stop(m,unknown=True)
        except BaseException:pass
        raise
    out=root/'output';out.mkdir(exist_ok=False)
    settings=V2.P.Settings(mode='S',profile_n=256,codec='fused',workload='light',call_limit=16896,
        workers=8,result_bytes=m['byte_limits']['result'],log_bytes=m['byte_limits']['log'])
    fault=Fault(dict(point='receipt.json.close')) if m['scenario']=='receipt_close' else Fault()
    ctl=Coordinator(settings,out,m['tasks'],16896,m['run_id'],fault,manifest=m);fault.control=ctl;ctl.parent_save_cost={}
    factory=ArtificialFactory(c,m);stream_factory=bits_factory
    if live:
        from .wrapper import ScientificFactory
        from .bits import ScientificBitsFactory
        factory=ScientificFactory(c,m,permit);stream_factory=ScientificBitsFactory(c,m,permit)
    phases=[];coverage_stop=False;checks=[]
    try:
        for size in (12<<20,4096,4096,65536,4096,4096):ctl.budget.reserve('result',size)
        _save(out,'candidate.json',c,ctl,fault);_save(out,'manifest.json',m,ctl,fault)
        for phase in range(6):
            if ctl.stopped.value:break
            tasks=[t for t in m['tasks'] if t['phase_index']==phase];mode=tasks[0]['condition'];ctl.condition=mode;store.enter(m)
            faults=((1,phase,'ack_exit'),) if m['scenario']=='unknown_ack' and phase==0 else ()
            ctl.settings=replace(ctl.settings,mode=mode,faults=faults)
            ctl.condition_event('PHASE_BARRIER_BEGIN',phase_index=phase)
            ctl.run_phase(tasks,c['runtime_root'],c['source_pins'],factory,stream_factory,None)
            selected=[r for r in ctl.results if r['task']['phase_index']==phase]
            okay=not ctl.stopped.value and len(selected)==8 and all(r['status']=='OK' for r in selected)
            coverage=okay and all(a['counts'].get('EVENT',0)>0 for row in selected for a in row['outcome']['axes'])
            phases.append(dict(phase_index=phase,condition=mode,phase=tasks[0]['phase'],pass_name=tasks[0]['pass_name'],
                complete=okay,pairs=len(selected),coverage=coverage if tasks[0]['phase']=='correctness' else None,
                resource_observation=resources()))
            ctl.condition_event('PHASE_BARRIER_END',phase_index=phase,complete=okay,coverage=coverage)
            if not okay:break
            from .offline import check_pairs
            checked=check_pairs(out,c,m,ctl.results,artificial=not live);checks.append(checked)
            _save(out,f'phase_{phase}_check.json',checked,ctl,fault)
            if tasks[0]['phase']=='correctness' and not coverage:
                coverage_stop=True;ctl.stop(dict(classification='INCONCLUSIVE_COVERAGE',phase_index=phase));break
            _save(out,f'phase_{phase}_receipt.json',dict(complete=True,phase_index=phase,manifest_binding=S.digest(m),execution_metadata=execution_meta,production_permission=False),ctl,fault)
            if live:K.preflight_runtime(c,m,permit)
            else:K.preflight(c,m)
        if not ctl.stopped.value:
            from .offline import check_pairs
            _save(out,'pair_validation.json',check_pairs(out,c,m,ctl.results,artificial=not live,require_all=True),ctl,fault)
    except BaseException as exc:
        ctl.stop(dict(parent_failure=repr(exc)))
        if ctl.active:ctl.run_phase([],c['runtime_root'],c['source_pins'],factory,stream_factory,None,drain_only=True)
    finally:
        store.stop(m)
        V2.close_storage(ctl);ctl.closed=True;ctl.budget.notify=ctl.post_notify
    report=ctl.report()
    if any('UNKNOWN' in k and v for k,v in report['states'].items()):store.stop(m,unknown=True)
    report.update(schema='b3_limited_pilot_live_bundle_v1' if live else 'b3_limited_pilot_artificial_bundle_v1',
        phases=phases,candidate_binding=S.digest(c),manifest_binding=S.digest(m),
        classification='LIVE_TECHNICAL_COMPLETE_AWAITING_REVIEW' if report['complete'] and live else
        'ARTIFICIAL_TECHNICAL_COMPLETE' if report['complete'] else 'INCONCLUSIVE_COVERAGE' if coverage_stop else 'TECHNICAL_STOP',
        parent_guard=observation())
    try:
        if report['complete']:
            receipt=dict(schema='b3_limited_pilot_receipt_v1',completed=16896,classification=report['classification'],
                candidate_binding=S.digest(c),manifest_binding=S.digest(m),execution_metadata=execution_meta,
                state_sha256=sha256((out/'trial_states.bin').read_bytes()).hexdigest(),trace_sha256=sha256((out/'operations.jsonl').read_bytes()).hexdigest(),
                store=store.snapshot(m),production_permission=False,storage_closed=True)
            report['receipt_sha256']=_save(out,'receipt.json',receipt,ctl,fault);store.link_receipt(m,report['receipt_sha256'])
        report['STOP_marker_sha256']=_save(out,'STOP_marker.json',dict(mandatory_STOP=True,claim=m['global_claim'],store=store.snapshot(m),
            classification=report['classification'],execution_metadata=execution_meta,receipt_sha256=report.get('receipt_sha256')),ctl,fault)
        _save(out,'final.json',report,ctl,fault,slot=12<<20,pre_reserved=True)
        _save(out,'delivery.json',dict(final_sha256=sha256((out/'final.json').read_bytes()).hexdigest(),artificial_only=not live,execution_metadata=execution_meta),ctl,fault,slot=4096,pre_reserved=True)
        _save(out,'caller.json',dict(delivery_sha256=sha256((out/'delivery.json').read_bytes()).hexdigest(),observed=True,artificial_only=not live,execution_metadata=execution_meta),ctl,fault,slot=4096,pre_reserved=True)
        if report['complete']:store.mark_completed(m)
        _save(out,'simple_run_state.json',store.snapshot(m),ctl,fault,slot=4096,pre_reserved=True)
        cpu,wall,ch=outer_begin
        parent=time.process_time_ns()-cpu;child=children()-ch
        outer=dict(schema='b3_pilot_outer_measurement_v1',parent_cpu_ns=parent,reaped_child_hierarchy_cpu_ns=child,total_cpu_ns=parent+child,
            global_monotonic_wall_ns=time.monotonic_ns()-wall,worker_END_own_CPU_ns=sum(r.get('process_cpu_ns') or 0 for r in report['pairs']),
            window='before admission/preflight/store creation through all joins/finalization/STOP/COMPLETED state export; this outer export excluded',
            child_cpu_includes_reaped_git_helpers=True,unreaped_resource_tracker_CPU=None,coarse_and_fine_added=False,
            resource_observation=resources(),parent_storage_costs=dict(ctl.parent_save_cost),
            observer_isolated_cpu_ns=None,IPC_unisolated_cpu_ns=None,postprocess_not_in_window=True)
        _save(out,'host_outer.json',outer,ctl,fault,slot=65536,pre_reserved=True)
    except BaseException as exc:
        preserve_finalization_failure(out,m,store,ctl,report,exc)
    return report


def run(c,m,root,store):
    begin=time.process_time_ns(),time.monotonic_ns(),children()
    return _run(c,m,root,store,None,begin)


def run_live(c,m,root,permit=None,outer_begin=None):
    from .activation import admit
    begin=outer_begin or (time.process_time_ns(),time.monotonic_ns(),children())
    if permit is None:permit=admit(c,m,root)
    K.preflight_runtime(c,m,permit)
    store=SimpleStore(root,permit=permit,candidate=c,manifest=m)
    try:return _run(c,m,root,store,permit,begin)
    except BaseException:
        if store._intent(m).exists() or store._run_intent(m).exists():store.stop(m,unknown=True)
        raise
