"""One owned TEST_ONLY claim -> S then P1 -> versioned pair/F01 -> Storage V2."""
from pathlib import Path
from dataclasses import replace
from hashlib import sha256
import resource,time,json
from .guard import install,require,observation
from . import contract as K
from .run_control import SimpleStore,mock_approval,private
from .wrapper import ArtificialFactory
from .bits import ArtificialBits
from .execution import Coordinator
from ..g6_w2c_b3_real_adapter_v1.bridge import Fault,_save
from .. import g6_w2c_b3_pilot_storage_v2 as V2
from ..g6_w2c_b3_live_preflight_v1 import spec as S
from ..g6_w2c_b3_real_adapter_v1 import codec as C


def bits_factory(arm,axis):install();return ArtificialBits(arm,axis)


def cpu_children():
    value=resource.getrusage(resource.RUSAGE_CHILDREN)
    return round((value.ru_utime+value.ru_stime)*1e9)


def _run(c,m,root,store,*,permit=None):
    begin_cpu,begin_wall,begin_children=time.process_time_ns(),time.monotonic_ns(),cpu_children()
    install();root=private(root)
    live=permit is not None
    if live:K.preflight_runtime(c,m,permit)
    else:K.preflight(c,m)
    require(type(store) is SimpleStore and store.root==root,'TEST_STORE_ONLY')
    require(Path(m['storage_destination'])==root/'output','STORAGE_DESTINATION')
    grant=None if live else mock_approval(m)
    try:
        if live:store.claim_live(c,m,permit)
        else:store.claim(m,grant)
    except BaseException:
        try:store.stop(m)
        except BaseException:pass
        raise
    out=root/'output';out.mkdir(exist_ok=False)
    settings=V2.P.Settings(mode='S',profile_n=16,codec='fused',workload='light',
        call_limit=511 if m.get('scenario','normal')=='call_cap' else 512,
        result_bytes=m['byte_limits']['result'],log_bytes=m['byte_limits']['log'])
    fault=Fault(dict(point='receipt.json.close')) if m.get('scenario','normal')=='receipt_close' else Fault()
    ctl=Coordinator(settings,out,m['tasks'],512,m['run_id'],fault,manifest=m)
    fault.control=ctl;ctl.parent_save_cost={};factory=ArtificialFactory(c,m);stream_factory=bits_factory
    if live:
        from .wrapper import ScientificFactory
        from .bits import ScientificBitsFactory
        factory=ScientificFactory(c,m,permit);stream_factory=ScientificBitsFactory(c,m,permit)
    conditions=[]
    try:
        for size in (4<<20,4096,4096,8192,4096):ctl.budget.reserve('result',size)
        _save(out,'candidate.json',c,ctl,fault);_save(out,'manifest.json',m,ctl,fault)
        for mode in ('S','P1'):
            if ctl.stopped.value:break
            store.enter(m);ctl.condition=mode
            faults=((1,0,'ack_exit'),) if mode=='S' and m.get('scenario','normal')=='unknown_ack' else ()
            ctl.settings=replace(ctl.settings,mode=mode,faults=faults)
            ctl.condition_event('CONDITION_BEGIN')
            tasks=[t for t in m['tasks'] if t['condition']==mode]
            ctl.run_phase(tasks,c['runtime_root'],c['source_pins'],factory,stream_factory,None)
            rows=[r for r in ctl.results if r['task']['condition']==mode]
            okay=not ctl.stopped.value and len(rows)==8 and all(r['status']=='OK' for r in rows)
            coverage=okay and all(axis['counts'].get('EVENT',0)>0 for r in rows for axis in r['outcome']['axes'])
            condition=dict(mode=mode,complete=okay,coverage=coverage,pairs=len(rows))
            conditions.append(condition);ctl.condition_event('CONDITION_END',complete=okay,coverage=coverage)
            if not okay:break
            _save(out,mode+'_receipt.json',dict(schema='ARTIFICIAL_CONDITION_RECEIPT',
                condition=condition,binding=S.digest(m),claim=m['global_claim'],production_permission=False),ctl,fault)
            if live:K.preflight_runtime(c,m,permit)
            else:K.preflight(c,m)
        if not ctl.stopped.value:
            from .offline import check_pairs
            checked=check_pairs(out,c,m,ctl.results,artificial=not live)
            _save(out,'pair_validation.json',checked,ctl,fault)
        ctl.condition='SHARED'
    except BaseException as exc:
        ctl.stop(dict(parent_failure=repr(exc)))
        if ctl.active:ctl.run_phase([],c['runtime_root'],c['source_pins'],factory,stream_factory,None,drain_only=True)
    finally:
        try:store.stop(m)
        except BaseException as exc:ctl.stop(dict(test_store_STOP_failure=repr(exc)))
        try:_save(out,'TEST_ONLY_snapshot.json',store.snapshot(m),ctl,fault)
        except BaseException as exc:ctl.stop(dict(snapshot_failure=repr(exc)))
        V2.close_storage(ctl);ctl.closed=True;ctl.budget.notify=ctl.post_notify
    report=ctl.report()
    if any('UNKNOWN' in k and v for k,v in report['states'].items()):store.stop(m,unknown=True)
    report.update(schema='b3_live_release_artificial_bundle_v1',conditions=conditions,
        candidate_binding=S.digest(c),manifest_binding=S.digest(m),guard=observation(),
        science_correctness_certified=False,live_API_compatibility='NOT_ESTABLISHED')
    if report['complete']:
        report['classification']='INCONCLUSIVE_COVERAGE' if any(not x['coverage'] for x in conditions) else ('LIVE_TECHNICAL_COMPLETE_AWAITING_REVIEW' if live else 'ARTIFICIAL_TECHNICAL_COMPLETE')
    else:
        report['classification']='ARTIFICIAL_SCIENTIFIC_MISMATCH_STOP' if 'ARTIFICIAL_SCIENTIFIC_MISMATCH' in C.driver_canonical(report['failures']) else 'TECHNICAL_STOP'
    try:
        if report['complete']:
            receipt=dict(schema='b3_TEST_ONLY_release_receipt_v1',completed=512,classification=report['classification'],
                candidate_binding=S.digest(c),manifest_binding=S.digest(m),source=c['source_release_candidate'],
                state_sha256=sha256((out/'trial_states.bin').read_bytes()).hexdigest(),
                trace_sha256=sha256((out/'operations.jsonl').read_bytes()).hexdigest(),
                store=store.snapshot(m),storage_closed=True,production_permission=False)
            digest=_save(out,'receipt.json',receipt,ctl,fault);store.link_receipt(m,digest)
            report['receipt_sha256']=digest
        report['STOP_marker_sha256']=_save(out,'STOP_marker.json',dict(mandatory_STOP=True,
            claim=m['global_claim'],store=store.snapshot(m),classification=report['classification'],
            receipt_sha256=report.get('receipt_sha256')),ctl,fault)
        _save(out,'final.json',report,ctl,fault,slot=4<<20,pre_reserved=True)
        _save(out,'delivery.json',dict(final_sha256=sha256((out/'final.json').read_bytes()).hexdigest(),
            artificial_only=not live),ctl,fault,slot=4096,pre_reserved=True)
        _save(out,'caller.json',dict(delivery_sha256=sha256((out/'delivery.json').read_bytes()).hexdigest(),
            observed=True,artificial_only=not live),ctl,fault,slot=4096,pre_reserved=True)
        parent=time.process_time_ns()-begin_cpu;workers=cpu_children()-begin_children
        outer=dict(schema='b3_host_outer_observation_v1',parent_cpu_ns=parent,all_worker_cpu_ns=workers,
            total_cpu_ns=parent+workers,outer_wall_ns=time.monotonic_ns()-begin_wall,
            window='before preflight/admission through all joins, STOP, storage close, receipt/delivery/caller close',
            observation_export_excluded_as_postprocessing=True,coarse_and_fine_added=False,
            boundary_byte_ledger_before_outer_export=dict(reserved=dict(ctl.budget.reserved),observed=dict(ctl.budget.observed)),
            post_storage_close_credits=list(ctl.post_close_credits),physical_power_loss_durability='NOT_ESTABLISHED')
        partitions={}
        for kind in ('coarse','fine'):
            partition=sum(sum(v['cpu_ns'] for v in (row.get(kind) or {}).values()) for row in report['pairs'])
            partitions[kind]=dict(worker_exclusive_cpu_ns=partition,
                outer_residual_cpu_ns=parent+workers-partition,
                residual_includes_parent_bootstrap_unobserved_and_observer_overhead=True)
        outer['exclusive_partitions_reconciled_separately']=partitions
        # Summarize credits rather than duplicating thousands of rows in this slot.
        outer['post_storage_close_credit_events']=len(outer.pop('post_storage_close_credits'))
        _save(out,'host_outer.json',outer,ctl,fault,slot=8192,pre_reserved=True)
        if report['complete']:store.mark_completed(m)
        _save(out,'simple_run_state.json',store.snapshot(m),ctl,fault,slot=4096,pre_reserved=True)
    except BaseException as exc:
        try:store.stop(m,unknown=True)
        except BaseException:pass
        report['complete']=False;report['classification']='INCOMPLETE_EVIDENCE';report['finalization_failure']=repr(exc)
    return report


def run(c,m,root,store):
    """Only artificial entry is usable in this preparation release."""
    try:return _run(c,m,root,store)
    except BaseException:
        if store._intent(m).exists() and store._run_intent(m).exists():store.stop(m,unknown=True)
        raise


def run_live(c,m,root,permit):
    from .gate import require_live
    require_live(permit,c,m)
    K.preflight_runtime(c,m,permit)
    store=SimpleStore(root,permit=permit,candidate=c,manifest=m)
    try:return _run(c,m,root,store,permit=permit)
    except BaseException:
        if store._intent(m).exists() and store._run_intent(m).exists():store.stop(m,unknown=True)
        raise
