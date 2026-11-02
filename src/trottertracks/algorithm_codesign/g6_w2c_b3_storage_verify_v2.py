"""Storage V2 supplemental checker. Never imports a producer or scheduler.

Fixed independent checker modules remain unchanged. This is author-run
supplemental verification; final independent acceptance belongs to local Codex.
"""
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from . import g6_w2c_b3_offline_result as V
from . import g6_w2c_b3_parallel_verifier_core as G
from . import g6_w2c_b3_parallel_offline_adapter as W
from . import g6_w2c_b3_v3_offline_adapter as B
from . import g6_w2c_b3_pilot_independent_trace as T
from . import g6_w2c_b3_pilot_independent_adapter as E
from . import g6_w2c_b3_pilot_independent_identity as I


def chain_only(path,run_id):
    previous=V.ZERO;seq=0
    with Path(path).open('rb')as f:
        for line in f:
            V.require(line.endswith(b'\n'),'PREFIX_PARTIAL_LINE')
            e=V.plain(line);raw=e['payload'].encode()
            V.require(e['previous']==previous and e['sha256']==sha256(previous.encode()+raw).hexdigest(),'PREFIX_CHAIN')
            value=V.plain(raw)
            V.require(value['seq']==seq and type(value['seq'])is int and value['run_id']==run_id,'PREFIX_SEQUENCE')
            V.require(json.dumps(value,separators=(',',':'),ensure_ascii=True).encode()==raw,'PREFIX_CANONICAL')
            previous=e['sha256'];seq+=1
    return dict(rows=seq,chain=previous)


def verify(output,c,inner,r):
    # Match saved JSON container types, retaining exact Fraction tags.
    r=W.safe_values(V.plain(json.dumps(r,default=lambda x:{'exact':V.canonical(x)})))
    output=Path(output);require=V.require
    require(c['schema']=='b3_pilot_storage_repair_candidate_v2' and c['contract_grants_execution']is False and
        c['run_ready']is False and c['science_release']is None,'CLOSED_V2_CANDIDATE')
    require(r['schema']=='b3_pilot_storage_result_v2','STORAGE_RESULT_V2')
    require(V.identity(c)==r['pilot_binding']==inner['pilot_candidate_binding'],'PILOT_BINDING')
    tasks,planned=G.plan(inner['settings']['profile_n']);ev=r['evidence_v2']
    require(planned==r['planned'] and sum(r['states'].values())==planned,'PLANNED_STATES')
    require(r['reserved']==planned-r['states']['NOT_STARTED'] and
        r['reserved']==sum(ev['next_indices'].values())<=min(planned,inner['settings']['call_limit']),
        'LIVE_RESERVATION_COUNTER_CURSOR_BINDING')
    require(r['started_observed']<=r['reserved'] and not ev['current_tickets'],'ENTRY_OR_UNDRAINED_TICKET')
    require(r['retry']==0 and type(r['retry'])is int and r['resume']is False and
        r['fake_claim_status']in('STOPPED','ADMITTED','RESERVED') and r['call_reservation_nonrefundable']is True,'ONE_SHOT_SPENT')
    if r['fake_claim_status']!='STOPPED'or not r.get('admission_stop')or r['admission_stop']['record_errors']:
        require(r['complete']is False,'FALSE_SUCCESS_WITH_ADMISSION_STOP_FAILURE')
    physical=dict(log=0,result=0)
    for path in output.iterdir():
        require(path.is_file()and not path.is_symlink(),'REGULAR_OUTPUT')
        channel='log'if path.name in('control.jsonl','operations.jsonl','trial_states.bin')else'result'
        physical[channel]+=path.stat().st_size
    budgets={}
    for channel in physical:
        reserved=r['output']['reserved'].get(channel,0);ack=r['output']['observed'].get(channel,0)
        require(0<=ack<=reserved<=r['output']['limits'][channel] and physical[channel]<=reserved,'BYTE_CAP')
        require(r['output']['unknown_credit'][channel]==reserved-ack,'UNKNOWN_CREDIT_NO_REFUND')
        budgets[channel]=dict(reserved=reserved,ack_observed=ack,physical_retained=physical[channel],
            unknown_credit=reserved-ack,refunded=False)
    snapshot=G.state_counts(output/'trial_states.bin',tasks)
    if ev['statefile_complete']:
        require(snapshot['counts']==r['states'],'GLOBAL_STATE_COUNTER_BINDING')
        calls=B.accounting_v3(snapshot,r,planned)
        require(snapshot['file_sha256']==ev['shadow_sha256'],'STATE_RAM_SHADOW_BINDING')
    else:
        require(ev['state_errors'] and r['complete']is False,'UNDECLARED_STATE_EVIDENCE_FAILURE')
        calls=dict(reserved=r['reserved'],started_observed=r['started_observed'],
            completed_producer_claim=r['states']['COMPLETED'],statefile_counts=snapshot['counts'],
            independent_state_equality_verified=False)
    path=output/'operations.jsonl';require(V.file_digest(path)==r['operation_trace_sha256'],'TRACE_PHYSICAL_SHA')
    prefix=ev['trace_prefix'];require(prefix and prefix['bytes']<=path.stat().st_size,'TRACE_PREFIX_LENGTH')
    require(V.file_digest(path,prefix['bytes'])==prefix['sha256'],'TRUSTED_PREFIX_SHA')
    with tempfile.TemporaryDirectory(prefix='b3-storage-prefix-')as d:
        target=Path(d)/'trusted.jsonl'
        with path.open('rb')as src,target.open('xb')as dest:
            remaining=prefix['bytes']
            while remaining:
                b=src.read(min(65536,remaining));require(b,'PREFIX_TRUNCATED');dest.write(b);remaining-=len(b)
        if ev['statefile_complete']:
            reader=T.Decisions(target,tasks,V.identity(inner))
            history=G.check_operations(iter(reader),tasks,V.identity(inner))
            stats=reader.stats
        else:
            stats=chain_only(target,V.identity(inner))
            history=dict(status='STATE_PERSISTENCE_INTERRUPTED_TRANSACTION_NOT_VERIFIED',outstanding=None)
        require(stats['rows']==prefix['rows']and stats['chain']==prefix['chain'],'PREFIX_CHAIN_ROWS')
    complete_evidence=ev['trace_complete']and ev['statefile_complete']and ev['journal_complete']and not ev['storage_errors']
    # A state persistence failure can interrupt an ENTER transaction after its
    # evidence row but before permission. Such a trace is never complete proof.
    if complete_evidence:
        require(prefix['bytes']==path.stat().st_size,'UNDECLARED_TRACE_TAIL')
        require(not history['outstanding'],'INCOMPLETE_OPERATION_HISTORY')
        require(history['counts'].get('reserved',0)==r['reserved']and
            history['counts'].get('started_observed',0)==r['started_observed']and
            history['counts'].get('completed',0)==r['states']['COMPLETED'],'TRACE_COUNTER_BINDING')
    else:require(r['complete']is False,'FALSE_SUCCESS_WITH_EVIDENCE_LOSS')
    pairs=[]
    for row in r['pairs']:
        t=row['task'];file=output/(t['key']+'.jsonl')
        if not file.exists():continue
        checked=W.verify_worker(file,t,row)
        require(checked['status']not in('INTEGRITY_MISMATCH','UNSUPPORTED_SCHEMA'),'PAIR_F01_INTEGRITY')
        committed=checked['engine_committed_trials']
        if ev['statefile_complete'] and committed is not None:
            require(snapshot['pairs'][t['key']]['COMPLETED']<=committed,'UNPROVEN_ENGINE_COMPLETION')
        pairs.append(checked)
    control=None
    if complete_evidence:
        control=E.control(output/'control.jsonl',tasks,r)
    if r['complete']:
        require(complete_evidence and not history['stopped']and len(pairs)==24 and
            r['states']['COMPLETED']==planned and all(x['status']=='VERIFIED_COMPLETE'for x in pairs),
            'FALSE_COMPLETE')
        receipt=V.plain((output/'pilot_receipt.json').read_text());summary=V.Reader(output/'pilot_summary.jsonl')
        docs=list(summary);require(len(docs)==1 and not summary.stats['incomplete_tail_bytes'],'SUMMARY')
        require(receipt['schema']=='b3_pilot_storage_receipt_v2'and
            receipt['summary_sha256']==summary.stats['file_sha256']and
            receipt['summary_chain']==summary.stats['chain_sha256'],'SUMMARY_RECEIPT')
        require(control['pilot_receipt']==receipt,'RECEIPT_JOURNAL')
        for obj in(receipt,docs[0].payload):
            require(obj['pilot_binding']==V.identity(c)and obj['inner_binding']==V.identity(inner),'RECEIPT_BINDING')
        for t in tasks:
            file=output/(t['key']+'.jsonl')
            for axis in('X','Y'):
                if t['layer']=='B':
                    require(V.cross_pair(output/(t['key'][:-1]+'A.jsonl'),file,axis)['status']=='PASS','AB_BYTES')
                if t['phase_index']:
                    require(V.cross_pair(output/f"correctness_baseline_{t['arm']}_{t['layer']}.jsonl",file,axis,prefix_only=True)['status']=='PASS','PREFIX_BYTES')
                if t['phase_index']==2:
                    require(V.cross_pair(output/f"profile_baseline_{t['arm']}_{t['layer']}.jsonl",file,axis)['status']=='PASS','PASS_BYTES')
        status='VERIFIED_ARTIFICIAL_COMPLETE'
    else:status='TECHNICAL_FAILURE'if complete_evidence else'INCOMPLETE_EVIDENCE'
    h=r['host_outer']
    require(all(type(h[k])is int and h[k]>=0 for k in('parent_cpu_ns','worker_cpu_ns','wall_ns')),'COST_BOUNDARY')
    return dict(schema='b3_storage_supplemental_verification_v2',status=status,calls=calls,
        history=history,physical=physical,budgets=budgets,complete_evidence=complete_evidence,
        state_counter_relation_verified=ev['statefile_complete'],
        total_cpu_ns=h['parent_cpu_ns']+h['worker_cpu_ns'],global_wall_ns=h['wall_ns'],
        comparison_eligible=r['complete'],fixed_checker_SOURCE_modified=False,
        fixed_full_adapter_supports_v2=False,local_Codex_final_acceptance_pending=True,
        author_run_supplemental_check=True,scientific_calls=0,mandatory_STOP=True)
