"""Independent V1 artificial pilot evidence checker; no producer imports."""
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from . import g6_w2c_b3_offline_result as V
from . import g6_w2c_b3_parallel_verifier_core as G
from . import g6_w2c_b3_parallel_offline_adapter as W


def operations(path):
    previous='0'*64
    with Path(path).open('rb') as f:
        for line in f:
            V.require(line.endswith(b'\n'),'TRACE_PARTIAL_LINE')
            e=V.plain(line.decode());raw=e['payload'].encode()
            V.require(e['previous']==previous,'TRACE_PREVIOUS')
            previous=sha256(previous.encode()+raw).hexdigest()
            V.require(e['sha256']==previous,'TRACE_CHAIN')
            r=V.plain(raw.decode())
            V.require(json.dumps(r,separators=(',',':'),ensure_ascii=True).encode()==raw,'TRACE_CANONICAL')
            yield r


def cost(report):
    h=report['host_outer']
    for k in ('parent_cpu_ns','worker_cpu_ns','wall_ns'):V.nonnegative(h[k])
    # Other timing partitions are diagnostics contained within these windows.
    if 'total_cpu_ns' in h:
        V.require(h['total_cpu_ns']==h['parent_cpu_ns']+h['worker_cpu_ns'],'CPU_DOUBLE_COUNT')
    return dict(total_cpu_ns=h['parent_cpu_ns']+h['worker_cpu_ns'],global_wall_ns=h['wall_ns'],
                pair_cpu_added=False,IPC_added=False,fine_added=False,
                machine_independent_arm_cost=False,failed_prefix_comparison_eligible=report['complete'])


def admission_snapshot(path,inner,*,expected_status):
    """Read a retained FAKE store only. Snapshot is not a race/transition log."""
    db=sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)
    try:
        V.require(db.execute('PRAGMA quick_check').fetchall()==[('ok',)],'FAKE_STORE_INTEGRITY')
        rows=db.execute('SELECT scope,claim_id,contract_digest,binding,status FROM claims').fetchall()
        V.require(len(rows)==1,'FAKE_SINGLE_SPENT_SCOPE')
        scope,claim,digest,binding,status=rows[0]
        b=V.plain(binding)
        V.require(scope==inner['scope'] and scope.startswith('ARTIFICIAL_') and claim==inner['claim_id'] and
            digest==V.identity(inner) and b['formal_digest']==digest and b['scope']==scope and
            b['claim_id']==claim and status==expected_status,'FAKE_STORE_BINDING')
        return dict(status='VERIFIED_FAKE_SPENT_SNAPSHOT',scope=scope,claim_status=status,
                    snapshot_proves_atomic_transition_history=False,real_approval_verified=False)
    finally:db.close()


def verify(root,c,inner,report,reference_root=None):
    root=Path(root).resolve();output=V.safe_path(root,inner['candidate']['output_relative'])
    V.require(c['schema']=='b3_limited_pilot_preparation_v1' and c['contract_grants_execution']is False and
        c['science_release']is None and c['artificial_only']is True,'PILOT_CLOSED_CONTRACT')
    V.require(V.identity(c)==inner['pilot_candidate_binding']==report['pilot_binding'],'PILOT_OUTER_BINDING')
    p=c['proposal'];n=p['profile_n']
    V.require(type(n)is int and n in (64,256,1024) and p['total_calls']==512+64*n and
        p['correctness_calls']==512 and p['online_calls']==64*n and p['modes']==['S','P1'] and
        c['retry']==0 and type(c['retry'])is int and c['resume']is False,'PILOT_SCOPE_ACCOUNTING')
    V.require(c['fixed']['arms']==list(V.ARMS) and c['fixed']['p']==['1/5','3/10','1/2'] and
        c['fixed']['x']=='5/7' and c['fixed']['m']==5 and c['fixed']['sigma']==1,'PILOT_SCIENCE_CONDITIONS')
    if reference_root is not None:
        base=Path(reference_root).resolve()
        for path,h in {**c['source_pins'],**c['input_pins']}.items():
            V.require(V.file_digest(V.safe_path(base,path))==h,'PILOT_FIXED_BYTES')
    V.require(report['operation_trace_sha256']==V.file_digest(output/'operations.jsonl'),'TRACE_FILE')
    tasks,planned=G.plan(inner['settings']['profile_n'])
    V.require(planned<=p['total_calls'],'PILOT_PLANNED_CALL_CAP')
    snapshot=G.state_counts(output/'trial_states.bin',tasks)
    V.require(snapshot['counts']==report['states'],'PILOT_FINAL_STATEFILE_BINDING')
    V.require(planned-snapshot['counts']['NOT_STARTED']==report['reserved'],'PILOT_STATE_RESERVED')
    history=G.check_operations(operations(output/'operations.jsonl'),tasks,V.identity(inner))
    V.require(history['counts'].get('reserved',0)<=min(planned,inner['settings']['call_limit']),'PILOT_CALL_CAP')
    for channel,limit in (('result',inner['settings']['result_bytes']),('log',inner['settings']['log_bytes'])):
        V.require(history['reserved_bytes'].get(channel,0)<=limit,'PILOT_BYTE_CAP')
    V.require(not history['outstanding'],'UNFINISHED_DECISION_LOG')
    retained={'log':0,'result':0}
    for path in output.iterdir():
        V.require(path.is_file() and not path.is_symlink(),'PILOT_OUTPUT_REGULAR_FILES')
        channel='log'if path.name in ('control.jsonl','trial_states.bin','operations.jsonl')else'result'
        retained[channel]+=path.stat().st_size
    for channel in retained:
        b=report['output']
        V.require(retained[channel]<=b['reserved'].get(channel,0)<=b['limits'][channel] and
            b['observed'].get(channel,0)<=b['reserved'].get(channel,0),'PILOT_RETAINED_BYTE_BUDGET')
    V.require(history['counts'].get('reserved',0)==report['reserved'] and
        history['counts'].get('completed',0)==report['states']['COMPLETED'] and
        history['counts'].get('started_observed',0)==report['started_observed'],'HISTORY_REPORT_BINDING')
    pairs=[]
    for row in report['pairs']:
        task=row['task'];path=output/(task['key']+'.jsonl')
        if path.exists():
            r=W.verify_worker(path,task,row)
            V.require(r['status']!='INTEGRITY_MISMATCH','PAIR_RAW_INTEGRITY:'+str(r.get('issues')))
            pairs.append(r)
    if report['complete']:
        V.require(not history['stopped'] and history['counts'].get('completed')==planned,'PILOT_NORMAL_HISTORY')
        V.require(len(pairs)==24 and all(p['status']=='VERIFIED_COMPLETE'for p in pairs),'PILOT_NORMAL_PAIRS')
        receipt=V.plain((output/'pilot_receipt.json').read_text())
        summary=V.Reader(output/'pilot_summary.jsonl');rows=list(summary)
        V.require(len(rows)==1 and not summary.stats['incomplete_tail_bytes'],'PILOT_SUMMARY')
        V.require(receipt['summary_sha256']==summary.stats['file_sha256'] and
            receipt['summary_chain']==summary.stats['chain_sha256'],'PILOT_RECEIPT_SUMMARY')
        prefix=receipt['trace_prefix']
        V.require(V.file_digest(output/'operations.jsonl',prefix['bytes'])==prefix['sha256'],'TRACE_PREFIX_RECEIPT')
        V.require(rows[0].payload['pilot_binding']==V.identity(c) and receipt['pilot_binding']==V.identity(c) and
            rows[0].payload['inner_binding']==V.identity(inner) and receipt['inner_binding']==V.identity(inner),
            'SUMMARY_RECEIPT_BINDING')
        for t in tasks:
            if t['layer']=='B':
                a=next(x for x in tasks if x['phase_index']==t['phase_index'] and x['arm']==t['arm'] and x['layer']=='A')
                for axis in ('X','Y'):
                    V.require(V.cross_pair(output/(a['key']+'.jsonl'),output/(t['key']+'.jsonl'),axis)['status']=='PASS',
                        'PILOT_AB_EVENT_IDENTITY')
            if t['phase_index']>0:
                correctness=next(x for x in tasks if x['phase_index']==0 and x['arm']==t['arm']and x['layer']==t['layer'])
                for axis in ('X','Y'):
                    V.require(V.cross_pair(output/(correctness['key']+'.jsonl'),output/(t['key']+'.jsonl'),axis,
                        prefix_only=True)['status']=='PASS','PILOT_CORRECTNESS_PREFIX')
            if t['phase_index']==2:
                baseline=next(x for x in tasks if x['phase_index']==1 and x['arm']==t['arm']and x['layer']==t['layer'])
                for axis in ('X','Y'):
                    V.require(V.cross_pair(output/(baseline['key']+'.jsonl'),output/(t['key']+'.jsonl'),axis)['status']=='PASS',
                        'PILOT_PASS_EVENT_IDENTITY')
    return dict(status='VERIFIED_ARTIFICIAL_COMPLETE'if report['complete']else'VERIFIED_ARTIFICIAL_STOP',
        scope='NEW_ARTIFICIAL_EVIDENCE_ONLY',history=history,cost=cost(report),pairs=pairs,
        statefile=snapshot,physical_retained_bytes=retained,
        historical_P07_P08_P11_resolved=False,real_approval_verified=False,mandatory_STOP=True)
