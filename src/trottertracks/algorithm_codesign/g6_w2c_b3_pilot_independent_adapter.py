"""Independent fixed pilot artificial adapter; normal path imports consumers only."""
from pathlib import Path
import json,tempfile,sqlite3,tarfile
from . import g6_w2c_b3_offline_result as V
from . import g6_w2c_b3_parallel_verifier_core as G
from . import g6_w2c_b3_parallel_offline_adapter as A
from . import g6_w2c_b3_v3_offline_adapter as B
from . import g6_w2c_b3_v3_offline_identity as I
from . import g6_w2c_b3_v3_offline_cost as C
from . import g6_w2c_b3_pilot_independent_identity as P
from .g6_w2c_b3_pilot_independent_trace import Decisions


def control(path,tasks,report):
    receipt=None
    with tempfile.TemporaryDirectory(prefix='pilot-control-independent-') as tmp:
        target=Path(tmp)/'control.jsonl'
        with target.open('x') as f:
            for i,(r,_) in enumerate(A.records(path)):
                if i==0:
                    V.require(r.get('schema')=='b3_parallel_control_v3','PILOT_PLAN_SCHEMA')
                    r=dict(r,schema='b3_parallel_artificial_v1')
                if r.get('event')=='PILOT_COMPLETION_RECEIPT':
                    V.require(receipt is None,'DUPLICATE_PILOT_RECEIPT');receipt={k:v for k,v in r.items() if k!='event'}
                    continue
                V.require(receipt is None,'RECEIPT_NOT_FINAL')
                f.write(json.dumps(r,default=lambda x:{'exact':V.canonical(x)},separators=(',',':'))+'\n')
        c=A.validate_control(target,tasks,report)
    c['pilot_receipt']=receipt;return c


def verify_output(output,candidate,inner,report,reference):
    output=Path(output);identity=reference.inner(candidate,inner)
    V.require(report.get('schema')=='b3_parallel_result_v3','PILOT_RESULT_SCHEMA')
    reference.legacy.validate('result',report)
    for k,v in dict(retry=0,resume=False,mandatory_STOP=True,artificial_only=True).items():
        V.require(I.strict_equal(report[k],v),'REPORT_CLOSED_FLAG:'+k)
    V.require(report['pilot_binding']==identity['pilot_digest'],'REPORT_PILOT_BINDING')
    V.require(report['settings']==inner['settings'],'INNER_SETTINGS')
    tasks,planned=G.plan(report['settings']['profile_n']);lookup={t['key']:t for t in tasks}
    V.require(planned<=candidate['proposal']['total_calls'],'PILOT_GLOBAL_BUDGET')
    journal=control(output/'control.jsonl',tasks,report)
    snapshot=G.state_counts(output/'trial_states.bin',tasks);calls=B.accounting_v3(snapshot,report,planned)
    V.require(report['contract_order']==list(lookup) and set(report['all_pair_states'])==set(lookup),'PAIR_COVERAGE_ORDER')
    keys=[r['task']['key'] for r in report['pairs']]
    V.require(len(set(keys))==len(keys) and keys==[t['key'] for t in tasks if t['key'] in keys],'PAIR_DUPLICATE_ORDER')
    rows={r['task']['key']:r for r in report['pairs']};pairs=[]
    allowed={t['key']+'.jsonl' for t in tasks}|{'control.jsonl','operations.jsonl','trial_states.bin','pilot_summary.jsonl','pilot_receipt.json'}
    inodes=set();physical=dict(log=0,result=0)
    for path in output.iterdir():
        V.require(path.is_file() and not path.is_symlink() and path.name in allowed,'OUTPUT_NAMESPACE_COLLISION')
        ino=(path.stat().st_dev,path.stat().st_ino);V.require(ino not in inodes,'OUTPUT_INODE_COLLISION');inodes.add(ino)
        channel='log' if path.name in ('control.jsonl','operations.jsonl','trial_states.bin') else 'result';physical[channel]+=path.stat().st_size
    for t in tasks:
        row=rows.get(t['key']);path=output/(t['key']+'.jsonl');pc=snapshot['pairs'][t['key']]
        if row:
            V.require(I.strict_equal(row['task'],t) and report['all_pair_states'][t['key']]==row['status'],'PAIR_TASK_OFFSET_STATE')
            f=(row.get('failure') or {}).get('trial_failure_provenance')
            if f is not None:reference.legacy.validate('f01',f)
        else:V.require(report['all_pair_states'][t['key']]=='NOT_STARTED','UNDISPATCHED_PAIR')
        if not path.exists():V.require(not pc['COMPLETED'],'COMPLETED_WITHOUT_PAIR');continue
        p=A.verify_worker(path,t,row)
        V.require(p['status'] not in ('INTEGRITY_MISMATCH','UNSUPPORTED_SCHEMA'),'PAIR_INTEGRITY:'+','.join(p['issues']))
        V.require(pc['COMPLETED']<=p['visible_trial_rows'],'STATE_EXCEEDS_PHYSICAL_ROWS')
        if p['engine_committed_trials'] is not None:V.require(pc['COMPLETED']<=p['engine_committed_trials'],'STATE_EXCEEDS_EMIT_COMMIT')
        p.update(task_key=t['key'],global_logical_completed=pc['COMPLETED']);pairs.append(p)
    receipt=None;summary=None;prefix=None
    if (output/'pilot_receipt.json').exists():
        receipt=V.plain((output/'pilot_receipt.json').read_text());prefix=receipt['trace_prefix']
        V.require(receipt==journal['pilot_receipt'],'RECEIPT_JOURNAL_BINDING')
        V.require(receipt.get('schema')=='b3_pilot_artificial_receipt_v1' and receipt.get('artificial_only') is True and
                  receipt.get('production_permission') is False,'PILOT_RECEIPT_FLAGS')
    else:V.require(journal['pilot_receipt'] is None,'MISSING_RECEIPT_FILE')
    decisions=Decisions(output/'operations.jsonl',tasks,identity['inner_digest'],output/'trial_states.bin',prefix)
    history=G.check_operations(iter(decisions),tasks,identity['inner_digest'])
    V.require(report['operation_trace_sha256']==decisions.stats['sha256'],'TRACE_FILE_SHA')
    V.require(history['counts'].get('reserved',0)==calls['reserved'] and
              history['counts'].get('started_observed',0)==calls['started_observed'] and
              history['counts'].get('completed',0)==calls['completed'],'TRACE_COUNTER_BINDING')
    V.require(not history['outstanding'],'UNFINISHED_TRACE_HISTORY')
    for state,n in decisions.witness_counts.items():V.require(n==snapshot['counts'][state],'TRACE_STATE_COUNT')
    output_budget={}
    for ch in ('result','log'):
        b=report['output'];reserved=V.nonnegative(b['reserved'].get(ch,0));ack=V.nonnegative(b['observed'].get(ch,0));limit=b['limits'][ch]
        V.require(limit==inner['settings'][ch+'_bytes'] and limit<=G.BUDGET[ch+'_bytes'] and
                  ack<=physical[ch]<=reserved<=limit,'GLOBAL_PHYSICAL_ACK_BUDGET:'+ch)
        trace_self=decisions.stats['bytes'] if ch=='log' else 0
        V.require(history['reserved_bytes'].get(ch,0)+trace_self==reserved,'TRACE_RESERVATION_ACCOUNTING:'+ch)
        V.require(history['ack_observed_bytes'].get(ch,0)+trace_self==ack,'TRACE_ACK_ACCOUNTING:'+ch)
        output_budget[ch]=dict(reserved=reserved,ack_observed=ack,physical_bytes=physical[ch],
                               unknown_credit=reserved-ack,refunded=False,trace_self_bytes=trace_self)
    if (output/'pilot_summary.jsonl').exists():
        reader=V.Reader(output/'pilot_summary.jsonl');docs=[]
        for x in reader:V.require(not docs,'MULTIPLE_SUMMARIES');docs.append(x.payload)
        V.require(docs and not reader.stats['incomplete_tail_bytes'],'PARTIAL_SUMMARY');summary=docs[0]
        V.require(summary['schema']=='b3_pilot_artificial_summary_v1' and receipt is not None,'SUMMARY_RECEIPT_REQUIRED')
        V.require(receipt['summary_sha256']==reader.stats['file_sha256'] and receipt['summary_chain']==reader.stats['chain_sha256'],'SUMMARY_RECEIPT_SHA')
        for doc in (receipt,summary):
            V.require(doc['pilot_binding']==identity['pilot_digest'] and doc['inner_binding']==identity['inner_digest'],'SUMMARY_BINDING')
        V.require(summary['trace']==prefix,'SUMMARY_TRACE_PREFIX')
        for k in ('schema','settings','planned','reserved','started_observed','states','pairs','failures','contract_order','completion_order','complete','parent_cpu_ns','worker_cpu_ns','total_cpu_ns','wall_ns'):
            V.require(I.strict_equal(summary['result'][k],report[k]),'SUMMARY_HOST_SNAPSHOT:'+k)
        V.require(summary['result']['output']['reserved']['log']<=report['output']['reserved']['log'],'SUMMARY_PREFINAL_BOUNDARY')
    complete=report['complete']
    if complete:
        V.require(calls['completed']==planned and calls['reserved']==planned and len(pairs)==24 and
                  all(r['status']=='OK' for r in report['pairs']) and not report['failures'] and
                  not history['stopped'],'FALSE_PILOT_COMPLETE')
        V.require(all(v['reserved']==v['ack_observed']==v['physical_bytes'] for v in output_budget.values()),'COMPLETE_WITH_UNKNOWN_CREDIT')
    V.require(report['representative_failure']==(report['failures'][0] if report['failures'] else None),'FIRST_FAILURE')
    for i,f in enumerate(report['failures']):V.require(f['parent_arrival']==i+1,'FAILURE_ARRIVAL')
    cap=1 if inner['settings']['mode']=='S' else(inner['settings']['inflight_cap'] if inner['settings']['mode']=='P2' else inner['settings']['workers'])
    V.require(0<=report['max_inflight']<=cap,'MAX_INFLIGHT')
    status='VERIFIED_COMPLETE' if complete and summary is not None and receipt is not None else ('INSUFFICIENT_EVIDENCE' if complete else 'TECHNICAL_FAILURE')
    if report['fake_claim_status']!='STOPPED' or report.get('log_errors') or report.get('secondary_stop_error'):status='TECHNICAL_FAILURE'
    if status=='VERIFIED_COMPLETE':
        for t in tasks:
            p=output/(t['key']+'.jsonl')
            for axis in ('X','Y'):
                if t['layer']=='B':
                    other=output/(t['key'][:-1]+'A.jsonl');V.require(V.cross_pair(other,p,axis)['status']=='PASS','AB_TRACE')
                if t['phase_index']:
                    other=output/f"correctness_baseline_{t['arm']}_{t['layer']}.jsonl"
                    V.require(V.cross_pair(other,p,axis,prefix_only=True)['status']=='PASS','CORRECTNESS_PREFIX')
                if t['phase_index']==2:
                    other=output/f"profile_baseline_{t['arm']}_{t['layer']}.jsonl"
                    V.require(V.cross_pair(other,p,axis)['status']=='PASS','PASS_TRACE')
    r=dict(schema='b3_limited_pilot_independent_result_v1',status=status,issues=[],identity=identity,
           report=report,pairs=pairs,calls=calls,state=snapshot,history=history,trace=decisions.stats,
           output_budget=output_budget,physical_bytes=physical,state_retained_bytes=planned,
           state_pwrite_traffic_bytes=report.get('state_update_io_bytes'),
           control=dict(started_workers=len(journal['started']),ended_workers=len(journal['ended']),
                        evidence_gaps=journal['evidence_gaps'],receipt_present=receipt is not None),
           historical_pending_reclassified=False,scientifically_adopted=False,mandatory_STOP=True)
    r['cost']=C.aggregate(r)
    if r['cost']['status']=='COST_INTEGRITY_MISMATCH':
        r['cost']['comparison_eligible']=False
        r['status']='INTEGRITY_MISMATCH';r['issues'].extend('COST:'+str(x) for x in r['cost'].get('issues',[]))
    return r


def verify(case_dir,reference):
    root=Path(case_dir).resolve();r=dict(schema='b3_limited_pilot_independent_result_v1',case=root.name,
        status='INSUFFICIENT_EVIDENCE',issues=[],historical_pending_reclassified=False,mandatory_STOP=True)
    try:
        c=V.plain((root/'pilot_contract.json').read_text());inner=V.plain((root/'inner_contract.json').read_text())
        identity=reference.inner(c,inner);snapshot=P.admission_snapshot(root/'FAKE_ONLY_claims.sqlite3',inner)
        obs=A.safe_values(V.plain((root/'observations.json').read_text()));report=obs.get('report')
        if report is None:
            if obs.get('status') is not None:V.require(obs['status']==snapshot['claim_status'],'OBSERVATION_LEDGER_STATUS')
            if root.name=='parent_crash':
                for k in ('host_total_cpu_ns','worker_cpu_ns'):V.require(obs[k] is None,'UNKNOWN_CRASH_COST_NOT_NULL')
            if root.name=='race':
                V.require(obs['contenders']==8 and len(obs['outcomes'])==8 and obs['outcomes'].count('ADMITTED')==1 and obs['outcomes'].count('DENIED')==7 and
                          obs['store_id']==snapshot['store_id'],'RACE_OBSERVATION_BINDING')
            r.update(admission=snapshot,identity=identity,observations=obs,
                     reason='FAKE snapshot verified; original approval/ordered race or crash transition history not retained',
                     atomic_transition_history_verified=False,cost=dict(total_cpu_ns=None,global_wall_ns=None,comparison_eligible=False))
            if (root/'artificial_raw.tar.gz').exists():
                with tempfile.TemporaryDirectory(prefix='pilot-independent-empty-raw-') as temp:
                    out=B.extract_archive(root/'artificial_raw.tar.gz',Path(temp)/'raw')
                    V.require(not any(p.stat().st_size for p in out.iterdir()),'UNEXPECTED_EARLY_FAILURE_BYTES')
                r.update(status='TECHNICAL_FAILURE',reason='Early log-budget failure; no state/trace/report retained, not a completed or fully observed history')
            return r
        with tempfile.TemporaryDirectory(prefix='pilot-independent-raw-') as temp:
            out=B.extract_archive(root/'artificial_raw.tar.gz',Path(temp)/'raw')
            r=verify_output(out,c,inner,report,reference)
        V.require(snapshot['claim_status']==report['fake_claim_status'],'REPORT_FAKE_CLAIM_SNAPSHOT')
        r.update(case=root.name,admission=snapshot,raw_archive_sha256=V.file_digest(root/'artificial_raw.tar.gz'))
    except V.Unsupported as e:r.update(status='UNSUPPORTED_SCHEMA',issues=[str(e)])
    except (V.Invalid,KeyError,IndexError,StopIteration,EOFError,ValueError,TypeError,OSError,sqlite3.Error,tarfile.TarError) as e:
        r.update(status='INSUFFICIENT_EVIDENCE' if isinstance(e,FileNotFoundError) else 'INTEGRITY_MISMATCH',issues=[str(e)])
    r.setdefault('case',root.name)
    return r
