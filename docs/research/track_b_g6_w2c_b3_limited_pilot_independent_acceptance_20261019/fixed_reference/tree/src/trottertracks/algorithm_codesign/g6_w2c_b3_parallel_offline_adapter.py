"""Versioned read-only adapter for the fixed published ARTIFICIAL P1 V1.

Never imports a producer, scheduler, authority or scientific module. The source
schema and final-state limitations are explicit; future integrated schema pending.
"""
from collections import Counter
from pathlib import Path
import json
import tempfile
from . import g6_w2c_b3_offline_result as V
from . import g6_w2c_b3_parallel_verifier_core as G

PUBLISHED='1cd2a7196bc48a4aa1a904d7bd351e3d8fbf45ff'
SCHEMA='R1_PARALLEL_8PAIR_ARTIFICIAL_RESULT_V1'


def safe_values(v):
    """Published safe_json's Fraction wrapper; other strings stay opaque."""
    if type(v)is dict:
        if set(v)=={'exact'}and type(v['exact'])is str:return V.decode(v['exact'])
        return{k:safe_values(x)for k,x in v.items()}
    if type(v)is list:return[safe_values(x)for x in v]
    return v


def records(path):
    with path.open('rb')as f:
        for line in f:
            V.require(line.endswith(b'\n'),'PARTIAL_CONTROL_LINE')
            p=V.plain(line);V.require(type(p)is dict,'CONTROL_OBJECT');yield safe_values(p),len(line)


def pair_summary(row):
    outcome=row.get('outcome')or(row.get('failure')or{}).get('committed_engine_summary_before_cleanup_failure')
    if outcome is None:return None
    return dict(outcome,**{k:row['task'][k]for k in('phase','pass_name','arm','layer')},
                output_rows=row['output_rows'],output_bytes=row['output_bytes'],
                output_chain_sha256=row['chain'],file_sha256=row['file_sha'])


def verify_worker(path,task,row):
    spec=dict(task,N=task['n'])
    f=(row.get('failure')or{}).get('trial_failure_provenance')if row else None
    summary=pair_summary(row)if row else None
    if f is not None:
        # Bridge only the old API's filename argument; private temp, no input writes.
        with tempfile.TemporaryDirectory(prefix='b3-offline-provenance-')as d:
            control=Path(d)/'f01.json';control.write_text(json.dumps(dict(kind='TYPED_FAILURE',trial_failure_provenance=V.canonical(f))))
            r=V.verify_pair(path,spec,summary=summary,failure_path=control)
    else:r=V.verify_pair(path,spec,summary=summary)
    if row and row.get('file_sha')is not None and'file'in r:
        for field,k in(('file_sha','file_sha256'),('chain','chain_sha256'),('output_bytes','bytes'),('output_rows','rows')):
            V.require(row[field]==r['file'][k],'WORKER_RAW_'+field.upper())
    engine_known=f is not None or summary is not None
    r['engine_committed_trials']=r.get('committed_trials')if engine_known else None
    if not engine_known:
        r['visible_trial_prefix']=r.pop('committed_trials',0)
        r['evidence']['engine_commit']=V.UNKNOWN
    if f is not None or(row and row['status']!='OK'):
        if r['status']not in('INTEGRITY_MISMATCH','UNSUPPORTED_SCHEMA'):r['status']='TECHNICAL_FAILURE'
    return r


def validate_control(path,tasks,report):
    lookup={t['key']:t for t in tasks};started={};ended={};completion=[];failures=[]
    phase=-1;open_phase=False;receipt=None;plan=None;bytes_before_receipt=None;total_bytes=0;stop_saved=False
    phase_stopped=False;gaps=[]
    incomplete_control=report is not None and(bool(report.get('log_errors'))or report['settings'].get('parent_fault')=='dispatch')
    for p,size in records(path):
        event=p.get('event')
        if event=='PLAN':
            V.require(plan is None and total_bytes==0 and p.get('schema')=='b3_parallel_artificial_v1','CONTROL_PLAN_SCHEMA')
            V.require(p['tasks']==tasks and p['planned']==sum(2*t['n']for t in tasks)and p['state_codes']=={s:i for i,s in enumerate(G.STATES)}and p['retry']==0 and p['resume']is False,'CONTROL_PLAN_BINDING');plan=p
        elif event=='PHASE_BEGIN':
            V.require(not open_phase and not phase_stopped and p['phase']==phase+1,'CONTROL_PHASE_BARRIER')
            if phase>=0:V.require(all(t['key']in ended and ended[t['key']]['status']=='OK'for t in tasks if t['phase_index']==phase),'NEXT_PHASE_AFTER_FAILURE_OR_MISSING_PAIR')
            phase=p['phase'];open_phase=True
        elif event=='WORKER_START':
            key=p['key'];V.require(key in lookup and key not in started and open_phase and lookup[key]['phase_index']==phase,'WORKER_START_IDENTITY_OR_DUPLICATE');started[key]=p['pid']
        elif event=='WORKER_END':
            row=p['report'];key=row['task']['key']
            V.require(key in lookup and key not in ended and row['task']==lookup[key],'WORKER_END_IDENTITY_OR_DUPLICATE')
            if key not in started:
                V.require(incomplete_control,'WORKER_END_WITHOUT_START');gaps.append('Missing WORKER_START:'+key)
            else:V.require(row['pid']==started[key],'WORKER_PID_BINDING')
            ended[key]=row;completion.append(key)
        elif event=='PHASE_END':
            V.require(open_phase and p['phase']==phase,'CONTROL_PHASE_END');open_phase=False;phase_stopped=p['stopped']
        elif event=='FAILURE_OBSERVED':
            r={k:v for k,v in p.items()if k!='event'}
            V.require(r['parent_arrival']==len(failures)+1,'FAILURE_ARRIVAL_ORDER');failures.append(r)
        elif event=='COMPLETION_RECEIPT':
            V.require(receipt is None and not failures and not open_phase and phase==2 and len(ended)==24 and all(x['status']=='OK'for x in ended.values()),'RECEIPT_BEFORE_ALL_WORKERS')
            receipt=p;bytes_before_receipt=total_bytes
        elif event=='MANDATORY_STOP':
            V.require(p['complete']is False and receipt is None,'FAILURE_RECEIPT_CONFLICT');stop_saved=True
        else:raise V.Unsupported('CONTROL_EVENT_SCHEMA:'+str(event))
        total_bytes+=size
    V.require(plan is not None,'MISSING_PLAN')
    if report is not None:
        V.require(plan['settings']==report['settings'],'PLAN_SETTINGS')
        if incomplete_control:
            saved=iter(report['completion_order'])
            V.require(all(any(y==x for y in saved)for x in completion),'COMPLETION_ORDER')
            if completion!=report['completion_order']:gaps.append('Some WORKER_END records not persisted')
        else:V.require(completion==report['completion_order'],'COMPLETION_ORDER')
        # Log budget failure can truncate later control rows; never declare them independently verified.
        V.require(failures==report['failures'][:len(failures)],'FAILURE_REPORT_BINDING')
        for key,row in ended.items():
            claim=next((r for r in report['pairs']if r['task']['key']==key),None)
            V.require(claim==row,'WORKER_CONTROL_REPORT_BINDING')
    return dict(plan=plan,started=started,ended=ended,failures=failures,completion_order=completion,
                receipt=receipt,bytes=total_bytes,bytes_before_receipt=bytes_before_receipt,stop_saved=stop_saved,
                trial_operations_logged=False,evidence_gaps=gaps)


def verify(manifest_path):
    manifest_path=Path(manifest_path).resolve();root=manifest_path.parent
    result=dict(schema='b3_parallel_independent_verification_v1',status='INSUFFICIENT_EVIDENCE',pairs=[],checks=[],issues=[],
                artificial_only=True,future_adapter='PENDING_SERVER_SCHEMA',mandatory_STOP=True)
    try:
        m=V.plain(manifest_path.read_text())
        if m.get('schema')!='b3_parallel_artificial_input_v1'or m.get('adapter')!='PUBLIC_P1_ARTIFICIAL_V1':raise V.Unsupported('ADAPTER_SCHEMA')
        V.require(m.get('artificial_only')is True and m.get('producer_commit')==PUBLISHED and type(m['run_id'])is str and m['run_id'],'FIXED_ARTIFICIAL_INPUT_REQUIRED')
        out=V.safe_path(root,m['output']);report=safe_values(V.plain(V.safe_path(root,m['report']).read_text()))if m.get('report')else None
        meta=safe_values(V.plain(V.safe_path(root,m['metadata']).read_text()))if m.get('metadata')else{}
        first=next(records(out/'control.jsonl'))[0];V.require(first.get('event')=='PLAN','FIRST_RECORD_PLAN')
        tasks,planned=G.plan(first['settings']['profile_n'])
        if report is not None and report.get('schema')!=SCHEMA:raise V.Unsupported('PRODUCER_RESULT_SCHEMA')
        if report is not None:
            V.require(report['retry']==0 and report['resume']is False and report['mandatory_STOP']is True and report['artificial_only']is True,'ONE_SHOT_STOP_FLAGS')
            V.require(report['contract_order']==[t['key']for t in tasks],'CONTRACT_ORDER')
            keys=[r['task']['key']for r in report['pairs']];V.require(len(set(keys))==len(keys),'DUPLICATE_PAIR_IDENTITY')
            V.require(keys==[t['key']for t in tasks if t['key']in keys],'PAIR_SUMMARY_ORDER')
            for row in report['pairs']:V.require(row['task']==next(t for t in tasks if t['key']==row['task']['key']),'TASK_OFFSET_OR_IDENTITY')
            V.require(set(report['all_pair_states'])=={t['key']for t in tasks},'ALL_PAIR_STATE_KEYS')
            for t in tasks:
                claim=next((r['status']for r in report['pairs']if r['task']['key']==t['key']),'NOT_STARTED')
                V.require(report['all_pair_states'][t['key']]==claim,'PAIR_STATUS_BINDING')
        control=validate_control(out/'control.jsonl',tasks,report)
        state=G.state_counts(out/'trial_states.bin',tasks);calls=G.accounting(state,report,planned)
        state_sha=(report or meta).get('state_file_sha256')
        if state_sha is not None:V.require(state_sha==state['file_sha256'],'STATEFILE_SHA')
        else:result['checks'].append(G.check('STATE_HASH_CLAIM','Producer external state SHA',None,'PENDING_EVIDENCE',V.RAW,commit=PUBLISHED))
        paths=m.get('pair_paths',{t['key']:t['key']+'.jsonl'for t in tasks})
        V.require(set(paths)=={t['key']for t in tasks},'PAIR_PATH_KEYS')
        resolved=[V.safe_path(out,paths[t['key']])for t in tasks]
        V.require(len(set(resolved))==24,'OUTPUT_PATH_COLLISION')
        inodes=[(p.stat().st_dev,p.stat().st_ino)for p in resolved if p.exists()]
        V.require(len(set(inodes))==len(inodes),'OUTPUT_INODE_COLLISION')
        rows={r['task']['key']:r for r in report['pairs']}if report else control['ended']
        actual_result=0
        for t,p in zip(tasks,resolved):
            row=rows.get(t['key']);pc=state['pairs'][t['key']]
            if p.exists():
                actual_result+=p.stat().st_size;r=verify_worker(p,t,row)
                V.require(r['status']not in('INTEGRITY_MISMATCH','UNSUPPORTED_SCHEMA'),'PAIR_INTEGRITY:'+t['key']+':'+','.join(r['issues']))
                engine=r['engine_committed_trials']
                if engine is not None:V.require(pc['COMPLETED']<=engine,'GLOBAL_COMPLETED_EXCEEDS_F01_COMMIT')
                visible=r.get('visible_trial_rows',0)
                with (out/'trial_states.bin').open('rb')as states:
                    states.seek(t['offset'])
                    i=0
                    for item in V.Reader(p):
                        if'trial_id'in item.payload:
                            code=states.read(1)[0]
                            V.require(code not in(0,6),'ROW_WITHOUT_GLOBAL_RESERVED_CALL');i+=1
                V.require(pc['COMPLETED']<=visible,'COMPLETED_STATE_WITHOUT_RAW_ROW')
                r.update(task_key=t['key'],global_logical_completed=pc['COMPLETED'],global_state_counts=pc)
                result['pairs'].append(r)
            elif pc['COMPLETED']:
                raise FileNotFoundError('Missing completed pair raw:'+t['key'])
        receipt=control['receipt'];summary_path=out/'summary.jsonl';summary=None
        if summary_path.exists():
            sr=V.Reader(summary_path);docs=[]
            for r in sr:V.require(not docs,'SUMMARY_MULTIPLE_ROWS');docs.append(r.payload)
            V.require(len(docs)==1 and not sr.stats['incomplete_tail_bytes'],'SUMMARY_INCOMPLETE');summary=docs[0]
            V.require(summary.get('schema')==SCHEMA,'SUMMARY_SCHEMA')
            if receipt is not None:V.require(receipt['summary_chain']==sr.stats['chain_sha256']and receipt['summary_sha']==sr.stats['file_sha256']and receipt['all_workers_joined']is True,'SUMMARY_RECEIPT_HASH')
            if report is not None:
                for k in('planned','reserved','started_observed','states','pairs','contract_order','completion_order','failures','complete','settings'):V.require(summary[k]==report[k],'SUMMARY_REPORT_'+k)
            actual_result+=summary_path.stat().st_size
            V.require(summary['output']['reserved'].get('result',0)==actual_result-summary_path.stat().st_size,'SUMMARY_PRE_FINAL_RESULT_SCOPE')
            if control['bytes_before_receipt']is not None:V.require(summary['output']['reserved'].get('log',0)==planned+control['bytes_before_receipt'],'SUMMARY_PRE_FINAL_LOG_SCOPE')
        actual_log=planned+control['bytes']
        budget=None
        if report is not None:
            output=report['output'];budget={}
            for channel,actual in(('result',actual_result),('log',actual_log)):
                reserved=V.nonnegative(output['reserved'].get(channel,0));observed=V.nonnegative(output['observed'].get(channel,0));limit=output['limits'][channel]
                V.require(limit==first['settings'][channel+'_bytes']and limit<=G.BUDGET[channel+'_bytes']and 0<=observed<=reserved<=limit,'GLOBAL_BYTE_BUDGET')
                V.require(observed<=actual<=reserved,'OBSERVED_RESERVED_ACTUAL_BYTES')
                budget[channel]=dict(reserved=reserved,ack_observed=observed,actual_file_bytes=actual,unacknowledged_credit=reserved-observed,unacknowledged_credit_refunded=False)
            V.require(report['representative_failure']==(report['failures'][0]if report['failures']else None),'REPRESENTATIVE_FAILURE')
        if report is not None:
            cap=report['settings']['inflight_cap']if report['settings']['mode']=='P2'else report['settings']['workers']
            V.require(0<=report['max_inflight']<=cap,'MAX_INFLIGHT_CAP')
            previous_time=None
            for i,f in enumerate(report['failures']):
                V.require(f['parent_arrival']==i+1,'FAILURE_ARRIVAL_SEQUENCE')
                now=V.nonnegative(f['parent_observed_ns'])
                V.require(previous_time is None or now>=previous_time,'PARENT_MONOTONIC_ARRIVAL');previous_time=now
                snapshot=f.get('inflight_snapshot')
                if snapshot is not None:
                    V.require(i==0 and len(snapshot)<=report['max_inflight'],'FIRST_FAILURE_INFLIGHT_SNAPSHOT')
                    ids=set()
                    for worker,ticket in snapshot.items():
                        t=next(x for x in tasks if x['key']==worker)
                        pieces=ticket['trial_id'].split('/');index=int(pieces[-1]);axis=pieces[-2]
                        V.require(ticket['worker']==worker and ticket['state']in('START_RESERVED','IN_FLIGHT')and axis in('X','Y')and 0<=index<t['n'],'SNAPSHOT_TICKET')
                        V.require(ticket['trial_id']==V.trial(dict(t,N=t['n']),axis,index)and ticket['index']==t['offset']+(0 if axis=='X'else t['n'])+index and ticket['trial_id']not in ids,'SNAPSHOT_GLOBAL_ID');ids.add(ticket['trial_id'])
        result.update(input_origin=m.get('fixture_origin','PUBLISHED_ARTIFICIAL_ARCHIVE'),run_id=m['run_id'],producer_schema=SCHEMA,producer_commit=PUBLISHED,state=state,calls=calls,
                      output_budget=budget,actual_result_bytes=actual_result,actual_log_bytes=actual_log,
                      state_retained_bytes=planned,state_pwrite_traffic_bytes=report.get('state_update_io_bytes')if report else None,
                      control=dict(started_workers=len(control['started']),ended_workers=len(control['ended']),failure_notifications=len(control['failures']),receipt_present=receipt is not None,stop_saved=control['stop_saved'],evidence_gaps=control['evidence_gaps']),
                      report=report,metadata=meta)
        for cid,observed in(('P01',True),('P03',calls),('P04',budget)):
            result['checks'].append(G.check(cid,'Consistent published evidence',observed,'PASS'if observed is not None else'PENDING_EVIDENCE',V.RAW,commit=PUBLISHED))
        result['checks'].append(G.check('P08','No RESERVE/ENTER after parent STOP','Final state/control has no per-operation history','PENDING_EVIDENCE',V.UNKNOWN,'Need ordered parent gate trace; statefile is not transition history.',commit=PUBLISHED))
        for cid,expected,observed,judgment,evidence,unresolved in (
            ('P02','State/F01 logical completion vs in-flight',calls,'PASS',V.RAW,None),
            ('P05','Lost worker/parent conservatively unknown',calls['unknown'],'PASS',V.RAW,'Physical child exit is only a saved producer claim.'),
            ('P06','Phase barrier in control log',control['evidence_gaps'],'PENDING_EVIDENCE'if control['evidence_gaps']else'PASS',V.RAW,None),
            ('P07','Atomic one-shot scope reservation and nonreuse',meta.get('retry_rejected'),'PENDING_EVIDENCE',V.CLAIM,'Flags/state alone do not prove atomic admission. Baseline runtime artificial regressions supplied separately.'),
            ('P10','Global summary/receipt hashes and join declaration',receipt is not None,'PASS'if receipt is not None else'PENDING_EVIDENCE',V.CLAIM,'all_workers_joined is a receipt declaration; control WORKER_END precedes join in producer.'),
            ('P11','Retry0/resume false',dict(retry=report.get('retry'),resume=report.get('resume'))if report else None,'PASS'if report else'PENDING_EVIDENCE',V.CLAIM,'Runtime claim-reuse refusal is separate artificial evidence.')):
            result['checks'].append(G.check(cid,expected,observed,judgment,evidence,unresolved,commit=PUBLISHED))
        if m.get('operations'):
            ops_path=V.safe_path(root,m['operations']);op_report=G.check_operations((p for p,_ in records(ops_path)),tasks,m['run_id'])
            V.require(op_report['counts'].get('reserved',0)==calls['reserved']and op_report['counts'].get('completed',0)==calls['completed'],'OP_GLOBAL_ACCOUNTING')
            result['normalized_operation_evidence']=op_report
            result['checks'].append(G.check('P08_ARTIFICIAL_OPERATIONS','No new permits after STOP',True,'PASS','Separate normalized artificial log; not published P1 control.',commit=None))
        complete=report is not None and report['complete']
        if complete:
            V.require(calls['completed']==planned and calls['reserved']==planned and len(report['pairs'])==24 and all(r['status']=='OK'for r in report['pairs'])and not report['failures'],'FALSE_GLOBAL_COMPLETION')
            ack_complete=budget is not None and all(x['reserved']==x['ack_observed']==x['actual_file_bytes']for x in budget.values())
            result['status']='VERIFIED_COMPLETE'if summary is not None and receipt is not None and ack_complete else'INSUFFICIENT_EVIDENCE'
            if not ack_complete:result['issues'].append('UNACKNOWLEDGED_CREDIT_NOT_COMPLETION_EVIDENCE')
            if result['status']=='VERIFIED_COMPLETE':
                # Cross-pair reuse without building a trial-ID set.
                by={t['key']:p for t,p in zip(tasks,resolved)}
                for arm in V.ARMS:
                    ref=by['correctness_baseline_'+arm+'_A']
                    for phase,pn in G.PHASES:
                        a=by[f'{phase}_{pn}_{arm}_A'];b=by[f'{phase}_{pn}_{arm}_B']
                        for axis in('X','Y'):
                            for x,y,prefix in((a,b,False),(ref,a,True),(ref,b,True),(by['profile_baseline_'+arm+'_A'],a,phase=='correctness')):
                                V.require(V.cross_pair(x,y,axis,prefix_only=prefix)['status']=='PASS','CROSS_PAIR_EVENT_HASH')
        else:result['status']='TECHNICAL_FAILURE'if report is not None or meta.get('case')=='parent_crash'else'VERIFIED_PARTIAL_PREFIX'
        if report is not None and len(control['failures'])<len(report['failures']):result['checks'].append(G.check('CONTROL_FAILURE_COVERAGE','All failure notices persisted',len(control['failures']),'PENDING_EVIDENCE',V.UNKNOWN,'Log budget or STOP save failure; not complete independent failure history.',commit=PUBLISHED))
    except V.Unsupported as e:result.update(status='UNSUPPORTED_SCHEMA',issues=[str(e)])
    except (V.Invalid,KeyError,IndexError,StopIteration,TypeError,ValueError,OSError)as e:result.update(status='INSUFFICIENT_EVIDENCE'if isinstance(e,FileNotFoundError)else'INTEGRITY_MISMATCH',issues=[str(e)])
    return result
