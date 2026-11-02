"""Independent Storage V2 offline acceptance. Imports consumer primitives only."""
from collections import Counter
from pathlib import Path
import json,sqlite3,tarfile,tempfile
from . import g6_w2c_b3_offline_result as V
from . import g6_w2c_b3_parallel_verifier_core as G
from . import g6_w2c_b3_parallel_offline_adapter as A
from . import g6_w2c_b3_v3_offline_adapter as B
from . import g6_w2c_b3_v3_offline_identity as I
from . import g6_w2c_b3_v3_offline_cost as C
from . import g6_w2c_b3_pilot_independent_identity as P
from . import g6_w2c_b3_storage_independent_trace_v2 as T

def journal(path,tasks,report):
    ev=report['evidence_v2'];raw=Path(path).read_bytes();lines=raw.splitlines(keepends=True);rows=[];line_bytes=[];markers=[];last_arrival=0
    for line in lines:
        if not line.endswith(b'\n'):
            V.require(not ev['journal_complete'],'JOURNAL_PARTIAL_COMPLETE');break
        row=A.safe_values(V.plain(line));rows.append(row);line_bytes.append(len(line))
        if row['event']=='PILOT_COMPLETION_RECEIPT':markers.append(dict(row))
        if row['event']=='FAILURE_OBSERVED':
            arrival=row['parent_arrival'];V.require(type(arrival) is int and last_arrival<arrival<=len(report['failures']),'V2_PARENT_ARRIVAL_ORDER');last_arrival=arrival
            V.require({k:v for k,v in row.items() if k!='event'}==report['failures'][arrival-1],'V2_PARENT_FAILURE_BINDING')
    V.require(len(markers)<=1,'V2_DUPLICATE_SUCCESS_MARKER')
    if rows:
        plan=rows[0];V.require(plan['event']=='PLAN' and plan['schema']=='b3_parallel_control_v3' and plan['settings']==report['settings'] and plan['tasks']==tasks,'V2_JOURNAL_PLAN')
    V.require(not ev['journal_complete'] or bool(rows),'V2_MISSING_COMPLETE_JOURNAL')
    core=None
    if ev['journal_complete']:
        with tempfile.TemporaryDirectory(prefix='b3-v2-independent-control-') as temp:
            p=Path(temp)/'control.jsonl'
            with p.open('x') as dst:
                for row in rows:
                    if row['event']=='PILOT_COMPLETION_RECEIPT':continue
                    if row['event']=='PLAN':row=dict(row,schema='b3_parallel_artificial_v1')
                    dst.write(json.dumps(row,separators=(',',':'),default=lambda x:{'exact':V.canonical(x)})+'\n')
            core=A.validate_control(p,tasks,report)
    if ev['stop_journal_saved']:V.require(any(r['event']=='FAILURE_OBSERVED' for r in rows),'V2_STOP_JOURNAL_WITNESS')
    return dict(marker={k:v for k,v in markers[0].items() if k!='event'} if markers else None,marker_final=bool(rows and rows[-1]['event']=='PILOT_COMPLETION_RECEIPT'),complete=ev['journal_complete'],rows=len(rows),line_bytes=line_bytes,core_validated=core is not None)

def documents(out,report,identity,trace_rows,control):
    complete=report['complete'];receipt=None;summary=None;diagnostic=None;gaps=[]
    rp=out/'pilot_receipt.json';sp=out/'pilot_summary.jsonl';dp=out/'final_diagnostic.json'
    if rp.exists() and rp.stat().st_size:
        receipt=V.plain(rp.read_bytes());V.require(receipt['schema']=='b3_pilot_storage_receipt_v2' and receipt['artificial_only'] is True and receipt['production_permission'] is False,'V2_RECEIPT_SCHEMA_FLAGS')
        for k in ('pilot','inner'):V.require(receipt[k+'_binding']==identity[k+'_digest'],'V2_RECEIPT_BINDING:'+k)
        # Validate a receipt prefix independently of the later final trace prefix.
        T.read_prefix(out/'operations.jsonl',receipt['trace_prefix'],identity['inner_digest'],False)
    if sp.exists() and sp.stat().st_size:
        reader=V.Reader(sp);docs=[r.payload for r in reader]
        V.require(len(docs)==1 and not reader.stats['incomplete_tail_bytes'],'V2_SUMMARY_ROWS')
        summary=docs[0];V.require(summary['schema']=='b3_pilot_storage_summary_v2','V2_SUMMARY_SCHEMA')
        for k in ('pilot','inner'):V.require(summary[k+'_binding']==identity[k+'_digest'],'V2_SUMMARY_BINDING:'+k)
        if receipt:
            V.require(receipt['summary_sha256']==reader.stats['file_sha256'] and receipt['summary_chain']==reader.stats['chain_sha256'] and summary['trace']==receipt['trace_prefix'],'V2_SUMMARY_RECEIPT_CHAIN')
        for k in ('schema','settings','planned','reserved','started_observed','states','pairs','contract_order','completion_order','parent_cpu_ns','worker_cpu_ns','total_cpu_ns','wall_ns'):
            # Parent timing grows until the final report; the snapshot values
            # are lower bounds, not equal observations at a different boundary.
            if k in ('parent_cpu_ns','wall_ns','total_cpu_ns'):V.require(summary['result'][k]<=report[k],'V2_SUMMARY_TIME_BOUNDARY:'+k)
            else:V.require(I.strict_equal(summary['result'][k],report[k]),'V2_SUMMARY_SNAPSHOT:'+k)
        V.require(summary['result']['output']['reserved']['result']<=report['output']['reserved']['result'],'V2_SUMMARY_PRE_FINAL_BUDGET')
    if dp.exists() and dp.stat().st_size:
        diagnostic=V.plain(dp.read_bytes());V.require(diagnostic['schema']=='b3_pre_close_diagnostic_v2' and diagnostic['complete'] is False and diagnostic['final_acceptance'] is False,'V2_DIAGNOSTIC_CANNOT_ACCEPT')
        V.require(diagnostic['reserved']==report['reserved'] and diagnostic['started']==report['started_observed'],'V2_DIAGNOSTIC_COUNTER')
        V.require({s:diagnostic['states'].get(s,0) for s in G.STATES}==report['states'],'V2_DIAGNOSTIC_STATE')
    marker=control['marker']
    if marker:V.require(receipt is not None and I.strict_equal(marker,receipt),'V2_MARKER_RECEIPT_BINDING')
    if complete:
        V.require(receipt is not None and summary is not None and diagnostic is not None and marker is not None and control['marker_final'],'V2_MISSING_FINAL_SUCCESS_WITNESS')
        V.require(summary['result']['complete'] is True,'V2_SUMMARY_FALSE_COMPLETE')
        # Nontrace result credit operations prove the saves precede marker.
        result_reserves=[r for r in trace_rows if r['op']=='WRITE_RESERVE' and r['channel']=='result']
        result_acks=[r for r in trace_rows if r['op']=='WRITE_ACK' and r['channel']=='result']
        expected=[sp.stat().st_size,rp.stat().st_size,dp.stat().st_size]
        V.require([r['amount'] for r in result_reserves[-3:]]==expected and [r['amount'] for r in result_acks[-3:]]==expected,'V2_FINAL_DOCUMENT_CHARGES')
        log_reserves=[r for r in trace_rows if r['op']=='WRITE_RESERVE' and r['channel']=='log']
        log_acks=[r for r in trace_rows if r['op']=='WRITE_ACK' and r['channel']=='log']
        # Initial state/PLAN credit precedes trace construction. Every later
        # log reservation/ACK maps to one physical journal line in order.
        expected_log=[report['planned']+control['line_bytes'][0],*control['line_bytes'][1:]]
        V.require([r['amount'] for r in log_reserves]==expected_log and [r['amount'] for r in log_acks]==expected_log,'V2_CONTROL_CREDIT_LINE_ORDER')
        V.require(result_reserves[-1]['seq']<result_acks[-1]['seq']<log_reserves[-1]['seq']<log_acks[-1]['seq'],'V2_DIAGNOSTIC_ACK_BEFORE_SUCCESS_MARKER')
    return dict(summary_present=summary is not None,receipt_present=receipt is not None,diagnostic_present=diagnostic is not None,marker_present=marker is not None,marker_is_sufficient_alone=False,full_success_join=complete)

def verify_output(output,candidate,inner,report,reference):
    out=Path(output);identity=reference.inner(candidate,inner)
    V.require(report['schema']=='b3_pilot_storage_result_v2','V2_RESULT_SCHEMA')
    V.require(report['pilot_binding']==identity['pilot_digest'] and I.strict_equal(report['settings'],inner['settings']),'V2_REPORT_BINDING')
    for k,v in dict(retry=0,resume=False,mandatory_STOP=True,artificial_only=True,call_reservation_nonrefundable=True).items():V.require(I.strict_equal(report[k],v),'V2_CLOSED_FLAG:'+k)
    V.require(type(report['complete']) is bool,'V2_COMPLETE_TYPE')
    tasks,planned=G.plan(inner['settings']['profile_n']);lookup={t['key']:t for t in tasks};ev=report['evidence_v2']
    for k in ('trace_complete','statefile_complete','journal_complete','stop_trace_saved','stop_journal_saved'):V.require(type(ev[k]) is bool,'V2_EVIDENCE_FLAG:'+k)
    V.require(ev['state_authority']=='MONOTONE_RAM_SHADOW_NOT_A_CLAIM_OF_PHYSICAL_DURABILITY','V2_STATE_AUTHORITY')
    ram=report['states'];V.require(set(ram)==set(G.STATES),'V2_STATE_COVERAGE')
    for x in ram.values():V.nonnegative(x)
    reserved=V.nonnegative(report['reserved']);started=V.nonnegative(report['started_observed'])
    V.require(report['planned']==planned and sum(ram.values())==planned and reserved==planned-ram['NOT_STARTED'] and reserved<=min(planned,inner['settings']['call_limit'],candidate['proposal']['total_calls']),'V2_RAM_CALL_ACCOUNTING')
    V.require(ram['COMPLETED']<=started<=reserved-ram['CANCELLED_BEFORE_START'],'V2_STARTED_BOUNDS')
    V.require(set(ev['next_indices'])==set(lookup) and sum(ev['next_indices'].values())==reserved,'V2_CURSOR_RESERVED_BINDING')
    for t in tasks:V.require(type(ev['next_indices'][t['key']]) is int and 0<=ev['next_indices'][t['key']]<=2*t['n'],'V2_CURSOR_RANGE')
    V.require(not ev['current_tickets'] and not ev['pair_next'],'V2_FINAL_TICKETS_NOT_DRAINED')
    V.require(report['representative_failure']==(report['failures'][0] if report['failures'] else None),'V2_FIRST_FAILURE')
    for i,f in enumerate(report['failures']):V.require(f['parent_arrival']==i+1,'V2_REPORT_ARRIVAL')
    V.require(report['contract_order']==list(lookup) and set(report['all_pair_states'])==set(lookup),'V2_PAIR_COVERAGE')
    keys=[r['task']['key'] for r in report['pairs']];V.require(len(keys)==len(set(keys)) and keys==[k for k in lookup if k in keys],'V2_PAIR_DUPLICATE_ORDER')
    rows={r['task']['key']:r for r in report['pairs']};physical=dict(log=0,result=0);inodes=set()
    allowed={t['key']+'.jsonl' for t in tasks}|{'control.jsonl','operations.jsonl','trial_states.bin','pilot_summary.jsonl','pilot_receipt.json','final_diagnostic.json'}
    for p in out.iterdir():
        V.require(p.is_file() and not p.is_symlink() and p.name in allowed,'V2_OUTPUT_NAMESPACE');ino=(p.stat().st_dev,p.stat().st_ino);V.require(ino not in inodes,'V2_INODE_ALIAS');inodes.add(ino)
        physical['log' if p.name in ('control.jsonl','operations.jsonl','trial_states.bin') else 'result']+=p.stat().st_size
    state_raw=(out/'trial_states.bin').read_bytes();state=None;calls=None;gaps=[]
    if ev['statefile_complete']:
        V.require(not ev['state_errors'],'V2_STATE_COMPLETE_WITH_ERRORS');state=G.state_counts(out/'trial_states.bin',tasks);calls=B.accounting_v3(state,report,planned)
        V.require(ev['shadow_sha256']==state['file_sha256'],'V2_SHADOW_PHYSICAL_SHA')
        for t in tasks:V.require(ev['next_indices'][t['key']]==2*t['n']-state['pairs'][t['key']]['NOT_STARTED'],'V2_CURSOR_PHYSICAL_BINDING')
    else:
        V.require(bool(ev['state_errors']),'V2_STATE_INCOMPLETE_WITHOUT_REASON');gaps.append('RAM shadow/counters are Producer claims; physical state consistency not independently established')
    if any(x.get('storage') in ('state','journal') and x.get('op')=='CLOSE' for x in ev['storage_errors']):gaps.append('Storage close failed; observed bytes do not establish successful final persistence')
    trace_rows,trace=T.read_prefix(out/'operations.jsonl',ev['trace_prefix'],identity['inner_digest'],ev['trace_complete'])
    V.require(report['operation_trace_sha256']==trace['sha256'],'V2_PHYSICAL_TRACE_SHA')
    history=T.inspect(trace_rows,tasks,identity['inner_digest'],report,state_raw,ev['statefile_complete'],ev['trace_complete'])
    if ev['stop_trace_saved']:V.require(any(x['op']=='STOP' for x in trace_rows),'V2_STOP_TRACE_WITNESS')
    if not ev['trace_complete']:gaps.append('Only trusted trace prefix validated; tail/RAM events are not independent transition evidence')
    if not ev['journal_complete']:gaps.append('Control journal incomplete; missing parent/worker transitions remain unknown')
    control=journal(out/'control.jsonl',tasks,report);pairs=[]
    for t in tasks:
        key=t['key'];row=rows.get(key);path=out/(key+'.jsonl')
        if row:
            V.require(I.strict_equal(row['task'],t) and report['all_pair_states'][key]==row['status'],'V2_PAIR_TASK_OFFSET_STATUS')
            f=(row.get('failure') or {}).get('trial_failure_provenance')
            if f is not None:reference.legacy.validate('f01',f)
        else:V.require(report['all_pair_states'][key]=='NOT_STARTED','V2_UNDISPATCHED_PAIR')
        if not path.exists():
            if state:V.require(not state['pairs'][key]['COMPLETED'],'V2_COMPLETED_WITHOUT_PAIR')
            continue
        p=A.verify_worker(path,t,row)
        V.require(p['status'] not in ('INTEGRITY_MISMATCH','UNSUPPORTED_SCHEMA'),'V2_PAIR_INTEGRITY:'+','.join(p['issues']))
        if state:
            n=state['pairs'][key]['COMPLETED'];V.require(n<=p['visible_trial_rows'],'V2_COMPLETE_EXCEEDS_PHYSICAL')
            if p['engine_committed_trials'] is not None:V.require(n<=p['engine_committed_trials'],'V2_COMPLETE_EXCEEDS_ENGINE_COMMIT')
        p.update(task_key=key,global_logical_completed=state['pairs'][key]['COMPLETED'] if state else None);pairs.append(p)
    budget={}
    for ch in ('log','result'):
        b=report['output'];credit=V.nonnegative(b['reserved'].get(ch,0));ack=V.nonnegative(b['observed'].get(ch,0));limit=b['limits'][ch]
        V.require(type(limit) is int and limit==inner['settings'][ch+'_bytes'] and ack<=physical[ch]<=credit<=limit<=G.BUDGET[ch+'_bytes'],'V2_PHYSICAL_ACK_BUDGET:'+ch)
        V.require(b['unknown_credit'][ch]==credit-ack,'V2_UNACKED_CREDIT_REFUND:'+ch)
        self_bytes=trace['bytes'] if ch=='log' else 0
        hr=history['reserved_bytes'].get(ch,0)+self_bytes;ha=history['ack_observed_bytes'].get(ch,0)+self_bytes
        if ev['trace_complete']:V.require(hr==credit and ha==ack,'V2_TRACE_BYTE_ACCOUNTING:'+ch)
        else:V.require(hr<=credit and ha<=ack,'V2_PREFIX_BYTE_LOWER_BOUND:'+ch)
        budget[ch]=dict(reserved=credit,ack_observed=ack,physical_bytes=physical[ch],unknown_credit=credit-ack,trace_self_bytes=self_bytes,refunded=False)
    docs=documents(out,report,identity,trace_rows,control)
    cap=1 if inner['settings']['mode']=='S' else inner['settings']['workers'];V.require(type(report['max_inflight']) is int and 0<=report['max_inflight']<=cap,'V2_MAX_INFLIGHT')
    stop=report['admission_stop'];V.require(stop is not None and stop['mandatory_STOP'] is True and stop['reservation_never_released'] is True,'V2_ADMISSION_STOP')
    complete=report['complete']
    if complete:
        V.require(not gaps and reserved==started==ram['COMPLETED']==planned and len(pairs)==24 and all(r['status']=='OK' for r in report['pairs']) and not report['failures'] and not history['stopped'],'V2_FALSE_COMPLETE')
        V.require(not ev['storage_errors'] and not ev['trace_errors'] and not report['log_errors'] and not stop['record_errors'] and report['fake_claim_status']=='STOPPED','V2_SUCCESS_WITH_SAVE_FAILURE')
        V.require(all(b['reserved']==b['ack_observed']==b['physical_bytes'] for b in budget.values()),'V2_SUCCESS_UNKNOWN_CREDIT')
        for t in tasks:
            p=out/(t['key']+'.jsonl')
            for axis in ('X','Y'):
                if t['layer']=='B':V.require(V.cross_pair(out/(t['key'][:-1]+'A.jsonl'),p,axis)['status']=='PASS','V2_AB_PAIR')
                if t['phase_index']:V.require(V.cross_pair(out/f"correctness_baseline_{t['arm']}_{t['layer']}.jsonl",p,axis,prefix_only=True)['status']=='PASS','V2_CORRECTNESS_PREFIX')
                if t['phase_index']==2:V.require(V.cross_pair(out/f"profile_baseline_{t['arm']}_{t['layer']}.jsonl",p,axis)['status']=='PASS','V2_BASELINE_INSTRUMENTED')
    status='VERIFIED_ARTIFICIAL_COMPLETE' if complete else ('INCOMPLETE_EVIDENCE' if gaps else 'TECHNICAL_FAILURE')
    result=dict(schema='b3_storage_independent_acceptance_v2',status=status,issues=[],identity=identity,calls=calls,ram_claim=dict(reserved=reserved,started=started,states=ram,independently_physically_proven=state is not None),state=state,history=history,trace=trace,control=control,documents=docs,output_budget=budget,physical_bytes=physical,pairs=pairs,evidence_gaps=gaps,report=report,state_retained_bytes=len(state_raw),state_pwrite_traffic_bytes=report['state_update_io_bytes'],mandatory_STOP=True,scientific_GO=False)
    # Cost core accepts its original status vocabulary; V2 evidence status is retained.
    cost_input=dict(result,status='VERIFIED_COMPLETE' if complete else 'TECHNICAL_FAILURE')
    result['cost']=C.aggregate(cost_input)
    V.require(result['cost']['status']!='COST_INTEGRITY_MISMATCH','V2_COST:'+','.join(result['cost'].get('issues',[])))
    result['cost']['comparison_eligible']=complete
    return result

def verify(case,reference):
    root=Path(case);result=dict(schema='b3_storage_independent_acceptance_v2',case=root.name,status='INCOMPLETE_EVIDENCE',issues=[],evidence_gaps=[],mandatory_STOP=True,scientific_GO=False)
    try:
        c=V.plain((root/'pilot_contract.json').read_bytes());inner=V.plain((root/'inner_contract.json').read_bytes());identity=reference.inner(c,inner)
        snapshot=P.admission_snapshot(root/'FAKE_ONLY_claims.sqlite3',inner)
        obs=A.safe_values(V.plain((root/'observations.json').read_bytes()));report=obs.get('report')
        if report is None:
            partial={}
            if (root/'artificial_raw.tar.gz').exists():
                with tempfile.TemporaryDirectory(prefix='b3-v2-incomplete-raw-') as temp:
                    out=B.extract_archive(root/'artificial_raw.tar.gz',Path(temp)/'raw')
                    partial['files']=[dict(name=p.name,bytes=p.stat().st_size,sha256=V.file_digest(p)) for p in sorted(out.iterdir())]
                    tasks,planned=G.plan(inner['settings']['profile_n'])
                    sp=out/'trial_states.bin'
                    if sp.exists() and sp.stat().st_size==planned:partial['state_bytes_snapshot']=G.state_counts(sp,tasks)
                    tp=out/'operations.jsonl'
                    if tp.exists():
                        raw=tp.read_bytes();lines=raw.splitlines(keepends=True);lines=[x for x in lines if x.endswith(b'\n')];prefix=b''.join(lines)
                        chain=V.plain(lines[-1])['sha256'] if lines else V.ZERO
                        _,stats=T.read_prefix(tp,dict(bytes=len(prefix),rows=len(lines),sha256=__import__('hashlib').sha256(prefix).hexdigest(),chain=chain),identity['inner_digest'],False)
                        partial['physical_chain']=stats;partial['trusted_acknowledged_prefix_established']=False
            result.update(identity=identity,admission=snapshot,partial_physical=partial,evidence_gaps=['Final host report absent; RAM and full transition/cost accounting unavailable'],cost=dict(total_cpu_ns=None,global_wall_ns=None,unknown_cost_is_zero=False,comparison_eligible=False))
            return result
        with tempfile.TemporaryDirectory(prefix='b3-v2-independent-raw-') as temp:
            out=B.extract_archive(root/'artificial_raw.tar.gz',Path(temp)/'raw');result=verify_output(out,c,inner,report,reference)
        V.require(snapshot['claim_status']==report['fake_claim_status'],'V2_REPORT_FAKE_LEDGER')
        result.update(case=root.name,admission=snapshot,raw_archive_sha256=V.file_digest(root/'artificial_raw.tar.gz'))
    except V.Unsupported as exc:result.update(status='UNSUPPORTED_SCHEMA',issues=[str(exc)])
    except (V.Invalid,KeyError,ValueError,TypeError,IndexError,OSError,sqlite3.Error,tarfile.TarError,EOFError) as exc:
        result.update(status='INCOMPLETE_EVIDENCE' if isinstance(exc,FileNotFoundError) else 'INTEGRITY_MISMATCH',issues=[str(exc)])
    result.setdefault('case',root.name)
    return result
