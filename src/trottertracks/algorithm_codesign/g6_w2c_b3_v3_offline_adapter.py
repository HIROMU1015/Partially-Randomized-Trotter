"""Independent, versioned offline adapter for fixed artificial integrated v3.

Only this consumer is extended. Old P1 V1/core and every producer byte remain
unchanged. Archived inputs are read-only and materialized in private owned temp.
"""
from copy import deepcopy
from pathlib import Path
import json
import gzip
import tarfile
import tempfile
from . import g6_w2c_b3_offline_result as V
from . import g6_w2c_b3_parallel_verifier_core as G
from . import g6_w2c_b3_parallel_offline_adapter as A
from . import g6_w2c_b3_v3_offline_identity as I


def extract_archive(archive,target):
    """No links/traversal/duplicate members; stream physical bytes, not all RAM."""
    target=Path(target);target.mkdir(exist_ok=False)
    names=set()
    # GzipFile bounds decompressed reads. tarfile's own r|gz decompressor can
    # retain a large expansion from a highly compressed input block.
    with gzip.open(archive,'rb') as stream,tarfile.open(fileobj=stream,mode='r|') as tar:
        for member in tar:
            name=Path(member.name)
            V.require(not name.is_absolute() and '..' not in name.parts and name.parts and
                      name.parts[0]=='artificial-output','ARCHIVE_PATH')
            V.require(member.isdir() or member.isfile(),'ARCHIVE_LINK_OR_SPECIAL_FILE')
            if member.isdir():
                (target/name).mkdir(parents=True,exist_ok=True);continue
            V.require(member.name not in names,'DUPLICATE_ARCHIVE_MEMBER');names.add(member.name)
            V.require(len(name.parts)==2,'ARCHIVE_NESTED_FILE')
            path=target/name;path.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as src,path.open('xb') as dst:
                while True:
                    chunk=src.read(1048576)
                    if not chunk:break
                    dst.write(chunk)
    return target/'artificial-output'


def control_v3(path,tasks,report,reference):
    """Explicit v3->legacy control normalization, with original bytes retained.

    Core phase/worker/failure semantics are reused; new binding/receipt schemas
    and original snapshot byte offsets are independently checked here.
    """
    binding=None;receipt=None;first=None;total=0;before_receipt=None;events=0
    saved_failures=[]
    partial=report is not None and bool(report.get('log_errors'))
    with tempfile.TemporaryDirectory(prefix='b3-v3-control-') as d:
        normalized=Path(d)/'control.jsonl'
        with normalized.open('xb') as dst:
            for row,size in A.records(path):
                event=row.get('event')
                if events==0:
                    V.require(event=='PLAN' and row.get('schema')=='b3_parallel_control_v3','V3_CONTROL_PLAN')
                    first=deepcopy(row);row=dict(row,schema='b3_parallel_artificial_v1')
                if event=='BUNDLE_BINDING':
                    V.require(binding is None and events==1 and row.get('schema')=='b3_parallel_admission_v3' and row.get('artificial_only') is True,'BUNDLE_BINDING_POSITION')
                    binding=row
                else:
                    if event=='FAILURE_OBSERVED':
                        original={k:v for k,v in row.items() if k!='event'}
                        arrival=original['parent_arrival']
                        V.require(type(arrival) is int and arrival>0 and
                                  (not saved_failures or arrival>saved_failures[-1]['parent_arrival']),
                                  'CONTROL_FAILURE_ORDER')
                        if partial:
                            V.require(arrival<=len(report['failures']) and
                                      original==report['failures'][arrival-1],'CONTROL_FAILURE_SUBSEQUENCE')
                            row=dict(row,parent_arrival=len(saved_failures)+1)
                        saved_failures.append(original)
                    if event=='COMPLETION_RECEIPT':
                        reference.validate('receipt',row)
                        V.require(receipt is None,'DUPLICATE_RECEIPT');receipt=row;before_receipt=total
                    # safe_values may contain exact Fraction in F01 reports.
                    data=json.dumps(row,separators=(',',':'),ensure_ascii=True,
                                    default=lambda x:{'exact':V.canonical(x)}).encode()+b'\n'
                    dst.write(data)
                total+=size;events+=1
        normalized_report=report
        if partial:
            normalized_report=dict(report,failures=[dict(f,parent_arrival=i+1) for i,f in enumerate(saved_failures)])
        c=A.validate_control(normalized,tasks,normalized_report)
    if partial and len(saved_failures)!=len(report['failures']):
        c['evidence_gaps'].append('Log failure: only a subsequence of parent failure notices persisted')
    c['original_failures']=saved_failures
    c.update(binding=binding,receipt=receipt,bytes=total,bytes_before_receipt=before_receipt,original_plan=first)
    return c


def check_binding(control,report,contract,reference):
    stored=control['binding'];V.require(stored is not None,'MISSING_BUNDLE_BINDING')
    for k in ('formal_digest','driver_digest','runtime_manifest_sha256','source_pins_sha256'):
        V.require(V.is_hash(stored.get(k)),'BINDING_DIGEST_SHAPE:'+k)
    expected=reference.contract(contract) if contract is not None else None
    if expected is not None:
        for k,value in expected.items():V.require(stored[k]==value,'CONTRACT_BINDING:'+k)
    claim=report.get('binding')
    if claim is not None:
        for k,value in claim.items():
            V.require(k in ('formal_digest','runtime_manifest_sha256','source_pins_sha256') and stored[k]==value,'HOST_BINDING:'+k)
    if report.get('runtime_manifest') is not None:
        runtime=report['runtime_manifest']
        reference.runtime(runtime,runtime['source_pins'])
        V.require(V.identity(runtime)==stored['runtime_manifest_sha256'] and V.identity(runtime['source_pins'])==stored['source_pins_sha256'],'HOST_RUNTIME_DIGEST')
    receipt=control['receipt']
    if receipt is not None:
        for k in ('formal_digest','runtime_manifest_sha256','source_pins_sha256'):
            V.require(receipt[k]==stored[k],'RECEIPT_BINDING:'+k)
        V.require(report.get('completion_receipt')==receipt,'HOST_RECEIPT_RECORD')
    return dict(complete_contract_present=contract is not None,independently_recomputed=expected,
                saved_binding=stored,contract_identity_evidence=V.RAW if expected else V.CLAIM,
                unavailable_input_bytes=['private artificial adoption.json','historical Python executable/libraries','private approval/SQLite store'],
                real_authorization_verified=False)


def accounting_v3(state,report,planned):
    """Reuse state totals; v3 also permits FAILED directly from START_RESERVED.

    A failed latest state alone cannot establish whether ENTER was granted.
    Keep that ambiguity; do not increment observed starts to match failed slots.
    """
    calls=G.accounting(state,None,planned)
    if report is None:return calls
    c=state['counts'];started=V.nonnegative(report['started_observed'])
    V.require(report['planned']==planned and I.strict_equal(report['states'],c) and
              report['reserved']==calls['reserved'],'GLOBAL_STATE_COUNTER_BINDING')
    minimum=c['COMPLETED']+c['STARTED']+c['IN_FLIGHT']
    maximum=minimum+c['FAILED']+c['TERMINATED_WITH_UNKNOWN_RESULT']
    V.require(minimum<=started<=maximum,'V3_STARTED_COMPLETED_UNKNOWN_BINDING')
    V.require(calls['reserved']<=min(planned,report['settings']['call_limit']),'CALL_CAP')
    calls.update(started_observed=started,failed_slot_entry_history='NOT_ESTABLISHED_FROM_LATEST_STATE')
    return calls


def _verify_directory(out,m,report,contract,reference):
    result=dict(schema='b3_v3_independent_verification_v1',status='INSUFFICIENT_EVIDENCE',issues=[],pairs=[],checks=[],
                run_id=m['run_id'],artificial_only=True,producer_public_commit=I.PUBLIC,measured_source_commit=I.MEASURED,mandatory_STOP=True)
    first=next(A.records(out/'control.jsonl'))[0]
    V.require(first.get('schema')=='b3_parallel_control_v3','V3_PLAN_SCHEMA')
    tasks,planned=G.plan(first['settings']['profile_n']);lookup={t['key']:t for t in tasks}
    crash=report.get('schema')=='b3_parallel_crash_observation_v3'
    if not crash:
        if report.get('schema')!='b3_parallel_result_v3':raise V.Unsupported('V3_RESULT_SCHEMA')
        reference.validate('result',report)
        V.require(report['settings']==first['settings'],'REPORT_PLAN_SETTINGS')
        keys=[r['task']['key'] for r in report['pairs']]
        V.require(len(set(keys))==len(keys) and keys==[t['key'] for t in tasks if t['key'] in keys],'PAIR_DUPLICATE_OR_CONTRACT_ORDER')
        V.require(report['contract_order']==[t['key'] for t in tasks],'CONTRACT_ORDER')
        V.require(set(report['all_pair_states'])==set(lookup),'PAIR_STATE_COVERAGE')
        for row in report['pairs']:
            V.require(I.strict_equal(row['task'],lookup[row['task']['key']]),'TASK_GLOBAL_OFFSET')
            V.require(report['all_pair_states'][row['task']['key']]==row['status'],'PAIR_STATUS_BINDING')
            f=(row.get('failure') or {}).get('trial_failure_provenance')
            if f is not None:reference.validate('f01',f)
        for t in tasks:
            if t['key'] not in keys:V.require(report['all_pair_states'][t['key']]=='NOT_STARTED','UNDISPATCHED_PAIR_STATE')
    else:
        V.require(report.get('complete') is False and report.get('mandatory_STOP') is True and
                  type(report.get('retry')) is int and report['retry']==0 and report.get('resume') is False,'CRASH_STOP_FLAGS')
        for k in ('started_observed','completed','parent_cpu_ns','worker_cpu_ns','total_cpu_ns'):
            V.require(report.get(k) is None,'CRASH_UNKNOWN_REPLACED:'+k)
        V.require(not (out/'summary.jsonl').exists(),'PARENT_CRASH_FAKE_SUMMARY')
    control=control_v3(out/'control.jsonl',tasks,None if crash else report,reference)
    state=G.state_counts(out/'trial_states.bin',tasks)
    calls=accounting_v3(state,None if crash else report,planned)
    if crash:
        V.require(report['raw_state_counts']==state['counts'] and calls['unknown']>=report['reserved_unknown_lower_bound'],'CRASH_STATE_UNKNOWN')
        calls.update(completed_reported=None,started_observed=None)
    identities=check_binding(control,report,contract,reference)
    if contract is not None:V.require(contract['settings']==first['settings'],'CONTRACT_SETTINGS')
    source_rows={r['task']['key']:r for r in report.get('pairs',[])}
    actual_result=0;present=set()
    allowed={t['key']+'.jsonl' for t in tasks}|{'summary.jsonl','control.jsonl','trial_states.bin'}
    V.require(all(p.is_file() and p.name in allowed for p in out.iterdir()),'UNKNOWN_OUTPUT_FILE_OR_COLLISION')
    for t in tasks:
        p=out/(t['key']+'.jsonl');pc=state['pairs'][t['key']];row=source_rows.get(t['key'])
        if not p.exists():
            V.require(not pc['COMPLETED'],'COMPLETED_STATE_MISSING_PAIR');continue
        inode=(p.stat().st_dev,p.stat().st_ino);V.require(inode not in present,'PAIR_OUTPUT_COLLISION');present.add(inode)
        actual_result+=p.stat().st_size
        r=A.verify_worker(p,t,row)
        V.require(r['status'] not in ('INTEGRITY_MISMATCH','UNSUPPORTED_SCHEMA'),'PAIR_INTEGRITY:'+t['key']+':'+','.join(r['issues']))
        if r['engine_committed_trials'] is not None:
            V.require(pc['COMPLETED']<=r['engine_committed_trials'],'STATE_EXCEEDS_F01_COMMIT')
        V.require(pc['COMPLETED']<=r['visible_trial_rows'],'STATE_EXCEEDS_PHYSICAL_PREFIX')
        with (out/'trial_states.bin').open('rb') as states:
            states.seek(t['offset'])
            for item in V.Reader(p):
                if 'trial_id' in item.payload:
                    b=states.read(1);V.require(b and b[0] not in (0,6),'PHYSICAL_ROW_WITHOUT_CALL_RESERVATION')
        r.update(task_key=t['key'],global_logical_completed=pc['COMPLETED'],global_state_counts=pc)
        result['pairs'].append(r)
    receipt=control['receipt'];summary=None;summary_file=None
    if (out/'summary.jsonl').exists():
        reader=V.Reader(out/'summary.jsonl');docs=[]
        for item in reader:V.require(not docs,'MULTIPLE_SUMMARY_ROWS');docs.append(item.payload)
        V.require(docs and not reader.stats['incomplete_tail_bytes'],'SUMMARY_INCOMPLETE')
        summary=docs[0];summary_file=reader.stats;reference.validate('result',summary)
        V.require(summary['binding']=={k:control['binding'][k] for k in ('formal_digest','runtime_manifest_sha256','source_pins_sha256')},'SUMMARY_BINDING')
        for k in ('schema','settings','planned','reserved','started_observed','states','pairs','failures','contract_order','completion_order','complete',
                  'parent_cpu_ns','worker_cpu_ns','total_cpu_ns','wall_ns','max_inflight','representative_failure'):
            V.require(summary[k]==report[k],'SUMMARY_HOST_SNAPSHOT:'+k)
        V.require(summary['measurement_condition']==report['measurement_condition'],'SUMMARY_CONDITION')
        if receipt is not None:
            V.require(receipt['summary_chain']==reader.stats['chain_sha256'] and receipt['summary_sha']==reader.stats['file_sha256'],'SUMMARY_RECEIPT_SHA')
        V.require(summary['output']['reserved'].get('result',0)==actual_result,'PRE_FINAL_RESULT_SNAPSHOT')
        if control['bytes_before_receipt'] is not None:
            V.require(summary['output']['reserved'].get('log',0)==planned+control['bytes_before_receipt'],'PRE_FINAL_LOG_SNAPSHOT')
        actual_result+=reader.stats['bytes']
    elif receipt is not None:
        result['issues'].append('RECEIPT_WITHOUT_SUMMARY')
    actual_log=planned+control['bytes'];budget=None
    if not crash:
        budget={}
        for ch,actual in (('result',actual_result),('log',actual_log)):
            saved=report['output'];reserved=V.nonnegative(saved['reserved'].get(ch,0));ack=V.nonnegative(saved['observed'].get(ch,0));limit=saved['limits'][ch]
            V.require(limit==first['settings'][ch+'_bytes'] and 0<=ack<=actual<=reserved<=limit<=G.BUDGET[ch+'_bytes'],'GLOBAL_BYTES:'+ch)
            budget[ch]=dict(reserved=reserved,ack_observed=ack,physical_bytes=actual,unknown_credit=reserved-ack,credit_refunded=False)
        V.require(report['representative_failure']==(report['failures'][0] if report['failures'] else None),'FIRST_PARENT_FAILURE')
        last_time=None
        for i,f in enumerate(report['failures']):
            V.require(f['parent_arrival']==i+1 and (last_time is None or f['parent_observed_ns']>=last_time),'PARENT_FAILURE_ORDER')
            last_time=f['parent_observed_ns']
            snapshot=f.get('inflight_snapshot')
            if snapshot is not None:
                V.require(i==0 and len(snapshot)<=report['max_inflight'],'FIRST_FAILURE_INFLIGHT_SNAPSHOT')
                ids=set()
                for key,ticket in snapshot.items():
                    V.require(key in lookup,'SNAPSHOT_WORKER');t=lookup[key]
                    parts=ticket['trial_id'].split('/');axis=parts[-2];index=int(parts[-1])
                    V.require(axis in ('X','Y') and 0<=index<t['n'] and
                              ticket['trial_id']==V.trial(dict(t,N=t['n']),axis,index) and
                              ticket['worker']==key and ticket['state'] in ('START_RESERVED','IN_FLIGHT') and
                              ticket['index']==t['offset']+(0 if axis=='X' else t['n'])+index and
                              ticket['trial_id'] not in ids,'SNAPSHOT_GLOBAL_ID');ids.add(ticket['trial_id'])
        mode=first['settings']['mode'];cap=1 if mode=='S' else(first['settings']['inflight_cap'] if mode=='P2' else first['settings']['workers'])
        V.require(0<=report['max_inflight']<=cap,'INFLIGHT_CAP')
    claim=report.get('fake_claim_status',report.get('claim_status'))
    success=not crash and report['complete']
    if success:
        V.require(calls['completed']==planned and calls['reserved']==planned and len(report['pairs'])==24 and
                  all(r['status']=='OK' for r in report['pairs']) and not report['failures'],'FALSE_COMPLETE')
        ack_complete=all(v['reserved']==v['ack_observed']==v['physical_bytes'] for v in budget.values())
        result['status']='VERIFIED_COMPLETE' if summary is not None and receipt is not None and contract is not None and ack_complete else 'INSUFFICIENT_EVIDENCE'
        if claim is not None and claim!='STOPPED' or report.get('log_errors') or report.get('secondary_stop_error'):
            result['status']='TECHNICAL_FAILURE';result['issues'].append('OUTER_STOP_OR_PERSISTENCE_INCOMPLETE')
        elif claim is None:
            result['status']='INSUFFICIENT_EVIDENCE';result['issues'].append('MISSING_FAKE_STOP_EVIDENCE')
        if result['status']=='VERIFIED_COMPLETE':
            for arm in V.ARMS:
                ref=out/('correctness_baseline_'+arm+'_A.jsonl')
                for phase,pn in G.PHASES:
                    a=out/f'{phase}_{pn}_{arm}_A.jsonl';b=out/f'{phase}_{pn}_{arm}_B.jsonl'
                    for axis in ('X','Y'):
                        for x,y,prefix in ((a,b,False),(ref,a,True),(ref,b,True),(out/('profile_baseline_'+arm+'_A.jsonl'),a,phase=='correctness')):
                            V.require(V.cross_pair(x,y,axis,prefix_only=prefix)['status']=='PASS','CROSS_PAIR_EVENT_HASH')
    else:result['status']='TECHNICAL_FAILURE'
    # Repeated FAIL/END notifications of the same trial are not additional calls.
    notifications={}
    for f in report.get('failures',[]):
        tid=f.get('trial_id')
        if tid is None:
            provenance=(f.get('terminal_failure') or {}).get('trial_failure_provenance')
            tid=(provenance or {}).get('current_attempt',{}).get('trial_id')
        if tid is not None:notifications[tid]=notifications.get(tid,0)+1
    if m.get('operations'):
        op=G.check_operations((p for p,_ in A.records(V.safe_path(Path(m['_input_root']),m['operations']))),tasks,m['run_id'])
        V.require(op['counts'].get('reserved',0)==calls['reserved'] and op['counts'].get('completed',0)==calls['completed'],'OPERATIONS_STATE_BINDING')
        result['normalized_artificial_operations']=op
    result.update(calls=calls,state=state,binding_evidence=identities,output_budget=budget,summary_file=summary_file,
                  actual_result_bytes=actual_result,actual_log_bytes=actual_log,state_retained_bytes=planned,
                  state_pwrite_traffic_bytes=report.get('state_update_io_bytes'),report=report,
                  control=dict(started_workers=len(control['started']),ended_workers=len(control['ended']),
                               receipt_present=receipt is not None,stop_saved=control['stop_saved'],evidence_gaps=control['evidence_gaps']),
                  fake_claim_status=claim,failure_notifications_by_trial=notifications,
                  independent_atomic_claim_proof=False,trial_operation_history_available=False)
    return result


def verify(manifest_path,reference_root):
    manifest_path=Path(manifest_path).resolve();root=manifest_path.parent
    result=dict(schema='b3_v3_independent_verification_v1',status='INSUFFICIENT_EVIDENCE',issues=[],pairs=[],checks=[],artificial_only=True,mandatory_STOP=True)
    try:
        m=V.plain(manifest_path.read_text())
        if m.get('schema')!='b3_v3_offline_input_v1':raise V.Unsupported('V3_ADAPTER_INPUT_SCHEMA')
        V.require(m.get('artificial_only') is True and m.get('producer_public_commit')==I.PUBLIC and
                  m.get('measured_source_commit')==I.MEASURED and type(m['run_id']) is str and m['run_id'],'FIXED_PUBLIC_ARTIFICIAL_INPUT')
        reference=I.Reference(reference_root)
        archive=V.safe_path(root,m['archive']);V.require(V.file_digest(archive)==m['archive_sha256'],'ARCHIVE_SHA')
        report=A.safe_values(V.plain(V.safe_path(root,m['host_report']).read_text()))
        contract=V.plain(V.safe_path(root,m['contract']).read_text()) if m.get('contract') else None
        m['_input_root']=str(root)
        with tempfile.TemporaryDirectory(prefix='b3-v3-offline-raw-') as temp:
            out=extract_archive(archive,Path(temp)/'raw')
            result=_verify_directory(out,m,report,contract,reference)
        result['input_archive_sha256']=m['archive_sha256']
    except V.Unsupported as e:result.update(status='UNSUPPORTED_SCHEMA',issues=[str(e)])
    except (V.Invalid,KeyError,IndexError,StopIteration,TypeError,ValueError,OSError,tarfile.TarError) as e:
        result.update(status='INSUFFICIENT_EVIDENCE' if isinstance(e,FileNotFoundError) else 'INTEGRITY_MISMATCH',issues=[str(e)])
    return result
