"""Read-only fake claim, physical byte ledger and post-close evidence join."""
from collections import Counter
from pathlib import Path
import sqlite3
import json
from . import g6_w2c_b3_offline_result as V
from .g6_w2c_b3_bundle_independent_identity_v1 import read,equal,hashed,SLOTS
from .g6_w2c_b3_bundle_independent_trace_v1 import scan

def claim(root,m,report=None):
    p=Path(root)/'FAKE_ONLY.sqlite3';V.require(p.is_file()and not p.is_symlink(),'FAKE_SNAPSHOT_REQUIRED')
    before=V.file_digest(p)
    with sqlite3.connect(p.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)as db:
        equal(db.execute('PRAGMA quick_check').fetchone(),('ok',),'FAKE_SQLITE_INTEGRITY')
        rows=db.execute('SELECT scope,claim,binding,status,spent FROM claims ORDER BY scope').fetchall()
        history=db.execute('SELECT seq,scope,event,binding FROM history ORDER BY seq').fetchall()
    equal(V.file_digest(p),before,'FAKE_SQLITE_READ_ONLY')
    V.require(len(rows)==1,'ONE_GLOBAL_CLAIM');r=rows[0]
    equal(r[:3],(m['global_scope'],m['global_claim'],hashed(m)),'GLOBAL_CLAIM_BINDING')
    equal(r[4],1,'CLAIM_SPENT');V.require(r[3]in ('SPENT','STOPPED'),'FAKE_STATUS')
    expected=[(1,m['global_scope'],'CLAIM_COMMITTED',hashed(m))]
    if r[3]=='STOPPED':expected.append((2,m['global_scope'],'STOP_COMMITTED',hashed(m)))
    equal(history,expected,'FAKE_COMMITTED_SNAPSHOT_HISTORY')
    snapshot=dict(schema='b3_FAKE_ONLY_snapshot_v1',claim=list(r),history=[list(x)for x in history],real_authority=False)
    if report is not None:equal(snapshot,report['fake_claim'],'FAKE_REPORT_SNAPSHOT')
    return dict(snapshot=snapshot,sha256=before,read_only=True,atomic_transition_history_independently_proven=False,real_authority=False)

def physical(out,root,tasks):
    logs={'trial_states.bin','control.jsonl','operations.jsonl'}
    names=logs|{t['key']+'.jsonl'for t in tasks}|{'candidate.json','manifest.json','S_receipt.json','P1_receipt.json','summary.json','diagnostic.json','FAKE_ONLY_snapshot.json','receipt.json','marker.json','final.json','delivery.json'}
    totals=Counter();files={};inodes=set()
    for p in out.iterdir():
        V.require(p.name in names and p.is_file()and not p.is_symlink(),'OUTPUT_NAMESPACE')
        ino=(p.stat().st_dev,p.stat().st_ino);V.require(ino not in inodes,'OUTPUT_INODE_ALIAS');inodes.add(ino)
        size=p.stat().st_size;totals['log'if p.name in logs else'result']+=size
        files[p.name]=dict(bytes=size,sha256=V.file_digest(p))
    observation=Path(root)/'external_observation.json'
    if observation.exists():
        V.require(observation.is_file()and not observation.is_symlink(),'CALLER_OBSERVATION_PATH')
        ino=(observation.stat().st_dev,observation.stat().st_ino);V.require(ino not in inodes,'CALLER_OUTPUT_ALIAS')
        totals['result']+=observation.stat().st_size
    return dict(totals=dict(totals),files=files,authority_input_harness_bytes_excluded=True,state_RAM_bytes_not_double_charged=True)

def ledger(m,r,h,trace,p,delivery,observation):
    postr=Counter();posta=Counter()
    for x in r['post_close_credits']:
        ch=x['channel'];V.require(ch in ('result','log'),'POST_CLOSE_CHANNEL');n=V.nonnegative(x['amount'])
        if x['op']=='WRITE_RESERVE':postr[ch]+=n
        else:
            equal(x['op'],'WRITE_ACK','POST_CLOSE_OP');posta[ch]+=n
            V.require(posta[ch]<=postr[ch],'POST_CLOSE_ACK_WITHOUT_CREDIT')
    attribution=h['by_condition'];result={}
    for ch in ('result','log'):
        b=r['output'];credit=V.nonnegative(b['reserved'].get(ch,0));ack=V.nonnegative(b['observed'].get(ch,0));retained=p['totals'].get(ch,0)
        equal(b['limits'][ch],m['byte_limits'][ch],'ONE_GLOBAL_BYTE_LIMIT:'+ch)
        V.require(ack<=credit<=m['byte_limits'][ch]and retained<=credit,'GLOBAL_PHYSICAL_RESERVED_CAP:'+ch)
        equal(b['unknown_credit'][ch],credit-ack,'UNKNOWN_CREDIT_NONREFUND:'+ch)
        projection=SLOTS if ch=='result'else 0
        if delivery and delivery['final_write_flush_close_observed']and observation and observation['bridge_returned_normally']:
            V.require(ack<=retained,'ACK_EXCEEDS_PHYSICAL:'+ch)
        reserve_lower=h['reserved_bytes'].get(ch,0)+postr[ch]+(trace['bytes']if ch=='log'else 0)
        ack_lower=h['ack_bytes'].get(ch,0)+posta[ch]+projection+(trace['bytes']if ch=='log'else 0)
        if r['evidence_v2']['trace_complete']:
            equal(reserve_lower,credit,'TRACE_POST_CLOSE_RESERVATION:'+ch);equal(ack_lower,ack,'TRACE_POST_CLOSE_ACK:'+ch)
        else:V.require(reserve_lower<=credit and ack_lower<=ack,'TRUSTED_PREFIX_BYTE_LOWER_BOUND:'+ch)
        attribution['SHARED']['reserved'][ch]+=postr[ch];attribution['SHARED']['observed'][ch]+=posta[ch]+projection
        result[ch]=dict(limit=m['byte_limits'][ch],reserved=credit,observed_or_projected=ack,physical_retained=retained,
                        unknown_credit=credit-ack,trace_self_bytes=trace['bytes']if ch=='log'else 0,refunded=False,
                        self_projection_independently_delivered=bool(delivery and delivery['final_write_flush_close_observed']and observation and observation['bridge_returned_normally']))
    if r['evidence_v2']['trace_complete']:
        for ch in ('result','log'):
            equal(sum(v['reserved'][ch]for v in attribution.values()),result[ch]['reserved'],'CONDITION_SHARED_RESERVATION_SUM:'+ch)
            equal(sum(v['observed'][ch]for v in attribution.values()),result[ch]['observed_or_projected'],'CONDITION_SHARED_ACK_SUM:'+ch)
    return result,attribution

def post_close(root,out,m,c,r,identity,admission,physical,gaps):
    binding=identity['manifest_digest'];run=m['run_id']
    observation=read(root/'external_observation.json')if(root/'external_observation.json').exists()else None
    delivery_path=out/'delivery.json';delivery=None
    if delivery_path.exists():
        try:delivery=read(delivery_path)
        except (json.JSONDecodeError,UnicodeDecodeError):
            V.require(observation is not None and observation.get('bridge_returned_normally')is False and bool(observation.get('exception')),'UNREADABLE_DELIVERY_WITHOUT_CALLER_FAILURE')
            gaps.append('Delivery bytes are incomplete; caller independently observed persistence failure')
    if delivery is None:gaps.append('External delivery after final close is absent')
    else:
        equal(delivery['schema'],'b3_bundle_external_delivery_v1','DELIVERY_SCHEMA')
        equal(delivery['run_id'],run,'DELIVERY_RUN');equal(delivery['manifest_binding'],binding,'DELIVERY_BINDING')
        equal(delivery['final_sha256'],V.file_digest(out/'final.json'),'DELIVERY_FINAL_HASH')
        equal(delivery['production_permission'],False,'DELIVERY_PERMISSION');equal(delivery['fsync_durability_verified'],False,'DELIVERY_DURABILITY')
        V.require(type(delivery['final_write_flush_close_observed'])is bool,'DELIVERY_CLOSE_TYPE')
        if not delivery['final_write_flush_close_observed']or delivery['post_final_errors']:gaps.append('Final close/persistence was not successfully observed')
    if observation is None:gaps.append('Caller observation after delivery close is absent')
    else:
        equal(observation['schema'],'b3_bundle_caller_observation_v1','CALLER_SCHEMA')
        equal(observation['run_id'],run,'CALLER_RUN');equal(observation['manifest_binding'],binding,'CALLER_BINDING')
        equal(observation['delivery_sha256'],V.file_digest(delivery_path)if delivery_path.exists()else None,'CALLER_DELIVERY_HASH')
        equal(observation['forbidden_imports'],[],'CALLER_SCIENCE_IMPORT')
        V.require(type(observation['bridge_returned_normally'])is bool,'CALLER_RETURN_TYPE')
        if not observation['bridge_returned_normally']or observation['exception']is not None:gaps.append('Caller observed bridge/delivery failure')
    for name in r['physical_files_before_final']:
        V.require(Path(name).name==name,'PRE_FINAL_INVENTORY_PATH')
        if not(out/name).exists():raise FileNotFoundError('Missing retained pre-final evidence: '+name)
    expected_files={n:v for n,v in physical['files'].items()if n not in ('final.json','delivery.json')}
    equal(r['physical_files_before_final'],expected_files,'PRE_FINAL_PHYSICAL_INVENTORY')
    if r['admission_stop_errors']or admission['snapshot']['claim'][3]!='STOPPED':gaps.append('Fake STOP was not persisted; SPENT is retained')
    if r['post_close_errors']:gaps.append('Receipt/marker persistence failed after storage close')
    return delivery,observation

def failed_final_witness(root,out,m):
    """A truncated final document supplies no RAM/cost proof, even with a marker."""
    delivery=read(out/'delivery.json');obs=read(root/'external_observation.json');binding=hashed(m)
    equal(delivery['schema'],'b3_bundle_external_delivery_v1','FAILED_FINAL_DELIVERY_SCHEMA')
    equal(obs['schema'],'b3_bundle_caller_observation_v1','FAILED_FINAL_CALLER_SCHEMA')
    for x in (delivery,obs):equal(x['manifest_binding'],binding,'FAILED_FINAL_MANIFEST');equal(x['run_id'],m['run_id'],'FAILED_FINAL_RUN')
    equal(delivery['final_sha256'],V.file_digest(out/'final.json'),'FAILED_FINAL_PHYSICAL_HASH')
    equal(delivery['final_write_flush_close_observed'],False,'FAILED_FINAL_NOT_OBSERVED')
    V.require(bool(delivery['post_final_errors']),'FAILED_FINAL_ERROR_WITNESS')
    equal(obs['delivery_sha256'],V.file_digest(out/'delivery.json'),'FAILED_FINAL_CALLER_HASH')
    equal(obs['forbidden_imports'],[],'FAILED_FINAL_SCIENCE_IMPORT')
    return dict(final_sha256=delivery['final_sha256'],delivery_sha256=obs['delivery_sha256'],full_RAM_or_transition_proof=False)

def success_documents(root,out,m,c,r,identity,claim,trace):
    binding=identity['manifest_digest'];run=m['run_id'];total=identity['planned']
    for name,n in [('final.json',4*1024*1024),('delivery.json',4096)]:equal((out/name).stat().st_size,n,'FINAL_SLOT_SIZE:'+name)
    equal((root/'external_observation.json').stat().st_size,4096,'CALLER_SLOT_SIZE')
    summary=read(out/'summary.json');receipt=read(out/'receipt.json');marker=read(out/'marker.json');diag=read(out/'diagnostic.json')
    for name,obj,schema in [('summary',summary,'b3_bundle_summary_v1'),('receipt',receipt,'b3_global_bundle_receipt_v1'),('marker',marker,'b3_bundle_post_close_marker_v1'),('diagnostic',diag,'b3_bundle_diagnostic_v1')]:
        equal(obj['schema'],schema,'POST_CLOSE_SCHEMA:'+name);equal(obj['run_id'],run,'POST_CLOSE_RUN:'+name)
    for obj in (summary,receipt,marker):equal(obj['manifest_binding'],binding,'POST_CLOSE_MANIFEST_BINDING')
    for obj in (summary,receipt):equal(obj['candidate_binding'],identity['candidate_digest'],'POST_CLOSE_CANDIDATE_BINDING')
    equal(summary['conditions'],r['conditions'],'SUMMARY_CONDITIONS')
    for k,v in [('planned',total),('reserved',r['reserved']),('started',r['started_observed']),('states',r['states'])]:equal(summary[k],v,'SUMMARY_COUNTER:'+k)
    scan(out/'operations.jsonl',summary['trace_prefix'],run)
    equal(diag['complete'],False,'DIAGNOSTIC_IS_NOT_SUCCESS');equal(diag['final_acceptance'],False,'DIAGNOSTIC_IS_NOT_ACCEPTANCE')
    equal(diag['reserved'],r['reserved'],'DIAGNOSTIC_RESERVED')
    equal({s:diag['states'].get(s,0)for s in r['states']},r['states'],'DIAGNOSTIC_STATE')
    for k in ('planned','reserved','completed'):equal(receipt[k],total,'GLOBAL_RECEIPT_COUNTER:'+k)
    equal(receipt['conditions'],['S','P1'],'GLOBAL_RECEIPT_CONDITIONS')
    equal(receipt['summary_sha256'],V.file_digest(out/'summary.json'),'RECEIPT_SUMMARY_HASH')
    equal(receipt['state_sha256'],V.file_digest(out/'trial_states.bin'),'RECEIPT_STATE_HASH')
    equal(receipt['trace_prefix'],r['evidence_v2']['trace_prefix'],'RECEIPT_FINAL_PREFIX')
    for k,v in [('storage_closed_before_receipt',True),('claim_status','STOPPED'),('production_permission',False)]:equal(receipt[k],v,'RECEIPT_POST_CLOSE:'+k)
    equal(marker['receipt_sha256'],V.file_digest(out/'receipt.json'),'MARKER_RECEIPT_HASH');equal(marker['receipt_closed_before_marker'],True,'MARKER_AFTER_RECEIPT_CLOSE')
    equal(read(out/'FAKE_ONLY_snapshot.json'),claim['snapshot'],'FINAL_FAKE_SNAPSHOT')
    return dict(full_success_join=True,marker_alone_sufficient=False,physical_durability_or_power_failure_proven=False,
                caller_observation_own_close_independently_proven=False)
