"""Read-only bundle consumer: never imports bridge, scheduler, stores or science.

Caller supplies a trusted candidate; an embedded candidate is not its own anchor.
Authored/tested by the same primary agent, not a separate reviewer.
"""
from collections import Counter
from pathlib import Path
from hashlib import sha256
import json,sqlite3
from .. import g6_w2c_b3_offline_result as V
from .. import g6_w2c_b3_parallel_verifier_core as G
from .. import g6_w2c_b3_parallel_offline_adapter as A
from .. import g6_w2c_b3_offline_cost as C
from .. import g6_w2c_b3_bundle_trace_v1 as T

CLOSED=dict(content_adopted=False,run_ready=False,science_release=None,scientific_adoption=None,contract_grants_execution=False,mandatory_STOP=True,artificial_only=True,retry=0,resume=False)

def _identity(value):return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
def _read(path):return A.safe_values(V.plain(Path(path).read_bytes()))
def _sha(path):return sha256(Path(path).read_bytes()).hexdigest()
def _same(a,b):return type(a)is type(b) and (set(a)==set(b)and all(_same(a[k],b[k])for k in a)if type(a)is dict else len(a)==len(b)and all(_same(x,y)for x,y in zip(a,b))if type(a)is list else a==b)

def plan(n):
    tasks=[];offset=0
    for j,mode in enumerate(('S','P1')):
        one,total=G.plan(n)
        for t in one:tasks.append(dict(t,key=mode+'__'+t['key'],condition=mode,phase_index=t['phase_index']+3*j,offset=t['offset']+offset))
        offset+=total
    return tasks,offset

def identity(c,m,trusted,source_root):
    V.require(_same(c,trusted),'TRUSTED_CANDIDATE_BINDING')
    V.require(c['schema']=='b3_closed_real_adapter_candidate_v1'and m['schema']=='b3_closed_real_adapter_manifest_v1','BUNDLE_SCHEMA')
    for obj in (c,m):
        for k,v in CLOSED.items():V.require(_same(obj[k],v),'CLOSED_FLAG:'+k)
    V.require(c['production_gate']=='NO_IMPLEMENTED_SCIENCE_DISPATCH' and c['fixed_order']==['S','P1'],'PRODUCTION_GATE')
    for p,h in c['source_pins'].items():V.require(_sha(V.safe_path(source_root,p))==h,'SOURCE_BYTES:'+p)
    for p,h in c['scientific_reference']['input_pins'].items():V.require(_sha(V.safe_path(source_root,p))==h,'INPUT_BYTES:'+p)
    for row in c['python']['libraries'].values():V.require(_sha(row['path'])==row['sha256'],'RUNTIME_LIBRARY_BYTES')
    V.require(_sha(c['python']['executable'])==c['python']['executable_sha256'],'RUNTIME_EXECUTABLE_BYTES')
    V.require(m['candidate_binding']==_identity(c) and c['source_release_candidate']==_identity(c['source_pins'])and m['source_release_candidate']==c['source_release_candidate'],'GLOBAL_SOURCE_CONTRACT_ID')
    p=c['proposal'];V.require(p['profile_n']==256 and p['cells']==16 and p['per_condition']==dict(correctness=256,baseline=4096,instrumented=4096,total=8448)and p['total_calls']==2*(16*16+16*256+16*256)==16896,'STATIC_16896_ARITHMETIC')
    V.require(p['result_bytes']==1<<36 and p['log_bytes']==1<<28 and p['codec']=='fused' and p['observer']=='cached_all_call_return'and p['mode_is_seed_domain']is False,'FIXED_COMPARISON')
    V.require(m['seed_namespace']=='ARTIFICIAL_REAL_ADAPTER_V1'and m['codec']=='fused'and m['observer']==p['observer']and m['workers']==8 and m['start_method']=='spawn','ARTIFICIAL_PREFIX_CONDITION')
    V.require(m['global_scope'].startswith('ARTIFICIAL_ONLY/')and m['global_claim']==_identity(['ARTIFICIAL_ONLY',m['global_scope'],_identity(c)])and m['run_id']==_identity(['RUN',m['global_scope'],_identity(c)]),'GLOBAL_CLAIM_RUN_ID')
    V.require(m['scientific_input']==_identity(c['scientific_reference']['scientific_reference']['input']),'SCIENTIFIC_INPUT_ID')
    tasks,total=plan(m['profile_n']);V.require(16<=m['profile_n']<=256 and m['tasks']==tasks and m['planned']==total and m['static_scientific_candidate_calls']==16896,'GLOBAL_PLAN')
    V.require(m['modes']in (['S'],['P1'],['S','P1'])and 0<=m['call_limit']<=total<=16896,'GLOBAL_CALL_PLAN')
    for ch in ('result','log'):V.require(type(m['byte_limits'][ch])is int and 0<m['byte_limits'][ch]<=p[ch+'_bytes'],'GLOBAL_BYTE_LIMIT')
    return tasks,total,_identity(m)

def _claim(root,m,report):
    p=root/'FAKE_ONLY.sqlite3';V.require(not p.is_symlink(),'FAKE_STORE_SYMLINK')
    with sqlite3.connect(p.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)as db:
        V.require(db.execute('PRAGMA quick_check').fetchone()[0]=='ok','FAKE_STORE_INTEGRITY')
        rows=db.execute('SELECT scope,claim,binding,status,spent FROM claims').fetchall();history=db.execute('SELECT seq,scope,event,binding FROM history ORDER BY seq').fetchall()
    V.require(len(rows)==1 and rows[0][:3]==(m['global_scope'],m['global_claim'],_identity(m))and rows[0][4]==1 and rows[0][3]in ('SPENT','STOPPED'),'ONE_GLOBAL_SPENT_CLAIM')
    V.require(history and history[0][1:]==(m['global_scope'],'CLAIM_COMMITTED',_identity(m))and len(history)==(2 if rows[0][3]=='STOPPED'else 1),'FAKE_CLAIM_HISTORY')
    if rows[0][3]=='STOPPED':V.require(history[1][1:]==(m['global_scope'],'STOP_COMMITTED',_identity(m)),'FAKE_STOP_HISTORY')
    snapshot=dict(schema='b3_FAKE_ONLY_snapshot_v1',claim=list(rows[0]),history=[list(x)for x in history],real_authority=False)
    if report is not None:V.require(_same(snapshot,report['fake_claim']),'FAKE_REPORT_BINDING')
    return snapshot

def _control(out,tasks,r,gaps):
    ev=r['evidence_v2'];rows=[]
    for line in (out/'control.jsonl').read_bytes().splitlines(keepends=True):
        if not line.endswith(b'\n'):V.require(not ev['journal_complete'],'CONTROL_PARTIAL_COMPLETE');break
        rows.append(A.safe_values(V.plain(line)))
    V.require(rows and rows[0]['event']=='PLAN' and rows[0]['tasks']==tasks and rows[0]['planned']==r['planned'],'CONTROL_GLOBAL_PLAN')
    starts={};ends={};arrivals=[];begun=[];ended={};open_phase=None;aborted=False
    for x in rows[1:]:
        event=x['event']
        if event=='WORKER_START':V.require(x['key']not in starts,'DUPLICATE_WORKER_START');starts[x['key']]=x['pid']
        elif event=='WORKER_END':
            row=x['report'];key=row['task']['key'];V.require(key not in ends,'CONTROL_DUPLICATE_WORKER_END')
            if key not in starts:
                V.require(aborted and bool(r['failures']),'CONTROL_WORKER_START_MISSING');gaps.append('Dispatch aborted before WORKER_START was saved')
            else:V.require(row['pid']==starts[key],'CONTROL_WORKER_ID')
            ends[key]=row
        elif event=='FAILURE_OBSERVED':
            V.require(x['parent_arrival']==len(arrivals)+1 and _same({k:v for k,v in x.items()if k!='event'},r['failures'][len(arrivals)]),'PARENT_ARRIVAL_BINDING');arrivals.append(x)
        elif event=='CONDITION_BEGIN':V.require(x['condition']not in begun and (not begun or ended[begun[-1]]is True),'CONTROL_CONDITION_ORDER');begun.append(x['condition'])
        elif event=='CONDITION_END':ended[x['condition']]=x['complete']
        elif event=='PHASE_BEGIN':V.require(open_phase is None,'CONTROL_PHASE_OVERLAP');open_phase=x['phase']
        elif event=='PHASE_ABORT':V.require(open_phase==x['phase'],'CONTROL_PHASE_ABORT');open_phase=None;aborted=True
        elif event=='PHASE_END':V.require(open_phase==x['phase'],'CONTROL_PHASE_END');open_phase=None
        else:raise V.Invalid('CONTROL_EVENT:'+event)
    bykey={x['task']['key']:x for x in r['pairs']}
    for key,x in ends.items():V.require(key in bykey and _same(x,bykey[key]),'CONTROL_PAIR_REPORT')
    if ev['journal_complete']:
        V.require(list(ends)==r['completion_order'] and len(arrivals)==len(r['failures']),'CONTROL_FINAL_REPORT')
    else:gaps.append('Journal incomplete; RAM transition declarations are not physical history')
    return dict(started=len(starts),ended=len(ends),condition_begin=begun,condition_end=ended)

def _cost(r,delivery,pairs):
    host=r['host_outer'];parent=V.nonnegative(host['parent_cpu_ns']);worker=V.nonnegative(host['worker_cpu_ns']);wall=V.nonnegative(host['wall_ns'])
    V.require(r['total_cpu_ns']==r['parent_cpu_ns']+r['worker_cpu_ns'],'CPU_DOUBLE_ADD')
    V.require(parent>=r['parent_cpu_ns'] and worker>=r['worker_cpu_ns'] and wall>=r['wall_ns'],'HOST_COST_BOUNDARY')
    known=0;result=[]
    for row in r['pairs']:
        if row['cpu_ns']is None:continue
        t=row['task'];total=[row['cpu_ns'],row['wall_ns']];p=C.partition(row['coarse'],row['reconciliation']);V.require(p['outer_total_ns']==total,'COARSE_OUTER')
        fine=None
        if row['fine']is not None:fine=C.partition(row['fine'],row['fine_reconciliation']);V.require(fine['outer_total_ns']==total,'FINE_OUTER')
        V.require((t['pass_name']=='instrumented')==(fine is not None),'OBSERVER_PASS')
        V.require(all(0<=row['control_rpc'].get(k,0)<=row[k]for k in ('cpu_ns','wall_ns')),'IPC_CONTAINED')
        known+=row['cpu_ns'];result.append(dict(key=t['key'],condition=t['condition'],total=total,setup=C.subtotal(row['coarse'],C.SETUP),online=C.subtotal(row['coarse'],C.ONLINE),output=C.subtotal(row['coarse'],C.OUTPUT),coarse=p,fine_alternative=fine,IPC_contained=row['control_rpc'],memory=row['memory'],io=row['io']))
    V.require(known<=worker,'PAIR_CPU_EXCEEDS_ALL_WORKER_CPU')
    V.require(sum(x['parent_cpu_ns']for x in r['conditions'])==r['condition_parent_cpu_ns']<=parent and sum(x['worker_cpu_ns']for x in r['conditions'])==r['condition_worker_cpu_ns']<=worker,'CONDITION_CPU_ALLOCATION')
    V.require(sum(x['wall_ns']for x in r['conditions'])<=wall,'CONDITION_WALL_SCOPE')
    if delivery:
        V.require(delivery['outer_parent_cpu_ns']>=parent and delivery['outer_wall_ns']>=wall and delivery['worker_cpu_ns']==worker,'FINAL_COST_WINDOW')
    return dict(parent_cpu_ns=parent,all_worker_cpu_ns=worker,total_cpu_ns=parent+worker,global_wall_ns=wall,
        outer_through_final_slot=delivery,conditions=r['conditions'],shared_parent_unallocated_ns=parent-r['condition_parent_cpu_ns'],pairs=result,
        observer_isolated_overhead_ns=None,parent_individual_IO_exclusive_cost_ns=None,unknown_cost_is_zero=False,coarse_and_fine_added=False,pair_or_IPC_added_to_outer=False,
        fixed_S_then_P1_order_confound=True,both_fused_gain_unidentifiable=True,scientific_CPU_extrapolation=False)

def verify_storage(root,trusted,source_root=None):
    root=Path(root);out=root/'output';source_root=Path(source_root or trusted['runtime_root'])
    result=dict(schema='b3_bundle_offline_acceptance_v1',status='INCOMPLETE_EVIDENCE',issues=[],evidence_gaps=[],scientific_GO=False,mandatory_STOP=True)
    try:
        c=_read(out/'candidate.json');m=_read(out/'manifest.json');tasks,total,binding=identity(c,m,trusted,source_root)
        if not(out/'final.json').exists():
            result.update(admission=_claim(root,m,None),evidence_gaps=['Final host evidence missing; no complete RAM/cost/close proof'],calls=None);return result
        r=_read(out/'final.json');ev=r['evidence_v2'];gaps=result['evidence_gaps']
        V.require(r['schema']=='b3_global_bundle_result_v1'and r['manifest_binding']==binding and r['candidate_binding']==_identity(c)and r['run_id']==m['run_id'],'FINAL_GLOBAL_IDENTITY')
        V.require(r['source_release_candidate']==c['source_release_candidate']and r['production_permission']is False and not r['science_imports'],'FINAL_PRODUCTION_GATE')
        V.require(r['retry']==0 and r['resume']is False and r['mandatory_STOP']is True and r['artificial_only']is True,'FINAL_STOP_FLAGS')
        claim=_claim(root,m,r)
        delivery=_read(out/'delivery.json')if(out/'delivery.json').exists()else None
        if delivery:
            V.require(delivery['schema']=='b3_bundle_external_delivery_v1'and delivery['run_id']==m['run_id']and delivery['manifest_binding']==binding and delivery['final_sha256']==_sha(out/'final.json'),'EXTERNAL_DELIVERY_BINDING')
            if not delivery['final_write_flush_close_observed']or delivery['post_final_errors']:gaps.append('Final witness close not successfully observed')
        else:gaps.append('External post-final-close delivery absent')
        observation=_read(root/'external_observation.json')if(root/'external_observation.json').exists()else None
        if observation:
            V.require(observation['schema']=='b3_bundle_caller_observation_v1'and observation['manifest_binding']==binding and observation['run_id']==m['run_id'],'CALLER_OBSERVATION_ID')
            V.require(observation['delivery_sha256']==(_sha(out/'delivery.json')if(out/'delivery.json').exists()else None),'CALLER_DELIVERY_HASH')
            if not observation['bridge_returned_normally']:gaps.append('Caller observed bridge/delivery persistence exception')
            V.require(not observation['forbidden_imports'],'CALLER_SCIENCE_IMPORT')
        else:gaps.append('Caller observation after delivery close absent')
        V.require(r['planned']==total and set(r['states'])==set(G.STATES)and sum(r['states'].values())==total and r['reserved']==total-r['states']['NOT_STARTED']<=m['call_limit'],'RAM_GLOBAL_CALL_ACCOUNTING')
        V.require(r['states']['COMPLETED']<=r['started_observed']<=r['reserved']-r['states']['CANCELLED_BEFORE_START'],'START_RESERVED_BOUNDARY')
        V.require(set(ev['next_indices'])=={t['key']for t in tasks}and sum(ev['next_indices'].values())==r['reserved']and not ev['current_tickets']and not ev['pair_next'],'CURSOR_TICKET_BINDING')
        state=(out/'trial_states.bin').read_bytes();V.require(len(state)==total,'STATEFILE_LENGTH');snapshot=None
        if ev['statefile_complete']:
            V.require(not ev['state_errors'],'STATE_FLAG_CONFLICT');snapshot=G.state_counts(out/'trial_states.bin',tasks)
            V.require(snapshot['counts']==r['states']and snapshot['file_sha256']==ev['shadow_sha256'],'RAM_PHYSICAL_WITNESS')
            for t in tasks:V.require(ev['next_indices'][t['key']]==2*t['n']-snapshot['pairs'][t['key']]['NOT_STARTED'],'PHYSICAL_CURSOR')
        else:gaps.append('RAM intent is not fully witnessed by physical state')
        rows,trace=T.read_prefix(out/'operations.jsonl',ev['trace_prefix'],m['run_id'],ev['trace_complete'])
        history=T.inspect(rows,tasks,m,r,state)
        if not ev['trace_complete']:gaps.append('Only trusted prefix is evidence; tail/unsaved intent remain unknown')
        if ev['storage_errors']or ev['trace_errors']or ev['state_errors']:gaps.append('Storage/flush/close failure retained')
        control=_control(out,tasks,r,gaps)
        V.require(r['representative_failure']==(r['failures'][0]if r['failures']else None),'REPRESENTATIVE_FAILURE')
        for i,f in enumerate(r['failures']):V.require(f['parent_arrival']==i+1,'PARENT_ARRIVAL_ORDER')
        bykey={x['task']['key']:x for x in r['pairs']};V.require(len(bykey)==len(r['pairs'])and r['contract_order']==[t['key']for t in tasks]and list(bykey)==[t['key']for t in tasks if t['key']in bykey],'PAIR_ID_ORDER')
        pairs=[]
        for t in tasks:
            row=bykey.get(t['key']);path=out/(t['key']+'.jsonl')
            if row:V.require(row['task']==t,'PAIR_GLOBAL_OFFSET_IDENTITY')
            if not path.exists():
                if snapshot:V.require(snapshot['pairs'][t['key']]['COMPLETED']==0,'MISSING_COMPLETED_PAIR')
                continue
            pc=A.verify_worker(path,t,row)
            V.require(pc['status']not in ('INTEGRITY_MISMATCH','UNSUPPORTED_SCHEMA'),'PAIR_F01:'+','.join(pc['issues']))
            if snapshot:
                n=snapshot['pairs'][t['key']]['COMPLETED'];V.require(n<=pc['visible_trial_rows']and(pc['engine_committed_trials']is None or n<=pc['engine_committed_trials']),'COMPLETE_REQUIRES_F01')
            pairs.append(dict(pc,task_key=t['key']))
        # Every output file belongs to one global budget; reject alias/extra paths.
        logs={'control.jsonl','operations.jsonl','trial_states.bin'}
        allowed=logs|{t['key']+'.jsonl'for t in tasks}|{'candidate.json','manifest.json','S_receipt.json','P1_receipt.json','summary.json','diagnostic.json','FAKE_ONLY_snapshot.json','receipt.json','marker.json','final.json','delivery.json'}
        physical=Counter();inodes=set()
        for p in out.iterdir():
            V.require(p.name in allowed and p.is_file()and not p.is_symlink(),'OUTPUT_NAMESPACE')
            ino=(p.stat().st_dev,p.stat().st_ino);V.require(ino not in inodes,'OUTPUT_ALIAS');inodes.add(ino)
            physical['log'if p.name in logs else'result']+=p.stat().st_size
        if (root/'external_observation.json').exists():physical['result']+=(root/'external_observation.json').stat().st_size
        for name,v in r['physical_files_before_final'].items():V.require((out/name).stat().st_size==v['bytes']and _sha(out/name)==v['sha256'],'FINAL_FILE_INVENTORY:'+name)
        postr=Counter();posta=Counter()
        for x in r['post_close_credits']:
            ch=x['channel'];V.require(ch in ('result','log'),'POST_CLOSE_CHANNEL')
            if x['op']=='WRITE_RESERVE':postr[ch]+=V.nonnegative(x['amount'])
            else:V.require(x['op']=='WRITE_ACK','POST_CLOSE_OP');posta[ch]+=V.nonnegative(x['amount']);V.require(posta[ch]<=postr[ch],'POST_CLOSE_ACK')
        budget={};by_condition={k:dict(reserved=Counter(),observed=Counter())for k in ('SHARED','S','P1')}
        for line,row in zip((out/'operations.jsonl').read_bytes()[:trace['trusted_bytes']].splitlines(keepends=True),rows):
            bucket=by_condition[row['condition']];bucket['reserved']['log']+=len(line);bucket['observed']['log']+=len(line)
            if row['op']=='WRITE_RESERVE':bucket['reserved'][row['channel']]+=row['amount']
            if row['op']=='WRITE_ACK':bucket['observed'][row['channel']]+=row['amount']
        for ch in ('result','log'):
            by_condition['SHARED']['reserved'][ch]+=postr[ch];by_condition['SHARED']['observed'][ch]+=posta[ch]+((4*1024*1024+8192)if ch=='result'else 0)
        for ch in ('result','log'):
            b=r['output'];credit=V.nonnegative(b['reserved'].get(ch,0));ack=V.nonnegative(b['observed'].get(ch,0));lim=b['limits'][ch]
            V.require(lim==m['byte_limits'][ch]and ack<=credit<=lim,'GLOBAL_BYTE_CAP')
            V.require(b['unknown_credit'][ch]==credit-ack,'UNKNOWN_CREDIT_NONREFUND')
            projected=(4*1024*1024+8192)if ch=='result'else 0
            # Projection is not evidence without a full final slot and delivery.
            if delivery and delivery['final_write_flush_close_observed'] and observation:V.require(ack<=physical[ch]<=credit,'ACK_PHYSICAL_GLOBAL_BUDGET')
            else:V.require(physical[ch]<=credit,'PARTIAL_PHYSICAL_CAP')
            selfbytes=trace['bytes']if ch=='log'else 0
            hr=history['reserved_bytes'].get(ch,0)+selfbytes+postr[ch];ha=history['ack_bytes'].get(ch,0)+selfbytes+posta[ch]+projected
            if ev['trace_complete']:V.require(hr==credit and ha==ack,'TRACE_POSTCLOSE_GLOBAL_ACCOUNTING:'+ch)
            else:V.require(hr<=credit and ha<=ack,'TRUSTED_PREFIX_BYTE_LOWER_BOUND')
            budget[ch]=dict(reserved=credit,observed_or_self_projected=ack,physical=physical[ch],unknown_credit=credit-ack,refunded=False)
        conditions=r['conditions'];V.require([x['condition']for x in conditions]==m['modes'][:len(conditions)],'CONDITION_STATUS_ORDER')
        for x in conditions:
            selected=[t for t in tasks if t['condition']==x['condition']];expected=256+32*m['profile_n'];V.require(x['planned']==expected,'CONDITION_CALL_ARITHMETIC')
            if snapshot:
                V.require(x['reserved']==sum(2*t['n']-snapshot['pairs'][t['key']]['NOT_STARTED']for t in selected)and x['completed']==sum(snapshot['pairs'][t['key']]['COMPLETED']for t in selected),'CONDITION_STATE_COUNTERS')
            if x['complete']:
                cr=_read(out/(x['condition']+'_receipt.json'));V.require(cr['manifest_binding']==binding and cr['run_id']==m['run_id']and all(_same(cr[k],v)for k,v in x.items()),'CONDITION_RECEIPT_BINDING')
        if r['admission_stop_errors']or claim['claim'][3]!='STOPPED':gaps.append('Fake STOP not persisted; spent reservation remains')
        if r['post_close_errors']:gaps.append('Post-close receipt/marker save failed')
        complete=r['complete'] and bool(delivery and delivery['final_write_flush_close_observed'] and not delivery['post_final_errors'] and observation and observation['bridge_returned_normally'])
        if complete:
            V.require(not gaps and m['modes']==['S','P1']and len(conditions)==2 and all(x['complete']for x in conditions)and r['reserved']==r['started_observed']==r['states']['COMPLETED']==total and len(pairs)==48 and not r['failures'],'FALSE_BUNDLE_COMPLETION')
            V.require(all(x['reserved']==x['physical']==x['observed_or_self_projected']for x in budget.values()),'SUCCESS_WITH_UNKNOWN_CREDIT')
            V.require((out/'final.json').stat().st_size==4*1024*1024 and(out/'delivery.json').stat().st_size==4096 and(root/'external_observation.json').stat().st_size==4096,'FINAL_SLOT_SIZE')
            summary=_read(out/'summary.json');receipt=_read(out/'receipt.json');marker=_read(out/'marker.json');diag=_read(out/'diagnostic.json')
            V.require(summary['schema']=='b3_bundle_summary_v1'and summary['run_id']==m['run_id']and summary['manifest_binding']==binding and summary['candidate_binding']==_identity(c)and summary['conditions']==conditions,'SUMMARY_IDENTITY')
            for k,v in [('planned',total),('reserved',r['reserved']),('started',r['started_observed']),('states',r['states'])]:V.require(summary[k]==v,'SUMMARY_COUNTER:'+k)
            T.read_prefix(out/'operations.jsonl',summary['trace_prefix'],m['run_id'],False)
            V.require(diag['schema']=='b3_bundle_diagnostic_v1'and diag['complete']is False and diag['final_acceptance']is False,'DIAGNOSTIC_IS_NOT_RECEIPT')
            V.require(receipt['schema']=='b3_global_bundle_receipt_v1'and receipt['manifest_binding']==binding and receipt['candidate_binding']==_identity(c)and receipt['run_id']==m['run_id']and receipt['conditions']==['S','P1']and receipt['planned']==receipt['reserved']==receipt['completed']==total,'GLOBAL_RECEIPT_ID_COUNTER')
            V.require(receipt['summary_sha256']==_sha(out/'summary.json')and receipt['state_sha256']==_sha(out/'trial_states.bin')and receipt['trace_prefix']==ev['trace_prefix'],'RECEIPT_FINAL_PHYSICAL_BINDING')
            V.require(receipt['storage_closed_before_receipt']is True and receipt['claim_status']=='STOPPED'and receipt['production_permission']is False,'POST_CLOSE_RECEIPT_GATE')
            V.require(marker['schema']=='b3_bundle_post_close_marker_v1'and marker['run_id']==m['run_id']and marker['manifest_binding']==binding and marker['receipt_sha256']==_sha(out/'receipt.json')and marker['receipt_closed_before_marker']is True,'MARKER_RECEIPT_JOIN')
            V.require(r['parent_messages']['RESERVE']==total and r['parent_messages']['ENTER']==total and not r['parent_messages'].get('WRITE_ACK',0),'COALESCED_IPC_COUNTS')
            # Verify full A/B/pass/mode equivalence and the correctness prefix.
            for t in tasks:
                path=out/(t['key']+'.jsonl')
                for axis in ('X','Y'):
                    if t['layer']=='B':V.require(V.cross_pair(out/(t['key'][:-1]+'A.jsonl'),path,axis)['status']=='PASS','AB_PREFIX')
                    if t['phase_index']%3:V.require(V.cross_pair(out/f"{t['condition']}__correctness_baseline_{t['arm']}_{t['layer']}.jsonl",path,axis,prefix_only=True)['status']=='PASS','CORRECTNESS_PREFIX')
                    if t['phase_index']%3==2:V.require(V.cross_pair(out/f"{t['condition']}__profile_baseline_{t['arm']}_{t['layer']}.jsonl",path,axis)['status']=='PASS','PASS_PREFIX')
                    if t['condition']=='P1':V.require(V.cross_pair(out/('S__'+t['key'].split('__',1)[1]+'.jsonl'),path,axis)['status']=='PASS','SP1_PREFIX')
            V.require(_read(out/'FAKE_ONLY_snapshot.json')==claim,'FINAL_FAKE_SNAPSHOT')
        cost=_cost(r,delivery,pairs)
        if observation:
            outer=observation['host_outer'];V.require(outer['parent_cpu_ns']>=cost['parent_cpu_ns'] and outer['wall_ns']>=cost['global_wall_ns'] and outer['worker_cpu_ns']==cost['all_worker_cpu_ns'],'CALLER_OUTER_COST_BOUNDARY')
            cost.update(global_wall_ns=outer['wall_ns'],parent_cpu_ns=outer['parent_cpu_ns'],total_cpu_ns=outer['parent_cpu_ns']+outer['worker_cpu_ns'],boundary='caller before bridge through delivery close; observation own persistence is postprocessing')
        status='VERIFIED_ARTIFICIAL_BUNDLE_COMPLETE'if complete else'INCOMPLETE_EVIDENCE'if gaps else'VERIFIED_ARTIFICIAL_CONDITION_ONLY'if m['modes']!=['S','P1']and conditions and all(x['complete']for x in conditions)else'TECHNICAL_FAILURE'
        result.update(status=status,calls=dict(planned=total,reserved=r['reserved'],started=r['started_observed'],completed=r['states']['COMPLETED'],unknown=r['states']['TERMINATED_WITH_UNKNOWN_RESULT'],cancelled=r['states']['CANCELLED_BEFORE_START']),state_physically_exact=snapshot is not None,history=history,trace=trace,control=control,output_budget=budget,condition_byte_attribution=by_condition,conditions=conditions,admission=claim,pairs=pairs,cost=cost)
    except (V.Invalid,V.Unsupported,KeyError,ValueError,TypeError,IndexError,OSError,sqlite3.Error)as exc:
        result.update(status='INCOMPLETE_EVIDENCE'if isinstance(exc,FileNotFoundError)else'INTEGRITY_MISMATCH',issues=[str(exc)])
    return result
