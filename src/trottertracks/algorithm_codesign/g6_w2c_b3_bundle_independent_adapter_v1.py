"""Local independent offline acceptance for the fixed Storage V2 global bundle.

Imports only independent consumer primitives. Producer assertions, reported PASS
and test expectations are never inputs to the acceptance decision.
"""
from pathlib import Path
import sqlite3
import json
from . import g6_w2c_b3_offline_result as V
from . import g6_w2c_b3_parallel_verifier_core as G
from . import g6_w2c_b3_parallel_offline_adapter as A
from .g6_w2c_b3_bundle_independent_identity_v1 import read,equal,SLOTS
from . import g6_w2c_b3_bundle_independent_trace_v1 as T
from . import g6_w2c_b3_bundle_independent_storage_v1 as S
from . import g6_w2c_b3_bundle_independent_cost_v1 as C

def journal(out,tasks,m,r,gaps):
    lookup={t['key']:t for t in tasks};starts={};ends={};arrivals=[];begun=[];ended={};phase=None;last=None;aborted=False;rows=0
    with (out/'control.jsonl').open('rb')as f:
        for line in f:
            if not line.endswith(b'\n'):
                V.require(not r['evidence_v2']['journal_complete'],'COMPLETE_JOURNAL_PARTIAL_TAIL');break
            x=A.safe_values(V.plain(line));event=x['event'];rows+=1
            if rows==1:
                equal(event,'PLAN','JOURNAL_FIRST_PLAN');equal(x['schema'],'b3_parallel_control_v3','JOURNAL_SCHEMA')
                equal(x['tasks'],tasks,'JOURNAL_TASKS');equal(x['planned'],m['planned'],'JOURNAL_PLANNED')
                equal(x['state_codes'],{s:i for i,s in enumerate(G.STATES)},'JOURNAL_CODES')
                equal(x['retry'],0,'JOURNAL_RETRY');equal(x['resume'],False,'JOURNAL_RESUME')
                for k,v in dict(mode='S',workers=8,profile_n=m['profile_n'],codec='fused',call_limit=m['call_limit'],result_bytes=m['byte_limits']['result'],log_bytes=m['byte_limits']['log']).items():equal(x['settings'][k],v,'JOURNAL_SETTINGS:'+k)
                continue
            if event=='CONDITION_BEGIN':
                V.require(len(begun)<len(m['modes'])and (not begun or ended.get(begun[-1])is True),'JOURNAL_P1_AFTER_S_FAILURE')
                equal(x['condition'],m['modes'][len(begun)],'JOURNAL_CONDITION_ORDER');begun.append(x['condition'])
            elif event=='CONDITION_END':
                V.require(begun and x['condition']==begun[-1]and x['condition']not in ended,'JOURNAL_CONDITION_END')
                V.require(type(x['complete'])is bool,'JOURNAL_CONDITION_COMPLETE_TYPE');ended[x['condition']]=x['complete']
            elif event=='PHASE_BEGIN':
                V.require(phase is None and begun,'JOURNAL_PHASE_BEGIN')
                want=(0 if begun[-1]=='S'else 3)if last is None else last+1
                equal(x['phase'],want,'JOURNAL_SIX_PHASE_BARRIER');phase=last=want
            elif event in ('PHASE_END','PHASE_ABORT'):
                equal(x['phase'],phase,'JOURNAL_PHASE_END');V.require(phase is not None,'JOURNAL_PHASE_NOT_OPEN');phase=None
                if event=='PHASE_ABORT':aborted=True
            elif event=='WORKER_START':
                key=x['key'];V.require(key in lookup and key not in starts and phase==lookup[key]['phase_index'],'JOURNAL_WORKER_START_IDENTITY')
                starts[key]=x['pid']
            elif event=='WORKER_END':
                row=x['report'];key=row['task']['key'];V.require(key in lookup and key not in ends,'JOURNAL_WORKER_END_DUPLICATE')
                equal(row['task'],lookup[key],'JOURNAL_TASK_OFFSET')
                if key not in starts:
                    V.require(aborted and bool(r['failures']),'JOURNAL_MISSING_WORKER_START')
                    gaps.append('Dispatch aborted before WORKER_START was physically saved')
                else:equal(row['pid'],starts[key],'JOURNAL_WORKER_PID')
                ends[key]=row
            elif event=='FAILURE_OBSERVED':
                i=len(arrivals);equal(x['parent_arrival'],i+1,'JOURNAL_FAILURE_ORDER')
                V.require(i<len(r['failures']),'JOURNAL_EXTRA_FAILURE')
                equal({k:v for k,v in x.items()if k!='event'},r['failures'][i],'JOURNAL_FAILURE_BINDING');arrivals.append(x)
            else:raise V.Unsupported('JOURNAL_EVENT:'+str(event))
    bykey={x['task']['key']:x for x in r['pairs']}
    for k,row in ends.items():V.require(k in bykey,'JOURNAL_EXTRA_PAIR');equal(row,bykey[k],'JOURNAL_PAIR_REPORT')
    if r['evidence_v2']['journal_complete']:
        V.require(rows>0,'MISSING_COMPLETE_JOURNAL');equal(list(ends),r['completion_order'],'JOURNAL_COMPLETION_ORDER');equal(len(arrivals),len(r['failures']),'JOURNAL_FAILURE_COVERAGE')
    else:gaps.append('Journal incomplete; no missing operations are inferred')
    if r['evidence_v2']['stop_journal_saved']:V.require(arrivals,'MISSING_STOP_JOURNAL_WITNESS')
    return dict(rows=rows,started=len(starts),ended=len(ends),conditions_begun=begun,conditions_ended=ended)

def verify(root,reference):
    root=Path(root);out=root/'output';result=dict(schema='b3_global_bundle_independent_acceptance_local_v1',case=root.name,
        status='INCOMPLETE_EVIDENCE',issues=[],evidence_gaps=[],scientific_GO=False,mandatory_STOP=True,
        real_scientific_imports=0,real_authority=False,comparison_eligible=False)
    gaps=result['evidence_gaps']
    try:
        cp=out/'candidate.json'if(out/'candidate.json').exists()else root/'candidate_input.json'
        mp=out/'manifest.json'if(out/'manifest.json').exists()else root/'manifest_input.json'
        c=read(cp);m=read(mp);identity=reference.identity(c,m);tasks=identity['tasks'];total=identity['planned']
        result['identity']={k:v for k,v in identity.items()if k!='tasks'}
        # Input copies bind the caller's request even when output never starts.
        if (root/'candidate_input.json').exists():equal(read(root/'candidate_input.json'),c,'CALLER_CANDIDATE_INPUT')
        if (root/'manifest_input.json').exists():equal(read(root/'manifest_input.json'),m,'CALLER_MANIFEST_INPUT')
        admission=S.claim(root,m);result['admission']=admission
        if not(out/'final.json').exists():
            gaps.append('Final host witness absent; RAM/full transitions/close and costs are not established')
            result.update(calls=None,cost=dict(total_cpu_ns=None,global_wall_ns=None,unknown_cost_is_zero=False,comparison_eligible=False))
            if out.exists():result['partial_physical_files']=[dict(name=p.name,bytes=p.stat().st_size,sha256=V.file_digest(p))for p in sorted(out.iterdir())if p.is_file()and not p.is_symlink()]
            return result
        try:r=A.safe_values(read(out/'final.json'))
        except (json.JSONDecodeError,UnicodeDecodeError):
            witness=S.failed_final_witness(root,out,m)
            gaps.append('Final document is unreadable; external failure observation binds its physical bytes')
            result.update(failed_final_witness=witness,calls=None,cost=dict(total_cpu_ns=None,global_wall_ns=None,unknown_cost_is_zero=False,comparison_eligible=False))
            return result
        ev=r['evidence_v2']
        equal(r['schema'],'b3_global_bundle_result_v1','GLOBAL_RESULT_SCHEMA')
        for k,v in dict(manifest_binding=identity['manifest_digest'],candidate_binding=identity['candidate_digest'],run_id=m['run_id'],source_release_candidate=c['source_release_candidate'],production_permission=False,retry=0,resume=False,mandatory_STOP=True,artificial_only=True).items():equal(r[k],v,'FINAL_IDENTITY:'+k)
        equal(r['science_imports'],[],'FINAL_SCIENCE_IMPORT');V.require(type(r['complete'])is bool,'COMPLETE_TYPE')
        for flag in ('trace_complete','statefile_complete','journal_complete','stop_trace_saved','stop_journal_saved'):V.require(type(ev[flag])is bool,'EVIDENCE_FLAG:'+flag)
        equal(ev['state_authority'],'MONOTONE_RAM_SHADOW_NOT_A_CLAIM_OF_PHYSICAL_DURABILITY','RAM_AUTHORITY_BOUNDARY')
        equal(set(r['states']),set(G.STATES),'RAM_STATE_KEYS')
        for n in r['states'].values():V.nonnegative(n)
        reserved=V.nonnegative(r['reserved']);started=V.nonnegative(r['started_observed'])
        equal(r['planned'],total,'RAM_PLANNED');equal(sum(r['states'].values()),total,'RAM_STATE_TOTAL')
        V.require(reserved==total-r['states']['NOT_STARTED']<=m['call_limit'],'RAM_GLOBAL_RESERVED')
        V.require(r['states']['COMPLETED']<=started<=reserved-r['states']['CANCELLED_BEFORE_START'],'RAM_STARTED_BOUNDARY')
        equal(set(ev['next_indices']),{t['key']for t in tasks},'CURSOR_KEYS')
        for t in tasks:V.require(type(ev['next_indices'][t['key']])is int and 0<=ev['next_indices'][t['key']]<=2*t['n'],'CURSOR_RANGE')
        equal(sum(ev['next_indices'].values()),reserved,'CURSOR_RESERVED_SUM')
        V.require(not ev['current_tickets']and not ev['pair_next'],'FINAL_UNDRAINED_TICKETS')
        equal(r['representative_failure'],r['failures'][0]if r['failures']else None,'FIRST_FAILURE_BINDING')
        for i,f in enumerate(r['failures']):equal(f['parent_arrival'],i+1,'FAILURE_PARENT_ARRIVAL')
        state_raw=(out/'trial_states.bin').read_bytes();equal(len(state_raw),total,'STATEFILE_LENGTH');snapshot=None
        if ev['statefile_complete']:
            equal(ev['state_errors'],[],'STATE_COMPLETE_WITH_ERRORS');snapshot=G.state_counts(out/'trial_states.bin',tasks)
            equal(snapshot['counts'],r['states'],'RAM_PHYSICAL_STATE');equal(snapshot['file_sha256'],ev['shadow_sha256'],'RAM_PHYSICAL_SHA')
            for t in tasks:equal(ev['next_indices'][t['key']],2*t['n']-snapshot['pairs'][t['key']]['NOT_STARTED'],'PHYSICAL_CURSOR')
        else:gaps.append('RAM intent is not fully witnessed by physical state')
        history=T.History(tasks,m,r,state_raw)
        trace=T.scan(out/'operations.jsonl',ev['trace_prefix'],m['run_id'],complete=ev['trace_complete'],visit=history)
        h=history.finish();gaps.extend(h['evidence_gaps'])
        if not ev['trace_complete']:gaps.append('Only the declared trusted prefix is evidence; tail and unsaved operations remain unknown')
        if ev['storage_errors']or ev['state_errors']or ev['trace_errors']or r['log_errors']:gaps.append('Storage/write/flush/close failure retained')
        control=journal(out,tasks,m,r,gaps)
        if ev['trace_complete']and ev['journal_complete']:
            equal(control['conditions_begun'],h['conditions_begun'],'TRACE_JOURNAL_CONDITION_BEGIN')
            equal(control['conditions_ended'],h['conditions_ended'],'TRACE_JOURNAL_CONDITION_END')
        keys=[row['task']['key']for row in r['pairs']];equal(r['contract_order'],[t['key']for t in tasks],'GLOBAL_CONTRACT_ORDER')
        V.require(len(set(keys))==len(keys),'DUPLICATE_PAIR');equal(keys,[t['key']for t in tasks if t['key']in keys],'PAIR_CONTRACT_ORDER')
        equal(set(r['all_pair_states']),set(r['contract_order']),'PAIR_STATE_COVERAGE')
        rows={x['task']['key']:x for x in r['pairs']};pairs=[]
        for t in tasks:
            key=t['key'];row=rows.get(key);path=out/(key+'.jsonl')
            if row:equal(row['task'],t,'PAIR_TASK_GLOBAL_OFFSET')
            equal(r['all_pair_states'][key],row['status']if row else'NOT_STARTED','PAIR_FINAL_STATUS')
            if not path.exists():
                if snapshot and snapshot['pairs'][key]['COMPLETED']:raise FileNotFoundError('Missing completed pair evidence: '+key)
                continue
            p=A.verify_worker(path,t,row)
            V.require(p['status']not in ('INTEGRITY_MISMATCH','UNSUPPORTED_SCHEMA'),'PAIR_F01:'+key+':'+','.join(p['issues']))
            if snapshot:
                n=snapshot['pairs'][key]['COMPLETED'];V.require(n<=p['visible_trial_rows'],'COMPLETED_WITHOUT_PHYSICAL_ROW')
                if p['engine_committed_trials']is not None:V.require(n<=p['engine_committed_trials'],'COMPLETED_WITHOUT_F01_COMMIT')
            p['task_key']=key;pairs.append(p)
        equal(r['fake_claim'],admission['snapshot'],'FAKE_SNAPSHOT_BINDING')
        phys=S.physical(out,root,tasks);delivery,observation=S.post_close(root,out,m,c,r,identity,admission,phys,gaps)
        budget,attribution=S.ledger(m,r,h,trace,phys,delivery,observation)
        conditions=r['conditions'];equal([x['condition']for x in conditions],m['modes'][:len(conditions)],'CONDITION_STATUS_ORDER')
        for x in conditions:
            mode=x['condition'];selected=[t for t in tasks if t['condition']==mode]
            V.require(type(x['complete'])is bool,'CONDITION_COMPLETE_TYPE');equal(x['planned'],256+32*m['profile_n'],'CONDITION_PLANNED')
            for k in ('reserved','started','completed'):V.nonnegative(x[k])
            if snapshot:
                equal(x['reserved'],sum(2*t['n']-snapshot['pairs'][t['key']]['NOT_STARTED']for t in selected),'CONDITION_RESERVED_STATE')
                equal(x['completed'],sum(snapshot['pairs'][t['key']]['COMPLETED']for t in selected),'CONDITION_COMPLETED_STATE')
            if ev['trace_complete']:equal(x['complete'],h['conditions_ended'].get(mode),'TRACE_CONDITION_STATUS')
            if x['complete']:
                cr=read(out/(mode+'_receipt.json'));equal(cr['schema'],'b3_condition_receipt_v1','CONDITION_RECEIPT_SCHEMA')
                equal(cr['run_id'],m['run_id'],'CONDITION_RECEIPT_RUN');equal(cr['manifest_binding'],identity['manifest_digest'],'CONDITION_RECEIPT_MANIFEST')
                for k,v in x.items():equal(cr[k],v,'CONDITION_RECEIPT:'+k)
        cr=sum(x['reserved']for x in conditions);cs=sum(x['started']for x in conditions)
        V.require(cr<=reserved and cs<=started,'CONDITION_GLOBAL_COUNTER_SUM')
        if cr!=reserved or cs!=started:gaps.append('Final condition status window was interrupted; shared counters remain reserved')
        # A storage exception may interrupt before a condition status is returned.
        # In that case final counters can exceed recorded condition windows; the
        # absence must remain a gap, never a new fabricated condition receipt.
        complete=bool(r['complete']and not gaps and delivery and delivery['final_write_flush_close_observed']and observation and observation['bridge_returned_normally'])
        documents=dict(full_success_join=False,marker_alone_sufficient=False)
        if complete:
            equal(m['modes'],['S','P1'],'BUNDLE_REQUIRES_BOTH_MODES');equal(len(conditions),2,'BUNDLE_CONDITION_COUNT')
            V.require(all(x['complete']for x in conditions)and not r['failures'],'BUNDLE_CONDITION_FAILURE')
            V.require(reserved==started==r['states']['COMPLETED']==total and len(pairs)==48,'FALSE_GLOBAL_COMPLETION')
            V.require(all(x['status']=='OK'for x in r['pairs'])and all(p['status']=='VERIFIED_COMPLETE'for p in pairs),'BUNDLE_PAIR_INCOMPLETE')
            equal(h['phase_history'],list(range(6)),'SUCCESS_SIX_PHASES')
            V.require(not h['stopped'],'SUCCESS_AFTER_STOP')
            for b in budget.values():V.require(b['reserved']==b['observed_or_projected']==b['physical_retained']and b['unknown_credit']==0,'SUCCESS_UNKNOWN_BYTE_CREDIT')
            documents=S.success_documents(root,out,m,c,r,identity,admission,trace)
            for t in tasks:
                path=out/(t['key']+'.jsonl')
                for axis in ('X','Y'):
                    others=[]
                    if t['layer']=='B':others.append((out/(t['key'][:-1]+'A.jsonl'),False,'AB_PREFIX'))
                    if t['phase_index']%3:others.append((out/(t['condition']+'__correctness_baseline_'+t['arm']+'_'+t['layer']+'.jsonl'),True,'CORRECTNESS_PREFIX'))
                    if t['phase_index']%3==2:others.append((out/(t['condition']+'__profile_baseline_'+t['arm']+'_'+t['layer']+'.jsonl'),False,'BASELINE_INSTRUMENTED_PREFIX'))
                    if t['condition']=='P1':others.append((out/('S__'+t['key'].split('__',1)[1]+'.jsonl'),False,'SP1_PREFIX'))
                    for other,prefix,code in others:equal(V.cross_pair(other,path,axis,prefix_only=prefix)['status'],'PASS',code)
            equal(r['parent_messages']['RESERVE'],total,'COALESCED_RESERVE_TRIAL_COUNT');equal(r['parent_messages']['ENTER'],total,'COALESCED_ENTER_TRIAL_COUNT')
            equal(r['parent_messages'].get('WRITE_ACK',0),0,'COALESCED_ACK_RPC_COUNT')
        condition_only=bool(not gaps and m['modes']!=['S','P1']and conditions and all(x['complete']for x in conditions))
        status='VERIFIED_ARTIFICIAL_BUNDLE_COMPLETE'if complete else'INCOMPLETE_EVIDENCE'if gaps else'VERIFIED_ARTIFICIAL_CONDITION_ONLY'if condition_only else'TECHNICAL_FAILURE'
        cost=C.aggregate(r,delivery,observation,pairs,comparison_eligible=complete)
        result.update(status=status,calls=dict(planned=total,reserved=reserved,started=started,completed=r['states']['COMPLETED'],unknown=r['states']['TERMINATED_WITH_UNKNOWN_RESULT'],cancelled=r['states']['CANCELLED_BEFORE_START']),
                      states=r['states'],state_physically_exact=snapshot is not None,trace=trace,history=h,control=control,documents=documents,
                      conditions=conditions,output_budget=budget,condition_byte_attribution=attribution,physical=phys,pairs=pairs,cost=cost,comparison_eligible=complete)
    except V.Unsupported as exc:result.update(status='UNSUPPORTED_SCHEMA',issues=[str(exc)],comparison_eligible=False)
    except (V.Invalid,KeyError,ValueError,TypeError,IndexError,OSError,sqlite3.Error,OverflowError,EOFError)as exc:
        result.update(status='INCOMPLETE_EVIDENCE'if isinstance(exc,FileNotFoundError)else'INTEGRITY_MISMATCH',issues=[str(exc)],comparison_eligible=False)
    return result
