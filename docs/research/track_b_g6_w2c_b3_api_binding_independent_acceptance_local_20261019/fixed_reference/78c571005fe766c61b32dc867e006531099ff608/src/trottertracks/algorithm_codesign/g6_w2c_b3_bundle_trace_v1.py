"""Bundle trace consumer. read_prefix derived unchanged from 77126767 V2.
Producer-free operation judgment, global identities and six phase barriers.
"""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
from . import g6_w2c_b3_offline_result as V
from . import g6_w2c_b3_parallel_verifier_core as G

def read_prefix(path,prefix,run_id,complete):
    raw=Path(path).read_bytes();limit=V.nonnegative(prefix['bytes'])
    V.require(limit<=len(raw),'TRUSTED_PREFIX_EXCEEDS_PHYSICAL')
    trusted=raw[:limit];V.require(not trusted or trusted.endswith(b'\n'),'TRUSTED_PREFIX_ROW_BOUNDARY')
    previous=V.ZERO;rows=[]
    for seq,line in enumerate(trusted.splitlines(keepends=True)):
        e=V.plain(line);V.require(set(e)=={'payload','previous','sha256'} and type(e['payload']) is str,'V2_TRACE_ENVELOPE')
        V.require(json.dumps(e,separators=(',',':'),ensure_ascii=True).encode()+b'\n'==line,'V2_TRACE_CANONICAL_ENVELOPE')
        payload=e['payload'].encode();r=V.plain(payload)
        V.require(json.dumps(r,separators=(',',':'),ensure_ascii=True).encode()==payload,'V2_TRACE_CANONICAL')
        V.require(e['previous']==previous,'V2_TRACE_PREVIOUS');previous=sha256(previous.encode()+payload).hexdigest()
        V.require(e['sha256']==previous,'V2_TRACE_CHAIN')
        V.require(type(r['seq']) is int and r['seq']==seq and r['run_id']==run_id,'V2_TRACE_SEQ_RUN')
        rows.append(r)
    V.require(type(prefix['rows']) is int and len(rows)==prefix['rows'] and previous==prefix['chain'] and sha256(trusted).hexdigest()==prefix['sha256'],'V2_TRUSTED_PREFIX_BINDING')
    if complete:V.require(limit==len(raw),'V2_COMPLETE_TRACE_HAS_TAIL')
    return rows,dict(bytes=len(raw),sha256=sha256(raw).hexdigest(),trusted_bytes=limit,trusted_rows=len(rows),chain=previous,tail_bytes=len(raw)-limit,tail_sha256=sha256(raw[limit:]).hexdigest(),untrusted_tail_used_as_history=False)

def inspect(rows,tasks,manifest,report,state_bytes):
    lookup={t['key']:t for t in tasks};cursor=Counter();current={};counts=Counter();terminal=Counter();credit=Counter();ack=Counter()
    stopped=False;phase=-1;opened=False;begun=[];ended={};ids=set();failed_states={r['index']for r in report['evidence_v2']['state_errors']}
    for r in rows:
        op=r['op'];condition=r['condition']
        if op.startswith('WRITE_'):
            ch=r['channel'];V.require(ch in ('log','result'),'TRACE_BYTE_CHANNEL');n=V.nonnegative(r['amount'])
            if op=='WRITE_RESERVE':credit[ch]+=n;V.require(credit[ch]<=manifest['byte_limits'][ch],'TRACE_GLOBAL_BYTE_CAP')
            else:V.require(op=='WRITE_ACK','TRACE_BYTE_OP');ack[ch]+=n;V.require(ack[ch]<=credit[ch],'TRACE_ACK_WITHOUT_CREDIT')
            continue
        if op=='CONDITION_BEGIN':
            V.require(not stopped and not opened and condition==manifest['modes'][len(begun)],'CONDITION_ORDER')
            V.require(not begun or ended.get(begun[-1])is True,'P1_AFTER_S_FAILURE');begun.append(condition);continue
        if op=='CONDITION_END':
            V.require(condition in begun and condition not in ended and not opened,'CONDITION_END_ORDER');ended[condition]=r['complete'];continue
        if op=='STOP':stopped=True;continue
        if op=='PHASE_BEGIN':
            want=(0 if condition=='S'else 3)if phase<0 else phase+1
            V.require(not stopped and not opened and r['phase']==want and condition==begun[-1],'GLOBAL_PHASE_BARRIER')
            if phase>=0:V.require(all(cursor[t['key']]==2*t['n']for t in tasks if t['phase_index']==phase),'PREVIOUS_PHASE_INCOMPLETE')
            phase=r['phase'];opened=True;continue
        if op=='PHASE_ABORT':V.require(opened and r['phase']==phase,'GLOBAL_PHASE_ABORT');opened=False;continue
        if op=='PHASE_END':V.require(opened and phase==r['phase'] and not current,'GLOBAL_PHASE_END');opened=False;continue
        V.require(op in ('RESERVE','ENTER','COMPLETE','FAIL','CANCEL','UNKNOWN'),'TRACE_OPERATION')
        key=r['task_key'];V.require(key in lookup,'UNKNOWN_TASK');t=lookup[key];tid=r['trial_id']
        V.require(condition==t['condition'] and r['global_trial_id']==manifest['run_id']+'/'+condition+'/'+tid,'GLOBAL_TRIAL_IDENTITY')
        if op=='RESERVE':
            i=cursor[key];index=t['offset']+i
            allowed_after_stop=index in failed_states and any(v.get('index')==index for v in report['failures'][0].get('inflight_snapshot',{}).values())if report['failures']else False
            V.require((not stopped or allowed_after_stop)and opened and t['phase_index']==phase,'RESERVE_AFTER_STOP_OR_PHASE')
            axis='X'if i<t['n']else'Y'
            V.require(i<2*t['n']and key not in current and tid==V.trial(dict(t,N=t['n']),axis,i%t['n']),'DUPLICATE_RESERVATION_OR_CURSOR')
            V.require(r['global_trial_id']not in ids,'GLOBAL_TRIAL_COLLISION');ids.add(r['global_trial_id'])
            current[key]=dict(tid=tid,index=index,entered=False);counts['reserved']+=1
            V.require(counts['reserved']<=manifest['call_limit'],'GLOBAL_CALL_CAP')
        else:
            V.require(key in current and current[key]['tid']==tid,'MISSING_CURRENT_TICKET');ticket=current[key];index=ticket['index']
            if op=='ENTER':V.require(not stopped and not ticket['entered'],'ENTER_AFTER_STOP_OR_DUPLICATE');ticket['entered']=True;counts['entry_attempts']+=1
            else:
                entered=ticket['entered']
                if op in ('COMPLETE','FAIL'):V.require(entered,'TERMINAL_WITHOUT_ENTRY')
                if op=='CANCEL':V.require(not entered or index in failed_states,'CANCEL_AFTER_GRANTED_ENTRY')
                state={'COMPLETE':'COMPLETED','FAIL':'FAILED','CANCEL':'CANCELLED_BEFORE_START','UNKNOWN':'TERMINATED_WITH_UNKNOWN_RESULT'}[op]
                if report['evidence_v2']['statefile_complete']:V.require(state_bytes[index]==G.STATES.index(state),'TERMINAL_PHYSICAL_WITNESS')
                terminal[state]+=1;cursor[key]+=1;current.pop(key)
        V.require(r['global_offset']==index,'GLOBAL_OFFSET')
    if report['evidence_v2']['trace_complete']:
        V.require(not current,'TRACE_UNFINISHED_TICKET');V.require(counts['reserved']==report['reserved'],'TRACE_COUNTER')
        if report['evidence_v2']['statefile_complete']:
            for state,n in terminal.items():V.require(n==report['states'][state],'TRACE_STATE_TOTAL')
            V.require(counts['entry_attempts']==report['started_observed'],'TRACE_GRANTED_START_COUNT')
    else:V.require(counts['reserved']<=report['reserved'],'TRACE_PREFIX_COUNTER')
    return dict(counts=dict(counts),terminal=dict(terminal),current=current,cursors=dict(cursor),reserved_bytes=dict(credit),ack_bytes=dict(ack),stopped=stopped,conditions_begun=begun,conditions_ended=ended)
