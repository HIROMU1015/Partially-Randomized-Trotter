"""Validate only the declared physically retained trusted V2 trace prefix."""
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

def inspect(rows,tasks,run_id,report,state_bytes,state_complete,trace_complete):
    """Reuse global core for normal semantics; explicit failed ENTER persistence is distinct."""
    ev=report['evidence_v2'];bykey={t['key']:t for t in tasks};cursor=Counter();current={};counts=Counter();witness=Counter();entry_attempts=0
    state_fail_indices={x['index'] for x in ev['state_errors']};normalized=[]
    for r in rows:
        r=dict(r);op=r['op']
        if op in ('RESERVE','ENTER','COMPLETE','FAIL','CANCEL','UNKNOWN'):
            key=r['task_key'];V.require(key in bykey,'V2_TRACE_TASK');t=bykey[key]
            if op=='RESERVE':
                i=cursor[key];V.require(key not in current and i<2*t['n'],'V2_DUPLICATE_RESERVE')
                V.require(r['trial_id']==V.trial(dict(t,N=t['n']),'X' if i<t['n'] else 'Y',i%t['n']),'V2_TRIAL_CURSOR_ID')
                current[key]=dict(index=t['offset']+i,trial_id=r['trial_id'],entered=False)
            else:
                V.require(key in current and current[key]['trial_id']==r['trial_id'],'V2_CURRENT_TICKET_ID');ticket=current[key]
                if op=='ENTER':
                    V.require(not any(x['op']=='STOP' for x in normalized),'ENTER_AFTER_STOP')
                    V.require(not ticket['entered'],'V2_DOUBLE_ENTER');ticket['entered']=True;entry_attempts+=1
                else:
                    state={'COMPLETE':'COMPLETED','FAIL':'FAILED','CANCEL':'CANCELLED_BEFORE_START','UNKNOWN':'TERMINATED_WITH_UNKNOWN_RESULT'}[op]
                    V.require(r.get('observed_source_state',state)==state or op=='UNKNOWN' and r.get('observed_source_state')=='FAILED','V2_TERMINAL_STATE')
                    if op=='CANCEL' and ticket['entered']:
                        V.require(ticket['index'] in state_fail_indices and not state_complete,'V2_CANCEL_AFTER_SUCCESSFUL_ENTER')
                        # Core ENTER denotes granted entry. Here persistence failed before grant.
                        previous=next(j for j in range(len(normalized)-1,-1,-1) if normalized[j]['op']=='ENTER' and normalized[j].get('trial_id')==r['trial_id'])
                        normalized.pop(previous)
                    if state_complete:
                        V.require(state_bytes[ticket['index']]==G.STATES.index(r.get('observed_source_state',state)),'V2_TERMINAL_PHYSICAL_WITNESS')
                    witness[state]+=1;cursor[key]+=1;current.pop(key)
            index=(current[key]['index'] if key in current else t['offset']+cursor[key]-1)
            if 'global_offset' in r:V.require(type(r['global_offset']) is int and r['global_offset']==index,'V2_GLOBAL_OFFSET')
        if op=='RESERVE' and any(x['op']=='STOP' for x in normalized):
            # RAM reservation predates failing state persistence; its trace
            # notification can follow STOP. This order is not independently
            # proven, so this path always remains INCOMPLETE_EVIDENCE.
            ticket=current[r['task_key']]
            V.require(not state_complete and bool(report['failures']),'V2_NEW_RESERVATION_AFTER_STOP')
            first=report['failures'][0]
            snap=first.get('inflight_snapshot',{}).get(r['task_key'],{})
            V.require(not state_complete and ticket['index'] in state_fail_indices and snap.get('index')==ticket['index'] and snap.get('trial_id')==r['trial_id'],'V2_NEW_RESERVATION_AFTER_STOP')
            at=next(i for i,x in enumerate(normalized) if x['op']=='STOP');normalized.insert(at,r)
        else:normalized.append(r)
    for seq,r in enumerate(normalized):r['seq']=seq
    history=G.check_operations(iter(normalized),tasks,run_id)
    if state_complete:
        for k in ('reserved','started_observed'):
            V.require(history['counts'].get(k,0)<=report[k],'V2_PREFIX_COUNTER_LOWER_BOUND:'+k)
        V.require(history['counts'].get('completed',0)<=report['states']['COMPLETED'],'V2_PREFIX_COMPLETE_LOWER_BOUND')
        if trace_complete:
            for k,field in [('reserved','reserved'),('started_observed','started_observed')]:V.require(history['counts'].get(k,0)==report[field],'V2_TRACE_COUNTER:'+k)
            V.require(not current,'V2_TRACE_UNFINISHED')
            for state,n in witness.items():V.require(n==report['states'][state],'V2_TRACE_STATE_TOTAL')
    history.update(terminal_state_witness=dict(witness),entry_attempts_in_prefix=entry_attempts,entry_granted_not_inferred_from_failed_state=True)
    return history
