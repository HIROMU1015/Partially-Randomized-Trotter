"""Streaming physical-prefix and six-phase bundle transition verification.

The trusted-prefix wire vocabulary is from the independent 77126767 V2 reader;
this version streams it and derives global identities with O(48 + in-flight) state.
No missing operation is synthesized from RAM declarations or untrusted tail bytes.
"""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
from . import g6_w2c_b3_offline_result as V
from . import g6_w2c_b3_parallel_verifier_core as G
from .g6_w2c_b3_bundle_independent_identity_v1 import equal

def scan(path,prefix,run_id,*,complete=False,visit=None):
    limit=V.nonnegative(prefix['bytes']);V.nonnegative(prefix['rows'])
    V.require(type(complete)is bool,'TRACE_COMPLETE_TYPE')
    previous=V.ZERO;trusted=sha256();physical=sha256();tail=sha256();size=rows=0
    with Path(path).open('rb')as f:
        while size<limit:
            line=f.readline(limit-size);V.require(line and line.endswith(b'\n'),'TRUSTED_PREFIX_ROW_BOUNDARY')
            size+=len(line);trusted.update(line);physical.update(line)
            env=V.plain(line)
            equal(set(env),{'payload','previous','sha256'},'TRACE_ENVELOPE_FIELDS')
            V.require(type(env['payload'])is str,'TRACE_PAYLOAD_TYPE')
            equal(json.dumps(env,separators=(',',':'),ensure_ascii=True).encode()+b'\n',line,'TRACE_ENVELOPE_CANONICAL')
            payload=env['payload'].encode();r=V.plain(payload)
            equal(json.dumps(r,separators=(',',':'),ensure_ascii=True).encode(),payload,'TRACE_CANONICAL')
            equal(env['previous'],previous,'TRACE_PREVIOUS');previous=sha256(previous.encode()+payload).hexdigest()
            equal(env['sha256'],previous,'TRACE_CHAIN');equal(r['seq'],rows,'TRACE_SEQ');equal(r['run_id'],run_id,'TRACE_RUN')
            if visit is not None:visit(r,len(line))
            rows+=1
        tail_bytes=0
        for block in iter(lambda:f.read(1024*1024),b''):tail.update(block);physical.update(block);tail_bytes+=len(block)
    equal(rows,prefix['rows'],'TRUSTED_PREFIX_ROWS');equal(previous,prefix['chain'],'TRUSTED_PREFIX_CHAIN');equal(trusted.hexdigest(),prefix['sha256'],'TRUSTED_PREFIX_SHA')
    V.require(not complete or tail_bytes==0,'COMPLETE_TRACE_HAS_TAIL')
    return dict(bytes=size+tail_bytes,sha256=physical.hexdigest(),trusted_bytes=size,trusted_rows=rows,chain=previous,
                tail_bytes=tail_bytes,tail_sha256=tail.hexdigest(),untrusted_tail_used_as_history=False)

class History:
    def __init__(self,tasks,manifest,report,state):
        self.tasks=tasks;self.m=manifest;self.r=report;self.state=state;self.ev=report['evidence_v2']
        self.lookup={t['key']:t for t in tasks};self.cursor=Counter();self.current={};self.counts=Counter();self.terminal=Counter()
        self.credit=Counter();self.ack=Counter();self.by_condition={c:dict(reserved=Counter(),observed=Counter())for c in ('SHARED','S','P1')}
        self.stopped=False;self.stop_rows=0;self.phase=None;self.last_phase=None;self.begun=[];self.ended={};self.phase_history=[];self.gaps=[]
        self.failed_indices={x['index']for x in self.ev['state_errors']}

    def __call__(self,r,size):
        op=r['op'];mode=r['condition'];V.require(mode in self.by_condition,'TRACE_CONDITION')
        bucket=self.by_condition[mode];bucket['reserved']['log']+=size;bucket['observed']['log']+=size
        if op in ('WRITE_RESERVE','WRITE_ACK'):
            channel=r['channel'];V.require(channel in ('result','log'),'BYTE_CHANNEL');n=V.nonnegative(r['amount'])
            if op=='WRITE_RESERVE':
                self.credit[channel]+=n;bucket['reserved'][channel]+=n
                V.require(self.credit[channel]<=self.m['byte_limits'][channel],'TRACE_GLOBAL_BYTE_CAP')
            else:
                self.ack[channel]+=n;bucket['observed'][channel]+=n
                V.require(self.ack[channel]<=self.credit[channel],'ACK_WITHOUT_RESERVATION')
            return
        if op=='STOP':self.stopped=True;self.stop_rows+=1;return
        if op=='CONDITION_BEGIN':
            V.require(not self.stopped and self.phase is None and len(self.begun)<len(self.m['modes']),'CONDITION_BEGIN_AFTER_STOP')
            equal(mode,self.m['modes'][len(self.begun)],'CONDITION_ORDER')
            V.require(not self.begun or self.ended.get(self.begun[-1])is True,'P1_AFTER_S_FAILURE')
            self.begun.append(mode);return
        if op=='CONDITION_END':
            V.require(self.begun and mode==self.begun[-1]and mode not in self.ended and self.phase is None,'CONDITION_END_ORDER')
            V.require(type(r['complete'])is bool,'CONDITION_COMPLETE_TYPE');self.ended[mode]=r['complete']
            if r['complete']:
                V.require(not self.stopped and not self.current,'FALSE_CONDITION_TRACE_COMPLETE')
                V.require(all(self.cursor[t['key']]==2*t['n']for t in self.tasks if t['condition']==mode),'CONDITION_TRACE_TRIAL_COVERAGE')
            return
        if op=='PHASE_BEGIN':
            V.require(not self.stopped and self.phase is None and self.begun and mode==self.begun[-1]and mode not in self.ended,'PHASE_BEGIN_AFTER_STOP')
            want=(0 if mode=='S'else 3)if self.last_phase is None else self.last_phase+1
            equal(r['phase'],want,'SIX_PHASE_BARRIER')
            if self.last_phase is not None:V.require(all(self.cursor[t['key']]==2*t['n']for t in self.tasks if t['phase_index']==self.last_phase),'PREVIOUS_PHASE_INCOMPLETE')
            self.phase=r['phase'];self.last_phase=self.phase;self.phase_history.append(self.phase);return
        if op in ('PHASE_END','PHASE_ABORT'):
            equal(r['phase'],self.phase,'PHASE_END_BINDING');V.require(self.phase is not None,'PHASE_END_WITHOUT_BEGIN')
            if op=='PHASE_END':
                V.require(not self.current,'PHASE_END_WITH_CURRENT_TICKET')
                if not self.stopped:V.require(all(self.cursor[t['key']]==2*t['n']for t in self.tasks if t['phase_index']==self.phase),'PHASE_END_BEFORE_COMPLETION')
            self.phase=None;return
        V.require(op in ('RESERVE','ENTER','COMPLETE','FAIL','CANCEL','UNKNOWN'),'UNKNOWN_BUNDLE_OPERATION')
        key=r['task_key'];V.require(key in self.lookup,'TRACE_TASK');t=self.lookup[key];tid=r['trial_id']
        equal(mode,t['condition'],'TASK_MODE');equal(r['global_trial_id'],self.m['run_id']+'/'+mode+'/'+tid,'GLOBAL_TRIAL_ID')
        if op=='RESERVE':
            i=self.cursor[key];index=t['offset']+i
            if self.stopped:
                snap=(self.r['failures'][0]if self.r['failures']else{}).get('inflight_snapshot',{}).get(key,{})
                V.require(not self.ev['statefile_complete']and index in self.failed_indices and snap.get('index')==index and snap.get('trial_id')==tid,'RESERVE_AFTER_STOP')
                self.gaps.append('Late RESERVE notification has only a failed-persistence RAM intent witness')
            V.require(self.phase==t['phase_index']and key not in self.current and i<2*t['n'],'RESERVE_PHASE_OR_DUPLICATE')
            equal(tid,V.trial(dict(t,N=t['n']),'X'if i<t['n']else'Y',i%t['n']),'LOCAL_TRIAL_CURSOR')
            self.current[key]=dict(tid=tid,index=index,entered=False);self.counts['reserved']+=1
            V.require(self.counts['reserved']<=self.m['call_limit'],'GLOBAL_CALL_CAP')
        else:
            V.require(key in self.current and self.current[key]['tid']==tid,'MISSING_OR_DUPLICATE_TERMINAL')
            ticket=self.current[key];index=ticket['index']
            if op=='ENTER':
                V.require(not self.stopped and not ticket['entered'],'ENTER_AFTER_STOP_OR_DUPLICATE');ticket['entered']=True;self.counts['entry_attempts']+=1
            else:
                if op in ('COMPLETE','FAIL'):V.require(ticket['entered'],'TERMINAL_WITHOUT_ENTRY')
                if op=='CANCEL'and ticket['entered']:
                    V.require(not self.ev['statefile_complete']and index in self.failed_indices,'CANCEL_AFTER_GRANTED_ENTRY')
                    self.counts['failed_entry_persistence']+=1
                state={'COMPLETE':'COMPLETED','FAIL':'FAILED','CANCEL':'CANCELLED_BEFORE_START','UNKNOWN':'TERMINATED_WITH_UNKNOWN_RESULT'}[op]
                observed=r.get('observed_source_state',state)
                V.require(observed==state or op=='UNKNOWN'and observed=='FAILED','TERMINAL_STATE_NAME')
                if self.ev['statefile_complete']:equal(self.state[index],G.STATES.index(observed),'TERMINAL_PHYSICAL_STATE')
                self.terminal[state]+=1;self.cursor[key]+=1;self.current.pop(key)
        equal(r['global_offset'],index,'GLOBAL_OFFSET')

    def finish(self):
        reserved=self.counts['reserved'];started=self.counts['entry_attempts']-self.counts['failed_entry_persistence']
        if self.ev['trace_complete']:
            V.require(not self.current,'TRACE_UNFINISHED_TICKETS');equal(reserved,self.r['reserved'],'TRACE_RESERVED_COUNTER')
            if self.ev['statefile_complete']:
                equal(started,self.r['started_observed'],'TRACE_GRANTED_START_COUNT')
                for s in G.STATES[4:]:equal(self.terminal[s],self.r['states'][s],'TRACE_TERMINAL_TOTAL:'+s)
        else:
            V.require(reserved<=self.r['reserved'],'PREFIX_RESERVATION_LOWER_BOUND')
        if self.ev['stop_trace_saved']:V.require(self.stop_rows>0,'MISSING_STOP_TRACE_WITNESS')
        return dict(counts=dict(self.counts),terminal=dict(self.terminal),current=self.current,cursors=dict(self.cursor),
                    stopped=self.stopped,phase_history=self.phase_history,conditions_begun=self.begun,conditions_ended=self.ended,
                    reserved_bytes=dict(self.credit),ack_bytes=dict(self.ack),by_condition=self.by_condition,evidence_gaps=self.gaps,
                    physical_worker_entry_time_proven=False,RAM_order_reconstructed=False)
