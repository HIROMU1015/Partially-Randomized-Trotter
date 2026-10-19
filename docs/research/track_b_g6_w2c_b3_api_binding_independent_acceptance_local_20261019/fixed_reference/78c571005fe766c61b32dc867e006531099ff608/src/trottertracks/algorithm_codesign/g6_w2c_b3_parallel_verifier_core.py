"""Producer-independent parallel evidence primitives. No scheduler/science imports."""
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from hashlib import sha256
from . import g6_w2c_b3_offline_result as V

STATES = ('NOT_STARTED','START_RESERVED','STARTED','IN_FLIGHT','COMPLETED','FAILED',
          'CANCELLED_BEFORE_START','TERMINATED_WITH_UNKNOWN_RESULT')
PHASES = (('correctness','baseline'),('profile','baseline'),('profile','instrumented'))
BUDGET = dict(correctness=256, baseline=15855296, instrumented=15855296, total=31710848,
              result_bytes=68719476736, log_bytes=268435456)


def plan(profile_n):
    """Independently implement the published fixed offset/identity contract."""
    tasks, offset = [], 0
    for phase_index,(phase,pass_name) in enumerate(PHASES):
        for i in range(8):
            arm,layer=V.ARMS[i//2],('A','B')[i%2]
            n=16 if phase_index==0 else (profile_n[arm] if isinstance(profile_n,dict) else profile_n)
            V.require(type(n)is int and n>=16,'PROFILE_N')
            tasks.append(dict(key=f'{phase}_{pass_name}_{arm}_{layer}',phase_index=phase_index,
                              phase=phase,pass_name=pass_name,arm=arm,layer=layer,pair_index=i,n=n,offset=offset))
            offset+=2*n
    return tasks,offset


def state_counts(path,tasks):
    """Read uint8 statefile sequentially; bounded buffer and 24 counters only."""
    total=Counter();pairs={};digest=sha256()
    with Path(path).open('rb')as f:
        for t in tasks:
            V.require(f.tell()==t['offset'],'STATE_OFFSET')
            remaining=2*t['n'];counts=Counter();noncomplete=False;not_started=False
            while remaining:
                b=f.read(min(65536,remaining));V.require(b,'STATEFILE_TRUNCATED')
                remaining-=len(b);digest.update(b)
                for code in b:
                    V.require(code<len(STATES),'STATE_CODE')
                    if code==4:V.require(not noncomplete,'COMPLETION_AFTER_UNFINISHED_TRIAL')
                    else:noncomplete=True
                    if code==0:not_started=True
                    else:V.require(not not_started,'RESERVATION_AFTER_NOT_STARTED_GAP')
                    counts[STATES[code]]+=1
            pairs[t['key']]={s:counts[s]for s in STATES};total.update(counts)
        V.require(not f.read(1),'STATEFILE_EXTRA_BYTES')
    return dict(counts={s:total[s]for s in STATES},pairs=pairs,retained_bytes=sum(total.values()),
                file_sha256=digest.hexdigest(),evidence=V.RAW)


def accounting(snapshot,report,planned):
    c=snapshot['counts'];reserved=planned-c['NOT_STARTED']
    V.require(sum(c.values())==planned and reserved<=BUDGET['total'],'GLOBAL_PLANNED_OR_RESERVED')
    if report is not None:
        V.require(report['planned']==planned and report['states']==c and report['reserved']==reserved,'GLOBAL_STATE_COUNTER_BINDING')
        started=V.nonnegative(report['started_observed'])
        known=c['COMPLETED']+c['FAILED']+c['STARTED']+c['IN_FLIGHT']
        V.require(known<=started<=known+c['TERMINATED_WITH_UNKNOWN_RESULT'],'STARTED_COMPLETED_UNKNOWN_BINDING')
        V.require(reserved<=min(planned,report['settings']['call_limit']),'CALL_CAP')
    else:started=None
    return dict(planned=planned,reserved=reserved,started_observed=started,completed=c['COMPLETED'],
                failed=c['FAILED'],cancelled_before_start=c['CANCELLED_BEFORE_START'],
                unknown=c['TERMINATED_WITH_UNKNOWN_RESULT']+c['START_RESERVED']+c['STARTED']+c['IN_FLIGHT'],
                not_started=c['NOT_STARTED'],reserved_is_actual_call_start=False,
                unknown_credit_refunded=False,latest_state_is_transition_history=False)


def check_operations(rows,tasks,run_id):
    """Optional normalized evidence API, NOT a claimed published producer log.

    O(24 pair cursors + outstanding workers); no all-trial ID set. Explicit seq,
    task key, local trial_id and op. Missing source evidence must remain pending.
    """
    lookup={t['key']:t for t in tasks};nexts=Counter();current={};totals=Counter()
    stopped=False;seq=0;phase=-1;open_phase=False;failures=[];reserved_bytes=Counter();observed_bytes=Counter()
    for r in rows:
        V.require(r['run_id']==run_id and r['seq']==seq,'OP_SEQUENCE_OR_RUN_ID');seq+=1
        op=r['op']
        if op=='PHASE_BEGIN':
            V.require(not stopped and not open_phase and r['phase']==phase+1,'PHASE_BARRIER_BEGIN')
            if phase>=0:V.require(all(nexts[t['key']]==2*t['n'] for t in tasks if t['phase_index']==phase),'PHASE_INCOMPLETE')
            phase=r['phase'];open_phase=True;continue
        if op=='PHASE_END':
            V.require(open_phase and not current and r['phase']==phase,'PHASE_BARRIER_END')
            open_phase=False;continue
        if op=='STOP':stopped=True;continue
        if op=='WRITE_RESERVE':
            amount=V.nonnegative(r['amount']);channel=r['channel'];V.require(channel in ('result','log'),'BYTE_CHANNEL')
            reserved_bytes[channel]+=amount;V.require(reserved_bytes[channel]<=BUDGET[channel+'_bytes'],'GLOBAL_BYTE_CAP');continue
        if op=='WRITE_ACK':
            channel=r['channel'];V.require(channel in ('result','log'),'BYTE_CHANNEL');observed_bytes[channel]+=V.nonnegative(r['amount'])
            V.require(observed_bytes[channel]<=reserved_bytes[channel],'ACK_WITHOUT_CREDIT');continue
        key=r['task_key'];V.require(key in lookup,'UNKNOWN_TASK');t=lookup[key]
        tid=r['trial_id'];spec=dict(t,N=t['n'])
        if op=='RESERVE':
            V.require(not stopped,'RESERVE_AFTER_STOP');V.require(open_phase and t['phase_index']==phase,'TRIAL_OUTSIDE_PHASE')
            i=nexts[key];axis='X'if i<t['n']else'Y';index=i if axis=='X'else i-t['n']
            V.require(i<2*t['n']and tid==V.trial(spec,axis,index)and key not in current,'DUPLICATE_OR_OUT_OF_ORDER_RESERVATION')
            current[key]=[tid,'START_RESERVED'];totals['reserved']+=1
            V.require(totals['reserved']<=BUDGET['total'],'CALL_CAP')
        elif op=='ENTER':
            V.require(not stopped,'ENTER_AFTER_STOP');V.require(current.get(key)==[tid,'START_RESERVED'],'DUPLICATE_OR_UNRESERVED_ENTRY')
            current[key][1]='IN_FLIGHT';totals['started_observed']+=1
        elif op in ('COMPLETE','FAIL','CANCEL','UNKNOWN'):
            V.require(key in current and current[key][0]==tid,'DUPLICATE_TERMINAL_OR_ID')
            state=current.pop(key)[1]
            if op in ('COMPLETE','FAIL'):V.require(state=='IN_FLIGHT','TERMINAL_BEFORE_ENTRY')
            if op=='CANCEL':V.require(state=='START_RESERVED','CANCEL_AFTER_ENTRY')
            nexts[key]+=1;totals[dict(COMPLETE='completed',FAIL='failed',CANCEL='cancelled_before_start',UNKNOWN='unknown')[op]]+=1
            if op=='UNKNOWN':stopped=True
            if op=='FAIL':failures.append(dict(seq=r['seq'],task_key=key,trial_id=tid));stopped=True
        else:raise V.Unsupported('NORMALIZED_OPERATION_SCHEMA')
    return dict(counts=dict(totals),outstanding=dict(current),stopped=stopped,failures=failures,
                representative_failure=failures[0]if failures else None,
                reserved_bytes=dict(reserved_bytes),ack_observed_bytes=dict(observed_bytes),
                evidence='NORMALIZED_ARTIFICIAL_OPERATION_LOG_ONLY',physical_exception_total_order_proven=False)


def check(check_id,expected,observed,judgment,evidence,unresolved=None,*,commit=None,schema='R1_PARALLEL_8PAIR_ARTIFICIAL_RESULT_V1'):
    return dict(check_id=check_id,target_SOURCE='published P1 prototype / independent verifier',fixed_commit=commit,
                schema_version=schema,expected=expected,observed=observed,judgment=judgment,evidence=evidence,unresolved=unresolved)


class FutureIntegratedAdapter:
    schema_status='PENDING_SERVER_SCHEMA'
    def verify(self,*args,**kwargs):
        return dict(status=self.schema_status,implemented=False,reason='Unpublished integrated producer schema is not inferred.')
