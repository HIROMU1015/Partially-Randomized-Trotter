"""Artificial-only storage repair V2. No production bridge or science import.

RAM accounting never calls a trace writer in the middle of a transaction.
Physical state and ordered evidence have explicit independent failure status.
"""
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import sys
import time

from . import g6_w2c_b3_pilot_candidate as OLD
A,P,H,D=OLD.A,OLD.P,OLD.H,OLD.D
SCHEMA='b3_pilot_storage_repair_candidate_v2'
DOC='docs/research/track_b_g6_w2c_b3_trace_reservation_accounting_repair_20261019/'
SOURCE='src/trottertracks/algorithm_codesign/g6_w2c_b3_pilot_storage_v2.py'
CHECKER='src/trottertracks/algorithm_codesign/g6_w2c_b3_storage_verify_v2.py'
SCHEMA_PATH=DOC+'storage_v2.schema.json'
production_entry=OLD.production_entry
ProductionFactory,ProductionBits=OLD.ProductionFactory,OLD.ProductionBits


def candidate(root,profile_n=256):
    c=OLD.candidate(root,profile_n)
    c['schema']=SCHEMA
    c['storage_protocol']='MONOTONE_ACCOUNTING_EXPLICIT_INCOMPLETE_EVIDENCE_V2'
    for path in (SOURCE,CHECKER,SCHEMA_PATH):
        c['source_pins'][path]=A.A.digest((Path(root)/path).read_bytes())
    return c


def validate(c):
    A.require(c['schema']==SCHEMA,'STORAGE_V2_CANDIDATE')
    A.require(Path(__file__).resolve()==Path(c['runtime_root'])/SOURCE,'LOADED_STORAGE_SOURCE')
    expected=candidate(c['runtime_root'],c['proposal']['profile_n'])
    A.require(type(c['proposal']['profile_n'])is int and c['proposal']['profile_n']in(64,256,1024),
              'FIXED_PROPOSAL_N')
    A.require(D.identity(c)==D.identity(expected),'STORAGE_V2_BINDING')
    for path,h in c['input_pins'].items():
        A.require(A.A.digest((Path(c['runtime_root'])/path).read_bytes())==h,'FIXED_INPUT_BYTES')
    A.require(not any(n.rsplit('.',1)[-1]in P.FORBIDDEN for n in sys.modules),'SCIENCE_IMPORTED')
    return D.identity(c)


class ArtificialAdmission(A.ArtificialAdmission):
    def __init__(self,c,*args,**kwargs):
        self.pilot=deepcopy(c)
        super().__init__(*args,**kwargs)

    def _check(self,digest):
        A.require(self.contract.get('pilot_candidate_binding')==validate(self.pilot),'PILOT_BINDING')
        return super()._check(digest)


class NoFault:
    def __call__(self,point,**context):pass


class Budget(P.ByteBudget):
    """Credit is irrevocable; notification cannot interrupt bookkeeping."""
    notify=None

    def raw_reserve(self,channel,n):
        super().reserve(channel,n)

    def raw_wrote(self,channel,n):
        D.need(channel in self.limit and type(n)is int and n>=0 and
               self.observed[channel]+n<=self.reserved[channel],'INVALID_WRITE_ACK')
        self.observed[channel]+=n

    def reserve(self,channel,n):
        self.raw_reserve(channel,n)
        if self.notify:self.notify('WRITE_RESERVE',channel=channel,amount=n)

    def wrote(self,channel,n):
        self.raw_wrote(channel,n)
        if self.notify:self.notify('WRITE_ACK',channel=channel,amount=n)


class Trace:
    """One charged unbuffered stream; a failed record ends the trusted prefix."""
    def __init__(self,path,budget,run_id,fault=NoFault()):
        self.budget,self.run_id,self.fault=budget,run_id,fault
        self.seq,self.previous,self.bytes=0,'0'*64,0
        self.file_sha=sha256();self.failed=False;self.errors=[]
        self.prefix=dict(bytes=0,rows=0,chain=self.previous,sha256=self.file_sha.hexdigest())
        self.fault('trace.open');self.file=Path(path).open('xb',buffering=0)

    def emit(self,op,**fields):
        if self.failed:raise P.Halt('TRACE_INCOMPLETE_NO_APPEND')
        try:
            self.fault('trace.hash',op=op,channel=fields.get('channel'))
            raw=P.safe_json(dict(seq=self.seq,run_id=self.run_id,op=op,**fields))
            digest=sha256(self.previous.encode()+raw).hexdigest()
            data=P.safe_json(dict(payload=raw.decode(),previous=self.previous,sha256=digest))+b'\n'
            self.budget.raw_reserve('log',len(data))
            view=memoryview(data)
            self.fault('trace.write',op=op,channel=fields.get('channel'))
            while view:
                n=self.file.write(view)
                D.need(type(n)is int and 0<n<=len(view),'TRACE_SHORT_WRITE')
                self.budget.raw_wrote('log',n);self.bytes+=n;self.file_sha.update(view[:n]);view=view[n:]
                self.fault('trace.after_write',op=op,remaining=len(view))
            self.fault('trace.flush',op=op,channel=fields.get('channel'));self.file.flush()
            self.seq+=1;self.previous=digest
            self.prefix=dict(bytes=self.bytes,rows=self.seq,chain=digest,sha256=self.file_sha.hexdigest())
        except BaseException as exc:
            self.failed=True;self.errors.append(dict(op=op,error=repr(exc)))
            raise

    def close(self):
        try:self.fault('trace.close');self.file.flush()
        except BaseException as exc:self.failed=True;self.errors.append(dict(op='CLOSE',error=repr(exc)))
        finally:self.file.close()


class States:
    """Authoritative RAM codes; failed physical state persistence is explicit.

    Each slot still consumes exactly one logical log byte. The shadow consumes
    an additional planned bytes of RAM, never a second retained-file charge.
    """
    def __init__(self,path,planned,budget,fault):
        self.fault=fault;self.mapping=bytearray(planned)
        self.counts=Counter(NOT_STARTED=planned);self.io_bytes=0;self.errors=[];self.notify=None
        budget.reserve('log',planned)
        self.fault('state.open');self.file=Path(path).open('x+b',buffering=0)
        try:self.fault('state.truncate');self.file.truncate(planned);budget.wrote('log',planned)
        except BaseException:self.file.close();raise

    def set(self,index,name):
        old=P.STATES[self.mapping[index]]
        D.need(old not in P.STATES[4:],'TERMINAL_TRIAL_REUSE')
        self.mapping[index]=P.CODE[name];self.counts[old]-=1;self.counts[name]+=1
        okay=True
        try:
            self.fault('state.write',state=name,index=index)
            n=os.pwrite(self.file.fileno(),bytes([P.CODE[name]]),index)
            D.need(n==1,'STATE_SHORT_WRITE');self.io_bytes+=n
            self.fault('state.after_write',state=name,index=index)
        except BaseException as exc:
            okay=False;self.errors.append(dict(index=index,old=old,intended=name,error=repr(exc)))
        if self.notify:self.notify(index,name,okay)
        return okay

    def close(self):self.file.close()


class Journal:
    def __init__(self,path,budget,fault):
        self.budget,self.fault=budget,fault;self.failed=False
        fault('journal.open');self.file=Path(path).open('xb',buffering=0)

    def emit(self,row):
        if self.failed:raise P.Halt('CONTROL_JOURNAL_INCOMPLETE')
        data=P.safe_json(row)+b'\n';self.budget.reserve('log',len(data));view=memoryview(data)
        try:
            self.fault('journal.write',event=row.get('event'))
            while view:
                n=self.file.write(view);D.need(type(n)is int and 0<n<=len(view),'JOURNAL_SHORT_WRITE')
                self.budget.wrote('log',n);view=view[n:]
                self.fault('journal.after_write',event=row.get('event'),remaining=len(view))
            self.fault('journal.flush',event=row.get('event'));self.file.flush()
        except BaseException:self.failed=True;raise


class Coordinator(P.BASE_COORDINATOR):
    """Same worker/F01 protocol; transactions contain no fallible trace calls."""
    def __init__(self,settings,target,tasks,planned,run_id,fault=NoFault()):
        # Copy only initialization, with versioned storage classes. Existing P
        # remains byte-for-byte unchanged, including its worker spawn entrypoint.
        self.before=D.clock();self.child_before=resource.getrusage(resource.RUSAGE_CHILDREN)
        self.settings,self.target,self.tasks,self.planned=settings,Path(target),tasks,planned
        self.next_indices={t['key']:0 for t in tasks}
        self.fault=fault;self.messages=Counter();self.trace=None
        self.storage_errors=[];self.stop_trace_saved=False;self.stop_journal_saved=False
        self.budget=Budget(settings.result_bytes,settings.log_bytes)
        self.journal=None;self.states=None
        self.ctx=P.mp.get_context('spawn')
        cap=settings.inflight_cap if settings.mode=='P2'else(1 if settings.mode=='S'else settings.workers)
        self.rendezvous=self.ctx.Barrier(cap)if settings.sync_failure else None
        self.stopped=self.ctx.RawValue('b',0)
        self.active,self.waiting,self.current={},[],{}
        self.results,self.failures,self.arrival,self.completion_order=[],[],0,[]
        self.reservations,self.started,self.max_inflight=0,0,0
        self.stop_ns=self.stop_cpu_ns=self.pre_stop_worker_cpu=None;self.pre_stop_cpu_valid=False
        self.log_errors,self.peak_rss,self.rss_samples=[],0,0;self.launch_cost=Counter()
        try:
            self.journal=Journal(self.target/'control.jsonl',self.budget,fault)
            self.states=States(self.target/'trial_states.bin',planned,self.budget,fault)
            self.journal.emit(dict(event='PLAN',schema='b3_parallel_control_v3',settings=P.asdict(settings),
                planned=planned,tasks=tasks,state_codes=P.CODE,retry=0,resume=False))
            self.trace=Trace(self.target/'operations.jsonl',self.budget,run_id,fault)
            initial_reserved=dict(self.budget.reserved);initial_observed=dict(self.budget.observed)
            for channel,n in initial_reserved.items():self._record('WRITE_RESERVE',channel=channel,amount=n)
            for channel,n in initial_observed.items():self._record('WRITE_ACK',channel=channel,amount=n)
            self.budget.notify=self._record;self.states.notify=self._direct_state_change
        except BaseException:
            if self.states:self.states.close()
            if self.journal:self.journal.file.close()
            if self.trace:self.trace.file.close()
            raise

    def _record(self,op,**fields):
        if self.trace is None or self.trace.failed:return False
        try:self.trace.emit(op,**fields);return True
        except BaseException as exc:
            self.trace.failed=True
            self.storage_errors.append(dict(storage='trace',op=op,error=repr(exc)))
            self.stopped.value=1
            if self.stop_ns is None:self.stop(dict(trace_failure=repr(exc),op=op))
            return False

    def _direct_state_change(self,index,name,okay):
        # Only inherited EOF cleanup invokes this hook directly. It has already
        # removed the ticket; no notification exception can interrupt cleanup.
        t=next(t for t in self.tasks if t['offset']<=index<t['offset']+2*t['n'])
        i=index-t['offset'];axis='X'if i<t['n']else'Y'
        if name=='TERMINATED_WITH_UNKNOWN_RESULT':
            self._record('UNKNOWN',task_key=t['key'],trial_id=f"{t['phase']}/{t['pass_name']}/{t['arm']}/{t['layer']}/{axis}/{i%t['n']}",observed_source_state=name)
        if not okay:self.stop(dict(state_persistence_failure=self.states.errors[-1]))

    def _state(self,index,name):
        notify=self.states.notify;self.states.notify=None
        try:okay=self.states.set(index,name)
        finally:self.states.notify=notify
        if not okay:self.stop(dict(state_persistence_failure=self.states.errors[-1]))
        return okay

    def log(self,row):
        if row.get('event')in('PHASE_BEGIN','PHASE_END'):
            self._record(row['event'],phase=row['phase'])
        try:self.journal.emit(row);return True
        except BaseException as exc:
            self.stopped.value=1;self.log_errors.append(repr(exc))
            self.storage_errors.append(dict(storage='control',event=row.get('event'),error=repr(exc)))
            if self.stop_ns is None:self.stop(dict(journal_failure=repr(exc)))
            return False

    def stop(self,detail):
        first=self.stop_ns is None
        if first:
            self.stopped.value=1
            self.stop_ns,self.stop_cpu_ns=time.monotonic_ns(),time.process_time_ns()
            alive,valid=P.owned_cpu([x['process'].pid for x in self.active.values()])
            reaped=resource.getrusage(resource.RUSAGE_CHILDREN)
            self.pre_stop_worker_cpu=alive+reaped.ru_utime+reaped.ru_stime-self.child_before.ru_utime-self.child_before.ru_stime
            self.pre_stop_cpu_valid=valid
        self.arrival+=1;record=dict(detail,parent_arrival=self.arrival,parent_observed_ns=time.monotonic_ns())
        if first:
            record['inflight_snapshot']={v['worker']:dict(v)for v in self.current.values()}
        self.failures.append(record)
        if first:self.stop_trace_saved=self._record('STOP',parent_arrival=self.arrival,detail=repr(detail))
        saved=self.log(dict(event='FAILURE_OBSERVED',**record))
        if first:self.stop_journal_saved=saved

    def _send(self,conn,okay,reason=None):
        try:conn.send(dict(ok=okay,**({'reason':reason}if reason else{})))
        except (BrokenPipeError,EOFError,OSError)as exc:self.stop(dict(reply_lost=repr(exc)))

    def reserve_start(self,conn,request):
        row=self.active[conn];t=row['task'];i=row['next']
        axis=D.AXES[i//t['n']]if i<2*t['n']else None
        tid=f"{t['phase']}/{t['pass_name']}/{t['arm']}/{t['layer']}/{axis}/{i%t['n']}"
        D.need(not self.stopped.value,'GLOBAL_STOP_NO_NEW_RESERVATION')
        D.need(axis is not None and request['trial_id']==tid and conn not in self.current,'TRIAL_ID_DUPLICATE_OR_ORDER')
        D.need(self.reservations<min(self.planned,self.settings.call_limit),'GLOBAL_CALL_BUDGET')
        index=t['offset']+i;D.need(self.states.mapping[index]==P.CODE['NOT_STARTED'],'TRIAL_ALREADY_RESERVED')
        # Non-I/O transaction first. A failed trace can no longer strand a slot
        # outside counter, cursor and ticket accounting.
        self.reservations+=1;row['next']+=1
        self.next_indices[t['key']]=row['next']
        self.current[conn]=dict(index=index,trial_id=tid,state='START_RESERVED',worker=t['key'],reserved_ns=time.monotonic_ns())
        self.max_inflight=max(self.max_inflight,len(self.current))
        okay=self._state(index,'START_RESERVED')
        logged=self._record('RESERVE',task_key=t['key'],trial_id=tid,observed_source_state='START_RESERVED')
        if self.settings.parent_fault=='after_reserved':os._exit(86)
        self._send(conn,okay and logged and not self.stopped.value,'STORAGE_STOP'if self.stopped.value else None)

    def _terminal(self,conn,op):
        ticket=self.current[conn];entered=ticket['state']=='IN_FLIGHT'
        if not entered and op!='UNKNOWN':op='CANCEL'
        state=dict(COMPLETE='COMPLETED',FAIL='FAILED',CANCEL='CANCELLED_BEFORE_START',UNKNOWN='TERMINATED_WITH_UNKNOWN_RESULT')[op]
        logged=self._record(op,task_key=ticket['worker'],trial_id=ticket['trial_id'],observed_source_state=state)
        if op=='COMPLETE'and not logged:state='TERMINATED_WITH_UNKNOWN_RESULT'
        self._state(ticket['index'],state)
        self.current.pop(conn)
        return logged

    def handle(self,conn,message):
        op=message['op'];row=self.active[conn];self.messages[op]+=1
        try:
            n=message.get('wrote_before',0)
            D.need(type(n)is int and n>=0,'INVALID_WRITE_ACK')
            if n:self.budget.wrote('result',n)
            previous=message.get('complete_before')
            if previous:
                D.need(self.current[conn]['trial_id']==previous and self.current[conn]['state']=='IN_FLIGHT','PREVIOUS_COMMIT_BINDING')
                self._terminal(conn,'COMPLETE')
            if op=='RESERVE':
                self.waiting.append((conn,message));self.drain_starts();return
            if op=='END':
                report=message['report']
                D.need(report['task']==row['task']and report['pid']==row['process'].pid,'WORKER_RESULT_IDENTITY')
                if report['status']!='OK':self.stop(dict(worker=row['task']['key'],terminal_failure=report['failure']))
                if conn in self.current:self._terminal(conn,'FAIL'if report['failure']else'UNKNOWN')
                self.results.append(report);self.completion_order.append(row['task']['key']);row['report']=report
                self.log(dict(event='WORKER_END',report=report));return
            if op=='WRITE_RESERVE':self.budget.reserve('result',message['amount'])
            elif op=='WRITE_ACK':self.budget.wrote('result',message['amount'])
            elif op=='FAIL':self.stop(dict(message,worker=row['task']['key']))
            elif op in('ENTER','START_CHECK','CANCEL','COMPLETE'):
                ticket=self.current[conn];D.need(ticket['trial_id']==message['trial_id'],'CURRENT_TRIAL_BINDING')
                if op in('ENTER','START_CHECK')and self.stopped.value:
                    self._terminal(conn,'CANCEL');self._send(conn,False,'CANCELLED_BEFORE_START');return
                if op=='ENTER':
                    D.need(ticket['state']=='START_RESERVED','DUPLICATE_ENTER')
                    okay=self._record('ENTER',task_key=ticket['worker'],trial_id=ticket['trial_id'],observed_source_state='IN_FLIGHT')
                    if okay:okay=self._state(ticket['index'],'STARTED')
                    if okay:okay=self._state(ticket['index'],'IN_FLIGHT')
                    if not okay:
                        self._terminal(conn,'CANCEL');self._send(conn,False,'ENTRY_NOT_GRANTED_STORAGE_FAILURE');return
                    ticket.update(state='IN_FLIGHT',worker_entry_ns=message['worker_entry_ns']);self.started+=1
                elif op=='CANCEL':
                    D.need(ticket['state']=='START_RESERVED','CANCEL_AFTER_ENTRY');self._terminal(conn,'CANCEL')
                elif op=='COMPLETE':
                    D.need(ticket['state']=='IN_FLIGHT','COMPLETE_WITHOUT_ENTRY');self._terminal(conn,'COMPLETE')
            else:raise P.Halt('UNKNOWN_WORKER_MESSAGE')
            self._send(conn,True)
        except BaseException as exc:
            self.stop(dict(worker=row['task']['key'],type=type(exc).__name__,reason=str(exc)))
            self._send(conn,False,str(exc))

    def report(self):
        r=super().report();r['schema']='b3_pilot_storage_result_v2'
        r['parent_messages']=dict(self.messages)
        r['evidence_v2']=dict(trace_complete=bool(self.trace and not self.trace.failed),
            trace_prefix=dict(self.trace.prefix)if self.trace else None,
            statefile_complete=not self.states.errors,journal_complete=not self.journal.failed,
            stop_trace_saved=self.stop_trace_saved,stop_journal_saved=self.stop_journal_saved,
            storage_errors=list(self.storage_errors),state_errors=list(self.states.errors),
            trace_errors=list(self.trace.errors)if self.trace else[],
            state_authority='MONOTONE_RAM_SHADOW_NOT_A_CLAIM_OF_PHYSICAL_DURABILITY',
            current_tickets=[dict(v)for v in self.current.values()],
            pair_next={v['task']['key']:v['next']for v in self.active.values()},
            shadow_sha256=sha256(self.states.mapping).hexdigest())
        r['evidence_v2']['next_indices']=dict(self.next_indices)
        r['output']['unknown_credit']={k:self.budget.reserved[k]-self.budget.observed[k]for k in self.budget.limit}
        r['complete']=r['complete']and not self.storage_errors and not self.states.errors and not self.trace.failed and not self.journal.failed
        return r


class ParentFile:
    """Inject at actual parent save boundaries, never at a scientific endpoint."""
    def __init__(self,file,fault,label):self.file,self.fault,self.label=file,fault,label
    def write(self,data):
        self.fault(self.label+'.write');n=self.file.write(data)
        self.fault(self.label+'.after_write',written=n)
        return n
    def flush(self):self.fault(self.label+'.flush');return self.file.flush()
    def close(self):return self.file.close()
    def __getattr__(self,name):return getattr(self.file,name)


def save_diagnostic(output,control,fault):
    fault('final_report.hash')
    data=P.safe_json(dict(schema='b3_pre_close_diagnostic_v2',complete=False,
        final_acceptance=False,reserved=control.reservations,started=control.started,
        states=dict(control.states.counts)))+b'\n'
    fault('final_report.open')
    f=P.ReservedFile(ParentFile((output/'final_diagnostic.json').open('xb',buffering=0),fault,'final_report'),
        lambda n:control.budget.reserve('result',n),lambda n:control.budget.wrote('result',n))
    try:
        view=memoryview(data)
        while view:
            n=f.write(view);D.need(type(n)is int and n>0,'DIAGNOSTIC_SHORT_WRITE');view=view[n:]
        f.flush()
    finally:f.close()


def close_storage(control):
    for label,close in (('trace',control.trace.close),('state',control.states.close)):
        try:
            if label!='trace':control.fault(label+'.close')
            close()
        except BaseException as exc:
            control.storage_errors.append(dict(storage=label,op='CLOSE',error=repr(exc)))
            try:close()
            except BaseException:pass
    if (control.trace.failed or control.storage_errors or control.states.errors or control.journal.failed)and control.stop_ns is None:
        control.stop(dict(storage_close_or_evidence_failure=True))
    # Save a newly observed close failure while the control journal is still
    # open. Never manufacture a second failure by logging after normal close.
    try:
        control.fault('journal.close');control.journal.file.close()
    except BaseException as exc:
        control.storage_errors.append(dict(storage='journal',op='CLOSE',error=repr(exc)))
        if control.stop_ns is None:control.stop(dict(journal_close_failure=repr(exc)))
        control.journal.failed=True
        try:control.journal.file.close()
        except BaseException:pass


def execute_artificial(c, inner, *,root,ledger,authority,endpoint_factory,bits_factory,
                       admission_fault=lambda point:None,watch=None,storage_fault=NoFault()):
    """Private fake authority only. No production dispatch or unseal parameter."""
    before=D.clock();ch=resource.getrusage(resource.RUSAGE_CHILDREN);io=P.own_io()
    control=None;result=None
    with H._execution_context(inner):
        port=ArtificialAdmission(c,inner,Path(root),ledger,authority,fault=admission_fault)
        try:
            with port:
                A.require(endpoint_factory.artificial is True,'ARTIFICIAL_ENDPOINT_REQUIRED')
                expected=Path(inner['runtime_manifest']['runtime_root'])/inner['runtime_manifest']['artificial_fixture']['path']
                A.require(Path(endpoint_factory.__call__.__code__.co_filename).resolve()==expected and
                    Path(bits_factory.__code__.co_filename).resolve()==expected and
                    A.A.digest(expected.read_bytes())==inner['input_pins']['artificial-fixture.py'],
                    'ARTIFICIAL_FACTORY_SOURCE_BINDING')
                digest=D.identity(inner['candidate'])
                port._check(digest);port.admit_artificial(digest);port._check(digest)
                output=A.A.exact_path(root,inner['candidate']['output_relative'])
                output.mkdir(parents=True,exist_ok=False)
                settings=P.Settings(**inner['settings']);settings.check()
                tasks,planned=P.plan_for(settings.profile_n)
                A.require(planned<=c['proposal']['total_calls'],'PILOT_GLOBAL_CALL_CAP')
                control=Coordinator(settings,output,tasks,planned,D.identity(inner),storage_fault)
                for phase in range(3):
                    if control.stopped.value:break
                    control.run_phase([t for t in tasks if t['phase_index']==phase],root,
                        inner['source_pins'],endpoint_factory,bits_factory,watch)
                    P.validate_success(control.results)
                    validate(c);A.check_runtime(root,inner['runtime_manifest'])
                result=control.report()
                if not result['complete']:raise P.BundleFailure(result)
                # Bind this prefix; later save-credit rows remain in the trace.
                trace_meta=dict(sha256=A.A.digest((output/'operations.jsonl').read_bytes()),
                    chain=control.trace.previous,rows=control.trace.seq,bytes=control.trace.bytes)
                summary=dict(schema='b3_pilot_storage_summary_v2',pilot_binding=validate(c),
                    inner_binding=D.identity(inner),trace=trace_meta,result=result)
                stream=D.Stream(output/'pilot_summary.jsonl',D.Meter(D.clock),byte_limit=settings.result_bytes,flush_rows=1)
                stream.file=ParentFile(stream.file,storage_fault,'summary')
                stream.file=P.ReservedFile(stream.file,lambda n:control.budget.reserve('result',n),
                    lambda n:control.budget.wrote('result',n))
                try:stream.emit(summary)
                finally:stream.close()
                receipt=dict(schema='b3_pilot_storage_receipt_v2',artificial_only=True,
                    pilot_binding=validate(c),inner_binding=D.identity(inner),
                    summary_sha256=stream.file_hash.hexdigest(),summary_chain=stream.previous,
                    trace_prefix=trace_meta,production_permission=False)
                storage_fault('receipt.open')
                f=P.ReservedFile(ParentFile((output/'pilot_receipt.json').open('xb',buffering=0),storage_fault,'receipt'),
                    lambda n:control.budget.reserve('result',n),lambda n:control.budget.wrote('result',n))
                try:
                    data=A.A.encoded(receipt);view=memoryview(data)
                    while view:
                        n=f.write(view);D.need(type(n)is int and n>0,'RECEIPT_SHORT_WRITE');view=view[n:]
                    f.flush()
                finally:f.close()
                save_diagnostic(output,control,storage_fault)
        except BaseException as exc:
            if control is not None:
                control.stop(dict(parent_failure=repr(exc)))
                if control.active:
                    control.run_phase([],root,inner['source_pins'],endpoint_factory,bits_factory,watch,drain_only=True)
                result=control.report()
                result['primary_failure']=repr(exc)
            else:raise
        else:
            # The one-shot STOP itself must also have persisted before the
            # success marker. An orphan receipt file is never sufficient.
            if control is not None and result is not None and result['complete']:
                control.log(dict(event='PILOT_COMPLETION_RECEIPT',**receipt))
        finally:
            if control is not None:
                close_storage(control)
    after=D.clock();endch=resource.getrusage(resource.RUSAGE_CHILDREN)
    if result is not None:
        primary=result.get('primary_failure')
        result=control.report()
        if primary:result['primary_failure']=primary
        result['output']=dict(limits=control.budget.limit,reserved=dict(control.budget.reserved),
            observed=dict(control.budget.observed),unknown_credit={k:control.budget.reserved[k]-control.budget.observed[k]for k in control.budget.limit})
        result['host_outer']=dict(parent_cpu_ns=after[0]-before[0],wall_ns=after[1]-before[1],
            worker_cpu_ns=round((endch.ru_utime+endch.ru_stime-ch.ru_utime-ch.ru_stime)*1e9),
            io={k:v-io.get(k,0)for k,v in P.own_io().items()})
        result['fake_claim_status']=ledger.row(inner['scope'])[4]
        result['admission_stop']=deepcopy(port._stop_report)
        if result['admission_stop'] is None or result['admission_stop']['record_errors']:
            result['complete']=False
        result['pilot_binding']=validate(c)
        result['operation_trace_sha256']=A.A.digest((output/'operations.jsonl').read_bytes())
        result['call_reservation_nonrefundable']=True
    return result
