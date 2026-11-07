"""Job-local START/STOP/budget adapters around the frozen worker and pool.

No new engine, sampler, price provider or manager. Locks protect admission and
receipts only, never a gate/compile body. All paths are private job outputs.
"""
from contextlib import contextmanager, ExitStack
from functools import wraps
from pathlib import Path
import json
import fcntl
import threading
import multiprocessing as mp
import os
import resource
import signal
import time
from unittest.mock import patch

from . import a1_n3_campaign as c
from . import a1_n3_connection as n3
from . import a1_n3_managed_runtime as managed
from . import a1_n3_lifecycle_observer as obs
from . import scv224_pilot as old
from . import scv_signed_pilot as pilot

_GATE = None
_ORIGINAL_WORKER = c.worker_trial


class CallGate:
    def __init__(self,out,stop,campaign,*,caps=None):
        self.out=Path(out);self.stop=stop;self.campaign=campaign
        self.caps=dict(pilot.CAPS if caps is None else caps);self.names=tuple(self.caps)
        ctx=mp.get_context('fork');self.lock=threading.RLock();self.local=threading.local();self.lock_pid=os.getpid()
        self.counts=ctx.Array('q',3*len(self.names),lock=False)
        self.sequence=ctx.Value('q',0,lock=False);self.claimed=set()
        self.planned=ctx.Array('b',448,lock=False);self.attempted=ctx.Array('b',448,lock=False)
        self.log=self.out/'ADAPTER_CALLS.jsonl';self.before_path=None

    @contextmanager
    def critical(self):
        # Parent signal handlers cannot interrupt the middle of a START record.
        previous=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGTERM,signal.SIGINT,signal.SIGHUP})
        # flock is released by the kernel if an owner dies. Open a distinct
        # descriptor per outer critical section (never reuse an inherited OFD).
        if self.lock_pid!=os.getpid():
            self.lock=threading.RLock();self.local=threading.local();self.lock_pid=os.getpid()
        try:
            with self.lock:
                depth=getattr(self.local,'depth',0);handle=None
                if depth==0:
                    handle=(self.out/'ADMISSION.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX)
                self.local.depth=depth+1
                try:yield
                finally:
                    self.local.depth=depth
                    if handle is not None:handle.close()
        finally:signal.pthread_sigmask(signal.SIG_SETMASK,previous)

    def _stopped(self):
        return self.stop.is_set() or any((self.out/n).exists() for n in ('STOP_REQUESTED.json','STOP.json'))

    def request_stop(self,reason):
        with self.critical():
            path=self.out/'STOP_REQUESTED.json'
            if not path.exists():managed.durable_new(path,dict(campaign=self.campaign,reason=str(reason),
                pid=os.getpid(),monotonic_ns=time.monotonic_ns(),retry=False))
            # Receipt precedes the event. No newer admission can cross this lock.
            if not self.stop.is_set():
                c.append(self.log,dict(event='STOP_REQUESTED',campaign=self.campaign,reason=str(reason),pid=os.getpid()))
            self.stop.set()

    def ensure_open(self):
        with self.critical():
            if self._stopped():raise RuntimeError('STOP: new work forbidden')

    @contextmanager
    def call(self,name,**fields):
        with self.critical():
            if self._stopped():raise RuntimeError('STOP before CALL_START: '+name)
            if name not in self.caps:raise ValueError('undeclared scientific call: '+name)
            i=self.names.index(name)*3
            if self.counts[i]>=self.caps[name]:
                self.request_stop('call budget exhausted: '+name)
                raise RuntimeError('call cap: '+name)
            self.counts[i]+=1;self.sequence.value+=1;serial=self.sequence.value
            c.append(self.log,dict(event='CALL_START',name=name,call_id=serial,campaign=self.campaign,
                pid=os.getpid(),monotonic_ns=time.monotonic_ns(),**fields))
        t=time.perf_counter()
        try:yield
        except BaseException as exc:
            with self.critical():
                self.counts[i+2]+=1
                c.append(self.log,dict(event='CALL_EXCEPTION',name=name,call_id=serial,pid=os.getpid(),
                    seconds=time.perf_counter()-t,type=type(exc).__name__))
                self.request_stop(name+': '+type(exc).__name__)
            raise
        else:
            with self.critical():
                self.counts[i+1]+=1
                c.append(self.log,dict(event='CALL_RETURN',name=name,call_id=serial,pid=os.getpid(),seconds=time.perf_counter()-t))

    def trial_slot(self,domain,trial):
        offset=0
        for d in pilot.domains():
            n=64 if d.stage=='train' else pilot.COUNTS[int(d.cell)]
            if d==domain:
                if type(trial)is not int or not 0<=trial<n:raise ValueError('fixed trial ID required')
                return offset+trial
            offset+=n
        raise ValueError('foreign trial domain')

    def accept_attempt(self,job):
        with self.critical():
            self.ensure_open();slot=self.trial_slot(job['domain'],job['trial'])
            ticket=__import__('hashlib').sha256((job['domain'].token+':'+str(job['trial'])).encode()).hexdigest()
            if not self.planned[slot] or self.attempted[slot] or job['ticket']!=ticket:
                self.request_stop('unplanned/duplicate/mixed worker attempt');raise ValueError('worker replay/identity forbidden')
            self.attempted[slot]=1
            c.append(self.log,dict(event='ATTEMPT_ACCEPTED',campaign=self.campaign,domain_token=job['domain'].token,
                trial_id=job['trial'],ticket=ticket,pid=os.getpid()))

    def snapshot(self):
        with self.critical():
            rows,tail=managed.audit.parse_records(self.log.read_bytes()) if self.log.exists() else ([],None)
            returns={n:set() for n in self.names};exceptions={n:set() for n in self.names}
            for row in rows:
                if row['event'] in ('CALL_RETURN','CALL_EXCEPTION'):
                    target=returns if row['event']=='CALL_RETURN' else exceptions
                    target[row['name']].add(row['call_id'])
            return dict(static_caps=self.caps,observed={n:dict(started=self.counts[3*i],
                returned=len(returns[n]),exceptions=len(exceptions[n]),
                in_flight=self.counts[3*i]-len(returns[n])-len(exceptions[n])) for i,n in enumerate(self.names)},
                incomplete_adapter_tail=tail,unresolved_means='UNKNOWN completion; never infer RETURN from shared memory alone',
                shared_counters={n:dict(started=self.counts[3*i],returned=self.counts[3*i+1],exceptions=self.counts[3*i+2])
                    for i,n in enumerate(self.names)},retry=0,extra_campaigns=0)


class Runtime(managed.Runtime):
    def __init__(self,out,gate):super().__init__(out,gate.stop);self.gate=gate
    def request_stop(self,reason):self.gate.request_stop(reason)


class CountedStreams:
    """Account only actual parent PRNG draws; worker replay is not a new draw."""
    def __init__(self,streams,gate):self.streams,self.gate=streams,gate
    def claim(self,domain,trial):
        self.gate.ensure_open()
        a,b=self.streams.claim(domain,trial)
        def count(source,kind):
            def draw():
                with self.gate.call('word',domain=domain.token,trial=trial,kind=kind):return source()
            return draw
        return count(a,'path'),count(b,'bit')


def prepare_adapter(original,gate):
    def prepared(sampler,streams,domain,trial,journal,words):
        # Short parent-only path generation/claim is linearized with STOP.
        with gate.critical():
            gate.ensure_open()
            if gate.before_path is not None:gate.before_path()
            quota=64 if domain.stage=='train' else pilot.COUNTS[int(domain.cell)]
            if domain not in pilot.domains() or type(trial)is not int or not 0<=trial<quota:
                raise ValueError('fixed domain/quota required')
            identity=(gate.campaign,domain.token,trial)
            if identity in gate.claimed:raise ValueError('trial replay forbidden')
            gate.claimed.add(identity);gate.planned[gate.trial_slot(domain,trial)]=1
            c.append(gate.out/'GLOBAL_TRIALS.jsonl',dict(campaign=gate.campaign,purpose=pilot.PURPOSE,
                domain_token=domain.token,trial_id=trial,event='TRIAL_ACCEPTED'))
            with gate.call('conditional_path',domain=domain.token,trial=trial):
                return original(sampler,streams,domain,trial,journal,words)
    return prepared


def _observe(stack,gate,owner,name,label,*,local=None):
    original=getattr(owner,name)
    @wraps(original)
    def call(*args,**kwargs):
        if label=='apply_exact':
            local['gates']+=1
            if local['gates']>955:
                gate.request_stop('per-probability max_gate_count exceeded');raise RuntimeError('955 gate cap')
        with gate.call(label,ticket=local.get('ticket') if local else None):return original(*args,**kwargs)
    stack.enter_context(patch.object(owner,name,call))


def process_memory():
    result={}
    try:
        for line in Path('/proc/self/smaps_rollup').read_text().splitlines():
            if line.startswith(('Rss:','Pss:','Private_Clean:','Private_Dirty:')):
                key,value,*_=line.split();result[key.rstrip(':')+'_bytes']=int(value)*1024
    except OSError:result['status']='UNKNOWN'
    return result


def worker_trial(job):
    """One frozen worker invocation plus finite counters, never a retry."""
    gate=_GATE
    if gate is None:raise RuntimeError('missing inherited campaign gate')
    gate.accept_attempt(job)
    d=job['domain'];out=gate.out;local=dict(gates=0,ticket=job['ticket'])
    c.append(out/f'RESOURCE_worker_{os.getpid()}.jsonl',dict(event='TRIAL_RESOURCE_START',ticket=job['ticket'],
        cpu_user=resource.getrusage(resource.RUSAGE_SELF).ru_utime,
        cpu_system=resource.getrusage(resource.RUSAGE_SELF).ru_stime,resources=obs.memory_snapshot(),process_memory=process_memory()))
    t=time.perf_counter()
    try:
        with gate.call('physical_trial',ticket=job['ticket']), ExitStack() as stack:
            facade,prices,sampler,scopes,certificate=c._SERVICES
            original=facade._MeasurementFacade__probability
            def probability(*args):
                with gate.call('probability',ticket=job['ticket']):return original(*args)
            stack.enter_context(patch.object(c,'_SERVICES',(n3.MeasurementFacade(probability,scopes),prices,sampler,scopes,certificate)))
            for owner,name,label in ((c.kernel.SavedEngine,'selected_circuit','selected_full_build'),
                (c.kernel,'apply_exact','apply_exact'),(c.kernel,'born_probability','Born'),
                (c.kernel,'threshold_count','bit_threshold'),(n3.SavedPrices,'price','price'),
                (c.module('rte_compiled_cost'),'transpile_and_measure_cost','compile')):
                _observe(stack,gate,owner,name,label,local=local)
            result=_ORIGINAL_WORKER(job)
            if 'reason' in result:gate.request_stop('worker failure: '+result['reason'])
            return result
    except BaseException as exc:
        gate.request_stop('worker exception: '+type(exc).__name__);raise
    finally:
        c.append(out/f'RESOURCE_worker_{os.getpid()}.jsonl',dict(event='TRIAL_RESOURCE_RETURN',ticket=job['ticket'],
            elapsed_wall_inclusive=time.perf_counter()-t,cpu_user=resource.getrusage(resource.RUSAGE_SELF).ru_utime,
            cpu_system=resource.getrusage(resource.RUSAGE_SELF).ru_stime,resources=obs.memory_snapshot(),
            process_memory=process_memory(),shared_pages_not_additive=True))


@contextmanager
def connection(gate):
    global _GATE
    if _GATE is not None:raise ValueError('nested or second campaign forbidden')
    _GATE=gate
    try:
        with patch.object(c,'prepare_trial',prepare_adapter(c.prepare_trial,gate)),patch.object(c,'worker_trial',worker_trial):yield
    finally:_GATE=None


class Collector:
    def __init__(self,raw,campaign,control,runtime,gate):
        self.raw,self.campaign,self.control,self.runtime,self.gate=raw,campaign,control,runtime,gate
        self.source_hash=pilot.check_source(raw.out,campaign,control)
        if (Path(runtime.out).resolve()!=Path(raw.out).resolve() or runtime.stop is not gate.stop
            or c._STOP_EVENT is not gate.stop or raw.sampler is not control.metadata.sampler
            or raw.trials or raw.costs):raise ValueError('fresh campaign/runtime/sampler binding required')
        if list(Path(raw.out).glob('worker_*.jsonl')) or any(p.is_file() for p in (Path(raw.out)/'cache').rglob('*')):
            raise ValueError('old worker/cache forbidden')

    def collect(self,axis,stage,quotas):
        self.gate.ensure_open()
        if pilot.check_source(self.raw.out,self.campaign,self.control)!=self.source_hash:raise ValueError('SOURCE changed')
        before,cbefore=len(self.raw.trials),len(self.raw.costs)
        rows,_=self.raw.group('A1' if stage=='train' else 'B1-OL',axis,stage,quotas)
        new=self.raw.trials[before:]
        return old.Envelope(self.campaign,pilot.PURPOSE,self.control.metadata.identity,self.control.identity,
            axis,stage,tuple(rows),tuple(self.raw.costs[cbefore:]),sum(r['cache_hit'] for r in new))

    def request_stop(self,reason):self.runtime.request_stop(reason)

    def finalize(self):
        errors=[]
        try:self.raw.pool.shutdown(wait=True,cancel_futures=True)
        except BaseException as exc:errors.append('drain '+str(exc))
        try:c.reconcile(self.raw.out)
        except BaseException as exc:errors.append('reconcile '+str(exc))
        attempts=self.runtime.classify()
        journal=self.raw.out/'JOURNAL.jsonl'
        rows,tail=managed.audit.parse_records(journal.read_bytes()) if journal.exists() else ([],None)
        trials=[dict(ticket=r['ticket'],campaign=self.campaign,domain=r['domain'],trial_id=r['trial'],
            classification=attempts.get(r['ticket'],'PLANNED_NO_DURABLE_WORKER_START')) for r in rows if r['event']=='TRIAL_PLANNED']
        priced=bits=known=None
        if not errors:
            account=json.loads((self.raw.out/'ACCOUNTING.json').read_text())['methods']
            if any(account[m]['bits_saved'] for m in ('B0','B1-O')):raise ValueError('foreign method in journal')
            bits=sum(r['bits_saved'] for r in account.values());priced=sum(r['priced_shots'] for r in account.values())
            known=sum(r['realized_RZ_priced'] for r in account.values())
        c.save(self.raw.out/'SIGNED_TAIL_ACCOUNTING.json',dict(trials=trials,bits_saved=bits,priced_trials=priced,known_RZ=known,
            total_RZ=known if not errors and tail is None and len(trials)==bits==priced else 'UNKNOWN',incomplete_parent_tail=tail,
            all_inflight_tails_included=True,unknown_not_zero=True,complete_campaign=False,
            authoritative_prices='ACCOUNTING.json' if not errors else 'UNKNOWN; inspect durable worker journals',errors=errors))
        c.save(self.raw.out/'BOUNDED_CALL_COUNTS.json',self.gate.snapshot())
        if errors:raise RuntimeError('; '.join(errors))
        counts=json.loads((self.raw.out/'CALL_COUNTS.json').read_text())
        if counts['trial_planned']>448 or counts['uniform53_words']>29120:raise ValueError('journal cap mismatch')
        return dict(authoritative_worker_reconciliation='ACCOUNTING.json',attempts='SIGNED_TAIL_ACCOUNTING.json',
            legacy_inclusive_worker_wall_intervals=self.raw.classical,
            legacy_cache_field='price minus compile, includes price circuit build; not pure cache',
            separate_classical_costs='ADAPTER_CALLS / RESOURCE_worker logs; nested intervals not additive')
