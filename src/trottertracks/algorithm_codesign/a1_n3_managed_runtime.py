"""Lifecycle adapter; no scientific inputs, numerical kernels or RNG here."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager, ExitStack
from functools import wraps
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from unittest.mock import patch

from . import a1_n3_lifecycle_observer as old
from . import a1_n3_interruption_diagnostics as audit

_WORKER_LOG = None
_WORKER_STOP = None


def durable_new(path, value):
    """Exclusive receipt, including parent-directory fsync; never overwrite."""
    path = Path(path)
    with path.open('x') as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def owned_signal(identity, number, *, group=False):
    """Only the still-matching job process/session recorded by this caller."""
    current = old.process_identity(identity['pid'])
    if any(current.get(k) != identity.get(k) for k in ('pid','start_ticks','boot_id','pgid','sid')):
        return 'IDENTITY_GONE_OR_CHANGED'
    status = Path('/proc',str(identity['pid']),'status').read_text()
    if int(next(x.split()[1] for x in status.splitlines() if x.startswith('Uid:'))) != os.getuid():
        raise PermissionError('refusing another UID')
    if group:
        if identity['pgid'] != identity['pid'] or identity['sid'] != identity['pid']:
            raise ValueError('job group was not a private new session')
        os.killpg(identity['pgid'],number)
    else:
        os.kill(identity['pid'],number)
    return 'SENT_TO_OWN_JOB'


def worker_identities(out):
    for path in Path(out).glob('WORKER_*.json'):
        if path.stem.removeprefix('WORKER_').isdigit():
            yield json.loads(path.read_text())


def cleanup_owned(out,identity):
    result=owned_signal(identity,signal.SIGTERM,group=True)
    records=[dict(role='parent_group',result=result)]
    if result=='IDENTITY_GONE_OR_CHANGED':
        for worker in worker_identities(out):
            if worker.get('pgid')!=identity.get('pgid') or worker.get('sid')!=identity.get('sid'):
                raise ValueError('worker outside recorded job session')
            records.append(dict(pid=worker['pid'],result=owned_signal(worker,signal.SIGTERM),
                exit_code='UNKNOWN: original wait owner is gone'))
    return records


def worker_init(out, stop, initializer, initargs):
    global _WORKER_LOG,_WORKER_STOP
    out = Path(out);_WORKER_LOG=out/f'OBS_worker_{os.getpid()}.jsonl';_WORKER_STOP=stop
    identity=old.process_identity(os.getpid())
    durable_new(out/f'WORKER_{os.getpid()}.json',identity)
    old.event(_WORKER_LOG,'WORKER_START',identity=identity,
              resources=old.memory_snapshot(cgroup_record=identity.get('cgroup','UNKNOWN')))
    def received(number, _frame):
        old.event(_WORKER_LOG,'SIGNAL_RECEIVED',signal=number)
        signal.signal(number,signal.SIG_DFL)
        os.kill(os.getpid(),number)
    for number in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
        signal.signal(number,received)
    if initializer is not None:
        initializer(*initargs)


def worker_call(function,args,kwargs):
    if _WORKER_STOP.is_set():
        raise RuntimeError('STOP before next owned task')
    with old.traced_call(_WORKER_LOG,'owned-task',function.__name__):
        return function(*args,**kwargs)


class Runtime:
    def __init__(self,out,stop=None):
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
        self.log=self.out/'OBS_parent.jsonl'
        self.stop=stop if stop is not None else mp.get_context('fork').Event()
        self.lock=threading.RLock()
        old.event(self.log,'PARENT_START',identity=old.process_identity(os.getpid()))

    def request_stop(self,reason):
        with self.lock:
            path=self.out/'STOP_REQUESTED.json'
            if not path.exists():
                durable_new(path,dict(reason=reason,monotonic_ns=time.monotonic_ns(),pid=os.getpid(),retry=False))
                old.event(self.log,'STOP_REQUESTED',reason=reason)
            self.stop.set()

    @contextmanager
    def signals(self):
        previous={}
        def received(number,_frame):
            old.event(self.log,'SIGNAL_RECEIVED',signal=number)
            self.request_stop('parent signal '+str(number))
            raise SystemExit(128+number)
        for number in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
            previous[number]=signal.signal(number,received)
        try:
            yield
        finally:
            for number,handler in previous.items():signal.signal(number,handler)

    def classify(self):
        bits,prices,started=set(),set(),set()
        incomplete=[]
        for path in self.out.glob('worker_*.jsonl'):
            rows,tail=audit.parse_records(path.read_bytes())
            if tail:incomplete.append(dict(file=path.name,**tail))
            for row in rows:
                ticket=row.get('ticket')
                if row['event']=='WORKER_STARTED':
                    if ticket in started:raise ValueError('duplicate attempt start')
                    started.add(ticket)
                elif row['event']=='BIT_SAVED':
                    if ticket in bits:raise ValueError('duplicate durable bit')
                    bits.add(ticket)
                elif row['event']=='PRICE_SAVED':
                    if ticket not in bits:raise ValueError('price before durable bit')
                    if ticket in prices:raise ValueError('duplicate durable price')
                    prices.add(ticket)
        result={t:('BIT_AND_PRICE_DURABLE' if t in prices else
                   'BIT_DURABLE_PRICE_MISSING' if t in bits else 'ATTEMPT_NO_DURABLE_BIT') for t in started}
        old.atomic_write_json(self.out/'ATTEMPTS.json',dict(trials=result,incomplete_tails=incomplete,
            UNKNOWN_not_zero=True,retry=False))
        return result


class ManagedPool:
    """Wrap the frozen Python3.12 pool without a second waitpid owner."""
    def __init__(self,runtime,**kwargs):
        self.runtime=runtime;self.processes={}
        initializer=kwargs.pop('initializer',None);initargs=kwargs.pop('initargs',())
        self.pool=ProcessPoolExecutor(**kwargs,initializer=worker_init,
            initargs=(runtime.out,runtime.stop,initializer,initargs))

    def submit(self,function,*args,**kwargs):
        if self.runtime.stop.is_set():raise RuntimeError('STOP before submit')
        future=self.pool.submit(worker_call,function,args,kwargs)
        # Private registry is explicitly version-bound and artificially tested.
        self.processes.update(self.pool._processes or {})
        def completed(f):
            if f.cancelled():return
            try:
                result=f.result()
                if isinstance(result,dict) and 'reason' in result:self.runtime.request_stop('worker failure result')
            except BaseException as exc:self.runtime.request_stop(type(exc).__name__)
        future.add_done_callback(completed)
        return future

    def shutdown(self,wait=True,cancel_futures=True):
        if self.runtime.stop.is_set():self.runtime.request_stop('STOP before pool drain')
        old.event(self.runtime.log,'POOL_DRAIN_START',stop_receipt=(self.runtime.out/'STOP_REQUESTED.json').exists())
        try:self.pool.shutdown(wait=wait,cancel_futures=cancel_futures)
        finally:
            codes={str(pid):old.wait_status(process.exitcode) for pid,process in self.processes.items()}
            old.atomic_write_json(self.runtime.out/'WORKER_EXIT_CODES.json',codes)
            old.event(self.runtime.log,'POOL_DRAIN_RETURN',exit_codes=codes)


@contextmanager
def connect_legacy(campaign,runtime):
    """Attach observers to a caller-provided frozen module, never run it.

    A future B runner must provide fresh output/seed/profile itself; the old
    fixed run() and its invocation marker are never called or changed here.
    """
    def factory(**kwargs):
        if getattr(campaign,'_STOP_EVENT',None) is not None:runtime.stop=campaign._STOP_EVENT
        return ManagedPool(runtime,**kwargs)
    def trace(function,label):
        @wraps(function)
        def wrapped(*args,**kwargs):
            log=_WORKER_LOG or runtime.log
            with old.traced_call(log,'owned-call',label):return function(*args,**kwargs)
        return wrapped
    targets=[(campaign.kernel.SavedEngine,'selected_circuit','selected_full_build'),
             (campaign.kernel.SavedEngine,'probability','exact_engine'),
             (campaign.kernel,'born_probability','Born'),
             (campaign.module('rte_compiled_cost'),'transpile_and_measure_cost','compile')]
    with ExitStack() as stack:
        stack.enter_context(patch.object(campaign,'ProcessPoolExecutor',factory))
        for owner,name,label in targets:
            stack.enter_context(patch.object(owner,name,trace(getattr(owner,name),label)))
        yield


def run_parent(argv,out,cwd,*,simulate_manager_loss=False):
    """One parent, no retry; the outer cron recorder owns this manager."""
    out=Path(out);log=out/'MANAGER.jsonl'
    old.event(log,'MANAGER_START',identity=old.process_identity(os.getpid()),restart=False)
    with (out/'parent.log').open('xb') as stream:
        child=subprocess.Popen(argv,cwd=cwd,stdin=subprocess.DEVNULL,stdout=stream,
                               stderr=subprocess.STDOUT,start_new_session=True)
        identity=old.process_identity(child.pid)
        durable_new(out/'PARENT_IDENTITY.json',identity)
        old.event(log,'PARENT_STARTED',identity=identity,
                  resources=old.memory_snapshot(cgroup_record=identity.get('cgroup','UNKNOWN')))
        if simulate_manager_loss:
            while not (out/'BIT_READY').exists():time.sleep(.01)
            os.kill(os.getpid(),signal.SIGKILL)
        while True:
            try:code=child.wait(timeout=.25);break
            except subprocess.TimeoutExpired:
                old.event(log,'HEARTBEAT',identity=old.process_identity(child.pid),
                    resources=old.memory_snapshot(cgroup_record=identity.get('cgroup','UNKNOWN')))
        status=old.wait_status(code)
        old.event(log,'PARENT_WAIT_STATUS',**status)
        if code!=0:
            runtime=Runtime(out);runtime.request_stop('parent abnormal exit')
            old.atomic_write_json(out/'OWN_CLEANUP.json',dict(records=cleanup_owned(out,identity)))
        stream.flush();os.fsync(stream.fileno())
    Runtime(out).classify()
    old.atomic_write_json(out/'PARENT_EXIT.json',status)
    return code
