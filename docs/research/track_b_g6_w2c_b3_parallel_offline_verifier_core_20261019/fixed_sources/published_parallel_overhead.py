"""NON-ADOPTED artificial-only overhead experiments; frozen P1 remains intact.

No production entrypoint, science dependency, authority or claim bootstrap.
Scoped patches apply only in the caller's dedicated artificial process.
"""
from collections import Counter
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

from . import g6_w2c_b3_parallel_candidate as P
D = P.D
BASE_OBSERVER, BASE_CLIENT, BASE_COORDINATOR, BASE_STATE = D.FunctionObserver, P.Client, P.Coordinator, P.StateFile


class ScopedObserver(BASE_OBSERVER):
    def __init__(self, paths, read_clock=D.clock):
        super().__init__([*paths, __file__], read_clock)


class CachedObserver(ScopedObserver):
    """Same call/return selection and exclusive partition; cached code labels.

    Each callback uses one clock read like the original Meter observer. No
    function is sampled, suppressed or shifted out of the measured bundle.
    The outer CPU still includes all callback work and residual overhead.
    """
    def __init__(self, paths, read_clock=D.clock):
        super().__init__(paths, read_clock)
        self.cache, self.read_clock = {}, read_clock

    def hook(self,frame,event,arg):
        if event not in ('call','return'):return
        code=frame.f_code
        try:descriptor=self.cache[id(code)]
        except KeyError:
            name=getattr(code,'co_qualname',code.co_name)
            key=(Path(code.co_filename).name+':'+str(code.co_firstlineno)+':'+name) \
                if code.co_filename in self.paths and name not in ('clock','delta','need') else None
            descriptor=[code,key,'self' in code.co_varnames or 'self' in code.co_freevars or 'self' in code.co_cellvars,None]
            self.cache[id(code)]=descriptor
        if descriptor[1] is None:return
        if descriptor[2] and isinstance(frame.f_locals.get('self'),(D.Meter,BASE_OBSERVER)):return
        now=self.read_clock()
        if event=='call':self.frames.append([frame,descriptor,now[0],now[1],0,0])
        else:
            if not self.frames or self.frames[-1][0] is not frame:raise D.Stop('PROFILE_STACK_MISMATCH')
            item=self.frames.pop();cpu,wall=now[0]-item[2],now[1]-item[3]
            if cpu<item[4] or wall<item[5]:raise D.Stop('NEGATIVE_EXCLUSIVE_COST')
            if self.frames:
                self.frames[-1][4]+=cpu;self.frames[-1][5]+=wall
            row=descriptor[3]
            if row is None:
                row=self.meter.rows.get(descriptor[1])
                if row is None:
                    row=dict(calls=0,cpu_ns=0,wall_ns=0);self.meter.rows[descriptor[1]]=row
                descriptor[3]=row
            row['calls']+=1;row['cpu_ns']+=cpu-item[4];row['wall_ns']+=wall-item[5]

class DisabledObserver(ScopedObserver):
    """Explicit diagnostic ablation, never an equivalent research observer."""
    def __enter__(self):
        D.need(sys.getprofile() is None,'EXISTING_THREAD_PROFILER')
        return self


class CoalescedClient(BASE_CLIENT):
    """Keep RESERVE and ENTER gates; attach prior completion/write ACK to RPC.

    No future call block or output block is preauthorized. Before every actual
    write the parent still reserves its exact bytes. Unacknowledged credit is
    never released. COMPLETE follows the same original F01 commit boundary.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ack_pending = 0

    def rpc(self, op, **fields):
        if op == 'WRITE_ACK':
            self.ack_pending += fields['amount']
            self.rpcs['coalesced_write_acks'] += 1
            return {'ok':True}
        if self.ack_pending:
            fields['wrote_before'] = self.ack_pending
            self.ack_pending = 0  # Never retry after send/ACK uncertainty.
        return super().rpc(op, **fields)

    def start(self, trial_id, mode):
        previous, self.pending = self.pending, None
        self.current = trial_id
        self.rpc('RESERVE',trial_id=trial_id,complete_before=previous)
        if self.reserved_exit is not None and trial_id.endswith('/X/'+str(self.reserved_exit)):
            os._exit(89)
        if self.stopped.value:
            self.rpc('CANCEL',trial_id=trial_id)
            raise P.Halt('CANCELLED_BEFORE_START')
        if mode == 'P2':
            self.rpc('START_CHECK',trial_id=trial_id)
            if self.stopped.value:
                self.rpc('CANCEL',trial_id=trial_id)
                raise P.Halt('CANCELLED_BEFORE_START')
        self.rpc('ENTER',trial_id=trial_id,worker_entry_ns=time.monotonic_ns(),
                 worker_cpu_ns=time.process_time_ns())


class AccountingCoordinator(BASE_COORDINATOR):
    def __init__(self, *args, **kwargs):
        self.messages = Counter()
        super().__init__(*args, **kwargs)

    def handle(self, conn, message):
        self.messages[message['op']] += 1
        try:
            amount = message.pop('wrote_before',0)
            D.need(type(amount) is int and amount>=0,'INVALID_WRITE_ACK')
            if amount:self.budget.wrote('result',amount)
            previous = message.pop('complete_before',None)
            if previous:
                ticket = self.current[conn]
                D.need(ticket['trial_id']==previous and ticket['state']=='IN_FLIGHT','PREVIOUS_COMMIT_BINDING')
                self.states.set(ticket['index'],'COMPLETED')
                del self.current[conn]
        except BaseException as exc:
            self.stop(dict(worker=self.active[conn]['task']['key'],reason=str(exc)))
            conn.send({'ok':False,'reason':str(exc)})
            return
        return super().handle(conn,message)

    def report(self):
        value = super().report()
        value['parent_messages'] = dict(self.messages)
        if hasattr(self.states,'timings'):
            value['diagnostic_state_cost'] = dict(self.states.timings)
        return value


class TimedState(BASE_STATE):
    def __init__(self,*args,**kwargs):
        self.timings = Counter()
        super().__init__(*args,**kwargs)

    def set(self,index,name):
        before = D.clock()
        try:return super().set(index,name)
        finally:
            cpu,wall = D.delta(before,D.clock())
            self.timings['calls'] += 1
            self.timings['cpu_ns'] += cpu
            self.timings['wall_ns'] += wall


@contextmanager
def install(transport='legacy',observer='legacy',diagnostic_state=False):
    D.need(transport in ('legacy','coalesced') and observer in ('legacy','cached','off'),'OVERHEAD_VARIANT')
    from contextlib import ExitStack
    with ExitStack() as stack:
        stack.enter_context(patch.object(D,'FunctionObserver',dict(legacy=ScopedObserver,cached=CachedObserver,off=DisabledObserver)[observer]))
        stack.enter_context(patch.object(P,'Coordinator',AccountingCoordinator))
        if transport=='coalesced':stack.enter_context(patch.object(P,'Client',CoalescedClient))
        if diagnostic_state:stack.enter_context(patch.object(P,'StateFile',TimedState))
        yield


def prepare(F,root,settings,transport,observer):
    """Bind experimental code and mode through fake authority input pins."""
    c,ledger,auth = F.fixture(root,settings)
    raw = Path(__file__).read_bytes()
    (root/'overhead-source.py').write_bytes(raw)
    policy = dict(artificial_only=True,transport=transport,observer=observer,
                  source_sha256=sha256(raw).hexdigest(),settings=F.asdict(settings))
    (root/'overhead-policy.json').write_text(json.dumps(policy,sort_keys=True))
    for name in ('overhead-source.py','overhead-policy.json'):
        c['input_pins'][name] = sha256((root/name).read_bytes()).hexdigest()
    return c,ledger,F.Q.authorize(c,root,ledger)


def invoke(F,root,c,ledger,auth,settings,*,transport='legacy',observer='legacy',diagnostic_state=False,**kwargs):
    policy = json.loads((root/'overhead-policy.json').read_text())
    D.need(policy['transport']==transport and policy['observer']==observer and
           policy['source_sha256']==sha256(Path(__file__).read_bytes()).hexdigest(),'OVERHEAD_SOURCE_POLICY')
    with install(transport,observer,diagnostic_state):
        return F.invoke(root,c,ledger,auth,settings,**kwargs)
