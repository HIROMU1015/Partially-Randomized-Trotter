"""NON-ADOPTED, ARTIFICIAL-ONLY 8-pair coordinator on the integrated F01 driver.

No production entrypoint/authority bootstrap. Parent owns all start and byte
reservations. Original scientific sources, Stream and run_pair stay unchanged.
"""
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from hashlib import sha256
import importlib.abc
import json
import mmap
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import resource
import sys
import time

sys.dont_write_bytecode = True
FORBIDDEN = ('g7_generator', 'g7_provider', 'return_aggregation', 'g6_f3_contract',
             'g6_f3_precision', 'g6_f3_generator', 'g9_native')


class BlockScience(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.rsplit('.', 1)[-1] in FORBIDDEN:
            raise RuntimeError('PARALLEL_ARTIFICIAL_ONLY_FORBIDS_SCIENCE_IMPORT:'+fullname)


sys.meta_path.insert(0, BlockScience())
from . import g6_w2c_b3_driver as D

STATES = ('NOT_STARTED', 'START_RESERVED', 'STARTED', 'IN_FLIGHT', 'COMPLETED',
          'FAILED', 'CANCELLED_BEFORE_START', 'TERMINATED_WITH_UNKNOWN_RESULT')
CODE = {name: i for i, name in enumerate(STATES)}
PHASES = (('correctness', 'baseline'), ('profile', 'baseline'), ('profile', 'instrumented'))


class Halt(RuntimeError):
    pass


class BundleFailure(Halt):
    def __init__(self, report):
        super().__init__('ARTIFICIAL_PARALLEL_BUNDLE_STOP')
        self.report = report


@dataclass(frozen=True)
class Settings:
    workers: int = 8
    mode: str = 'P1'
    inflight_cap: int = 2
    profile_n: int = 64
    workload: str = 'mixed'
    result_bytes: int = 1 << 36
    log_bytes: int = 1 << 28
    call_limit: int = 31710848
    faults: tuple = ()  # (pair_index, phase index, stage, trial index)
    parent_fault: str = ''
    collision: bool = False

    def check(self):
        D.need(self.mode in ('S', 'P1', 'P2') and 1 <= self.workers <= 8, 'SCHEDULER_SETTINGS')
        D.need(self.profile_n >= 16 and self.result_bytes > 0 and self.log_bytes > 0, 'ARTIFICIAL_SETTINGS')
        D.need(self.workload in ('light','mixed','heavy'), 'ARTIFICIAL_WORKLOAD')
        D.need(1 <= self.inflight_cap <= 8 and 0 <= self.call_limit <= 31710848, 'CALL_POLICY')


def safe_json(value):
    return json.dumps(value, separators=(',', ':'), ensure_ascii=True,
                      default=lambda x: {'exact': D.audit.canonical(D.exact(x))}).encode()


def own_io():
    try:
        return {k: int(v) for k,v in (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}
    except OSError:
        return {}


def plan_for(n):
    """Deterministic byte offsets; N-sized state storage, O(workers) Python state."""
    plan, offset = [], 0
    for phase_index, (phase, pass_name) in enumerate(PHASES):
        for pair_index in range(8):
            arm, layer = D.ARMS[pair_index//2], D.LAYERS[pair_index % 2]
            count = 16 if phase_index == 0 else (n[arm] if isinstance(n, dict) else n)
            plan.append(dict(key=f'{phase}_{pass_name}_{arm}_{layer}', phase_index=phase_index,
                             phase=phase, pass_name=pass_name, arm=arm, layer=layer,
                             pair_index=pair_index, n=count, offset=offset))
            offset += 2*count
    return plan, offset


class ByteBudget:
    def __init__(self, result, log):
        self.limit = {'result': result, 'log': log}
        self.reserved = Counter()
        self.observed = Counter()

    def reserve(self, channel, amount):
        D.need(channel in self.limit and type(amount) is int and amount >= 0, 'BYTE_RESERVATION')
        D.need(self.reserved[channel]+amount <= self.limit[channel], 'GLOBAL_'+channel.upper()+'_BYTE_LIMIT')
        self.reserved[channel] += amount

    def wrote(self, channel, amount):
        self.observed[channel] += amount
        D.need(self.observed[channel] <= self.reserved[channel], 'WRITE_WITHOUT_RESERVATION')


class Journal:
    def __init__(self, path, budget):
        self.file, self.budget, self.failed = Path(path).open('xb', buffering=0), budget, False

    def emit(self, row):
        data = safe_json(row)+b'\n'
        self.budget.reserve('log', len(data))
        try:
            view = memoryview(data)
            while view:
                size = self.file.write(view)
                D.need(type(size) is int and size > 0, 'JOURNAL_SHORT_WRITE')
                self.budget.wrote('log', size)
                view = view[size:]
        except BaseException:
            self.failed = True
            raise


class StateFile:
    def __init__(self, path, planned, budget):
        budget.reserve('log', planned)  # Logical retained bytes, not cumulative pwrite traffic.
        self.file = Path(path).open('x+b', buffering=0)
        try:
            self.file.truncate(planned)
            budget.wrote('log', planned)
            self.mapping = mmap.mmap(self.file.fileno(), planned)
        except BaseException:
            self.file.close()
            raise
        self.counts = Counter({'NOT_STARTED': planned})
        self.io_bytes = 0

    def set(self, index, name):
        old = STATES[self.mapping[index]]
        D.need(old not in ('COMPLETED', 'FAILED', 'CANCELLED_BEFORE_START', 'TERMINATED_WITH_UNKNOWN_RESULT'),
               'TERMINAL_TRIAL_REUSE')
        D.need(os.pwrite(self.file.fileno(), bytes([CODE[name]]), index) == 1, 'STATE_SHORT_WRITE')
        self.counts[old] -= 1
        self.counts[name] += 1
        self.io_bytes += 1

    def close(self):
        self.mapping.close()
        self.file.close()


class Client:
    def __init__(self, conn, stopped, parent_pid, reserved_exit=None):
        self.conn, self.stopped, self.parent_pid = conn, stopped, parent_pid
        self.pending, self.current, self.rpcs = None, None, Counter()
        self.reserved_exit = reserved_exit

    def rpc(self, op, **fields):
        before = D.clock()
        try:
            if os.getppid() != self.parent_pid:
                raise Halt('PARENT_GONE_NO_NEW_WORK')
            self.conn.send(dict(op=op, **fields))
            reply = self.conn.recv()
            if not reply.get('ok'):
                raise Halt(reply.get('reason', 'GLOBAL_STOP'))
            return reply
        except (EOFError, BrokenPipeError, ConnectionResetError) as exc:
            raise Halt('PARENT_CONNECTION_LOST') from exc
        finally:
            elapsed = D.delta(before, D.clock())
            self.rpcs['calls'] += 1
            self.rpcs['cpu_ns'] += elapsed[0]
            self.rpcs['wall_ns'] += elapsed[1]

    def confirm(self):
        if self.pending:
            self.rpc('COMPLETE', trial_id=self.pending)
            if self.current == self.pending:
                self.current = None
            self.pending = None

    def start(self, trial_id, mode):
        self.confirm()  # Next engine boundary proves F01 committed the previous emit.
        self.current = trial_id
        self.rpc('RESERVE', trial_id=trial_id)
        if self.reserved_exit is not None and trial_id.endswith('/X/'+str(self.reserved_exit)):
            os._exit(89)
        if self.stopped.value:
            self.rpc('CANCEL', trial_id=trial_id)
            raise Halt('CANCELLED_BEFORE_START')
        if mode == 'P2':
            self.rpc('START_CHECK', trial_id=trial_id)
            if self.stopped.value:
                self.rpc('CANCEL', trial_id=trial_id)
                raise Halt('CANCELLED_BEFORE_START')
        self.rpc('ENTER', trial_id=trial_id, worker_entry_ns=time.monotonic_ns(),
                 worker_cpu_ns=time.process_time_ns())

    def fail(self, reason, stage):
        return self.rpc('FAIL', reason=str(reason), stage=stage, trial_id=self.current,
                        worker_observed_ns=time.monotonic_ns(), worker_cpu_ns=time.process_time_ns())


class ReservedFile:
    """Before-write global reservation; unused short-write credit stays local.

    Normal partial writes do not double-charge. Failed/unknown credit is retained
    and never reassigned, avoiding an unsafe refund after an uncertain write.
    """
    def __init__(self, file, reserve, wrote, stage_fault=''):
        self.file, self.reserve, self.wrote, self.credit = file, reserve, wrote, 0
        self.fault, self.partial = stage_fault, False

    def write(self, data):
        if len(data) > self.credit:
            self.reserve(len(data)-self.credit)
            self.credit = len(data)
        if self.fault == 'write' and self.partial:
            raise OSError('ARTIFICIAL_PARTIAL_WRITE_FAILURE')
        view = data[:13] if self.fault == 'write' and not self.partial else data
        size = self.file.write(view)
        if size:
            self.credit -= size
            self.wrote(size)
        if self.fault == 'write':
            self.partial = True
        return size

    def flush(self):
        if self.fault == 'flush':
            raise OSError('ARTIFICIAL_FLUSH_FAILURE')
        return self.file.flush()

    def close(self):
        return self.file.close()


class EngineStream:
    def __init__(self, stream, client):
        self.stream, self.client = stream, client

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def emit(self, payload):
        self.stream.emit(payload)
        if 'trial_id' in payload:
            self.client.pending = payload['trial_id']


class ControlledEndpoint:
    def __init__(self, endpoint, client, mode):
        self.endpoint, self.client, self.mode = endpoint, client, mode

    def shared_identity(self):
        return self.endpoint.shared_identity()

    def sample(self, bits, trial_id):
        self.client.start(trial_id, self.mode)
        try:
            return self.endpoint.sample(bits, trial_id)
        except BaseException as exc:
            self.client.fail(exc, 'SAMPLING')
            raise

    def angle_for(self, result):
        try:
            return self.endpoint.angle_for(result)
        except BaseException as exc:
            self.client.fail(exc, 'ANGLE_AND_REQUERY')
            raise


def _worker(task, settings, target, conn, inherited, stopped, root, pins,
            endpoint_factory, bits_factory, watch):
    for parent_end in inherited:
        parent_end.close()
    reserved_exit = next((f[3] for f in settings.faults if f[:3] ==
                          (task['pair_index'], task['phase_index'], 'reserved_exit')), None)
    client = Client(conn, stopped, os.getppid(), reserved_exit)
    before, meter, stream, outcome, failure = D.clock(), D.Meter(D.clock), None, None, None
    baseline = D.memory_snapshot()
    io_before = own_io()
    observer = D.FunctionObserver([str(Path(root)/p) for p in pins if p.endswith('.py')] +
                                  [D.__file__, __file__, endpoint_factory.__code__.co_filename])
    try:
        with observer if task['pass_name'] == 'instrumented' else nullcontext():
            with meter.span('setup.source_check'):
                D.check_pins(root, pins)
            with meter.span('setup.constructor_and_preflight'):
                endpoint = ControlledEndpoint(endpoint_factory(task, settings), client, settings.mode)
            name = 'COLLISION' if settings.collision else task['key']
            stream = D.Stream(Path(target)/(name+'.jsonl'), meter,
                              byte_limit=settings.result_bytes, flush_rows=4096)
            fault = next((f[2] for f in settings.faults if f[0] == task['pair_index'] and
                          f[1] == task['phase_index'] and f[2] in ('write', 'flush')), '')
            stream.file = ReservedFile(stream.file,
                lambda n: client.rpc('WRITE_RESERVE', amount=n),
                lambda n: client.rpc('WRITE_ACK', amount=n), fault)
            cells = [dict(arm=task['arm'], layer=task['layer'], axis=a, N=task['n']) for a in D.AXES]
            def bits(axis):
                client.confirm()
                return bits_factory(task['arm'], axis)
            outcome = D.run_pair(endpoint, bits, cells, EngineStream(stream, client), meter,
                                 phase=task['phase'], pass_name=task['pass_name'], correctness_trials=16)
            client.confirm()
            stream.close()
    except BaseException as exc:
        failure = getattr(exc, 'b3_failure_provenance', None)
        if failure and client.pending and failure['last_completed_trial'] and \
                failure['last_completed_trial']['trial_id'] == client.pending:
            try:
                client.confirm()
            except BaseException:
                pass
        try:
            client.fail(exc, failure['current_attempt']['stage'] if failure else 'WORKER_SETUP_OR_CLEANUP')
        except BaseException:
            pass
        failure = {'type': type(exc).__name__, 'reason': str(exc), 'trial_failure_provenance': failure,
                   'committed_engine_summary_before_cleanup_failure': outcome}
        outcome = None
    finally:
        if stream and not stream.closed:
            try:
                stream.close()
            except BaseException as exc:
                if failure is None:
                    failure = {'type': type(exc).__name__, 'reason': str(exc)}
        total = D.delta(before, D.clock())
        row = dict(task=task, status='FAILED' if failure else 'OK', outcome=outcome, failure=failure,
                   pid=os.getpid(), cpu_ns=total[0], wall_ns=total[1], process_cpu_ns=time.process_time_ns(),
                   coarse=meter.rows, reconciliation=meter.reconcile(total),
                   fine=observer.meter.rows if task['pass_name'] == 'instrumented' else None,
                   fine_reconciliation=observer.meter.reconcile(total) if task['pass_name'] == 'instrumented' else None,
                   memory_baseline=baseline, memory=D.memory_snapshot(), control_rpc=dict(client.rpcs),
                   io={k:v-io_before.get(k,0) for k,v in own_io().items()},
                   output_bytes=stream.bytes if stream else 0, output_rows=stream.rows if stream else 0,
                   chain=stream.previous if stream else None, file_sha=stream.file_hash.hexdigest() if stream else None)
        try:
            conn.send({'op': 'END', 'report': row})
        except (BrokenPipeError, EOFError, OSError):
            pass
        if watch is not None:
            watch[task['phase_index']*8+task['pair_index']] = 1
        conn.close()


def owned_cpu(pids):
    ticks, valid = 0, True
    for pid in pids:
        try:
            text = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
            ticks += int(text[11])+int(text[12])
        except (OSError, ValueError):
            valid = False
    return ticks/os.sysconf('SC_CLK_TCK'), valid


class Coordinator:
    def __init__(self, settings, target, tasks, planned):
        self.before = D.clock()
        self.child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        self.settings, self.target, self.tasks, self.planned = settings, Path(target), tasks, planned
        self.budget = ByteBudget(settings.result_bytes, settings.log_bytes)
        self.journal = Journal(self.target/'control.jsonl', self.budget)
        try:
            self.states = StateFile(self.target/'trial_states.bin', planned, self.budget)
        except BaseException:
            self.journal.file.close()
            raise
        self.ctx = mp.get_context('fork')
        self.stopped = self.ctx.RawValue('b', 0)  # Parent-only writer; RPC is the authoritative gate.
        self.active, self.waiting, self.current = {}, [], {}
        self.results, self.failures, self.arrival, self.completion_order = [], [], 0, []
        self.reservations, self.started, self.max_inflight = 0, 0, 0
        self.stop_ns = self.stop_cpu_ns = self.pre_stop_worker_cpu = None
        self.pre_stop_cpu_valid = False
        self.log_errors, self.peak_rss, self.rss_samples = [], 0, 0
        self.launch_cost = Counter()
        try:
            self.journal.emit({'event': 'PLAN', 'schema': 'b3_parallel_artificial_v1', 'settings': asdict(settings),
                               'planned': planned, 'tasks': tasks, 'state_codes': CODE, 'retry': 0, 'resume': False})
        except BaseException:
            self.states.close()
            self.journal.file.close()
            raise

    def log(self, row):
        try:
            self.journal.emit(row)
        except BaseException as exc:
            self.log_errors.append(repr(exc))
            self.stopped.value = 1
            raise

    def stop(self, detail):
        self.arrival += 1
        record = dict(detail, parent_arrival=self.arrival, parent_observed_ns=time.monotonic_ns())
        self.failures.append(record)
        if self.stop_ns is None:
            self.stopped.value = 1
            self.stop_ns, self.stop_cpu_ns = time.monotonic_ns(), time.process_time_ns()
            alive, valid = owned_cpu([x['process'].pid for x in self.active.values()])
            reaped = resource.getrusage(resource.RUSAGE_CHILDREN)
            self.pre_stop_worker_cpu = alive + reaped.ru_utime+reaped.ru_stime - self.child_before.ru_utime-self.child_before.ru_stime
            self.pre_stop_cpu_valid = valid
            record['inflight_snapshot'] = {value['worker']: dict(value) for value in self.current.values()}
        try:
            self.log(dict(event='FAILURE_OBSERVED', **record))
        except BaseException:
            pass

    def sample_rss(self):
        total = 0
        for pid in [os.getpid()]+[x['process'].pid for x in self.active.values()]:
            try:
                for line in Path(f'/proc/{pid}/status').read_text().splitlines():
                    if line.startswith('VmRSS:'):
                        total += int(line.split()[1])*1024
                        break
            except OSError:
                pass
        self.rss_samples += 1
        self.peak_rss = max(self.peak_rss, total)

    def reserve_start(self, conn, request):
        row = self.active[conn]
        task = row['task']
        expected_index = row['next']
        axis = D.AXES[expected_index//task['n']] if expected_index < 2*task['n'] else None
        local = expected_index % task['n']
        expected_id = f"{task['phase']}/{task['pass_name']}/{task['arm']}/{task['layer']}/{axis}/{local}"
        D.need(not self.stopped.value, 'GLOBAL_STOP_NO_NEW_RESERVATION')
        D.need(request['trial_id'] == expected_id and axis is not None and conn not in self.current, 'TRIAL_ID_DUPLICATE_OR_ORDER')
        D.need(self.reservations < min(self.planned, self.settings.call_limit), 'GLOBAL_CALL_BUDGET')
        index = task['offset']+expected_index
        D.need(self.states.mapping[index] == CODE['NOT_STARTED'], 'TRIAL_ALREADY_RESERVED')
        self.states.set(index, 'START_RESERVED')
        self.reservations += 1
        row['next'] += 1
        self.current[conn] = dict(index=index, trial_id=expected_id, state='START_RESERVED',
                                  reserved_ns=time.monotonic_ns(), worker=task['key'])
        self.max_inflight = max(self.max_inflight, len(self.current))
        if self.settings.parent_fault == 'after_reserved':
            os._exit(86)  # Artificial crash of this one owned coordinator.
        conn.send({'ok': True})

    def drain_starts(self):
        cap = 1 if self.settings.mode == 'S' else (self.settings.inflight_cap if self.settings.mode == 'P2' else self.settings.workers)
        while self.waiting and (self.stopped.value or len(self.current) < cap):
            conn, request = self.waiting.pop(0)
            if conn not in self.active:
                continue
            try:
                self.reserve_start(conn, request)
            except BaseException as exc:
                self.stop({'type': type(exc).__name__, 'reason': str(exc), 'worker': self.active[conn]['task']['key']})
                conn.send({'ok': False, 'reason': str(exc)})

    def handle(self, conn, message):
        op, row = message['op'], self.active[conn]
        if op == 'RESERVE':
            self.waiting.append((conn, message))
            self.drain_starts()
            return
        if op == 'END':
            report = message['report']
            D.need(report['task'] == row['task'] and report['pid'] == row['process'].pid, 'WORKER_RESULT_IDENTITY')
            if report['status'] != 'OK':
                self.stop(dict(worker=row['task']['key'], terminal_failure=report['failure']))
            if conn in self.current:
                ticket = self.current.pop(conn)
                self.states.set(ticket['index'], 'FAILED' if report['failure'] else 'TERMINATED_WITH_UNKNOWN_RESULT')
            self.results.append(report)
            self.completion_order.append(row['task']['key'])
            row['report'] = report
            try:
                self.log({'event': 'WORKER_END', 'report': report})
            except BaseException as exc:
                self.stop({'terminal_log_error': repr(exc)})
            return
        try:
            if op == 'WRITE_RESERVE':
                self.budget.reserve('result', message['amount'])
            elif op == 'WRITE_ACK':
                self.budget.wrote('result', message['amount'])
            elif op == 'FAIL':
                self.stop(dict(message, worker=row['task']['key']))
            elif op in ('ENTER', 'START_CHECK', 'CANCEL', 'COMPLETE'):
                ticket = self.current[conn]
                D.need(ticket['trial_id'] == message['trial_id'], 'CURRENT_TRIAL_BINDING')
                if op in ('ENTER', 'START_CHECK') and self.stopped.value:
                    self.states.set(ticket['index'], 'CANCELLED_BEFORE_START')
                    del self.current[conn]
                    raise Halt('CANCELLED_BEFORE_START')
                if op == 'ENTER':
                    D.need(ticket['state'] == 'START_RESERVED', 'DUPLICATE_ENTER')
                    self.states.set(ticket['index'], 'STARTED')
                    self.states.set(ticket['index'], 'IN_FLIGHT')
                    ticket.update(state='IN_FLIGHT', worker_entry_ns=message['worker_entry_ns'])
                    self.started += 1
                elif op == 'CANCEL':
                    D.need(ticket['state'] == 'START_RESERVED', 'CANCEL_AFTER_ENTRY')
                    self.states.set(ticket['index'], 'CANCELLED_BEFORE_START')
                    del self.current[conn]
                elif op == 'COMPLETE':
                    D.need(ticket['state'] == 'IN_FLIGHT', 'COMPLETE_WITHOUT_ENTRY')
                    self.states.set(ticket['index'], 'COMPLETED')
                    del self.current[conn]
            else:
                raise Halt('UNKNOWN_WORKER_MESSAGE')
            conn.send({'ok': True})
        except BaseException as exc:
            self.stop(dict(worker=row['task']['key'], type=type(exc).__name__, reason=str(exc)))
            conn.send({'ok': False, 'reason': str(exc)})

    def run_phase(self, tasks, root, pins, endpoint_factory, bits_factory, watch, drain_only=False):
        next_task = 0
        concurrency = 1 if self.settings.mode == 'S' else self.settings.workers
        if not drain_only:
            try:
                self.log({'event': 'PHASE_BEGIN', 'phase': tasks[0]['phase_index'], 'time_ns': time.monotonic_ns()})
            except BaseException as exc:
                self.stop({'phase_log_error': repr(exc)})
        while self.active or (next_task < len(tasks) and not self.stopped.value):
            while len(self.active) < concurrency and next_task < len(tasks) and not self.stopped.value:
                task = tasks[next_task]
                next_task += 1
                parent, child = self.ctx.Pipe()
                inherited = [*self.active.keys(), parent]
                process = self.ctx.Process(target=_worker, args=(task, self.settings, self.target, child,
                    inherited, self.stopped, root, pins, endpoint_factory, bits_factory, watch))
                launch_before = D.clock()
                process.start()
                launch_elapsed = D.delta(launch_before, D.clock())
                self.launch_cost['cpu_ns'] += launch_elapsed[0]
                self.launch_cost['wall_ns'] += launch_elapsed[1]
                self.launch_cost['count'] += 1
                child.close()
                self.active[parent] = dict(process=process, task=task, next=0, report=None)
                if self.settings.parent_fault == 'dispatch' and next_task == 2:
                    raise Halt('ARTIFICIAL_PARENT_DISPATCH_FAILURE')
                try:
                    self.log({'event': 'WORKER_START', 'key': task['key'], 'pid': process.pid})
                except BaseException as exc:
                    self.stop({'worker_start_log_error': repr(exc)})
            if not self.active:
                break
            ready = wait(list(self.active)+[r['process'].sentinel for r in self.active.values()], timeout=.05)
            for conn in list(self.active):
                row = self.active[conn]
                if conn in ready or (row['process'].sentinel in ready and conn.poll()):
                    try:
                        message = conn.recv()
                        self.handle(conn, message)
                    except EOFError:
                        if row['report'] is None:
                            self.stop(dict(worker=row['task']['key'], reason='WORKER_EOF_RESULT_UNKNOWN'))
                            if conn in self.current:
                                ticket = self.current.pop(conn)
                                self.states.set(ticket['index'], 'TERMINATED_WITH_UNKNOWN_RESULT')
                            row['report'] = dict(task=row['task'], pid=row['process'].pid,
                                status='LOST', outcome=None, failure={'reason': 'TERMINATED_WITH_UNKNOWN_RESULT'},
                                cpu_ns=None, wall_ns=None, output_bytes=None)
                            self.results.append(row['report'])
                    except BaseException as exc:
                        self.stop(dict(worker=row['task']['key'], parent_handler_error=repr(exc)))
                        try:
                            conn.send({'ok': False, 'reason': repr(exc)})
                        except OSError:
                            pass
                if row['report'] is not None:
                    row['process'].join()
                    if row['process'].exitcode:
                        self.stop(dict(worker=row['task']['key'], exitcode=row['process'].exitcode))
                    conn.close()
                    self.active.pop(conn)
            self.drain_starts()
            if self.rss_samples % 128 == 0:
                self.sample_rss()
            else:
                self.rss_samples += 1
        if not drain_only:
            try:
                self.log({'event': 'PHASE_END', 'phase': tasks[0]['phase_index'], 'stopped': bool(self.stopped.value)})
            except BaseException as exc:
                self.stop({'phase_end_log_error': repr(exc)})

    def report(self):
        elapsed = D.delta(self.before, D.clock())
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        child_s = usage.ru_utime+usage.ru_stime-self.child_before.ru_utime-self.child_before.ru_stime
        by_key = {r['task']['key']: r['status'] for r in self.results}
        return dict(schema='R1_PARALLEL_8PAIR_ARTIFICIAL_RESULT_V1', settings=asdict(self.settings),
                    planned=self.planned, reserved=self.reservations, started_observed=self.started,
                    states={name:self.states.counts[name] for name in STATES}, max_inflight=self.max_inflight,
                    complete=not self.stopped.value and self.states.counts['COMPLETED'] == self.planned,
                    failures=self.failures, representative_failure=self.failures[0] if self.failures else None,
                    completion_order=self.completion_order, contract_order=[t['key'] for t in self.tasks],
                    all_pair_states={t['key']: by_key.get(t['key'], 'NOT_STARTED') for t in self.tasks},
                    pairs=sorted(self.results, key=lambda r: (r['task']['phase_index'], r['task']['pair_index'])),
                    parent_cpu_ns=elapsed[0], wall_ns=elapsed[1], worker_cpu_ns=round(child_s*1e9),
                    total_cpu_ns=elapsed[0]+round(child_s*1e9),
                    output=dict(limits=self.budget.limit, reserved=dict(self.budget.reserved), observed=dict(self.budget.observed)),
                    peak_sampled_rss_bytes=self.peak_rss, rss_is_shared_page_sum_not_unique_memory=True,
                    log_errors=self.log_errors, state_update_io_bytes=self.states.io_bytes,
                    worker_launch_cost=dict(self.launch_cost),
                    stop_to_join_ns=time.monotonic_ns()-self.stop_ns if self.stop_ns else None,
                    post_stop_parent_cpu_ns=time.process_time_ns()-self.stop_cpu_ns if self.stop_cpu_ns else None,
                    post_stop_worker_cpu_estimate_ns=round(max(0, child_s-self.pre_stop_worker_cpu)*1e9)
                        if self.stop_ns and self.pre_stop_cpu_valid else None,
                    post_stop_cpu_resolution_ns=round(self.settings.workers/os.sysconf('SC_CLK_TCK')*1e9),
                    retry=0, resume=False, mandatory_STOP=True, artificial_only=True)


def validate_success(rows):
    traces, prefixes, passes = {}, {}, {}
    for row in rows:
        task, result = row['task'], row['outcome']
        D.need(row['status'] == 'OK' and result is not None, 'PAIR_NOT_COMPLETED')
        D.need(sum(result['counts'].values()) == 2*task['n'], 'PAIR_N_NOT_COMPLETE')
        for axis in result['axes']:
            key = task['arm'], axis['axis']
            digest = axis['first_correctness_prefix_sha256']
            D.need(key not in prefixes or prefixes[key] == digest, 'CORRECTNESS_PREFIX_MISMATCH')
            prefixes[key] = digest
        key = task['phase_index'], task['arm']
        D.need(key not in traces or traces[key] == result['trace_sha256'], 'A_B_TRACE_MISMATCH')
        traces[key] = result['trace_sha256']
        if task['phase_index']:
            key = task['arm'], task['layer']
            D.need(key not in passes or passes[key] == result['trace_sha256'], 'BASELINE_INSTRUMENTED_TRACE_MISMATCH')
            passes[key] = result['trace_sha256']


def execute_artificial_bundle(candidate, *, root, output, approval_port, settings,
                              endpoint_factory, bits_factory, watch=None):
    """Test-only host injection target, never a production parallel execute API."""
    settings.check()
    D.need(getattr(approval_port, 'artificial', False) is True, 'ARTIFICIAL_PORT_ONLY')
    guard = D.Boundary()
    control = None
    try:
        receipt = guard.enter(candidate, approval_port)
        D.need(receipt.get('artificial_only') is True, 'ARTIFICIAL_RECEIPT_ONLY')
        root, output = Path(root), Path(output)
        D.need(root.is_relative_to(Path('/tmp')) and root.name.startswith('b3-r1-artificial-'), 'PRIVATE_ARTIFICIAL_ROOT_ONLY')
        D.need(output == root/candidate['output_relative'] and not output.exists(), 'EXCLUSIVE_OUTPUT_NAMESPACE')
        D.check_pins(root, candidate['source_pins'])
        policy_document = (root/'parallel-policy.json').read_text()
        policy = json.loads(policy_document)
        D.need(policy['settings'] == json.loads(json.dumps(asdict(settings))), 'PARALLEL_POLICY_BINDING')
        D.need(policy['source_sha256'] == sha256(Path(__file__).read_bytes()).hexdigest(), 'PARALLEL_SOURCE_IDENTITY')
        D.need(policy['fixture_sha256'] == sha256(Path(endpoint_factory.__code__.co_filename).read_bytes()).hexdigest(),
               'ARTIFICIAL_FIXTURE_IDENTITY')
        if settings.parent_fault == 'before_output':
            raise Halt('ARTIFICIAL_OUTPUT_BEFORE')
        output.mkdir()
        tasks, planned = plan_for(settings.profile_n)
        control = Coordinator(settings, output, tasks, planned)
        for phase_index in range(3):
            if control.stopped.value:
                break
            control.run_phase([t for t in tasks if t['phase_index'] == phase_index], root,
                              candidate['source_pins'], endpoint_factory, bits_factory, watch)
            try:
                validate_success(control.results)
                D.check_pins(root, candidate['source_pins'])
                D.need((root/'parallel-policy.json').read_text() == policy_document, 'POLICY_CHANGED')
            except BaseException as exc:
                control.stop(dict(phase=phase_index, validation_error=repr(exc)))
        report = control.report()
        if not report['complete'] or len(report['pairs']) != 24:
            raise BundleFailure(report)
        final_meter = D.Meter(D.clock)
        final_before = D.clock()
        final_stream = D.Stream(output/'summary.jsonl', final_meter, byte_limit=settings.result_bytes, flush_rows=1)
        final_stream.file = ReservedFile(final_stream.file, lambda n: control.budget.reserve('result', n),
                                         lambda n: control.budget.wrote('result', n))
        try:
            final_stream.emit(report)
        finally:
            final_stream.close()
        control.log({'event': 'COMPLETION_RECEIPT', 'summary_chain': final_stream.previous,
                     'summary_sha': final_stream.file_hash.hexdigest(), 'all_workers_joined': True})
        report['finalization'] = dict(coarse=final_meter.rows,
            elapsed_ns=D.delta(final_before, D.clock()), receipt_write_is_shared=True)
        report['output'] = dict(limits=control.budget.limit, reserved=dict(control.budget.reserved), observed=dict(control.budget.observed))
        return report
    except BaseException as exc:
        if control is not None:
            if not isinstance(exc, BundleFailure):
                control.stop({'parent_failure': repr(exc)})
            if control.active:
                control.run_phase([], root, candidate['source_pins'], endpoint_factory,
                                  bits_factory, watch, drain_only=True)
            try:
                if settings.parent_fault == 'stop_save':
                    raise OSError('ARTIFICIAL_STOP_SAVE_FAILURE')
                control.log({'event': 'MANDATORY_STOP', 'complete': False, 'states': dict(control.states.counts)})
            except BaseException as secondary:
                exc.secondary_stop_error = repr(secondary)
            if isinstance(exc, BundleFailure):
                exc.report = control.report()
                if hasattr(exc, 'secondary_stop_error'):
                    exc.report['secondary_stop_error'] = exc.secondary_stop_error
            if not isinstance(exc, BundleFailure):
                wrapped = BundleFailure(control.report())
                wrapped.primary = repr(exc)
                raise wrapped from exc
        raise
    finally:
        if control is not None:
            control.states.close()
            control.journal.file.close()
        guard.stop()
