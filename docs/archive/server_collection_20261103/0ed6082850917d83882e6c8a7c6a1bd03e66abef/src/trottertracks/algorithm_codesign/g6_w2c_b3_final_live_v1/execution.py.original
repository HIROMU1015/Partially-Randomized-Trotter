"""Versioned worker/pair projection. Frozen originals are never patched.

Only pair coverage classification and the worker target/import path differ.
F01 publication, RESERVE/ENTER/COMPLETE, cleanup and Storage V2 are retained.
"""
from .guard import install
install()
from pathlib import Path
from collections import Counter
from hashlib import sha256
from contextlib import nullcontext
from multiprocessing.connection import wait
import os,sys,time
from .. import g6_w2c_b3_parallel_driver as D
from .. import g6_w2c_b3_parallel_scheduler as P
from ..g6_w2c_b3_real_adapter_v1.bridge import Coordinator as PreviousCoordinator

audit,need,identity,canonical,exact,AXES=D.audit,D.need,D.identity,D.canonical,D.exact,D.AXES
INPUT=D.INPUT
memory_snapshot=D.memory_snapshot
Client,BlockScience,CachedObserver=P.Client,P.BlockScience,P.CachedObserver
ControlledEndpoint,ReservedFile,EngineStream=P.ControlledEndpoint,P.ReservedFile,P.EngineStream
Halt,own_io=P.Halt,P.own_io

def run_pair(endpoint, bits_factory, cells, stream, meter, *, phase, pass_name,
             correctness_trials, memory_read=memory_snapshot):
    """Streaming execution engine with supplied endpoints. No RNG construction here.

    B accepts angle work per accepted event. A returns frozen ratios only.
    All completed/failed prefixes are retained; failure halts the whole pair.
    """
    need(len(cells) == 2 and tuple(c['axis'] for c in cells) == AXES and
         all((c['arm'], c['layer'], c['N']) ==
             (cells[0]['arm'], cells[0]['layer'], cells[0]['N']) for c in cells), 'PAIR_AXIS_N_BINDING')
    # Published as one reference only after emit returns. Candidate hash/counter
    # copies never mutate the last completed prefix on a failed attempt.
    committed = {'counts': Counter(), 'digest': sha256(), 'last_trial': None}
    axis_summaries = [];max_rss = None
    current_result = current_event = pending = None

    def context(cell=None, trial=None, stage='PAIR_IDENTITY'):
        return {'axis': cell['axis'] if cell else None, 'trial_index': trial,
                'trial_id': (f"{phase}/{pass_name}/{cell['arm']}/{cell['layer']}/{cell['axis']}/{trial}"
                             if cell is not None and trial is not None else None),
                'stage': stage, 'result_acquired': False, 'event_present': False,
                'record_constructed': False, 'candidate_ready': False,
                'stream_emit_started': False, 'stream_emit_returned': False}

    def stream_prefix():
        # Observed writes only: no reread or fsync, and a partial tail may exist.
        try:
            h = getattr(stream, 'file_hash', None)
            return {'complete_rows': getattr(stream, 'rows', None),
                    'written_bytes_including_possible_partial_tail': getattr(stream, 'bytes', None),
                    'last_complete_chain_sha256': getattr(stream, 'previous', None),
                    'written_bytes_sha256': (h.hexdigest() if h is not None and
                                              getattr(stream, 'file_hash_covers_written_bytes', True) else None),
                    'file_hash_covers_written_bytes': getattr(stream, 'file_hash_covers_written_bytes', None),
                    'failure': getattr(stream, 'failed', None),
                    'readback_verified': False, 'fsync_durability_verified': False}
        except BaseException as error:
            return {'status': 'UNAVAILABLE', 'type': type(error).__name__, 'reason': str(error)}

    current = context()
    try:
        with meter.span('setup.share_identity'):
            shared = endpoint.shared_identity()
        for cell in cells:
            current_result = current_event = pending = None
            current = context(cell, stage='AXIS_IDENTITY')
            committed = committed | {'axis_counts': Counter(), 'axis_digest': sha256(),
                                     'prefix_digest': sha256()}
            with meter.span('setup.axis_identity'):
                need(endpoint.shared_identity() == shared, 'X_Y_SETUP_SHARING_MISMATCH')
            current['stage'] = 'BITS_FACTORY'
            bits = bits_factory(cell['axis'])
            current['stage'] = 'TRIAL_BUDGET'
            n = correctness_trials if phase == 'correctness' else cell['N']
            need(type(n) is int and n > 0, 'INVALID_TRIAL_BUDGET')
            for trial in range(n):
                current_result = current_event = pending = None
                current = context(cell, trial, 'SAMPLING')
                trial_id = current['trial_id']
                with meter.span('online.generator_and_audit') as trial_span:
                    current_result = result = endpoint.sample(bits, trial_id)
                    current['result_acquired'] = True
                    current['stage'] = 'RESULT_ACCESS'
                    current_event = result.event
                    current['event_present'] = current_event is not None
                    current['stage'] = 'RESULT_VALIDATION'
                    result.validate()
                    trial_span[0] = 'online.'+result.kind+'.generator_and_audit'
                if phase == 'correctness' and result.kind == 'EVENT':
                    current['stage'] = 'EVENT_VALIDATION'
                    with meter.span('correctness.event_invariants'):
                        audit.event_witness(result.event, full=cell['arm'] == 'full_return',
                                            degree_cap=INPUT['m'], labels=3)
                current['stage'] = 'RECORD_CONSTRUCTION'
                payload = {'trial_id': trial_id, 'axis': cell['axis'], 'kind': result.kind,
                           'event': result.event, 'evidence': None, 'angle': None,
                           'generator_exclusive_ns': trial_span[3]}
                if result.kind == 'EVENT' and cell['layer'] == 'B':
                    current['stage'] = 'ANGLE_AND_REQUERY'
                    with meter.span('online.EVENT.angle_and_requery'):
                        a = endpoint.angle_for(result)
                        payload['angle'] = getattr(a, 'document', getattr(a, 'text', a))
                with meter.span('online.'+result.kind+'.evidence'):
                    current['stage'] = 'EVIDENCE_CONSTRUCTION'
                    if isinstance(result.evidence, dict):
                        payload['evidence'] = {k: getattr(v, 'document', v)
                                               for k, v in result.evidence.items()}
                    else:payload['evidence'] = result.evidence
                    current['record_constructed'] = True
                    event_text = canonical({'kind': result.kind, 'event': result.event}).encode()
                    current['stage'] = 'CANDIDATE_DIGEST'
                    pending = {'counts': committed['counts'].copy(),
                               'digest': committed['digest'].copy(),
                               'axis_counts': committed['axis_counts'].copy(),
                               'axis_digest': committed['axis_digest'].copy(),
                               'prefix_digest': committed['prefix_digest'].copy(),
                               'last_trial': {'axis': cell['axis'], 'trial_id': trial_id,
                                              'trial_index': trial}}
                    pending['axis_digest'].update(event_text);pending['digest'].update(event_text)
                    if trial < correctness_trials:pending['prefix_digest'].update(event_text)
                    pending['counts'][result.kind] += 1;pending['axis_counts'][result.kind] += 1
                    if phase == 'profile' and trial >= correctness_trials:
                        # All evidence is constructed/hashed; retain exact full
                        # documents for first 16/axis plus typed-failure context.
                        payload = {'trial_id': trial_id, 'axis': cell['axis'], 'kind': result.kind,
                                   'event_sha256': sha256(event_text).hexdigest(),
                                   'evidence_sha256': identity(payload['evidence']),
                                   'angle_sha256': identity(payload['angle']),
                                   'generator_exclusive_ns': trial_span[3],
                                   'retention': 'HASHED_FULL_DOCUMENTS_NOT_RETAINED'}
                    current['record_constructed'] = True
                    current['candidate_ready'] = True
                current['stage'] = 'STREAM_EMIT'
                current['stream_emit_started'] = True
                stream.emit(payload)
                current['stream_emit_returned'] = True
                committed = pending  # Single publication of counts, hashes and last trial.
                current['stage'] = 'TRIAL_COMPLETED'
                if (trial+1) % 4096 == 0:
                    current['stage'] = 'MEMORY_OBSERVATION'
                    with meter.span('observe.memory'):
                        sample = memory_read()
                        rss = sample.get('VmRSS_bytes')
                        if rss is not None:max_rss = max(max_rss or 0, rss)
                    stream.emit({'memory_sample': sample, 'completed_trial': trial+1, 'axis': cell['axis']})
            current_result = current_event = pending = None
            current = context(cell, stage='AXIS_COMPLETION')
            axis_summaries.append({'axis': cell['axis'], 'N_executed': n, 'counts': dict(committed['axis_counts']),
                                   'event_trace_sha256': committed['axis_digest'].hexdigest(),
                                   'first_correctness_prefix_sha256': committed['prefix_digest'].hexdigest(),
                                   'H_bits_supplied': getattr(bits, 'consumed_bits', None)})
            # Versioned candidate: fixed N completes; parent classifies missing coverage.
            # No extra sampling, retry, or scientific success is inferred.
        current = context(stage='FINAL_IDENTITY')
        with meter.span('setup.final_identity'):
            need(endpoint.shared_identity() == shared, 'FINAL_STATE_CHANGED')
        return {'counts': dict(committed['counts']), 'axes': axis_summaries, 'trace_sha256': committed['digest'].hexdigest(),
                'shared_identity': shared, 'population_cost_certificate': False,
                'periodic_max_observed_RSS_bytes': max_rss,
                'RSS_sampling_is_not_continuous_peak': True}
    except BaseException as exc:
        completed = (current['trial_id'] is not None and committed['last_trial'] is not None and
                     committed['last_trial']['trial_id'] == current['trial_id'])
        failure = {'kind': 'TYPED_FAILURE', 'type': type(exc).__name__, 'reason': str(exc),
                   'failure_provenance_schema': 'g6_w2c_b3_f01_failure_v1',
                   'completed_prefix': dict(committed['counts']), 'completed_axes': axis_summaries,
                   'trace_sha256': committed['digest'].hexdigest(),
                   'trace_scope': 'COMPLETED_TRIALS_AFTER_NORMAL_EMIT_RETURN',
                   'last_completed_trial': committed['last_trial'],
                   'current_attempt': current | {'trial_completed': completed},
                   'inflight_event': current_event,
                   'candidate_trace_sha256': (pending['digest'].hexdigest()
                                               if current['candidate_ready'] and pending is not None else None),
                   'stream_prefix_before_failure_record': stream_prefix(), 'mandatory_STOP': True,
                   'is_random_zero': False, 'retry': 0, 'resume': False}
        if hasattr(exc, 'to_dict'):
            try:failure['frozen_failure_evidence'] = exc.to_dict()
            except BaseException as error:
                failure['frozen_failure_evidence_error'] = {'type': type(error).__name__, 'reason': str(error)}
        # Preserve the original exception and context even if saving failure fails.
        try:exc.b3_failure_provenance = failure
        except BaseException:pass
        try:stream.emit(failure)
        except BaseException as error:
            failure['failure_record_write_error'] = {'type': type(error).__name__, 'reason': str(error)}
            failure['stream_prefix_after_failure_record_attempt'] = stream_prefix()
        raise

def worker(task, settings, target, conn, inherited, stopped, root, pins,
            endpoint_factory, bits_factory, watch, rendezvous=None):
    install()
    if not endpoint_factory.artificial:
        from .gate import require_live
        require_live(endpoint_factory.permit,endpoint_factory.c,endpoint_factory.m)
    D.SERIALIZATION_MODE = settings.codec
    if endpoint_factory.artificial:
        sys.meta_path.insert(0, BlockScience())
    for parent_end in inherited:
        parent_end.close()
    reserved_exit = next((f[3] for f in settings.faults if tuple(f[:3]) ==
                          (task['pair_index'], task['phase_index'], 'reserved_exit')), None)
    client = Client(conn, stopped, os.getppid(), reserved_exit)
    before, meter, stream, outcome, failure = D.clock(), D.Meter(D.clock), None, None, None
    baseline = D.memory_snapshot()
    io_before = own_io()
    observer = CachedObserver([str(Path(root)/p) for p in pins if p.endswith('.py')] +
                                  [str(Path(endpoint_factory.runtime_manifest['runtime_root'])/p) for p in pins if p.endswith('.py')] +
                                  [D.__file__, __file__, endpoint_factory.__call__.__code__.co_filename])
    try:
        # Diagnostics on the same fixed correctness calls; no additional pass.
        with observer:
            with meter.span('setup.source_check'):
                D.check_pins(root, pins)
                from ..g6_w2c_b3_parallel_admission import check_runtime
                check_runtime(root, endpoint_factory.runtime_manifest)
            with meter.span('setup.constructor_and_preflight'):
                endpoint = ControlledEndpoint(endpoint_factory(task, settings), client, settings.mode)
            name = 'COLLISION' if settings.collision else task['key']
            stream = D.Stream(Path(target)/(name+'.jsonl'), meter,
                              byte_limit=settings.result_bytes, flush_rows=settings.flush_rows)
            fault = next((f[2] for f in settings.faults if f[0] == task['pair_index'] and
                          f[1] == task['phase_index'] and f[2] in ('write', 'flush', 'ack_exit')), '')
            stream.file = ReservedFile(stream.file,
                lambda n: client.rpc('WRITE_RESERVE', amount=n),
                lambda n: client.rpc('WRITE_ACK', amount=n), fault)
            cells = [dict(arm=task['arm'], layer=task['layer'], axis=a, N=task['n']) for a in D.AXES]
            def bits(axis):
                client.confirm()
                return bits_factory(task['arm'], axis)
            if rendezvous is not None:
                original_sample = endpoint.endpoint.sample
                def sample(bits, trial_id):
                    if trial_id.endswith('/X/0'):
                        rank = rendezvous.wait(timeout=60)
                        if rank == 0 or settings.sync_failure == 'all':
                            raise Halt('ARTIFICIAL_SYNCHRONIZED_FAILURE')
                        until = time.monotonic() + .1
                        while time.monotonic() < until: pass
                    return original_sample(bits, trial_id)
                endpoint.endpoint.sample = sample
            outcome = run_pair(endpoint, bits, cells, EngineStream(stream, client), meter,
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
                else:
                    failure.setdefault('cleanup_errors',[]).append({'type':type(exc).__name__,'reason':str(exc)})
        total = D.delta(before, D.clock())
        row = dict(task=task, status='FAILED' if failure else 'OK', outcome=outcome, failure=failure,
                   pid=os.getpid(), cpu_ns=total[0], wall_ns=total[1], process_cpu_ns=time.process_time_ns(),
                   coarse=meter.rows, reconciliation=meter.reconcile(total),
                   fine=observer.meter.rows,
                   fine_reconciliation=observer.meter.reconcile(total),
                   memory_baseline=baseline, memory=D.memory_snapshot(), control_rpc=dict(client.rpcs),
                   io={k:v-io_before.get(k,0) for k,v in own_io().items()},
                   output_bytes=stream.bytes if stream else 0, output_rows=stream.rows if stream else 0,
                   chain=stream.previous if stream else None, file_sha=stream.file_hash.hexdigest() if stream else None,
                   stream_prefix=dict(complete_rows=stream.rows,written_bytes=stream.bytes,last_complete_hash=stream.previous,
                       file_sha256=stream.file_hash.hexdigest(),file_hash_covers_written_bytes=stream.file_hash_covers_written_bytes,
                       stream_failure=stream.failed,closed=stream.closed,readback_verified=False,fsync_durability_verified=False) if stream else None,
                   start_method='spawn',unknown_bit_operations=None)
        try:
            conn.send({'op': 'END', 'report': row})
        except (BrokenPipeError, EOFError, OSError):
            pass
        if watch is not None:
            watch[task['phase_index']*8+task['pair_index']] = 1
        conn.close()

class Coordinator(PreviousCoordinator):
    def run_phase(self,*args,drain_only=False,**kwargs):
        try:return self._versioned_phase(*args,drain_only=drain_only,**kwargs)
        except BaseException:
            if not drain_only:
                self._record('PHASE_ABORT',phase=args[0][0]['phase_index'] if args and args[0] else None)
            raise

    def _versioned_phase(self, tasks, root, pins, endpoint_factory, bits_factory, watch, drain_only=False):
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
                inherited = []  # spawn inherits no authority or unrelated pipe endpoints
                process = self.ctx.Process(target=worker, args=(task, self.settings, self.target, child,
                    inherited, self.stopped, root, pins, endpoint_factory, bits_factory, watch, self.rendezvous))
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
