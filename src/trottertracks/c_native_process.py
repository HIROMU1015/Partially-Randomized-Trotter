"""Opt-in, spawn-isolated native samples; no scientific work on import.

One worker owns its Qiskit state and runs all checks for a sample. Immutable
reference matrices use read-only mmap; no shared Qiskit mutex is needed.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import json
import multiprocessing
import os
from pathlib import Path
import resource
import time

VERSION = 'native_complete_sample_spawn_mmap_v1'
_STATE = None


class NativeWorkerError(RuntimeError):
    pass


class _Cancelled(RuntimeError):
    pass


@contextmanager
def _spawn_environment():
    # Spawn imports numpy before the initializer. Limit only our child processes,
    # restoring the parent's environment afterwards; no package/OS changes.
    keys = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'RAYON_NUM_THREADS')
    previous = {k: os.environ.get(k) for k in keys}
    try:
        os.environ.update({k: '1' for k in keys})
        yield
    finally:
        for key, value in previous.items():
            if value is None: os.environ.pop(key, None)
            else: os.environ[key] = value


def pack_context(out, a, prepared):
    """Cache immutable numeric context once per candidate, not once per sample."""
    import numpy as np
    if not prepared or prepared['arm'] is not a['arm'] or a['arm'].general is not None:
        raise ValueError('PROCESS_REQUIRES_PREPARED_RELOCATED_ARM')
    cached = prepared.get('process_context')
    if cached is not None:
        if cached['arm'] is not a['arm']:
            raise ValueError('PROCESS_CONTEXT_CANDIDATE')
        return cached
    folder = Path(out) / 'shared_context'; folder.mkdir()
    ctx = prepared['ctx']; keys = tuple(ctx['atoms'])
    np.save(folder/'reference.npy', np.stack([ctx['left'], ctx['right'],
            *(ctx['atoms'][key] for key in keys)]), allow_pickle=False)
    shell_keys = tuple((label, kind) for label in ('head', 'X', 'Y')
                       for kind in ('raw', 'compiled'))
    np.save(folder/'shells.npy', np.stack([prepared['shells'][l][k]
            for l, k in shell_keys]), allow_pickle=False)
    cached = dict(arm=a['arm'], folder=str(folder.resolve()), atoms=keys,
                  shell_keys=shell_keys)
    prepared['process_context'] = cached
    return cached


def _initialize(payload_path, stopped):
    global _STATE
    import pickle
    with Path(payload_path).open('rb') as stream: payload = pickle.load(stream)
    import numpy as np
    from threadpoolctl import threadpool_limits
    from .c1_h4_strong_baseline import StrongBuilder
    packed = payload['packed']; folder = Path(packed['folder'])
    refs = np.load(folder/'reference.npy', mmap_mode='r', allow_pickle=False)
    shells = np.load(folder/'shells.npy', mmap_mode='r', allow_pickle=False)
    ctx = dict(left=refs[0], right=refs[1],
               atoms={key: refs[i+2] for i, key in enumerate(packed['atoms'])},
               event_matrices={})
    shell_context = {label: {} for label in ('head', 'X', 'Y')}
    for (label, kind), matrix in zip(packed['shell_keys'], shells):
        shell_context[label][kind] = matrix
    # StrongBuilder callbacks are process-local; warmed circuits are transferred
    # without the unpickleable parent callbacks/locks.
    builder = StrongBuilder()
    builder.__dict__.update(payload['builder'])
    _STATE = dict(payload, ctx=ctx, shells=shell_context, builder=builder,
                  stopped=stopped, limits=threadpool_limits(limits=1))


def _sample(i):
    from .c_native_sample import run_sample
    state = _STATE; records = []; failure = None
    tick = time.monotonic(); cpu = time.process_time()
    def call(kind, label, fn):
        nonlocal failure
        if state['stopped'].is_set(): raise _Cancelled('DEPENDENT_WORK_STOPPED')
        start = time.monotonic()
        try:
            value = fn()
        except _Cancelled:
            raise
        except BaseException as exc:
            if failure is None:
                failure = dict(kind=kind, label=label, error=repr(exc))
            state['stopped'].set()
            raise
        records.append(dict(kind=kind, label=label, wall_seconds=time.monotonic()-start))
        return value
    builder = state['builder']; builder.science_call = call
    # PrecisionBuilder's two-argument callback counts a cache miss; its parent
    # historically uses a no-op here. StrongBuilder's numerical calls use call.
    status = 'PASS'; value = None
    try:
        value = run_sample(i, state['contract'], state['out'], call, state['a'],
                           state['policy'], state['ctx'], state['psi'], builder,
                           state['shells'])
    except _Cancelled:
        status = 'CANCELLED'
    except BaseException as exc:
        state['stopped'].set(); status = 'FAIL'
        if failure is None: failure = dict(kind='native_sample', label=str(i), error=repr(exc))
    return dict(sample=i, value=value, status=status, failure=failure, calls=records,
                pid=os.getpid(), wall_seconds=time.monotonic()-tick,
                CPU_seconds=time.process_time()-cpu,
                peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                reference_readonly=not state['ctx']['left'].flags.writeable,
                relative_cache=builder.relative_cache if status=='PASS' else {})


def run_samples(c, out, call, a, policy, prepared, *, indices=range(32)):
    """Production passes exactly indices 0..31; smaller sets support toy tests."""
    import pickle
    from . import c_pr_integrated_performance as perf
    indices = tuple(indices)
    if (not indices or len(set(indices)) != len(indices)
            or any(type(i) is not int or i < 0 or i >= policy['samples'] for i in indices)):
        raise ValueError('NATIVE_SAMPLE_INDICES')
    packed = pack_context(out, a, prepared)
    snapshot = {k: v for k, v in vars(prepared['builder']).items()
                if k not in ('call', 'science_call')}
    payload = dict(contract=c, out=Path(out),
                   a={k: a[k] for k in ('arm', 'tail', 'dist', 'config')},
                   policy=policy, psi=prepared['psi'], packed={k:v for k,v in packed.items() if k!='arm'},
                   builder=snapshot)
    # Fail before scheduling any sample if a new unpickleable field is added.
    # Send only a small path during spawn. Sending the warmed Qiskit payload
    # through each spawn pipe can block the parent until that child imports
    # Qiskit, serializing process startup. This file is generated by this run.
    payload_path = (Path(out)/'worker_context.pkl').resolve()
    with payload_path.open('xb') as stream: pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    plan = perf.m4.integrated.resource_plan(task_count=len(indices),
                                           full_dimension=len(prepared['psi']))
    # Spawn also pays Python/Qiskit/context residency. The thread scheduler's
    # dense-work estimate alone understates this (toy CDF8 measured ~1.2 GiB).
    # Use measured parent residency plus its existing dense scratch estimate;
    # this is a conservative workload estimate, never a fixed resource quota.
    process_bytes = plan['estimated_bytes_per_candidate'] + resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    available = plan['available_memory_bytes']
    if available is not None:
        plan['workers'] = min(plan['workers'], max(1, available//process_bytes))
    plan['estimated_bytes_per_worker'] = process_bytes
    plan.update(backend=VERSION, start_method='spawn', Qiskit_shared_mutex=False,
                BLAS_threads_per_worker=1, samples=list(indices),
                reference_storage='read-only mmap', shared_context=packed['folder'])
    perf.io.save(Path(out)/'native_resource_plan.json', plan)
    context = multiprocessing.get_context('spawn'); stopped = context.Event()
    rows = {}; failures = []; cancelled = []; reports = []
    try:
        with _spawn_environment():
            with ProcessPoolExecutor(max_workers=plan['workers'], mp_context=context,
                                     initializer=_initialize, initargs=(str(payload_path), stopped)) as pool:
                futures = [pool.submit(_sample, i) for i in indices]
                for future in as_completed(futures):
                    try: report = future.result()
                    except BaseException:
                        stopped.set()
                        for pending in futures: pending.cancel()
                        raise
                    # Replay completed-call bookkeeping only, never numerical work.
                    # Worker stage times remain in worker_samples.jsonl.
                    for record in report['calls']:
                        try: call(record['kind'], record['label'], lambda: None)
                        except BaseException:
                            stopped.set()
                            for pending in futures: pending.cancel()
                            raise
                    prepared['builder'].relative_cache.update(report.pop('relative_cache'))
                    if report['status'] == 'PASS':
                        i, value = report.pop('value'); rows[i] = value
                    elif report['status'] == 'FAIL': failures.append(report['failure'])
                    else: cancelled.append(report['sample'])
                    reports.append(report)
                    with (Path(out)/'worker_samples.jsonl').open('a') as stream:
                        stream.write(json.dumps(report)+'\n')
                    with (Path(out)/'native_progress.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(completed=len(rows), failed=len(failures),
                            cancelled=len(cancelled), worker_CPU_seconds=sum(x['CPU_seconds'] for x in reports)))+'\n')
    except BaseException:
        stopped.set(); call.stop.set(); raise
    if failures:
        failure = failures[0]
        def failed(): raise NativeWorkerError(str(failure))
        try: call(failure['kind'], failure['label'], failed)
        finally: call.stop.set()
    if cancelled or len(rows) != len(indices):
        call.stop.set(); raise NativeWorkerError('INCOMPLETE_NATIVE_COHORT')
    return [rows[i] for i in indices], plan, reports


def native(root, c, out, call, a, *, sampling=None, prepared=None):
    from . import c_pr_integrated_performance as perf
    from .c_full_unitary import VERSION as evaluator
    policy = dict(perf.SAMPLING if sampling is None else sampling)
    steps = policy['events_per_sample']
    if (policy['samples'] != 32 or type(steps) is not int or steps < 1
            or a['config'].rte_steps != steps):
        raise ValueError('NATIVE_SAMPLING_CONFIG')
    if not prepared or prepared['arm'] is not a['arm']:
        raise ValueError('PROCESS_REQUIRES_PREPARED_RELOCATED_ARM')
    tick = time.monotonic(); cpu = time.process_time()
    for leaf in ('circuits', 'samples', 'references'): (Path(out)/leaf).mkdir()
    rows, plan, reports = run_samples(c, out, call, a, policy, prepared)
    calibration = call('calibration_summary', 'fixed32 new-candidate cohort',
                       lambda: perf.b2.calibration(rows))
    perf.io.save(Path(out)/'native_timing.json', dict(samples=len(rows), events_per_sample=steps,
        cohort=policy['cohort'], backend=VERSION, wall_seconds=time.monotonic()-tick,
        CPU_seconds=time.process_time()-cpu, worker_CPU_seconds=sum(x['CPU_seconds'] for x in reports),
        resource_plan=plan, parent_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        max_worker_peak_RSS_KiB=max(x['peak_RSS_KiB'] for x in reports),
        worker_pids=sorted({x['pid'] for x in reports}), full_operator_evaluator=evaluator,
        full_input_output_dimension=2*len(prepared['psi']), actual_inverse_circuits_independently_evaluated=True))
    return rows, calibration
