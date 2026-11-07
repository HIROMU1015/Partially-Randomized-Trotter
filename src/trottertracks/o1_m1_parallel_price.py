"""Spawn workers for the fixed matched price primitives; no input I/O on import.

MC draws/reduction retain the serial ordering. This module never interrupts,
resumes or edits a scientific job. Molecular use needs its execution scope.
"""
from concurrent.futures import ProcessPoolExecutor
from collections import deque
from dataclasses import replace
from functools import lru_cache
from itertools import islice
import importlib
import multiprocessing
import os
from pathlib import Path
import pickle
import resource
import sys
from types import ModuleType, SimpleNamespace


def _bridge():
    # Existing passive package pattern: avoid eager unrelated research imports.
    if 'trotterlib' not in sys.modules:
        package=ModuleType('trotterlib')
        package.__path__=[str(Path(__file__).resolve().parents[1]/'trotterlib')]
        sys.modules['trotterlib']=package
    return importlib.import_module('trotterlib.o1_m1_resource_bridge')


def _worker_init():
    # Outer parallelism runs up to one task/available CPU. Small BLAS work in
    # each compiler worker must not start another CPU-sized thread pool.
    for variable in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','RAYON_NUM_THREADS'):
        os.environ[variable]='1'
    if os.environ.get('XDG_CACHE_HOME'):
        os.environ['XDG_CACHE_HOME']=str(Path(os.environ['XDG_CACHE_HOME'])/str(os.getpid()))
    if os.environ.get('MPLCONFIGDIR'):
        os.environ['MPLCONFIGDIR']=str(Path(os.environ['MPLCONFIGDIR'])/str(os.getpid()))
    _bridge()


@lru_cache(maxsize=1)
def _restore(payload):
    return pickle.loads(payload)


def _mc_task(task):
    payload,config,distribution,q,events,compiler,compile_function,axis=task
    bridge=_bridge()
    circuit=bridge.build_controlled_trajectory(_restore(payload),config,distribution,q=q,events=events)
    if axis is not None:
        values={axis:_one_axis(circuit,axis,compiler,compile_function)}
        return os.getpid(),values,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    pair=bridge._compile_pair(circuit,compiler,compile_function=compile_function or bridge.transpile_and_measure_cost,backend=None)
    # Send metrics, not the large compiled circuits, back to the parent.
    return os.getpid(),{axis:tuple(getattr(cost,k) for k in bridge.RPE_COST_METRICS)
                           for axis,cost in pair.items()},resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _one_axis(circuit,axis,compiler,compile_function):
    import math
    bridge=_bridge()
    wrapper=bridge.wrap_controlled_no_prep(circuit,ancilla_qubit=circuit.num_qubits-1,
                                          axis=axis,include_measurement=True)
    fingerprint=bridge.canonical_qiskit_circuit_fingerprint(wrapper)
    cost=(compile_function or bridge.transpile_and_measure_cost)(wrapper,compiler,
        backend=None,actual_circuit_fingerprint=fingerprint)
    values=tuple(getattr(cost,k) for k in bridge.RPE_COST_METRICS)
    if cost.compiler!=compiler or any(not math.isfinite(float(v)) or v<0 for v in values):
        raise ValueError('Compile context or metric differs')
    return values


def _axis_task(task):
    payload,axis,compiler,compile_function=task
    values=_one_axis(_restore(payload),axis,compiler,compile_function)
    return os.getpid(),values,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _available_memory_kb():
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                       if line.startswith('MemAvailable:')))
    # Respect the actual unified cgroup limit, including any limiting ancestor.
    base=Path('/sys/fs/cgroup')
    entries=[line.split('::',1)[1] for line in Path('/proc/self/cgroup').read_text().splitlines()
             if line.startswith('0::')]
    current=(base/entries[0].lstrip('/')).resolve() if entries else base
    if not current.is_relative_to(base): current=base
    while current.is_relative_to(base):
        maximum,used=current/'memory.max',current/'memory.current'
        if maximum.exists() and used.exists():
            limit=maximum.read_text().strip()
            if limit!='max': available=min(available,max(0,(int(limit)-int(used.read_text()))//1024))
        if current==base: break
        current=current.parent
    return available


def memory_worker_limit(tasks,cpus,peak_kb,available_kb):
    """2x observed peak allows compiler/trajectory variation; no fixed RAM cap."""
    if peak_kb<=0 or available_kb<2*peak_kb:
        raise RuntimeError('Insufficient observed memory headroom for the worker batch')
    return min(tasks,cpus,available_kb//(2*peak_kb))


class ParallelMatchedNoPrepCostProvider:
    """C0 backend=None only. Scientific cost records match the serial provider.

    Default workers=min(available CPUs, tasks). MC uses paired trajectory tasks
    when CPUs<=sample_count; otherwise axes split too (64 tasks for 32 draws).
    The split axes receive the same event tuple. Deterministic has two tasks.
    Only spawned worker processes receive the numerical-thread setting.
    """
    def __init__(self,compiler,*,max_workers=None,compile_function=None):
        bridge=_bridge()
        if max_workers is not None:
            max_workers=bridge.require_integer_count(max_workers,name='max_workers',minimum=1)
        self.provider=bridge.MatchedNoPrepCostProvider(compiler)
        self.compiler,self.max_workers,self.compile_function=compiler,max_workers,compile_function
        self.last_execution=None

    def _iter_map(self,function,tasks,*,memory_probe=False,task_count=None):
        """Lazy ordered dispatch with at most one pending future per worker.

        Reuses the existing spawn initializer and measured-memory admission.
        Task generation and result reduction never require a batch-sized list.
        """
        available=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 1)
        limit=min(available,self.max_workers or available)
        tasks=iter(tasks);probe=None;completed=0;pids=set();workers=0;peak_pending=0
        sentinel=object()
        first_task=next(tasks,sentinel)
        if first_task is sentinel:
            self.last_execution=dict(start_method='spawn',configured_workers=0,task_count=0,
                worker_pids=[],reduction='submission order',numerical_threads_per_worker=1,
                memory_probe=None,peak_pending_futures=0)
            return
        if memory_probe:
            with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),
                                     initializer=_worker_init) as executor:
                first=executor.submit(function,first_task).result()
            completed+=1;pids.add(first[0])
            free=_available_memory_kb()
            probe=dict(pilot_peak_rss_kb=first[2],available_memory_kb=free,memory_peak_multiplier=2)
            yield first[1]
            first_task=next(tasks,sentinel)
            if first_task is not sentinel:
                limit=memory_worker_limit(limit if task_count is None else task_count-1,limit,first[2],free)
        try:
            if first_task is not sentinel:
                workers=min(limit,limit if task_count is None else task_count-completed)
                with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                                         initializer=_worker_init) as executor:
                    pending=deque([executor.submit(function,first_task)])
                    del first_task
                    for task in islice(tasks,workers-1):
                        pending.append(executor.submit(function,task))
                    peak_pending=len(pending)
                    while pending:
                        pid,value,_=pending.popleft().result()
                        completed+=1;pids.add(pid)
                        yield value
                        task=next(tasks,sentinel)
                        if task is not sentinel:
                            pending.append(executor.submit(function,task))
                        peak_pending=max(peak_pending,len(pending))
        finally:
            self.last_execution=dict(start_method='spawn',configured_workers=workers,task_count=completed,
                worker_pids=sorted(pids),reduction='submission order',numerical_threads_per_worker=1,
                memory_probe=probe,peak_pending_futures=peak_pending)

    def _map(self,function,tasks,*,memory_probe=False):
        """Compatibility collection API; streaming callers use _iter_map."""
        return list(self._iter_map(function,tasks,memory_probe=memory_probe,
            task_count=len(tasks) if hasattr(tasks,'__len__') else None))

    def _metric(self,values):
        return SimpleNamespace(**dict(zip(_bridge().RPE_COST_METRICS,values,strict=True)))

    def monte_carlo(self,preparation,config,distribution,*,q,sample_count,seed):
        bridge=_bridge()
        from trotterlib.rte import iter_sample_rte_events
        q=bridge.require_integer_count(q,name='q',minimum=1)
        sample_count=bridge.require_integer_count(sample_count,name='sample_count',minimum=2)
        seed=bridge.require_integer_count(seed,name='seed')
        bridge._validate_law(preparation,config,distribution)
        length=q*config.rte_steps
        stream=iter_sample_rte_events(preparation.tail.components,distribution,
                                     sample_count=sample_count*length,seed=seed)
        payload=pickle.dumps(preparation,protocol=pickle.HIGHEST_PROTOCOL)
        available=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 1)
        split_axes=(self.max_workers or available)>sample_count
        def tasks():
            for _ in range(sample_count):
                events=tuple(islice(stream,length))
                for axis in (('cosine','sine') if split_axes else (None,)):
                    yield (payload,config,distribution,q,events,self.compiler,self.compile_function,axis)
        accumulators={axis:bridge.CompiledMetricAccumulator(weighted=False) for axis in ('cosine','sine')}
        # map returns submission order, regardless of worker completion order.
        for pair in self._iter_map(_mc_task,tasks(),memory_probe=True,
                task_count=sample_count*(2 if split_axes else 1)):
            for axis,values in pair.items(): accumulators[axis].update(self._metric(values))
        self.last_execution['granularity']='axis' if split_axes else 'paired_trajectory'
        result=self.provider._result(accumulators,context_hash=preparation.context_hash,
            arm_hash=preparation.preparation_hash,q=q,delta_time=config.evolution_time,
            law_hash=bridge._law_hash(config),method='fresh_iid_matched_full_no_prep_monte_carlo')
        errors={axis:bridge.circuit_cost_from_metric_statistics(acc.finalize(),compiler=self.compiler,
            estimate_kind='compiled_cost_standard_error',use_standard_error=True,fidelity_level=5)
            for axis,acc in accumulators.items()}
        return replace(result,cosine_standard_error=errors['cosine'],sine_standard_error=errors['sine'],
            classical_sample_count=sample_count,
            cost_model_fingerprint=bridge.digest(dict(context=result.cost_model_fingerprint,
                method=result.evaluation_method,sample_count=sample_count,seed=seed)),
            metadata=result.metadata+(('classical_price_seed',seed),('fresh_events_per_trajectory',length),
                ('paired_axes',True),('price_uncertainty','sample_SE_only; no confidence interval')))

    def deterministic(self,circuit,*,context_hash,arm_hash,q,delta_time):
        import math
        bridge=_bridge()
        q=bridge.require_integer_count(q,name='q',minimum=1)
        if not context_hash or not arm_hash or not math.isfinite(delta_time) or delta_time<=0:
            raise ValueError('Explicit deterministic context and positive time required')
        payload=pickle.dumps(circuit,protocol=pickle.HIGHEST_PROTOCOL)
        axes=('cosine','sine')
        results=self._map(_axis_task,[(payload,axis,self.compiler,self.compile_function) for axis in axes])
        accumulators={axis:bridge.CompiledMetricAccumulator(weighted=True) for axis in axes}
        for axis,values in zip(axes,results,strict=True):
            accumulators[axis].update(self._metric(values),weight=1.)
        return self.provider._result(accumulators,context_hash=context_hash,arm_hash=arm_hash,q=q,
            delta_time=delta_time,law_hash='deterministic_no_RTE',method='deterministic_full_no_prep')
