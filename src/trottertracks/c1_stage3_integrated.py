"""Stage 3 one-shot orchestration. Import and byte preflight do no science.

Permission is given in chat. This module records scientific calls and results,
not proof of permission. Numerical work exists only behind run()/execute().
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from fractions import Fraction
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import sys
import threading
import time
import traceback

from . import c1_stage1_execution as io
from . import c1_stage3_accuracy as accuracy
from .c1_stage3_connection import compose_serial_cost


@dataclass(frozen=True)
class Candidate:
    arm: str
    family: str
    s: int
    r: int = 0
    K: int = 0

    @property
    def id(self):
        return f'{self.arm}_{self.family}_s{self.s}_r{self.r}_K{self.K}'


def candidates():
    return tuple(Candidate(arm, family, s, r, k)
        for arm in ('CDF', 'XDF')
        for family in ('D', 'PR') for s in (1, 4)
        for r in ((0,) if family == 'D' else (4, 16))
        for k in ((0,) if family == 'D' else (0, 2)))


def preflight(root, contract_path, input_path):
    """Bytes, JSON, versions only. NPZ/QPY are never decoded here."""
    root, contract_path, input_path = map(lambda p: Path(p).resolve(),
                                         (root, contract_path, input_path))
    c = io.read(contract_path)
    if c['scope'] != 'C1_STAGE3_INTEGRATED_CA_V1' or c['new_fit_calls'] != 0 or c['retry'] or c['resume']:
        raise ValueError('STAGE3_CONTRACT_SCOPE')
    if (c['maximum_logical_scientific_units'] != 5637
            or sum(v['maximum'] for v in c['scientific_calls'].values()) != 5637
            or c['candidate_ids'] != [x.id for x in candidates()]):
        raise ValueError('STAGE3_REGISTERED_GRID_OR_CALLS')
    p = c['proposal']
    if (p['accuracy_target']['energy_ha'] != .00159
            or p['accuracy_target']['df_reserve_fraction'] != 1/8
            or p['accuracy_target']['base_time'] != .1
            or p['rounds'] != list(range(13)) or p['PR_LD'] != 2
            or p['law_model'] != 'ideal_finite_RTE_law'
            or p['target_state'] != 'own_DF_ground_state_oracle'
            or p['state_preparation_included']):
        raise ValueError('STAGE3_ADOPTED_CORE')
    if io.sha(root/c['source_manifest']) != c['source_manifest_sha256']:
        raise ValueError('STAGE3_SOURCE_MANIFEST')
    source = io.read(root/c['source_manifest'])
    for name, want in source['files'].items():
        if io.sha(root/name) != want: raise ValueError('SOURCE: '+name)
    if io.sha(input_path) != c['input_sha256']: raise ValueError('ORIGINAL_INPUT')
    for name, want in c['saved_inputs'].items():
        if io.sha(root/name) != want: raise ValueError('SAVED_INPUT: '+name)
    packages = {name: importlib.metadata.version(name) for name in c['runtime_required_packages']}
    if packages != c['runtime_required_packages'] or sys.version_info[:2]!=(3,11):
        raise ValueError('NUMERICAL_ENVIRONMENT')
    return dict(root=root, input=input_path, contract_path=contract_path,
                contract=c, source=source, runtime=packages)


class Stopped(RuntimeError):
    pass


def progress(output,label,phase,payload):
    """Preserve completed scientific values even when this candidate fails."""
    with (Path(output)/'candidates'/(label+'.progress.jsonl')).open('a') as f:
        f.write(json.dumps(dict(phase=phase,**payload),allow_nan=False)+'\n')


class Calls:
    """Thread-safe reuse of the existing scientific call journal."""
    def __init__(self, output, limits, maximum):
        self.output, self.maximum = Path(output), maximum
        self.counts, self.lock, self.stop = Counter(), threading.Lock(), threading.Event()
        self.mark = io.journal(output, 'science', limits)
        self.first_failure = None

    def __call__(self, kind, label, function):
        with self.lock:
            if self.stop.is_set(): raise Stopped('Dependent work stopped')
            if sum(self.counts.values()) >= self.maximum: raise ValueError('TOTAL_CALL_LIMIT')
            self.mark(kind, label)
            self.counts[kind] += 1
        tick = time.monotonic()
        try:
            value = function()
        except BaseException as error:
            with self.lock:
                self.stop.set()
                if self.first_failure is None:
                    self.first_failure = dict(kind=kind, label=label,
                        error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
                self._finish(kind, label, tick, 'FAIL')
            raise
        with self.lock: self._finish(kind, label, tick, 'COMPLETE')
        return value

    def _finish(self, kind, label, tick, status):
        with (self.output/'science_calls.jsonl').open('a') as f:
            f.write(json.dumps(dict(kind=kind, label=label, event=status,
                                   wall_seconds=time.monotonic()-tick))+'\n')


def resource_plan(task_count=20, full_dimension=256):
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
    available = None
    if Path('/proc/meminfo').exists():
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'): available = int(line.split()[1])*1024
    quota_cpus=None
    if Path('/sys/fs/cgroup/cpu.max').exists():
        quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota!='max':quota_cpus=max(1,math.ceil(int(quota)/int(period)))
    if Path('/sys/fs/cgroup/memory.max').exists():
        limit=Path('/sys/fs/cgroup/memory.max').read_text().strip()
        if limit!='max':
            current=int(Path('/sys/fs/cgroup/memory.current').read_text())
            available=min(available or int(limit),max(0,int(limit)-current))
    # 64 controlled dense work arrays per independent candidate, shared input
    # arrays excluded. This is a workload estimate, not a CPU/RAM quota.
    per_task = 64*16*(2*full_dimension)**2
    workers = min(cpus, quota_cpus or cpus, task_count,
                  max(1, available//per_task) if available is not None else task_count)
    return dict(workers=workers, available_affinity_cpus=cpus, available_memory_bytes=available,
                quota_cpus=quota_cpus, estimated_bytes_per_candidate=per_task,
                BLAS_threads_per_worker=1 if workers>1 else None,
                policy='Independent candidates; process-local BLAS coordination; no shared settings changed')


def _feasibility(row, target):
    """Reject only the registered sufficient bound, never zero-fill synthesis."""
    a = row['attenuation']
    if not math.isfinite(a) or a < 0: raise ValueError('INVALID_ATTENUATION')
    if a == 0: return dict(status='INFEASIBLE', reason='zero observed radius at binary64')
    known = row['pf_operator_error']+row['rte_corrected_operator_error']
    if not math.isfinite(known) or known < 0: raise ValueError('NONFINITE_REFERENCE')
    if known > math.sin(target.beta_total-target.beta_stat):
        return dict(status='INFEASIBLE', reason='PF/RTE registered sufficient bound exceeds phase room')
    return dict(status='POTENTIALLY_FEASIBLE', synthesis_error_unknown=True)


def _evaluate(packet, call, label):
    """Same three existing functions as evaluate_evidence_packet, charged once.

    Calling the old opaque entry again would repeat numerical assessments.
    Its packet schema and semantics are preserved without editing old SOURCE.
    """
    target = accuracy.AccuracyTarget(**packet['target'])
    plans = [call('final_round_assessment', f'{label}:m{row["m"]}',
        lambda row=row: accuracy.assess_round(target, accuracy.RoundEvidence(**row)))
        for row in packet['rounds']]
    energy = call('energy_assessment', label, lambda: accuracy.assess_energy(target,
        packet['df_error_bound_ha'], plans, state_model=packet['state_model'],
        alias_branch_verified=packet['alias_branch_verified']))
    totals = call('resource_aggregate', label, lambda: accuracy.aggregate_totals(energy, plans,
        packet['costs'], cost_scope=packet['cost_scope']))
    return dict(accuracy=energy, rounds=plans, resources=totals, mandatory_stop=True)


def _candidate(backend, candidate, proposal, call, output):
    label = candidate.id
    target = accuracy.AccuracyTarget(**proposal['accuracy_target'])
    kernel = call('kernel_reference', label, lambda: backend.kernel(candidate))
    references, feasible = [], []
    for m in proposal['rounds']:
        ref = call('round_reference', f'{label}:m{m}', lambda m=m: backend.reference(candidate, kernel, m))
        references.append(ref)
        feasible.append(call('feasibility_assessment', f'{label}:m{m}', lambda ref=ref: _feasibility(ref, target)))
        progress(output,label,'reference',dict(m=m,reference=ref,feasibility=feasible[-1]))
    df_error = backend.df_errors[candidate.arm]
    bad = any(x['status'] == 'INFEASIBLE' for x in feasible) or df_error > target.df_cap
    calibration = None
    qualification = dict(synthesis_error_per_kernel=None, synthesis_error_shell=None,
                         status='UNRESOLVED', reason='Dependent cost skipped for ineligible candidate')
    if not bad:
        calibration = backend.calibrate(candidate, kernel, call)
        qualification = call('synthesis_qualification', label,
                             lambda: backend.qualify(candidate, calibration))
    rows, costs = [], []
    for m, ref in zip(proposal['rounds'], references):
        n = (1<<m)*candidate.s
        syn = qualification.get('synthesis_error_per_kernel')
        shell = qualification.get('synthesis_error_shell')
        syn = None if syn is None or shell is None else min(2., n*syn+shell)
        rows.append(dict(m=m, attenuation=ref['attenuation'],
            pf_operator_error=ref['pf_operator_error'], rte_corrected_operator_error=ref['rte_corrected_operator_error'],
            synthesis_signal_error=syn, finite_law_signal_error=0., target_id=label,
            state_model=proposal['target_state'], alias_branch_verified=True, iid_per_shot=True,
            evidence_kind='numerically_qualified_pilot', law_model=proposal['law_model']))
        if calibration is not None:
            for axis in ('X', 'Y'):
                costs.append(call('serial_cost_expansion', f'{label}:m{m}:{axis}',
                    lambda m=m, axis=axis: compose_serial_cost(calibration['kernel'],
                        calibration['shells'][axis], m=m, substeps=candidate.s, target_id=label)))
    packet = dict(target=proposal['accuracy_target'], df_error_bound_ha=df_error,
        state_model=proposal['target_state'], alias_branch_verified=True, rounds=rows,
        costs=costs, cost_scope=proposal['cost_scope'])
    evaluated = _evaluate(packet, call, label)
    status = ('INFEASIBLE' if bad else 'PASS' if evaluated['accuracy']['same_accuracy'] else
              'UNRESOLVED' if any(x['shots'] is None and x['status'].startswith('UNRESOLVED')
                                  for x in evaluated['rounds']) else 'INFEASIBLE')
    result = dict(candidate=asdict(candidate), id=label, status=status,
        feasibility=feasible, qualification=qualification, calibration=calibration,
        evidence=packet, evaluation=evaluated, raw_signals=[ref.get('raw_signal_parts') for ref in references],
        DF_errors_are_different=True, state_preparation_included=False)
    io.save(Path(output)/'candidates'/f'{label}.json', result)
    return result


def run(backend, proposal, output, limits, *, maximum=5637, workers=None):
    """Integrated pipeline shared by the real saved-H4 and artificial backends."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'candidates').mkdir()
    calls = Calls(output, limits, maximum)
    start, cpu = time.monotonic(), time.process_time()
    completed, selected, virtual, comparisons = {}, {}, {}, {}
    plan = resource_plan()
    if workers is not None: plan['workers'] = workers  # tests or an explicit current-job choice
    try:
        backend.execution_plan=plan
        backend.prepare(calls)
        io.save(output/'prepared.json', backend.prepared_summary())
        preparation_workers=plan['workers']
        # Probe again AFTER shared CAR/atom/exact matrices occupy memory;
        # aggregate worker memory must fit the resources still available.
        plan=resource_plan()
        if workers is not None:plan['workers']=workers
        plan['preparation_workers']=preparation_workers
        backend.execution_plan=plan
        # Sharing readonly inputs avoids process serialization; independent
        # candidate SVD/expm/compiler workloads release the GIL.
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1 if plan['workers']>1 else None):
            with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
                def task(c):
                    try:return _candidate(backend,c,proposal,calls,output)
                    except BaseException as error:
                        calls.stop.set()
                        io.save(output/'candidates'/f'{c.id}.json',dict(id=c.id,candidate=asdict(c),
                            status='STOPPED_DEPENDENCY' if isinstance(error,Stopped) else 'FAIL',
                            error_type=type(error).__name__,error=str(error)))
                        raise
                futures = {pool.submit(task, c): c for c in candidates()}
                try:
                    for future in as_completed(futures):
                        c = futures[future]
                        completed[c.id] = future.result()
                except BaseException:
                    calls.stop.set()
                    for future in futures: future.cancel()
                    raise
        for arm in ('CDF', 'XDF'):
            for family in ('D', 'PR'):
                key = arm+'_'+family
                def choose(arm=arm, family=family):
                    valid = [r for r in completed.values() if r['candidate']['arm']==arm
                        and r['candidate']['family']==family and r['status']=='PASS'
                        and r['evaluation']['resources'].get('totals') is not None]
                    return min(valid, key=lambda r:(r['evaluation']['resources']['totals']['rz_count'],
                        r['candidate']['s'],r['candidate']['r'],r['candidate']['K'])) if valid else None
                chosen = calls('candidate_selection', key, choose)
                selected[key] = chosen['id'] if chosen else None
                if chosen is not None:
                    virtual[key] = calls('virtual_rpe_batch', key, lambda chosen=chosen:
                        backend.virtual(chosen))
        for family in ('D', 'PR'):
            a, b = selected['CDF_'+family], selected['XDF_'+family]
            comparisons[family] = calls('cdf_xdf_comparison', family,
                lambda a=a,b=b: accuracy.compare_totals(completed[a]['evaluation']['resources'] if a else {},
                                                       completed[b]['evaluation']['resources'] if b else {}))
        statuses = Counter(r['status'] for r in completed.values())
        result = dict(status='COMPLETE_REGISTERED_PILOT', candidate_status_counts=dict(statuses),
            candidate_ids=[c.id for c in candidates()], selected=selected, virtual=virtual, comparisons=comparisons,
            conclusion='UNRESOLVED_QUALIFICATION' if statuses['UNRESOLVED'] else
                'INFEASIBLE_REGISTERED_GRID' if statuses['INFEASIBLE']==20 else 'CONDITIONAL_RESULTS_AVAILABLE',
            unconditional_total_resource_claim=False, new_fit=0, optimizer=0, retry=0, resume=0,
            scientific_units=sum(calls.counts.values()), call_counts=dict(calls.counts), mandatory_stop=True)
        io.save(output/'result.json', result)
        return result
    except BaseException as error:
        calls.stop.set()
        io.save(output/'failure.json', dict(status='SCIENTIFIC_FAIL', first_failure=calls.first_failure,
            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
            completed_candidate_ids=list(completed), call_counts=dict(calls.counts)))
        raise
    finally:
        io.save(output/'resources.json', dict(plan=plan, wall_seconds=time.monotonic()-start,
            process_cpu_seconds=time.process_time()-cpu, peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        io.save(output/'STOP.json', dict(mandatory_stop=True, scientific_units=sum(calls.counts.values()),
            call_counts=dict(calls.counts), failed=calls.stop.is_set(), retry=0, resume=0, new_fit=0))


def execute(verified, output):
    """Real science: call only after separate explicit chat execution permission."""
    v = preflight(verified['root'], verified['contract_path'], verified['input'])
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(v['root']/'artifacts/research_c_c1_stage3_science'):
        raise ValueError('FRESH_STAGE3_OUTPUT_REQUIRED')
    for key, leaf in [('XDG_CACHE_HOME',''),('MPLCONFIGDIR','matplotlib'),('NUMBA_CACHE_DIR','numba'),('TMPDIR','tmp')]:
        path=v['root']/'.cache/stage3-integrated'/leaf; path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    os.environ['QISKIT_PARALLEL']='FALSE'  # only this job; candidate pool owns parallelism
    sys.pycache_prefix=str(v['root']/'.cache/stage3-integrated/pycache')
    from .c1_stage3_molecular import MolecularBackend
    backend=MolecularBackend(v,output)
    limits={k:row['maximum'] for k,row in v['contract']['scientific_calls'].items()}
    try:
        return run(backend,v['contract']['proposal'],output,limits)
    finally:
        if output.exists():
            io.save(output/'contract.json',v['contract']);io.save(output/'SOURCE.json',v['source'])
            io.save(output/'runtime.json',v['runtime'])
