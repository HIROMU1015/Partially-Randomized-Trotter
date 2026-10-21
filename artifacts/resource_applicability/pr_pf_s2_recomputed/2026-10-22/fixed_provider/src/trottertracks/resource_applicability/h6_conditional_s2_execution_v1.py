"""One disposable cost worker, eight fresh conditional groups, mandatory STOP.

Frozen supervisor and cost builder/compiler are reused without editing them.
The legacy leaf name cost_confirmation_fast_XXXX belongs only to the new
conditional namespace; results carry the new phase and cannot be pooled.
"""
from contextlib import contextmanager
from copy import deepcopy
import math
import os
from pathlib import Path
import statistics
import sys
import threading
import time

from . import h6_conditional_s2_contract_v1 as c
from .h6_conditional_s2_sampling_v1 import make_conditioned_steps,pilot_seed,conditional_binomial
from .h6_confirmation_fast_execution_v1 import supervise
from .h6_matched_execution_v2 import install_limits,Progress
from .h6_matched_execution_v3 import ResourceTraceV3
from .h6_server_resources_v1 import (THREAD_KEYS,install_parent_guard,process_identity,process_memory)
from .ax2b_supplement_records_v1 import AtomicWriter
from .ax2a_preparation import digest
from .h6_matched_reuse_v3 import COMPILER_HASH


def group_path(out,index):return Path(out)/f'cost_confirmation_fast_{index:04d}'


@contextmanager
def conditional_port_scope(task,expected_distribution):
    """Only one child/thread; class alias restored on every success/failure."""
    from . import h6_roundoff_cost_port_v1 as port
    if threading.active_count()!=1:raise ValueError('CONDITIONAL_WORKER_ONE_PYTHON_THREAD')
    original=port.MatchedPort
    class ConditionalPort(original):
        def sampled_steps(self,cell,seed):
            if (digest(cell)!=digest(c.CELL) or seed!=task['seed'] or task['phase']!=c.PHASE
                    or self.plan['T']!=.8 or self.ham.n_qubits!=12):raise ValueError('CONDITIONAL_COST_SCOPE')
            self.calls.take('trajectory');self.calls.take('occurrence',cell['q'])
            steps,receipt=make_conditioned_steps(self.preps[cell['id']],T=self.plan['T'],q=cell['q'],r=cell['r'],
                seed=seed,ancilla_qubit=self.ham.n_qubits,expected_distribution=expected_distribution)
            events=[[e.to_dict() for e in s.rte_occurrence.events] for s in steps]
            self.writer.write('conditional_sampling_receipt.json',dict(receipt,task=task,event_digest=digest(events)))
            return steps
    port.MatchedPort=ConditionalPort
    try:yield
    finally:port.MatchedPort=original


def worker(root,directory,run_root):
    root=Path(root);out=c.private_path(Path(run_root).absolute());directory=c.private_path(Path(directory).absolute())
    m=c.read(out/'frozen_manifest.json');g=c.read(out/'authorization.json');c.validate_grant(m,g);c.validate_scope(root,m)
    if directory.parent!=out or str(out)!=m['exclusive_output']['absolute_path']:raise ValueError('PILOT_WORKER_PATH')
    if (c.read(out/'authorization_source.json')!=g or c.read(out/'launch_binding.json')!=dict(
            manifest_digest=digest(m),authorization_digest=digest(g),authorization_source_sha256=c.file_hash(out/'authorization_source.json'))):
        raise ValueError('PILOT_WORKER_GRANT')
    install_parent_guard(c.read(out/'execution_claim.json')['coordinator_identity'])
    binding=c.read(directory/'task_binding.json');r=m['proposed_resources'];slot=binding['slot']
    if type(slot) is not int or slot!=0:raise ValueError('PILOT_WORKER_SLOT')
    expected=dict(manifest_digest=digest(m),index=binding['index'],task=binding['task'],slot=slot,
        cpus=r['cost_cpu_sets'][slot],threads=1,address_space_bytes=r['cost_address_space_bytes'])
    if (digest(binding)!=digest(expected) or {'index':binding['index'],'task':binding['task']} not in m['registrations']
            or directory!=group_path(out,binding['index'])):raise ValueError('PILOT_WORKER_TASK')
    if set(m['source_hashes'])!=set(c.sources(root)):raise ValueError('PILOT_WORKER_CLOSURE')
    for p,h in m['source_hashes'].items():
        if c.file_hash(c.safe_path(root,p))!=h:raise ValueError('PILOT_WORKER_SOURCE')
    base=m['base_manifest']
    for p,h in base['input_identity']['parent_raw_hashes'].items():
        if c.file_hash(c.safe_path(root,p))!=h:raise ValueError('PILOT_WORKER_INPUT')
    if c.environment()!=base['environment']:raise ValueError('PILOT_WORKER_ENVIRONMENT')
    if any(os.environ.get(k)!='1' for k in THREAD_KEYS) or os.environ.get('QISKIT_PARALLEL')!='FALSE':
        raise ValueError('PILOT_WORKER_THREADS')
    limits=install_limits(binding['cpus'],r['cost_address_space_bytes'],32*2**20)
    writer=AtomicWriter(directory,byte_cap=32*2**20);task=binding['task']
    writer.write('worker_claim.json',dict(source_commit=m['source_commit'],task_digest=digest(task),resource_limits=limits,
        actual_cpus=sorted(os.sched_getaffinity(0)),threads=1,retry=False,resume=False))
    started=time.monotonic();trace=ResourceTraceV3(writer.write,lambda:time.monotonic()-started)
    reserve=bytearray(2**20);reason=None;status='TASK_COMPLETE'
    try:
        with trace.bind(),trace.stage('cost_task',cell_id=task['cell_id'],phase=c.PHASE,replica=task['replica']):
            from .h6_roundoff_cost_port_v1 import cost_task
            from .h6_roundoff_preparation_v1 import read_preparations,use_preparations
            from .h6_hash_reuse_v1 import use_fingerprint_reuse
            c.roundoff_binding(root,m);ref=m['roundoff_preparation_binding']
            preparations,archive=read_preparations(ref['archive_path'],ref['archive_sha256'])
            if (archive['binding']['input_identity']!=base['input_identity'] or archive['binding']['profile_sha256']!=base['profile']['sha256']
                    or archive['binding']['compiler']!=base['plan']['compiler_proposed']):raise ValueError('PILOT_ARCHIVE_BINDING')
            effective=deepcopy(base)
            for k,v in dict(compile=2,control_probe=0,trajectory=1,occurrence=8,primitive=0,reference_matvec=0).items():
                effective['plan']['caps_proposed'][k]=v
            with use_preparations(preparations),use_fingerprint_reuse() as cache,conditional_port_scope(task,m['prior_binding']['finite_distribution']):
                cost_task(root,effective,task,writer,Progress(writer),trace)
                writer.write('hash_reuse_statistics.json',cache.snapshot())
        for p,h in m['source_hashes'].items():
            if c.file_hash(c.safe_path(root,p))!=h:raise ValueError('PILOT_SOURCE_AFTER')
    except Exception as exc:
        del reserve;reason=type(exc).__name__+':'+str(exc)[:512];status='TASK_STOP'
        try:trace.failure(exc)
        except Exception:pass
    writer.write('worker_terminal.json',dict(status=status,reason=reason,task=task,source_commit=m['source_commit'],
        memory=process_memory(os.getpid()),mandatory_stop=True,next_stage_authorized=False),terminal=True)
    return 0 if reason is None else 1


def admit_group(directory,task,*,source,finite_distribution):
    directory=Path(directory);term=c.read(directory/'worker_terminal.json');claim=c.read(directory/'worker_claim.json')
    binding=c.read(directory/'task_binding.json');result=c.read(directory/'cost_result.json')
    trajectory=c.read(directory/'trajectory.json');condition=c.read(directory/'conditional_sampling_receipt.json')
    if (task['phase']!=c.PHASE or digest(task['cell'])!=digest(c.CELL)
            or task['seed']!=pilot_seed(task['cell_id'],task['replica']) or task['validate'] is not False or task['ordinary'] is not False
            or term['status']!='TASK_COMPLETE' or term['reason'] is not None
            or term['source_commit']!=source or claim['source_commit']!=source
            or claim['task_digest']!=digest(task) or claim['resource_limits']['rlimit_as']!=[32*2**30]*2
            or any(digest(x['task'])!=digest(task) for x in (term,binding,result,trajectory,condition))):
        raise ValueError('PILOT_GROUP_IDENTITY')
    events=trajectory['events'];orders=[e['taylor_order'] for s in events for e in s]
    if (len(events)!=8 or any(len(s)!=2 for s in events) or len(orders)!=16 or 2 not in orders
            or any(type(k) is not int or k not in (0,2) for k in orders)
            or trajectory['observed_orders']!=orders or trajectory['finite_distribution']!=finite_distribution
            or digest(events)!=trajectory['event_digest'] or condition['event_digest']!=trajectory['event_digest']
            or condition['original_finite_distribution']!=finite_distribution or condition['observed_orders']!=orders
            or condition['master_seed']!=task['seed'] or condition['order2_count']!=orders.count(2)
            or condition['stratum']!='s2' or condition['whole_trajectory'] is not True
            or any(condition[k] is not False for k in ('quantum_event_weights_changed','normalization_changed','component_selection_probability_changed'))):
        raise ValueError('PILOT_GROUP_CONDITIONING')
    mass=conditional_binomial(16,finite_distribution['order_probabilities'][1])['s2_probability']
    probability=math.prod(e['event_probability'] for s in events for e in s)
    if not math.isclose(condition['s2_probability'],mass,rel_tol=1e-14) or not math.isclose(
            condition['conditional_event_sequence_probability'],probability/mass,rel_tol=1e-13,abs_tol=1e-300):
        raise ValueError('PILOT_GROUP_PROBABILITY')
    rows=result['rows'];keys=[]
    for row in rows:
        key=(row['control'],row['axis']);keys.append(key)
        if (any(type(row[k]) is not type(task[k]) or row[k]!=task[k] for k in ('cell_id','phase','replica','seed'))
                or row['event_digest']!=trajectory['event_digest'] or row['compiler_hash']!=COMPILER_HASH
                or len(row['fingerprint'])!=64 or set(row['metrics'])!={'RZ','CX','RZ_depth','CX_depth','depth','size'}
                or any(type(v) is not int or v<0 for v in row['metrics'].values())
                or row!=c.read(directory/f'wrapper_{key[0]}_{key[1]}.json')):raise ValueError('PILOT_WRAPPER_IDENTITY')
    if len(keys)!=2 or set(keys)!={('symmetric_directional','cosine'),('symmetric_directional','sine')}:
        raise ValueError('PILOT_PAIRED_AXES')
    if digest(result['calls_attempted'])!=digest(dict(compile=2,control_probe=0,trajectory=1,occurrence=8,primitive=0,reference_matvec=0)):
        raise ValueError('PILOT_TASK_CALL_CAP')
    return result


def calls_after_run(m,groups):
    if any(type(k) is not int or not 0<=k<8 for k in groups):raise ValueError('PILOT_ACCOUNTING_GROUP_SCOPE')
    a=m['cumulative_accounting'];known=deepcopy(a['known_calls']);reserved=deepcopy(a['reservations'])
    for group in groups.values():
        for k,v in group['calls_attempted'].items():known[k]+=v
    pending=8-len(groups)
    for k,v in dict(compile=2,control_probe=0,trajectory=1,occurrence=8,primitive=0,reference_matvec=0).items():
        if v:reserved[k]=reserved.get(k,0)+pending*v
    if any(known[k]+reserved.get(k,0)>a['proposed_caps'][k] for k in known):raise ValueError('PILOT_CUMULATIVE_CALL_CAP')
    return dict(known_calls=known,reservations=reserved,historical_unknown_actual_totals=None,
        reservations_are_not_actual_totals=True,unaccepted_new_task_actual_totals=None if pending else {},
        unaccepted_new_tasks_fully_reserved=pending,budget_reset=False,failed_calls_refunded=False)


def result_statistics(groups):
    rows=[r for i in sorted(groups) for r in groups[i]['rows']];stats={}
    for axis in ('cosine','sine'):
        stats[axis]={}
        for metric in ('RZ','CX','RZ_depth','CX_depth','depth','size'):
            values=[r['metrics'][metric] for r in rows if r['axis']==axis];n=len(values)
            stats[axis][metric]=dict(n=n,mean=statistics.mean(values),min=min(values),max=max(values),
                sample_sd=statistics.stdev(values) if n>1 else None,
                standard_error=statistics.stdev(values)/math.sqrt(n) if n>1 else None)
    return dict(conditional_mean_scope='s2 only; weighted by saved pi_s2 in a separate analysis; not pooled with exploration/confirmation',
        groups=len(groups),wrappers=len(rows),axes=stats,population_mean_certified=False,cost_mean_tail_bound=None,
        formal_winner_certified=False,scientific_comparison_conditions_changed=False)


def orchestrate(root,out,m,writer,fresh):
    r=m['proposed_resources'];base=m['base_manifest'];report=None;reason=None;groups={}
    limits=install_limits(r['coordinator_cpu_set'],r['coordinator_address_space_bytes'],
        base['plan']['caps_proposed']['output_bytes'],hard_address_bytes=r['coordinator_hard_address_space_bytes'])
    writer.write('execution_claim.json',dict(coordinator_identity=process_identity(os.getpid()),resource_limits=limits,fresh=fresh))
    try:
        def admission(directory,task):return admit_group(directory,task,source=m['source_commit'],finite_distribution=m['prior_binding']['finite_distribution'])
        def command(row,directory,slot):return [sys.executable,'-I','-P','-B',str(Path(root)/c.RUNNER),'--worker',str(directory),'--run-root',str(out)]
        report=supervise(root,out,m,command=command,admission=admission)
        writer.write('pilot_resource_summary.json',report)
        for row in m['registrations']:
            if row['index'] in report['completed_indices']:groups[row['index']]=admission(group_path(out,row['index']),row['task'])
        writer.write('aggregate_calls.json',calls_after_run(m,groups))
        if report['first_stop_reason']:raise RuntimeError(report['first_stop_reason'])
        if (set(groups)!=set(range(8)) or not report['all_dispatched_reaped'] or report['cleanup_errors']
                or not report['ownership_cleanup_verified'] or not report['log_and_journal_closed']):
            raise ValueError('PILOT_INCOMPLETE_OR_CLEANUP')
        writer.write('conditional_resource_statistics.json',result_statistics(groups))
        c.validate_plan(root,m)
        writer.write('pilot_complete_receipt.json',dict(group_count=8,wrapper_count=16,source_commit=m['source_commit'],
            conditional_sampling_only=True,rows_digest=digest([r for i in sorted(groups) for r in groups[i]['rows']]),
            population_mean_certified=False,next_stage_authorized=False,mandatory_stop=True))
    except Exception as exc:
        reason=type(exc).__name__+':'+str(exc)[:512]
        # Unknown/partial dispatch is never credited or refunded after a STOP.
        if not (Path(out)/'aggregate_calls.json').exists():
            writer.write('aggregate_calls.json',calls_after_run(m,groups))
    writer.write('execution_terminal.json',dict(status='H6_CONDITIONAL_S2_PILOT_COMPLETE_MANDATORY_STOP' if reason is None else 'H6_CONDITIONAL_S2_PILOT_STOP',
        reason=reason,source_commit=m['source_commit'],resource_summary=report,mandatory_stop=True,
        retry=False,resume=False,next_stage_authorized=False,H8_authorized=False,population_winner='UNDETERMINED_POPULATION_MEAN'),terminal=True)
    return 0 if reason is None else 1
