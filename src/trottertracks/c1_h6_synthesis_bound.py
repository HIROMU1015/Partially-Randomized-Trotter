"""Dormant saved-QPY complete-sector bound runner; no new circuit/PF/fit."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import math
import resource
import time
import traceback
from . import c1_h6_native as native
from . import c1_subspace_synthesis_bound as bounds

io=native.io
SCOPE='C1_H6_COMPLETE_SECTOR_SYNTHESIS_BOUND_V1'
FAMILY='artifacts/research_c_c1_h6_synthesis_bound'
LIMITS=dict(saved_bundle_load=1,sector_basis=1,reference_check=4,qpy_structure=8,
    route_grouping=1,state_action_columns=6400,kernel_bound=8,round_bound=104,candidate_summary=8)
MAX_CALLS=sum(LIMITS.values())
ALLOWANCES=dict(state_component=1e-10,state_norm_drift=1e-10,reference_unitarity_component=1e-10,
    global_phase_fit=False,rigorous_enclosure=None)
JOBS=native.JOBS


def preflight(root,path):
    root=Path(root);c=io.read(path)
    if (c['scope']!=SCOPE or c['status']!='UNAPPROVED_SCIENCE_CONTRACT'
            or c['conditions']!=native.saved.CONDITIONS or c['allowances']!=ALLOWANCES
            or c['call_limits']!=LIMITS or c['maximum_logical_units']!=MAX_CALLS
            or c['method']!=bounds.VERSION or c['candidate_order']!=[f'{a}:s{s}' for a,s in native.saved.CANDIDATES]
            or c['routes']!=list(native.ROUTES)
            or any(c[k] for k in ['new_fit','optimizer','DF_recomputation','PF_recomputation','build_compile',
                                  'CA_evaluation','shots_evaluation','total_cost_evaluation','retry','resume'])):
        raise ValueError('SYNTHESIS_BOUND_SCOPE')
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    for group in [io.read(root/c['source_manifest'])['files'],c['saved_inputs']]:
        for f,h in group.items():
            if io.sha(root/f)!=h:raise ValueError('IDENTITY: '+f)
    for p,w in c['evaluation_packages'].items():
        if importlib.metadata.version(p)!=w:raise ValueError('ENVIRONMENT: '+p)
    required=[c['native_result'],c['PF_result'],*c['PF_kernels'].values(),*c['compiled_circuits'].values()]
    if any(p not in c['saved_inputs'] for p in required):raise ValueError('UNPINNED_INPUT')
    if (set(c['PF_kernels'])!={f'{a}:s{s}' for a,s in native.saved.CANDIDATES}
            or set(c['compiled_circuits'])!={f'{a}_s{s}_{r}' for a,s,r in JOBS}):
        raise ValueError('REGISTERED_INPUT_COVERAGE')
    return c


def circuit_identity(circuit):
    """Exact P0 IR identity, including declared phase; circuit name excluded.

    Only official real-parameter unitary gates are accepted. No Operator matrix
    or numerical action is computed by this structural check.
    """
    from qiskit.circuit.library import RZGate,SXGate,XGate,CXGate
    allowed={x.__name__:x for x in [RZGate,SXGate,XGate,CXGate]}
    if circuit.num_clbits or circuit.parameters:raise ValueError('BOUND_P0_UNITARY_REQUIRED')
    if circuit.layout is not None and circuit.layout.final_index_layout(filter_ancillas=True)!=list(range(circuit.num_qubits)):
        raise ValueError('NONIDENTITY_LAYOUT')
    phase=float(circuit.global_phase)
    if not math.isfinite(phase):raise ValueError('NONFINITE_PHASE')
    h=hashlib.sha256();h.update(json.dumps([circuit.num_qubits,phase.hex()]).encode())
    for item in circuit.data:
        op=item.operation;cls=op.base_class
        if (cls.__name__ not in allowed or cls is not allowed[cls.__name__] or item.clbits
                or getattr(op,'condition',None) is not None):raise ValueError('UNSUPPORTED_OR_NONUNITARY_P0_GATE')
        params=[float(x) for x in op.params]
        if not all(math.isfinite(x) for x in params):raise ValueError('NONFINITE_GATE')
        entry=[cls.__name__,[x.hex() for x in params],[circuit.find_bit(q).index for q in item.qubits]]
        h.update(json.dumps(entry,separators=(',',':')).encode());h.update(b'\n')
    return dict(signature=h.hexdigest(),declared_phase=phase,
        structural_unitarity='exact ideal semantics of official rz/sx/x/cx real-parameter gates',
        floating_point_state_action_error_enclosed=False)


def column_batch(calls,label,function,count):
    """Use existing journal/STOP, charge logical columns independently of batching."""
    if type(count) is not int or count<1:raise ValueError('COLUMN_CALL_COUNT')
    with calls.lock:
        if calls.stop.is_set():raise RuntimeError('DEPENDENT_WORK_STOPPED')
        if sum(calls.counts.values())+count>calls.maximum:raise ValueError('TOTAL_CALL_LIMIT')
        calls.mark('state_action_columns',label,count);calls.counts['state_action_columns']+=count
    tick=time.monotonic()
    try:value=function()
    except BaseException as error:
        with calls.lock:
            calls.stop.set()
            if calls.first_failure is None:calls.first_failure=dict(kind='state_action_columns',label=label,
                error=str(error),traceback=traceback.format_exc())
            calls._finish('state_action_columns',label,tick,'FAIL')
        raise
    with calls.lock:calls._finish('state_action_columns',label,tick,'COMPLETE')
    return value


def run(root,path,output):
    import numpy as np
    from qiskit import qpy
    from concurrent.futures import ThreadPoolExecutor
    from threadpoolctl import threadpool_limits
    root,output=Path(root),Path(output);c=preflight(root,path)
    output.mkdir(parents=True,exist_ok=False);(output/'groups').mkdir();(output/'routes').mkdir()
    calls=native.saved.molecule.integrated.Calls(output,LIMITS,MAX_CALLS)
    start=time.monotonic();cpu=time.process_time();io.save(output/'contract.json',c)
    try:
        def load():
            previous=io.read(root/c['native_result']);pf=io.read(root/c['PF_result'])
            if (previous['status']!='H6_NATIVE_FINITE_PROBE_COST_COMPLETE'
                    or pf['status']!='H6_SAVED_RANK14_SECTOR_PF_COMPLETE'):raise ValueError('SAVED_EVIDENCE_STATUS')
            shells={k:v['unitary_distance_pilot'] for k,v in previous['shells'].items()}
            if set(shells)!=set(['head','X','Y']) or any(v is None or not math.isfinite(v) or v<0 for v in shells.values()):
                raise ValueError('SHELL_FULL_2X2_EVIDENCE_REQUIRED')
            kernels={}
            for key,f in c['PF_kernels'].items():
                with np.load(root/f,allow_pickle=False) as a:kernels[key]=np.asarray(a['pf'])
            return kernels,shells,pf
        kernels,shells,pf=calls('saved_bundle_load','saved four PF kernels/full2x2 shells/52 PF rows',load)
        def sector():
            n=c['conditions']['atoms'];d=native.saved.molecule.stage1.legacy_df_module()
            basis=d.PhysicalSector.spin_sector(n_qubits=2*n,nelec_alpha=c['conditions']['Nalpha'],nelec_beta=c['conditions']['Nbeta'])
            return native.saved.molecule.stage1.bit_reverse_indices(2*n)[basis.basis_indices]
        indices=calls('sector_basis','original CAR sector ordering, no bilinears/operator',sector)
        dimension=c['conditions']['sector_dimension'];system=1<<(2*c['conditions']['atoms'])
        if len(indices)!=dimension:raise ValueError('SECTOR_DIMENSION')
        refs={}
        for key,w in kernels.items():
            refs[key]=calls('reference_check',key,lambda w=w:bounds.reference_info(w))
            if w.shape!=(dimension,dimension):raise ValueError('REFERENCE_DIMENSION')
        circuits={};identities={}
        for arm,s,route in JOBS:
            label=f'{arm}_s{s}_{route}'
            def read(label=label):
                with (root/c['compiled_circuits'][label]).open('rb') as f:items=qpy.load(f)
                if len(items)!=1 or items[0].num_qubits!=2*c['conditions']['atoms']+1:raise ValueError('QPY_SCHEMA')
                return items[0],circuit_identity(items[0])
            circuits[label],identities[label]=calls('qpy_structure',label,read)
        def group():
            groups={};aliases={}
            for arm,s,route in JOBS:
                label=f'{arm}_s{s}_{route}';key=f'{arm}:s{s}'
                token=(key,c['saved_inputs'][c['PF_kernels'][key]],identities[label]['signature'])
                groups.setdefault(token,[]).append(label)
            values=list(groups.values())
            for labels in values:
                for label in labels:aliases[label]=labels[0]
            return values,aliases
        groups,aliases=calls('route_grouping','same arm/s + same target + exact compiled IR only',group)
        io.save(output/'route_identity.json',dict(identities=identities,groups=groups,aliases=aliases))
        plan=native.saved.molecule.integrated.resource_plan(task_count=len(groups),full_dimension=dimension)
        threads=max(1,min(plan['available_affinity_cpus'],plan['quota_cpus'] or plan['available_affinity_cpus'])//plan['workers'])
        plan.update(Aer_threads_per_worker=threads,batch_columns=threads,
            batch_policy='one active experiment per available worker thread; streaming full outputs, cached conversion')
        io.save(output/'resource_plan.json',plan)
        if plan['available_memory_bytes'] is not None and plan['available_memory_bytes']<plan['estimated_bytes_per_candidate']:
            raise ValueError('INSUFFICIENT_AVAILABLE_MEMORY_FOR_ESTIMATED_WORKSPACE')
        def evaluate(labels):
            label=labels[0];arm,ss,_=label.split('_');key=f'{arm}:{ss}'
            stream=bounds.ColumnStream(2*dimension,2*system);engine=native.action.StateActions(threads)
            try:
                for lo in range(0,2*dimension,threads):
                    columns=list(range(lo,min(lo+threads,2*dimension)))
                    def batch():
                        initial,expected=bounds.basis_batch(indices,system,kernels[key],columns)
                        actual=engine.apply(circuits[label],initial)
                        # Preserve every finite evaluated column even when the
                        # existing correctness gate rejects this batch.
                        stream.add(columns,actual,expected)
                        check=native.action.state_gate(actual,expected,label=f'{label}:columns{lo}')
                        with (output/'groups'/(label+'_batches.jsonl')).open('a') as f:
                            f.write(json.dumps(dict(columns=columns,max_entry=check['max_entry'],norm_drift=check['norm_drift']))+'\n')
                    column_batch(calls,f'{label}:{lo}',batch,len(columns))
            finally:io.save(output/'groups'/(label+'_columns.json'),stream.finish())
            value=stream.finish()
            if value['status']!='COMPLETE':raise ValueError('INCOMPLETE_BASIS')
            return label,dict(value,**refs[key])
        with threadpool_limits(limits=1 if plan['workers']>1 else None):
            with ThreadPoolExecutor(max_workers=plan['workers']) as pool:measured=dict(pool.map(evaluate,groups))
        routes={}
        for arm,s,route in JOBS:
            label=f'{arm}_s{s}_{route}';key=f'{arm}:s{s}'
            kernel=calls('kernel_bound',label,lambda label=label:dict(measured[aliases[label]],evaluated_route=aliases[label],route=label))
            rows=pf['candidates'][key]['rounds']
            if [r['m'] for r in rows]!=list(range(13)):raise ValueError('PF_ROUND_IDENTITY')
            rounds=[]
            for row in rows:
                rounds.append(calls('round_bound',f'{label}:m{row["m"]}',lambda row=row:
                    bounds.round_bound(kernel,shells,s=s,m=row['m'],saved_pf_error=row['pf_operator_error'])))
            def summarize():
                return dict(label=label,kernel=kernel,rounds=rounds,
                    status='POTENTIALLY_PHASE_ELIGIBLE_NUMERICAL' if all(r['status']=='POTENTIALLY_PHASE_ELIGIBLE_NUMERICAL' for r in rounds)
                        else 'INFEASIBLE_REGISTERED_BOUND',CA_achieved=None,shots=None,total_cost=None)
            routes[label]=calls('candidate_summary',label,summarize)
            io.save(output/'routes'/(label+'.json'),routes[label])
        io.save(output/'result.json',dict(status='H6_NUMERICAL_RESTRICTED_BOUND_COMPLETE',routes=routes,
            evaluated_groups=len(groups),evidence_level=bounds.LEVEL,rigorous_enclosure=None,
            CA_achieved=None,shots=None,total_cost=None,new_fit=0,build_compile=0,retry=0,resume=0,mandatory_stop=True))
    except BaseException as error:
        calls.stop.set();io.save(output/'failure.json',dict(error=str(error),first_failure=calls.first_failure));raise
    finally:
        io.save(output/'STOP.json',dict(failed=calls.stop.is_set(),scientific_units=sum(calls.counts.values()),call_counts=dict(calls.counts),retry=0,resume=0,mandatory_stop=True))
        io.save(output/'resources.json',dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
