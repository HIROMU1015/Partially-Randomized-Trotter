"""Dormant saved-H6 native correctness/cost pilot; no shots or CA evaluation."""
from dataclasses import asdict,replace
from fractions import Fraction
from pathlib import Path
import importlib.metadata
import math
import resource
import time

from . import c1_h6_saved_pf as saved
from . import c1_h6_state_action as action
from . import c1_stage2_native as native
from . import c1_stage3_connection as connection
from . import c1_stage3_deterministic_accuracy as det
from .c1_stage3_cost_compat import CompileSerialCalls

io=saved.io
SCOPE='C1_H6_RANK14_NATIVE_PROBE_COST_V1'
FAMILY='artifacts/research_c_c1_h6_native'
ROUTES=('literal','shared')
SHELLS=('head','X','Y')
JOBS=tuple((arm,s,route) for arm,s in saved.CANDIDATES for route in ROUTES)
LIMITS=dict(saved_bundle_load=1,native_model=2,sector_context=1,sector_generators=2,
    gaussian_build=30,local_reference=30,circuit_build=41,local_action=30,
    compile=11,shell_assessment=3,kernel_action=8,adjoint_action=8,
    hadamard_action=8,metrics=11,round_synthesis=104,cost_comparison=8)
MAX_CALLS=sum(LIMITS.values())
ALLOWANCES=dict(state_component=1e-10,state_norm_drift=1e-10,signal_complex=2e-10,
    local_time='1/20',global_phase_fit=False,probe_seed=action.PROBE_SEED,
    state_coverage='control0 full-Fock random; control1 sector basis/complex/own-ground; control+ own-ground',
    synthesis_policy='finite probe residuals are not global operator norm bounds')


def preflight(root,path):
    root=Path(root);c=io.read(path)
    from .pr_dictionary_validation.fixture import CompilerProfile
    profile=asdict(CompilerProfile());profile['basis_gates']=list(profile['basis_gates'])
    if (c['scope']!=SCOPE or c['status']!='UNAPPROVED_SCIENCE_CONTRACT'
            or c['conditions']!=saved.CONDITIONS or c['allowances']!=ALLOWANCES
            or c['compiler']!=profile or c['call_limits']!=LIMITS
            or c['maximum_logical_units']!=MAX_CALLS or c['routes']!=list(ROUTES)
            or c['candidate_order']!=[f'{a}:s{s}' for a,s in saved.CANDIDATES]
            or c['state_action']!=action.VERSION
            or any(c[k] for k in ('new_fit','optimizer','PF_recomputation','DF_error_recomputation',
                                  'shots_evaluation','total_cost_evaluation','retry','resume'))):
        raise ValueError('H6_NATIVE_SCOPE')
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    for f,w in io.read(root/c['source_manifest'])['files'].items():
        if io.sha(root/f)!=w:raise ValueError('SOURCE: '+f)
    for f,w in c['saved_inputs'].items():
        if io.sha(root/f)!=w:raise ValueError('SAVED_INPUT: '+f)
    required=[c['original_input'],c['saved_optimizer'],c['PF_result'],*c['factor_arrays'].values(),
        *c['saved_sector_rows'].values(),*c['ground_states'].values(),*c['PF_kernels'].values()]
    if any(f not in c['saved_inputs'] for f in required):raise ValueError('UNPINNED_INPUT')
    for p,w in c['evaluation_packages'].items():
        if importlib.metadata.version(p)!=w:raise ValueError('ENVIRONMENT: '+p)
    return c


def load_bundle(root,c):
    import numpy as np
    packet,factors,rows,optimizer=saved.load_saved(root,c)
    for arm in saved.ARMS:
        saved.saved_DF_gate(factors[arm],rows[arm],optimizer if arm=='CDF' else None,
            rank=c['conditions']['rank'],n=len(packet['h']),budget=c['conditions']['DF_reserve_Ha'])
    previous=io.read(root/c['PF_result'])
    if previous['status']!='H6_SAVED_RANK14_SECTOR_PF_COMPLETE':raise ValueError('SAVED_PF_STATUS')
    ground={};kernels={};d=c['conditions']['sector_dimension']
    for arm,path in c['ground_states'].items():
        with np.load(root/path,allow_pickle=False) as a:ground[arm]=a['state']
        if ground[arm].shape!=(d,) or abs(np.linalg.norm(ground[arm])-1)>1e-10:raise ValueError('GROUND_STATE_SCHEMA')
    for label,path in c['PF_kernels'].items():
        with np.load(root/path,allow_pickle=False) as a:kernels[label]=a['pf']
        candidate=previous['candidates'][label]
        if (kernels[label].shape!=(d,d) or candidate['summary']['status']!='POTENTIALLY_FEASIBLE'
                or [r['m'] for r in candidate['rounds']]!=list(range(13))):raise ValueError('SAVED_PF_SCHEMA')
    saved.molecule.stage1.finite_arrays(*ground.values(),*kernels.values())
    return packet,factors,ground,kernels


def prepare_model(packet,data,arm,c):
    from .c1_general_k_adapter import from_ffsim_number_arrays
    model=from_ffsim_number_arrays(one_body_tensor=saved.molecule.stage1.corrected_kappa(packet['h'],packet['g']),
        diag_coulomb_mats=data['J'],orbital_rotations=data['U'],constant=float(packet['constant']),
        z_representation=False,frame_ids=[f'F{i}' for i in range(len(data['J']))],
        group_ids=[f'F{i}' for i in range(len(data['J']))],source_identity=c['saved_inputs'][c['factor_arrays'][arm]],
        input_identity=c['saved_inputs'][c['original_input']],one_body_role='ffsim_corrected_kappa',enable_general_k=True)
    # Both saved J/U arms use the same general-K number emitter. No new XDF
    # square-root decomposition or special optimization is applied to one arm.
    return replace(native.prepare_cdf(model),label=arm)


def shell_unitary(circuit):
    """Only the ancilla's 2x2 matrix; any system operation is a failure."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    qc=action.measured_tail(circuit) if circuit.num_clbits else circuit
    small=QuantumCircuit(1);small.global_phase=qc.global_phase
    for item in qc.data:
        if item.operation.name=='barrier':continue
        if [qc.find_bit(q).index for q in item.qubits]!=[qc.num_qubits-1]:raise ValueError('SHELL_SYSTEM_OPERATION')
        small.append(item.operation,[0])
    return Operator(small).data


def round_profile(kernel,shells,occurrences):
    # Never feed a finite-probe maximum into the old global telescoping formula.
    out=det.round_synthesis(dict(unitary_distance_pilot=None),shells,occurrences)
    return dict(**out,qualification='UNRESOLVED_FINITE_PROBES',kernel_operator_norm_bound=None,
        finite_probe_maximum=kernel['raw_reference']['max_entry'],
        CA_achieved=None,shots=None,total_cost=None)


def compare_metrics(left,right,label):
    metrics=('rz_count','rz_depth','cx_count','cx_depth','total_depth','circuit_size')
    return dict(label=label,scope='one P0 controlled native S2 kernel, no state preparation',
        left=left,right=right,difference_left_minus_right={k:left[k]-right[k] for k in metrics},
        RZ_ratio_left_over_right=left['rz_count']/right['rz_count'] if right['rz_count'] else None,
        CA_achieved=None,total_resource_superiority=None)


def run(root,contract_path,output):
    import numpy as np
    from scipy.linalg import expm
    from concurrent.futures import ThreadPoolExecutor
    from threadpoolctl import threadpool_limits
    from qiskit import qpy
    from .pr_dictionary_validation.fixture import CompilerProfile
    from .pr_dictionary_validation.gates import compiled_wrapper,signal_gate
    from .pr_dictionary_validation.f2_execution import circuit_metrics
    root,output=Path(root),Path(output);c=preflight(root,contract_path)
    output.mkdir(parents=True,exist_ok=False)
    for name in ('circuits','local','candidates'):(output/name).mkdir()
    calls=saved.molecule.integrated.Calls(output,LIMITS,MAX_CALLS);compile_calls=CompileSerialCalls(calls)
    start=time.monotonic();cpu=time.process_time();profile=CompilerProfile()
    plan=saved.molecule.integrated.resource_plan(task_count=8,full_dimension=c['conditions']['sector_dimension'])
    plan['Aer_threads_per_worker']=max(1,(min(plan['available_affinity_cpus'],plan['quota_cpus'] or plan['available_affinity_cpus']))//plan['workers'])
    io.save(output/'contract.json',c);io.save(output/'resource_plan.json',plan)
    def qsave(label,kind,qc):
        with (output/'circuits'/f'{label}_{kind}.qpy').open('xb') as f:qpy.dump(qc,f)
    try:
        packet,factors,ground,kernels=calls('saved_bundle_load','fixed H6/PF/ground, no new fit/norm',lambda:load_bundle(root,c))
        ctx=calls('sector_context','sector400',lambda:saved.molecule.SectorCAR(len(packet['h'])))
        modes=2*len(packet['h']);D=1<<modes
        indices=saved.molecule.stage1.bit_reverse_indices(modes)[ctx.sector.basis_indices]
        models={};generators={};builder=native.NativeBuilder()
        for arm in saved.ARMS:
            models[arm]=calls('native_model',arm,lambda arm=arm:prepare_model(packet,factors[arm],arm,c))
            generators[arm]=calls('sector_generators',arm,lambda arm=arm:ctx.generators(packet['h'],packet['g'],packet['constant'],factors[arm]['J'],factors[arm]['U']))
            for block in models[arm].blocks:
                calls('gaussian_build',arm+':'+block.block_id,lambda arm=arm,block=block:builder._basis(models[arm],block))
        # Build shared cached basis circuits sequentially; heavy independent
        # local exponentials/state batches and candidate actions run in parallel.
        local_jobs=[]
        rng=np.random.default_rng(action.PROBE_SEED);v=rng.normal(size=ctx.d)+1j*rng.normal(size=ctx.d);v/=np.linalg.norm(v)
        psi=action.sector_embedding(indices,v,D);initial=np.r_[psi,psi][None,:]/math.sqrt(2)
        for arm in saved.ARMS:
            for index in range(len(models[arm].blocks)):
                label=f'{arm}_block{index}'
                qc=calls('circuit_build',label,lambda arm=arm,index=index:builder.local(models[arm],index,Fraction(1,20)))
                qsave(label,'raw',qc);local_jobs.append((arm,index,label,qc))
        def local_job(job):
            arm,index,label,qc=job
            reference=calls('local_reference',label,lambda:expm(-.05j*generators[arm][index]))
            target=action.sector_embedding(indices,reference@v,D)
            expected=np.r_[psi,target][None,:]/math.sqrt(2)
            report=calls('local_action',label,lambda:action.state_gate(
                action.StateActions(plan['Aer_threads_per_worker']).apply(qc,initial),expected,label=label))
            io.save(output/'local'/(label+'.json'),report);return report
        with threadpool_limits(limits=1 if plan['workers']>1 else None):
            with ThreadPoolExecutor(max_workers=plan['workers']) as pool:local_reports=list(pool.map(local_job,local_jobs))
        raw_shells={};compiled_shells={};shell_reports={};shell_metrics={}
        H=np.array([[1,1],[1,-1]],complex)/math.sqrt(2)
        for name in SHELLS:
            def make(name=name):
                head,tail=connection.readout_segments(modes,'Y' if name=='Y' else 'X')
                return head if name=='head' else tail
            raw=calls('circuit_build','shell_'+name,make)
            compiled=compile_calls('compile','shell_'+name,lambda raw=raw:compiled_wrapper(raw,profile))
            qsave('shell_'+name,'raw',raw);qsave('shell_'+name,'compiled',compiled)
            ideal=H@np.diag([1,-1j]) if name=='Y' else H
            shell_reports[name]=calls('shell_assessment',name,lambda raw=raw,compiled=compiled,ideal=ideal,name=name:
                det.numerical_segment(shell_unitary(raw),shell_unitary(compiled),ideal,
                    dict(operator_component_abs=1e-10,unitarity_component_abs=1e-10),'ancilla_'+name))
            shell_metrics[name]=calls('metrics','shell_'+name,lambda compiled=compiled:circuit_metrics(compiled))
            raw_shells[name]=raw;compiled_shells[name]=compiled
        io.save(output/'shells.json',dict(reports=shell_reports,metrics=shell_metrics))
        circuits={}
        for arm,s,route in JOBS:
            label=f'{arm}_s{s}_{route}';h=Fraction(1,10*s)
            circuits[arm,s,route]=calls('circuit_build',label,lambda arm=arm,h=h,route=route:builder.ordered(models[arm],h,optimized=route=='shared'))
            qsave(label,'raw',circuits[arm,s,route])
        def candidate_job(key):
            arm,s,route=key;label=f'{arm}_s{s}_{route}';raw=circuits[key];ideal=kernels[f'{arm}:s{s}']
            compiled=compile_calls('compile',label,lambda:compiled_wrapper(raw,profile));qsave(label,'compiled',compiled)
            engine=action.StateActions(plan['Aer_threads_per_worker'])
            inputs,expected=action.probes(indices,ground[arm],D,ideal)
            def assess():
                a=engine.apply(raw,inputs);b=engine.apply(compiled,inputs)
                report=dict(raw_reference=action.state_gate(a,expected,label='raw:'+label),
                    compiler=action.state_gate(b,a,label='compiled/raw:'+label),
                    compiled_reference=action.state_gate(b,expected,label='compiled/ref:'+label),
                    probe_count=5,operator_norm_bound=None,rigorous_enclosure=None)
                return report,a,b
            kernel_report,a,b=calls('kernel_action',label,assess)
            io.save(output/'candidates'/(label+'_kernel.json'),kernel_report)
            def adjoint():
                return dict(raw=action.state_gate(engine.apply(native.actual_adjoint(raw),a[-1:]),inputs[-1:],label='raw actual adjoint'),
                    compiled=action.state_gate(engine.apply(native.actual_adjoint(compiled),b[-1:]),inputs[-1:],label='compiled actual adjoint'))
            inverse_report=calls('adjoint_action',label,adjoint)
            io.save(output/'candidates'/(label+'_adjoint.json'),inverse_report)
            def hadamard():
                start_state=np.r_[action.sector_embedding(indices,ground[arm],D),np.zeros(D)][None,:]
                target=ground[arm]
                for _ in range(s):target=ideal@target
                target_signal=complex(np.vdot(ground[arm],target));signals={}
                for kind,qc,shell in [('raw',raw,raw_shells),('compiled',compiled,compiled_shells)]:
                    state=engine.apply(shell['head'],start_state)
                    for _ in range(s):state=engine.apply(qc,state)
                    signals[kind]={axis:float(action.ancilla_z(engine.apply(action.measured_tail(shell[axis]),state))[0]) for axis in ('X','Y')}
                return dict(raw_reference=signal_gate(signals['raw']['X'],signals['raw']['Y'],target_signal),
                    compiled_raw=signal_gate(signals['compiled']['X'],signals['compiled']['Y'],complex(signals['raw']['X'],signals['raw']['Y'])),
                    compiled_reference=signal_gate(signals['compiled']['X'],signals['compiled']['Y'],target_signal),
                    macro_occurrences=s,total_time=.1,coverage='own-ground signal at m0, not all13 rounds')
            signal_report=calls('hadamard_action',label,hadamard)
            io.save(output/'candidates'/(label+'_XY.json'),signal_report)
            metrics=calls('metrics',label,lambda:circuit_metrics(compiled))
            rounds=[]
            for m in range(13):
                row=calls('round_synthesis',f'{label}:m{m}',lambda m=m:dict(m=m,**round_profile(kernel_report,shell_reports,s*(1<<m))))
                rounds.append(row)
            value=dict(arm=arm,s=s,route=route,kernel=kernel_report,adjoint=inverse_report,XY=signal_report,
                metrics=metrics,round_synthesis=rounds,CA_achieved=None,shots=None,total_cost=None)
            io.save(output/'candidates'/(label+'.json'),value);return label,value
        with threadpool_limits(limits=1 if plan['workers']>1 else None):
            with ThreadPoolExecutor(max_workers=plan['workers']) as pool:results=dict(pool.map(candidate_job,JOBS))
        comparisons=[]
        for s in (1,4):
            for route in ROUTES:
                label=f'CDF_XDF_s{s}_{route}'
                comparisons.append(calls('cost_comparison',label,lambda s=s,route=route,label=label:compare_metrics(
                    results[f'CDF_s{s}_{route}']['metrics'],results[f'XDF_s{s}_{route}']['metrics'],label)))
        for arm,s in saved.CANDIDATES:
            label=f'{arm}_s{s}_literal_shared'
            comparisons.append(calls('cost_comparison',label,lambda arm=arm,s=s,label=label:compare_metrics(
                results[f'{arm}_s{s}_literal']['metrics'],results[f'{arm}_s{s}_shared']['metrics'],label)))
        io.save(output/'result.json',dict(status='H6_NATIVE_FINITE_PROBE_COST_COMPLETE',local_reports=local_reports,
            shells=shell_reports,candidates=results,comparisons=comparisons,global_synthesis_qualification='UNRESOLVED_FINITE_PROBES',
            CA_achieved=None,shots=None,total_cost=None,new_fit=0,optimizer=0,retry=0,resume=0,mandatory_stop=True))
    except BaseException as e:
        calls.stop.set();io.save(output/'failure.json',dict(error=str(e),first_failure=calls.first_failure));raise
    finally:
        io.save(output/'STOP.json',dict(mandatory_stop=True,failed=calls.stop.is_set(),retry=0,resume=0,
            scientific_units=sum(calls.counts.values()),call_counts=dict(calls.counts)))
        io.save(output/'resources.json',dict(wall_seconds=time.monotonic()-start,process_CPU_seconds=time.process_time()-cpu,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,compile_completed=compile_calls.completed,
            compile_queue_wait_seconds=compile_calls.wait_seconds))
