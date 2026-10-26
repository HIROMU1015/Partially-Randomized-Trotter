"""Opt-in B2 cost calibration. Imports/preflight never decode molecular arrays.

Reuses paired-CDF preparation, the corrected signed-event bridge, precise
Gaussian/P0 emitters, raw CAR, finite-RTE sampler, metrics and call journal.
No B1 rerun, fit, PF/DF norm, shots or total-resource evaluation lives here.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from fractions import Fraction
import importlib.metadata
import math
from pathlib import Path
import resource
import subprocess
import threading
import time
from types import SimpleNamespace

from . import c_pr_aware_df_stage_b as b
from . import c_pr_aware_df_stage_b_runner as b1

io, integrated = b1.io, b1.integrated
VERSION = 'paired_CDF_B2_full_controlled_calibration_v1'
SCOPE = 'C_PR_AWARE_DF_STAGE_B2_NATIVE_COST_V1'
FAMILY = 'artifacts/research_c_pr_aware_df_stage_b2'
FRESH = FAMILY + '/2026-10-27/run01'
SAMPLES = 32
MASTER = 2026102702
COHORT = 'B2_fixed32_per_arm_v1'
POLICY = dict(norb=4, rank=8, Nalpha=2, Nbeta=2, sector_dimension=36,
    system_qubits=8, ancilla_qubits=1, controlled_dimension=512, ld=2,
    delta='1/10', s=4, h='1/40', r=16, K=0, rounds=list(range(13)),
    CA_Ha=.00159, DF_reserve_Ha=.00019875,
    frame_order=list(range(8)), original_g_kappa=True, scalar='nuclear once',
    oracle='own_DF_ground_state_oracle', state_preparation='EXCLUDED',
    calibration_samples_per_arm=SAMPLES, sample_unit='one complete h-step controlled PR kernel',
    quantum_occurrence_policy='fresh independent trajectory at every coherent occurrence; not executed',
    measured_cost_scope='kernel only; head/X/Y shell costs recorded separately, no cross-segment fusion',
    native_policy=b.NATIVE_POLICY, head_tail_cross_fusion=False,
    law='existing PCG64 binary64 finite-RTE implementation; ideal-law distance unqualified',
    retry=0, resume=0, new_fit=0, optimizer=0, DF_PF_recalculation=0,
    quantum_shots=0, total_resources=0)
ALLOWANCES = dict(operator_component_abs=1e-10, unitarity_component_abs=1e-10,
    signal_complex=2e-10, normalization=1e-12, mass_relative=1e-12, mass_absolute=1e-14,
    rtol=0, global_phase_fit=False, rigorous_enclosure=None,
    qualification='full controlled binary64 numerical pilot per sampled trajectory; not all-law enclosure')
LIMITS = dict(input_load=1, evidence_load=1, fit_load=2, native_model=2,
    law_prepare=2, law_gate=2, ground_load=2, car_context=1, raw_factor=18,
    reference_context=2, sector_reference_gate=2, gaussian_build=18, relative_gaussian=8,
    shell_build=3, shell_assessment=3, shell_metrics=3,
    trajectory_sample=64, kernel_build=64, trajectory_reference=64,
    raw_operator_gate=64, compile=67, compiled_operator_gate=64,
    adjoint_pair_gate=64, XY_gate=64, kernel_metrics=64,
    calibration_summary=2, comparison=1)
MAX_CALLS = sum(LIMITS.values())
SHELLS = ('head', 'X', 'Y')


def profile_dict():
    from .pr_dictionary_validation.fixture import CompilerProfile
    value = asdict(CompilerProfile()); value['basis_gates'] = list(value['basis_gates'])
    return value


def evidence_gate(c, root):
    """Saved JSON/status provenance only; no reanalysis or array decoding."""
    stage_a, bounds = b1.saved_gate(c, root)
    records = {k:io.read(Path(root)/p) for k,p in c['B1_records'].items()}
    result, stop, source = (records[k] for k in ('result', 'STOP', 'SOURCE'))
    if (result['status'] != 'B1_MECHANISM_EVALUATION_COMPLETE' or stop['failed']
            or stop['scientific_units'] != 141 or stop['call_counts'] != b1.LIMITS
            or stop['retry'] or stop['resume'] or not stop['mandatory_stop']
            or source['execution_commit'] != '7da0fdb3ff7c2e5e22c7d8658f6980077993c5d9'):
        raise ValueError('B1_PROVENANCE_OR_STOP')
    for role in b.ARMS:
        arm = result['arms'][role]
        if (arm['classification']['status'] != 'POTENTIALLY_FEASIBLE'
                or arm['saved_DF_bound_Ha'] != bounds[role]
                or [r['m'] for r in arm['rounds']] != list(range(13))
                or any(r['feasibility']['status'] != 'POTENTIALLY_FEASIBLE' for r in arm['rounds'])
                or records[role+'_ground']['status'] != 'PASS'):
            raise ValueError('B1_ARM_OR_GROUND_EVIDENCE')
    return dict(Stage_A=stage_a, B1=records, saved_DF_bounds_Ha=bounds)


def preflight(root, path, input_path):
    root = Path(root).resolve(); c = io.read(path)
    from .c1_h6_native_precision import VERSION as gaussian
    from .c1_h6_compiler_precision import VERSION as phase
    if (c['scope'] != SCOPE or c['status'] != 'UNAPPROVED_PROPOSAL'
            or c['implementation'] != VERSION or c['policy'] != POLICY
            or c['allowances'] != ALLOWANCES or c['compiler'] != profile_dict()
            or c['gaussian'] != gaussian or c['compiler_phase'] != phase
            or c['call_limits'] != LIMITS or c['maximum_logical_units'] != MAX_CALLS
            or c['fresh_output'] != FRESH
            or c['sampling'] != sampling_policy()):
        raise ValueError('B2_SCOPE_OR_POLICY')
    if io.sha(input_path) != c['original_input']['sha256']:
        raise ValueError('ORIGINAL_INPUT')
    if io.sha(root/c['source_manifest']) != c['source_manifest_sha256']:
        raise ValueError('SOURCE_MANIFEST')
    for group in (io.read(root/c['source_manifest'])['files'], c['saved_inputs']):
        for name, want in group.items():
            p = (root/name).resolve()
            if not p.is_relative_to(root) or io.sha(p) != want:
                raise ValueError('SOURCE_OR_INPUT: '+name)
    required = {c['stage_A_input'], *c['final_exports'].values(), *c['stage_A_records'].values(),
        *c['B1_records'].values(), *c['B1_references'].values(), *c['B1_grounds'].values()}
    if (set(c['final_exports']) != set(b.ARMS) or set(c['B1_references']) != set(b.ARMS)
            or set(c['B1_grounds']) != set(b.ARMS) or not required <= c['saved_inputs'].keys()):
        raise ValueError('UNPINNED_B2_INPUT')
    for p in c['final_exports'].values(): b1.export_headers(root/p)
    evidence_gate(c, root)
    for name, want in c['evaluation_packages'].items():
        if importlib.metadata.version(name) != want: raise ValueError('ENVIRONMENT: '+name)
    return c


def sampling_policy():
    return dict(samples_per_arm=SAMPLES, master=MASTER, cohort=COHORT,
        PRNG='numpy.random.PCG64', implementation='existing iter_sample_rte_events',
        seed_key='(master,cohort,B2_SOURCE_manifest_SHA256), arm, final_JU_SHA256, -1, B2_CALIBRATION_ONLY, sample_index, 0',
        independence='separate arm/source/cohort/sample streams; no paired-variance claim',
        adaptive_samples=False, support_enumeration=False,
        ideal_law_distance=None, guaranteed_confidence_interval=None)


def sample_key(arm, sample, source_digest):
    if type(sample) is not int or not 0 <= sample < SAMPLES: raise ValueError('SAMPLE_INDEX')
    master = (MASTER, COHORT, source_digest)
    seed = b.connection.trajectory_seed(master, arm.role, arm.model.source_identity,
        -1, 'B2_CALIBRATION_ONLY', sample, 0)
    return master, seed


def sample_trajectory(arm, tail, distribution, config, sample, source_digest):
    master, seed = sample_key(arm, sample, source_digest)
    events = b.independent_occurrence(arm, tail, distribution, config, master=master,
        round_index=-1, axis='B2_CALIBRATION_ONLY', shot=sample, occurrence=0)
    if config is None or len(events) != POLICY['r'] or any(e.taylor_order != 0 for e in events):
        raise ValueError('REGISTERED_K0_R16_TRAJECTORY')
    from .c1_stage3_qualification import trajectory_key
    return events, dict(arm=arm.role, sample=sample, seed=seed, cohort=COHORT,
        SOURCE_sha256=source_digest, final_JU_sha256=arm.model.source_identity,
        microsteps=len(events), trajectory_key=trajectory_key(events),
        quantum_shots=0, reused_as_coherent_repetitions=False)


def reference_context(arm, generators, car, h):
    """Share immutable CAR atoms and existing head exponentials over 32 samples."""
    from scipy.sparse import csr_matrix
    d = len(generators[0])
    ctx = SimpleNamespace(d=d, bilinears={(p,q):csr_matrix(car[p,q])
        for p in range(arm.model.modes) for q in range(arm.model.modes) if p%2 == q%2})
    atoms = b.atom_operators(ctx, arm)
    half = tuple((i,Fraction(h)/2) for i in range(arm.plan.ld+1)); cache = {}
    left = b.reference._ordered_exponentials(generators, half, cache)
    right = b.reference._ordered_exponentials(generators, tuple(reversed(half)), cache)
    for value in (left, right, *atoms.values()): value.setflags(write=False)
    return dict(atoms=atoms, left=left, right=right, generators=generators)


def trajectory_reference(arm, ctx, h, events):
    """Same CAR/event_unitary product as B1 glue, with head exponentials cached."""
    import numpy as np
    from trotterlib.rte import event_unitary
    middle = np.eye(len(ctx['left']), dtype=complex)
    for event in events: middle = event_unitary(event,ctx['atoms']) @ middle
    return np.exp(-1j*float(h)*arm.plan.extracted_identity) * ctx['right'] @ middle @ ctx['left']


def law_gate(arm, tail, ctx, saved):
    import numpy as np
    if not math.isclose(tail.lambda_r,saved['lambda_R'],rel_tol=ALLOWANCES['mass_relative'],abs_tol=ALLOWANCES['mass_absolute']):
        raise ValueError('B1_LAMBDA_CHANGED')
    if (arm.plan.extracted_identity != saved['identity'] or len(arm.plan.atoms) != saved['atoms']
            or [x.probability for x in tail.components] != [x['probability'] for x in saved['components']]):
        raise ValueError('B1_LAW_OR_IDENTITY_CHANGED')
    d = len(ctx['left']); total = arm.plan.extracted_identity*np.eye(d,dtype=complex)
    for atom in arm.plan.atoms: total += abs(float(atom.coefficient))*ctx['atoms'][atom.atom_id]
    return b.component_check(total, sum(ctx['generators'][arm.plan.ld+1:]),
        ALLOWANCES['operator_component_abs'], label='full CAR tail / signed law + identity')


def raw_gate(raw, system_reference):
    from .c1_h4_native_fair import decode, controlled
    import numpy as np
    R = decode(raw); target = controlled(system_reference)
    checks = dict(reference=b.component_check(R,target,ALLOWANCES['operator_component_abs'],label='raw/full controlled CAR'),
        unitarity=b.component_check(R.conj().T@R,np.eye(len(R)),ALLOWANCES['unitarity_component_abs'],label='raw unitary'))
    return R, target, checks


def compiled_gate(compiled, raw_matrix, target):
    from .c1_h4_native_fair import decode
    import numpy as np
    C = decode(compiled)
    return C, dict(reference=b.component_check(C,target,ALLOWANCES['operator_component_abs'],label='compiled/full controlled CAR'),
        raw=b.component_check(C,raw_matrix,ALLOWANCES['operator_component_abs'],label='compiled/raw'),
        unitarity=b.component_check(C.conj().T@C,np.eye(len(C)),ALLOWANCES['unitarity_component_abs'],label='compiled unitary'))


def adjoint_pair_gate(raw, compiled, R, C, target):
    from .c1_h4_native_fair import decode
    import numpy as np
    circuits = {key:b.native.actual_adjoint(qc) for key,qc in [('raw',raw),('compiled',compiled)]}
    checks = {}
    for key, matrix in [('raw',R),('compiled',C)]:
        value = decode(circuits[key])
        checks[key] = dict(reference=b.component_check(value,target.conj().T,ALLOWANCES['operator_component_abs'],label=key+' actual adjoint/ref'),
            returned=b.component_check(value@matrix,np.eye(len(matrix)),ALLOWANCES['operator_component_abs'],label=key+' adjoint return'))
    return circuits, checks


def shell_build(modes, label):
    head, tail = b.connection.readout_segments(modes, 'X' if label == 'head' else label)
    return head if label == 'head' else tail


def shell_assessment(raw, compiled, label):
    from .c1_h4_native_fair import decode
    from .c1_h6_native import shell_unitary
    import numpy as np
    R, C = decode(raw), decode(compiled)
    H = np.array([[1.,1.],[1.,-1.]])/math.sqrt(2)
    expected = H if label in ('head','X') else H@np.diag([1.,-1j])
    structure = {key:b.component_check(shell_unitary(qc),expected,ALLOWANCES['operator_component_abs'],
        label=key+' ancilla-only shell:'+label) for key,qc in [('raw',raw),('compiled',compiled)]}
    return dict(raw=R, compiled=C), dict(structure=structure,
        compiled_raw=b.component_check(C,R,ALLOWANCES['operator_component_abs'],label='shell compiled/raw:'+label))


def XY_gate(R, C, system_reference, psi, shells):
    import numpy as np
    from .pr_dictionary_validation.gates import signal_gate
    if abs(np.vdot(psi,psi)-1) > ALLOWANCES['normalization']: raise ValueError('GROUND_NORMALIZATION')
    initial = np.r_[psi,np.zeros_like(psi)]; expected = complex(np.vdot(psi,system_reference@psi))
    signals = {}; norms = {}
    for kind, matrix in [('raw',R),('compiled',C)]:
        observed = []; errors = []
        for axis in ('X','Y'):
            state = shells[axis][kind] @ matrix @ shells['head'][kind] @ initial
            errors.append(float(abs(np.vdot(state,state)-1)))
            d = len(psi); observed.append(float(np.sum(abs(state[:d])**2)-np.sum(abs(state[d:])**2)))
        if max(errors) > ALLOWANCES['signal_complex']: raise ValueError('XY_NORM')
        report = signal_gate(*observed,expected,ALLOWANCES['signal_complex'])
        signals[kind] = dict(report, complex_parts=observed); norms[kind] = errors
    return dict(status='PASS',signals=signals,norm_errors=norms,reference_parts=[expected.real,expected.imag],
        state='saved own-DF ground oracle; preparation excluded',attenuation_applied=False,
        interpretation='single unitary trajectory signal, not the B1 mean signal or CA test')


def calibration(samples):
    from .c1_stage3_accuracy import METRICS
    point = b.cost_point(samples,scope='complete_controlled_partial_S2_kernel',independent_trajectories=True)
    n = len(samples); means = point['sample_mean']
    return dict(point, sample_variance=None if n<2 else {
        k:math.fsum((x[k]-means[k])**2 for x in samples)/(n-1) for k in METRICS},
        observed_min={k:min(x[k] for x in samples) for k in METRICS},
        observed_max={k:max(x[k] for x in samples) for k in METRICS},
        uncertainty='descriptive SE only; rare trajectories can be missed; observed extrema are not support bounds',
        finite_bit_law_error=None, CA_achieved=None, shots=None, total_resources=None)


def comparison(results):
    from .c1_stage3_accuracy import METRICS
    means = {a:results[a]['sample_mean'] for a in b.ARMS}
    return dict(status='B2_FINITE_BIT_NATIVE_CALIBRATION_COMPLETE',arms=results,
        proposed_minus_baseline={k:means['proposed'][k]-means['baseline'][k] for k in METRICS},
        proposed_over_baseline={k:means['proposed'][k]/means['baseline'][k] if means['baseline'][k] else None for k in METRICS},
        comparison='fixed-cohort descriptive comparison only; no paired or population superiority inference',
        ideal_law_mean=None,cost_support=None,whole_law_synthesis_bound=None,implemented_law_error=None,
        CA_achieved=None,shots=None,total_resources=None,guaranteed_advantage=None,mandatory_stop=True)


def execute(c, root, output, call, *, samples_per_arm=SAMPLES):
    """One pipeline, including mock/synthetic E2E; production sample count is fixed by run()."""
    import numpy as np
    from qiskit import qpy
    from threadpoolctl import threadpool_limits
    from .c1_h4_strong_baseline import StrongBuilder
    from .c1_h4_native_fair import basis_request, compile_checked
    from .c1_stage3_cost_compat import CompileSerialCalls
    from .pr_dictionary_validation.fixture import CompilerProfile
    from .pr_dictionary_validation.f2_execution import circuit_metrics
    root, output = Path(root), Path(output); h = Fraction(POLICY['h'])
    for leaf in ('circuits','samples','references'): (output/leaf).mkdir()
    def qsave(label,kind,qc):
        with (output/'circuits'/f'{label}_{kind}.qpy').open('xb') as f: qpy.dump(qc,f)
    p = call('input_load','Stage A h/g/constant snapshot',lambda:b1.load(root/c['stage_A_input'],('h','g','constant')))
    evidence = call('evidence_load','Stage A and B1 read only; no PF/DF recomputation',lambda:evidence_gate(c,root))
    b.stage1.finite_arrays(*p.values())
    n = len(p['h']); modes = 2*n; d = 1<<modes
    car = call('car_context','full Fock CAR, no per-gate sector projection',lambda:b.stage1.car_bilinears(modes))
    serial = CompileSerialCalls(call); profile = CompilerProfile(); profile.validate()
    shells = {}; shell_reports = {}
    for label in SHELLS:
        raw = call('shell_build',label,lambda label=label:shell_build(modes,label)); qsave(label,'raw',raw)
        compiled = serial('compile',label,lambda:compile_checked(raw,profile)); qsave(label,'compiled',compiled)
        shells[label], report = call('shell_assessment',label,lambda:shell_assessment(raw,compiled,label))
        report['metrics'] = call('shell_metrics',label,lambda:circuit_metrics(compiled))
        shell_reports[label] = report
    io.save(output/'shells.json',shell_reports)
    prepared = {}; locks = {}
    for role in b.ARMS:
        data = call('fit_load',role,lambda role=role:b1.load(root/c['final_exports'][role],('J','U')))
        arm = call('native_model',role,lambda:b.prepare_arm(**p,**data,role=role,
            source_identity=c['saved_inputs'][c['final_exports'][role]],input_identity=c['original_input']['sha256'],ld=POLICY['ld']))
        tail,config,distribution = call('law_prepare',role,lambda:b.configure(arm,h,POLICY['r'],POLICY['K']))
        if config is None: raise ValueError('REGISTERED_NONZERO_TAIL_REQUIRED')
        G = []
        for i in range(len(arm.J)+1):
            G.append(call('raw_factor',f'{role}:{i}',lambda i=i:b.stage1.raw_base(b.stage1.corrected_kappa(p['h'],p['g']),p['constant'],car)
                if i == 0 else b.stage1.raw_frame(arm.J[i-1],arm.U[i-1],car)[0]))
        def sector_gate():
            saved = b1.load(root/c['B1_references'][role],('generators','full_input_indices'))
            indices = np.asarray(saved['full_input_indices'])
            if (indices.ndim != 1 or len(set(indices.tolist())) != len(indices)
                    or np.any(indices<0) or np.any(indices>=d)):
                raise ValueError('B1_FULL_SECTOR_MAPPING')
            reports = [b.component_check(g[np.ix_(indices,indices)],sg,ALLOWANCES['operator_component_abs'],
                label=f'{role}:full CAR/saved B1 factor{i}') for i,(g,sg) in enumerate(zip(G,saved['generators'],strict=True))]
            return indices, reports
        indices,reports = call('sector_reference_gate',role,sector_gate); io.save(output/(role+'_sector_gate.json'),reports)
        def load_ground():
            psi = b1.load(root/c['B1_grounds'][role],('sector_state',))['sector_state']
            if psi.shape != (len(indices),) or not np.isfinite(psi).all() or abs(np.vdot(psi,psi)-1)>ALLOWANCES['normalization']:
                raise ValueError('SAVED_GROUND_SCHEMA')
            full = np.zeros(d,complex); full[indices] = psi; return full
        psi = call('ground_load',role,load_ground)
        ctx = call('reference_context',role,lambda:reference_context(arm,G,car,h))
        report = call('law_gate',role,lambda:law_gate(arm,tail,ctx,evidence['B1'][role+'_tail']))
        io.save(output/(role+'_law_gate.json'),report)
        builder = StrongBuilder(relative_fusion=True,call=call)
        for block in arm.model.blocks:
            call('gaussian_build',role+':'+block.block_id,lambda block=block:basis_request(builder,arm.model,block))
        for schedule in (tuple(range(arm.plan.ld+1)),tuple(reversed(range(arm.plan.ld+1)))):
            for i,j in zip(schedule,schedule[1:]): builder.transition(arm.model,arm.model.blocks[i],arm.model.blocks[j])
        io.save(output/(role+'_basis.json'),dict(bases=builder.basis_reports,head_relative=builder.fusion_records,
            policy=b.NATIVE_POLICY,head_tail_cross_fusion=False))
        prepared[role] = (arm,tail,config,distribution,ctx,psi,builder); locks[role] = threading.Lock()
    plan = integrated.resource_plan(task_count=2*samples_per_arm,full_dimension=d)
    plan.update(compile='existing process-local P0 mutex',build='per-arm lock protects mutable Gaussian caches',
        reference='immutable per-arm CAR atoms/head exponentials shared across sample tasks',GPU='CPU Qiskit Operator; full512 only, no SVD')
    io.save(output/'resource_plan.json',plan)
    def task(job):
        role,sample = job; label = f'{role}_{sample:02d}'
        arm,tail,config,distribution,ctx,psi,builder = prepared[role]
        events,record = call('trajectory_sample',label,lambda:sample_trajectory(arm,tail,distribution,config,sample,c['source_manifest_sha256']))
        record['events'] = [e.to_dict() for e in events]; io.save(output/'samples'/(label+'_trajectory.json'),record)
        # Charge/check STOP after acquiring the mutable-builder lock, exactly
        # like the existing compile mutex: waiting tasks cannot bypass FAIL.
        with locks[role]:
            raw = call('kernel_build',label,lambda:b.build_kernel(arm,h,events,config,builder=builder))
        qsave(label,'raw',raw)
        ref = call('trajectory_reference',label,lambda:trajectory_reference(arm,ctx,h,events))
        b1.save_arrays(output/'references'/(label+'.npz'),system_reference=ref)
        R,target,raw_report = call('raw_operator_gate',label,lambda:raw_gate(raw,ref))
        io.save(output/'samples'/(label+'_raw_gate.json'),raw_report)
        compiled = serial('compile',label,lambda:compile_checked(raw,profile)); qsave(label,'compiled',compiled)
        C,compiled_report = call('compiled_operator_gate',label,lambda:compiled_gate(compiled,R,target))
        io.save(output/'samples'/(label+'_compiled_gate.json'),compiled_report)
        adjoints,adj_report = call('adjoint_pair_gate',label,lambda:adjoint_pair_gate(raw,compiled,R,C,target))
        for kind,qc in adjoints.items(): qsave(label,kind+'_adjoint',qc)
        io.save(output/'samples'/(label+'_adjoint.json'),adj_report)
        signal = call('XY_gate',label,lambda:XY_gate(R,C,ref,psi,shells))
        io.save(output/'samples'/(label+'_XY.json'),signal)
        metrics = call('kernel_metrics',label,lambda:circuit_metrics(compiled))
        io.save(output/'samples'/(label+'_metrics.json'),metrics)
        return role,sample,metrics
    rows = {a:{} for a in b.ARMS}; failures = []
    with threadpool_limits(limits=plan['BLAS_threads_per_worker']):
        with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
            futures = {pool.submit(task,(a,i)):(a,i) for a in b.ARMS for i in range(samples_per_arm)}
            for future in as_completed(futures):
                try:
                    role,index,metrics = future.result(); rows[role][index] = metrics
                except BaseException as error:
                    call.stop.set(); failures.append(error)
    if failures: raise failures[0]
    results = {}
    for role in b.ARMS:
        results[role] = call('calibration_summary',role,lambda role=role:calibration([rows[role][i] for i in range(samples_per_arm)]))
        io.save(output/(role+'_calibration.json'),results[role])
    result = call('comparison','fixed-cohort kernel calibration only',lambda:comparison(results))
    io.save(output/'result.json',dict(result,shells=shell_reports,B1_evidence=c['B1_records']['result'],
        resource_plan=plan,compile_completed=serial.completed,compile_wait_seconds=serial.wait_seconds))
    return result


def run(root, path, input_path, output):
    root = Path(root).resolve(); c = preflight(root,path,input_path); output = Path(output).resolve()
    if output != root/FRESH or output.exists(): raise ValueError('FRESH_B2_NO_RETRY_RESUME')
    output.mkdir(parents=True); call = integrated.Calls(output,LIMITS,MAX_CALLS); start = time.monotonic()
    try:
        io.save(output/'contract.json',c)
        io.save(output/'SOURCE.json',dict(io.read(root/c['source_manifest']),
            execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()))
        return execute(c,root,output,call)
    except BaseException as error:
        call.stop.set(); io.save(output/'failure.json',dict(error=str(error),first_failure=call.first_failure)); raise
    finally:
        io.save(output/'STOP.json',dict(failed=call.stop.is_set(),scientific_units=sum(call.counts.values()),
            call_counts=dict(call.counts),retry=0,resume=0,mandatory_stop=True))
        io.save(output/'resources.json',dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time(),
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
