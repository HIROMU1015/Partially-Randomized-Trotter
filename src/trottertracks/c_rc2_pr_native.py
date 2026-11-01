"""New saved RC2/PR native calibration, not the old M4/B2 runner.

Passive imports/preflight; real arrays, trajectories and circuits require
separate scientific chat permission. Finite-cohort residuals are not all-law
bounds. No fit, PF/RTE/ground/shot recalculation or total-cost entry.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
import importlib.metadata
from pathlib import Path
import resource
import subprocess
import time

from . import c_rc2_pr_conditional_shots as shots
from . import c_pr_aware_df_stage_b2 as b2
from . import c_pr_aware_df_m2 as m2

stage_b, io, save = shots.stage_b, shots.io, shots.save
b = stage_b.b
BASE='1542bf0bece5a10f20a3c5f28101df170515ec94'
SHOTS_SOURCE='e88fa599af7724f34e11a0f2eab959f92135cf13'
VERSION='saved_rc2_pr_native_fixed32_full512_v1'
SCOPE='C_RC2_PR_NATIVE_FINITE_COHORT_V1'
FAMILY='artifacts/research_c_rc2_pr_native/2026-11-01'
FRESH=FAMILY+'/run01'
ROLES=stage_b.ROLES
SAMPLING=dict(samples_per_arm=32,master=2026110107,cohort='RC2_PR_native_r128_fixed32_v1',
    axis='RC2_PR_NATIVE_ONLY',r=128,K=0,PRNG='numpy.random.PCG64',paired=False,
    adaptive=False,old_cohort_pooled=False,quantum_shots=0)
ALLOWANCES=b2.ALLOWANCES
LIMITS=dict(evidence_load=1,identity_gate=1,input_load=1,car_context=1,
    fit_load=2,native_model=2,law_prepare=2,sector_reference_load=2,ground_load=2,
    raw_factor=18,full_sector_gate=2,reference_context=2,law_gate=2,
    gaussian_build=18,relative_gaussian=8,shell_build=3,compile=67,
    shell_assessment=3,shell_metrics=3,trajectory_sample=64,kernel_build=64,
    trajectory_reference=64,raw_operator_gate=64,compiled_operator_gate=64,
    adjoint_pair_gate=64,XY_gate=64,kernel_metrics=64,calibration_summary=2,comparison=1)
MAX_CALLS=sum(LIMITS.values())


def evidence(root,c):
    return {k:io.read(Path(root)/p) for k,p in c['records'].items()}


def identity_gate(saved,c):
    B={k:saved['B_'+k] for k in ('result','STOP','SOURCE','contract','manifest','rc2_ground','pr_ground')}
    shots.identity_gate(B,c)
    result,stop,source=(saved['shots_'+k] for k in ('result','STOP','SOURCE'))
    if (result['status']!='STAGE_C_SHOTS_COMPLETE' or stop['failed']
            or stop['scientific_units']!=105 or stop['call_counts']!=shots.LIMITS
            or stop['retry'] or stop['resume'] or not stop['mandatory_stop']
            or source['execution_commit']!=SHOTS_SOURCE
            or saved['shots_contract']['identities']!=c['identities']):
        raise ValueError('SHOTS_SOURCE_IDENTITY')
    for role in ROLES:
        for lane in shots.LANES:
            record=result['arms'][role][lane]
            want='EVIDENCE_UNQUALIFIED' if lane=='evidence' else 'CONDITIONAL_SHOTS_AVAILABLE'
            if record['identity']!=c['identities'][role] or record['shots_weight']['status']!=want:
                raise ValueError('SAVED_SHOTS_ROLE_LANE')
    return dict(status='PASS',Stage_B_SOURCE=shots.STAGE_B_SOURCE,shots_SOURCE=SHOTS_SOURCE,
        optimizer_qualification='UNQUALIFIED',DF_budget='SAVED_PASS',shots_recalculated=False)


def preflight(root,path):
    root=Path(root).resolve();c=io.read(path)
    for k,want in dict(scope=SCOPE,implementation=VERSION,base_commit=BASE,
            sampling=SAMPLING,allowances=ALLOWANCES,compiler=b2.profile_dict(),
            policy=stage_b.POLICY,call_limits=LIMITS,maximum_logical_units=MAX_CALLS,
            fresh_output=FRESH,retry=0,resume=0).items():
        if c[k]!=want:raise ValueError('NATIVE_FIXED_SCOPE:'+k)
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    for group in (io.read(root/c['source_manifest'])['files'],c['saved_inputs']):
        for p,h in group.items():
            target=(root/p).resolve()
            if not target.is_relative_to(root) or io.sha(target)!=h:raise ValueError('SOURCE_OR_INPUT:'+p)
    required={c['input'],*c['exports'].values(),*c['sector_references'].values(),
        *c['grounds'].values(),*c['tails'].values(),*c['laws'].values(),*c['records'].values()}
    if not required<=c['saved_inputs'].keys():raise ValueError('UNPINNED_NATIVE_INPUT')
    for role in ROLES:
        identity=c['identities'][role]
        if (identity['role']!=role or identity['export']!=c['exports'][role]
                or identity['sha256']!=c['saved_inputs'][c['exports'][role]]):
            raise ValueError('NATIVE_ARM_IDENTITY')
        stage_b.old.export_headers(root/c['exports'][role])
    identity_gate(evidence(root,c),c)
    for k,want in c['packages'].items():
        if importlib.metadata.version(k)!=want:raise ValueError('ENVIRONMENT:'+k)
    if (root/FRESH).exists():raise ValueError('FRESH_NATIVE_NO_RETRY_RESUME')
    return c


def sample(arm,tail,distribution,config,index,digest):
    from .c1_stage3_qualification import trajectory_key
    if (arm.role not in ROLES or type(index)is not int or not 0<=index<32
            or config is None or config.rte_steps!=128 or config.finite_taylor_order!=0):
        raise ValueError('NATIVE_REGISTERED_SAMPLE')
    master=(SAMPLING['master'],SAMPLING['cohort'],digest)
    seed=b.connection.trajectory_seed(master,arm.role,arm.model.source_identity,-1,SAMPLING['axis'],index,0)
    events=b.independent_occurrence(arm,tail,distribution,config,master=master,
        round_index=-1,axis=SAMPLING['axis'],shot=index,occurrence=0)
    if len(events)!=128 or any(x.taylor_order!=0 for x in events):raise ValueError('NATIVE128_K0_EVENTS')
    return events,dict(arm=arm.role,index=index,seed=seed,cohort=SAMPLING['cohort'],
        SOURCE_sha256=digest,final_JU_sha256=arm.model.source_identity,
        microsteps=128,trajectory_key=trajectory_key(events),events=[x.to_dict() for x in events],
        paired=False,reused_as_coherent_repetitions=False,old_cohort_pooled=False,quantum_shots=0)


def sector_load(path):
    import numpy as np
    values=stage_b.old.load(path,('generators','H'))
    sector=b.stage1.legacy_df_module().PhysicalSector.spin_sector(n_qubits=8,nelec_alpha=2,nelec_beta=2)
    values['full_input_indices']=b.stage1.bit_reverse_indices(8)[sector.basis_indices]
    if (values['generators'].shape!=(9,36,36) or values['H'].shape!=(36,36)
            or not all(np.isfinite(values[k]).all() for k in ('generators','H'))):
        raise ValueError('SAVED_SECTOR_SCHEMA')
    return values


def law_gate(root,c,role,arm,tail,ctx):
    previous=io.read(Path(root)/c['tails'][role]);law=io.read(Path(root)/c['laws'][role])
    current=[dict(id=x.component_id,probability=x.probability,sign=x.coefficient_sign) for x in tail.components]
    if (current!=law['event_probabilities'] or law['status']!='PASS'
            or law['orders']!=[0] or previous['status']!='PASS'):
        raise ValueError('SAVED_COMPONENT_ID_SIGN_ORDER_PROBABILITY')
    adapted=dict(previous,components=[dict(probability=x['probability']) for x in law['event_probabilities']])
    return dict(status='PASS',full_CAR=b2.law_gate(arm,tail,ctx,adapted),
        catalog='same new arm/component IDs/signs/order/probabilities',implemented_law_error=None)


def compare(calibrations,shells):
    from .c1_stage3_accuracy import METRICS
    means={r:calibrations[r]['sample_mean'] for r in ROLES}
    return dict(status='RC2_PR_FINITE_COHORT_NATIVE_COMPLETE',arms=calibrations,shells=shells,
        PR_minus_RC2={k:means['pr'][k]-means['rc2'][k] for k in METRICS},
        measured_scope='one complete h=.1 controlled partial-S2 kernel; external ancilla-only head/X/Y shells separate',
        qualification='32 fixed nonpaired trajectories per arm, full512 numerical correctness per trajectory; descriptive finite-bit mean/SE only',
        whole_law_synthesis_bound=None,implemented_law_error=None,cost_support=None,
        ideal_law_mean=None,actual_CA_achieved=None,shots=None,conditional_interrogation_totals=None,
        guaranteed_advantage=None,mandatory_stop=True)


def execute(root,c,out,call):
    import numpy as np
    from qiskit import qpy
    from threadpoolctl import threadpool_limits
    from .c_pr_aware_df_stage_b2_compat import SharedQiskitCalls
    from .c1_h4_strong_baseline import StrongBuilder
    from .c1_h4_native_fair import basis_request,compile_checked
    from .pr_dictionary_validation.fixture import CompilerProfile
    from .pr_dictionary_validation.f2_execution import circuit_metrics
    out=Path(out)
    for leaf in ('circuits','samples','references'):(out/leaf).mkdir()
    saved=call('evidence_load','B and shots provenance only',lambda:evidence(root,c))
    save(out/'identity.json',call('identity_gate','new RC2/PR only',lambda:identity_gate(saved,c)))
    p=call('input_load','original h/g/scalar, no new integrals',lambda:stage_b.old.load(Path(root)/c['input'],('h','g','constant')))
    if p['h'].shape!=(4,4) or p['g'].shape!=(4,)*4 or p['constant'].shape!=():raise ValueError('PHYSICAL_INPUT_SCHEMA')
    b.stage1.finite_arrays(*p.values())
    car=call('car_context','full256 CAR',lambda:b.stage1.car_bilinears(8))
    shared=SharedQiskitCalls(call);profile=CompilerProfile();profile.validate();h=Fraction(1,10)
    def qsave(label,kind,qc):
        with (out/'circuits'/f'{label}_{kind}.qpy').open('xb') as f:qpy.dump(qc,f)
    shells={};shell_reports={}
    for label in ('head','X','Y'):
        raw=call('shell_build',label,lambda label=label:b2.shell_build(8,label));qsave(label,'raw',raw)
        compiled=shared('compile',label,lambda:compile_checked(raw,profile));qsave(label,'compiled',compiled)
        shells[label],report=call('shell_assessment',label,lambda:b2.shell_assessment(raw,compiled,label))
        report['metrics']=call('shell_metrics',label,lambda:circuit_metrics(compiled))
        report['sharing']='exact same ancilla-only shell/register/profile for both arms; no old shell price transfer'
        shell_reports[label]=report;save(out/(label+'_shell.json'),report)
    prepared={}
    for role in ROLES:
        data=call('fit_load',role,lambda role=role:stage_b.old.load(Path(root)/c['exports'][role],('J','U')))
        arm=call('native_model',role,lambda:stage_b.prepare_candidate(p,data,identity=c['identities'][role]))
        tail,config,dist=call('law_prepare',role,lambda:b.configure(arm,h,128,0))
        sector=call('sector_reference_load',role,lambda:sector_load(Path(root)/c['sector_references'][role]))
        G=[]
        for i in range(9):
            G.append(call('raw_factor',f'{role}:{i}',lambda i=i:b.stage1.raw_base(
                b.stage1.corrected_kappa(p['h'],p['g']),p['constant'],car) if i==0
                else b.stage1.raw_frame(arm.J[i-1],arm.U[i-1],car)[0]))
        ix,reports=call('full_sector_gate',role,lambda:m2.sector_match(G,sector));save(out/(role+'_sector.json'),reports)
        def ground():
            values=stage_b.old.load(Path(root)/c['grounds'][role],('energy','sector_state'))
            v=values['sector_state']
            if (values['energy'].shape!=() or not np.isfinite(values['energy'])
                    or v.shape!=(36,) or not np.isfinite(v).all()
                    or abs(np.vdot(v,v)-1)>ALLOWANCES['normalization']):
                raise ValueError('SAVED_GROUND_SCHEMA')
            psi=np.zeros(256,complex);psi[ix]=v;return psi
        psi=call('ground_load',role,ground)
        ctx=call('reference_context',role,lambda:b2.reference_context(arm,G,car,h))
        save(out/(role+'_law.json'),call('law_gate',role,lambda:law_gate(root,c,role,arm,tail,ctx)))
        builder=StrongBuilder(relative_fusion=True,call=call)
        for block in arm.model.blocks:
            call('gaussian_build',role+':'+block.block_id,lambda block=block:basis_request(builder,arm.model,block))
        for schedule in (tuple(range(3)),tuple(reversed(range(3)))):
            for i,j in zip(schedule,schedule[1:]):builder.transition(arm.model,arm.model.blocks[i],arm.model.blocks[j])
        save(out/(role+'_basis.json'),dict(bases=builder.basis_reports,head_relative=builder.fusion_records,
            cross_head_tail_fusion=False,policy=b.NATIVE_POLICY))
        prepared[role]=(arm,tail,config,dist,ctx,psi,builder)
    def task(role,index):
        arm,tail,config,dist,ctx,psi,builder=prepared[role];label=f'{role}_{index:02d}'
        events,record=call('trajectory_sample',label,lambda:sample(arm,tail,dist,config,index,c['source_manifest_sha256']))
        save(out/'samples'/(label+'_trajectory.json'),record)
        raw=shared('kernel_build',label,lambda:b.build_kernel(arm,h,events,config,builder=builder));qsave(label,'raw',raw)
        ref=call('trajectory_reference',label,lambda:b2.trajectory_reference(arm,ctx,h,events))
        stage_b.old.save_arrays(out/'references'/(label+'.npz'),system_reference=ref)
        R,target,report=call('raw_operator_gate',label,lambda:b2.raw_gate(raw,ref));save(out/'samples'/(label+'_raw.json'),report)
        compiled=shared('compile',label,lambda:compile_checked(raw,profile));qsave(label,'compiled',compiled)
        C,report=call('compiled_operator_gate',label,lambda:b2.compiled_gate(compiled,R,target));save(out/'samples'/(label+'_compiled.json'),report)
        adjoints,report=call('adjoint_pair_gate',label,lambda:b2.adjoint_pair_gate(raw,compiled,R,C,target))
        for kind,qc in adjoints.items():qsave(label,kind+'_adjoint',qc)
        save(out/'samples'/(label+'_adjoint.json'),report)
        save(out/'samples'/(label+'_XY.json'),call('XY_gate',label,lambda:b2.XY_gate(R,C,ref,psi,shells)))
        metrics=call('kernel_metrics',label,lambda:circuit_metrics(compiled));save(out/'samples'/(label+'_metrics.json'),metrics)
        return role,index,metrics
    plan=stage_b.old.integrated.resource_plan(task_count=64,full_dimension=256)
    plan.update(Qiskit='process-local build/compile/actual-adjoint mutex',reference='immutable per-arm fullCAR caches; full512 all entries, no SVD')
    save(out/'resource_plan.json',plan);rows={r:{} for r in ROLES}
    with threadpool_limits(limits=plan['BLAS_threads_per_worker']),shared.adjoint_builds():
        with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
            futures=[pool.submit(task,r,i) for r in ROLES for i in range(32)]
            try:
                for future in as_completed(futures):
                    role,index,value=future.result();rows[role][index]=value
            except BaseException:
                call.stop.set()
                for future in futures:future.cancel()
                raise
    calibrations={}
    for role in ROLES:
        calibrations[role]=call('calibration_summary',role,lambda role=role:b2.calibration([rows[role][i] for i in range(32)]))
        save(out/(role+'_calibration.json'),calibrations[role])
    result=call('comparison','two new finite cohorts; no interrogation total',lambda:compare(calibrations,shell_reports))
    save(out/'result.json',dict(result,QPY_count=262,compile_count=67,
        Qiskit_lock_wait_seconds=shared.wait_seconds,completed_build_compile=shared.completed))
    return result


def run(root,path,output):
    root=Path(root).resolve();c=preflight(root,path);output=Path(output).resolve()
    if output!=root/FRESH or output.exists():raise ValueError('FRESH_NATIVE_NO_RETRY_RESUME')
    output.mkdir(parents=True);calls=stage_b.old.integrated.Calls(output,LIMITS,MAX_CALLS);start=time.monotonic()
    try:
        save(output/'contract.json',c);save(output/'SOURCE.json',dict(io.read(root/c['source_manifest']),
            execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()))
        return execute(root,c,output,calls)
    except BaseException as error:
        calls.stop.set();save(output/'failure.json',dict(status='TECHNICAL_FAILURE',error=str(error),first_failure=calls.first_failure));raise
    finally:
        save(output/'STOP.json',dict(failed=calls.stop.is_set(),scientific_units=sum(calls.counts.values()),call_counts=dict(calls.counts),retry=0,resume=0,mandatory_stop=True))
        save(output/'resources.json',dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time(),peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
