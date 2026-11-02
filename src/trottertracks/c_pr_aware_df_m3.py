"""One saved Proposed arm: gated macrostep pilot using existing B1/B2/C1.

Imports and byte preflight do no molecular evaluation. Old runners/contracts
are never restarted. No new sampler, PF engine, compiler or shot solver.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
import importlib.metadata
import json
import math
from pathlib import Path
import resource
import subprocess
import time

from . import c1_stage1_execution as io
from . import c1_stage3_accuracy as accuracy
from . import c1_stage3_integrated as integrated

VERSION='Proposed_PR_macrostep_s1_r64_JSON_catalog_v2'
SCOPE='C_PR_AWARE_DF_M3_S1_R64_REPAIR_V2'
BASE='def3748fc33f923f1a684a5d3ee0608fd7818a44'
FAMILY='artifacts/research_c_pr_aware_df_M3/2026-10-28'
FRESH=FAMILY+'/run02'
TARGET=dict(energy_ha=.00159,df_reserve_fraction=1/8,base_time=.1,
    beta_total=math.pi/6,beta_stat=math.pi/12,alpha_axis=.01/26,alpha_total=.01)
POLICY=dict(atoms=4,distance_A=1.,basis='STO-3G',rank=8,Nalpha=2,Nbeta=2,
    sector_dimension=36,system_modes=8,controlled_dimension=512,arm='proposed',
    frame_order=list(range(8)),ld=2,PF='partial ordered symmetric S2',delta='1/10',
    s=1,h='1/10',r=64,K=0,rounds=list(range(13)),original_g_kappa=True,
    scalar='nuclear once',oracle='own_DF_ground_state_oracle',state_preparation='EXCLUDED',
    native='B2 precision bases;relative head;identical-basis tail;no cross-head/tail fusion')
ALLOWANCES=dict(component_abs=1e-10,unitarity_abs=1e-10,signal_abs=2e-10,
    normalization=1e-12,ground_residual=1e-10,mass_relative=1e-12,mass_absolute=1e-14,
    algebra_absolute=1e-12,rtol=0,global_phase_fit=False,strict_enclosure=None)
SAMPLING=dict(samples=32,master=2026102803,cohort='M3_Proposed_s1_r64_fixed32_v1',
    axis='M3_CALIBRATION_ONLY',PRNG='numpy.random.PCG64',events_per_sample=64,
    source_key='(master,cohort,M3_SOURCE_digest),arm,JU_hash,-1,axis,sample,0',
    old_cohort_pooled=False,paired=False,adaptive=False,ideal_IID_proof=None)
A_LIMITS=dict(evidence_load=1,input_load=1,fit_load=1,native_model=1,
    sector_reference_load=1,sector_context=1,general_factor_gate=9,ground_load=1,
    ground_gate=1,law_prepare=1,tail_gate=1,algebra_gate=1,kernel_reference=1,
    exact_evolution=13,round_reference=13,round_classification=13,A_summary=1)
B_LIMITS=dict(car_context=1,raw_factor=9,full_sector_gate=1,reference_context=1,
    full_law_gate=1,gaussian_build=9,relative_gaussian=4,shell_load=6,shell_assessment=3,
    trajectory_sample=32,kernel_build=32,trajectory_reference=32,raw_operator_gate=32,
    compile=32,compiled_operator_gate=32,adjoint_pair_gate=32,XY_gate=32,kernel_metrics=32,
    calibration_summary=1,delta_cost_comparison=1)
C_LIMITS=dict(target_validation=1,cost_calibration=1,serial_cost=26,
    signal_qualification=3,round_accuracy=39,energy_assessment=3,total_aggregation=3,
    calibration_uncertainty=3,resource_comparison=4,C_summary=1)
LIMITS=dict(**A_LIMITS,**B_LIMITS,**C_LIMITS,result_summary=1)
MAX_CALLS=sum(LIMITS.values())


def preflight(root,path):
    """Hashes/schema/packages only; no arrays, circuits or saved-value analysis."""
    from .c_pr_aware_df_stage_c import SCENARIOS
    root=Path(root).resolve();c=io.read(path)
    for key,want in dict(scope=SCOPE,implementation=VERSION,base_commit=BASE,
            policy=POLICY,target=TARGET,allowances=ALLOWANCES,sampling=SAMPLING,
            stage_call_limits=dict(A=A_LIMITS,B=B_LIMITS,C=C_LIMITS),call_limits=LIMITS,
            maximum_logical_units=MAX_CALLS,fresh_output=FRESH,retry=0,resume=0,
            scenarios=list(SCENARIOS)).items():
        if c[key]!=want:raise ValueError('M3_FIXED_CONDITIONS:'+key)
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    for files in (io.read(root/c['source_manifest'])['files'],c['saved_inputs']):
        for name,want in files.items():
            p=(root/name).resolve()
            if not p.is_relative_to(root) or io.sha(p)!=want:raise ValueError('SOURCE_OR_INPUT:'+name)
    for package,version in c['packages'].items():
        if importlib.metadata.version(package)!=version:raise ValueError('ENVIRONMENT:'+package)
    if (root/FRESH).exists():raise ValueError('FRESH_M3_NO_RETRY_RESUME')
    return c


def evidence(root,c):
    from . import c_pr_aware_df_m2 as m2
    saved=m2.evidence(root,c)
    if (saved['M2']['status']!='M2_SAME_JU_DETERMINISTIC_COMPLETE'
            or saved['M2_STOP']['failed'] or saved['M2_STOP']['scientific_units']!=253
            or saved['M2_SOURCE']['execution_commit']!='a6ffc23d94f6488a4769596f089a91c895d49214'
            or saved['M2']['candidates']['proposed']['export_sha256']!=c['saved_inputs'][c['exports']['proposed']]):
        raise ValueError('SAVED_M2_SOURCE_IDENTITY')
    if (c['saved_inputs'][c['input']]!='1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db'
            or c['saved_inputs'][c['exports']['proposed']]!='f47a9fbf86918458ea0db6549468f2041b3b87bba6649ff70234467a442cd227'):
        raise ValueError('M3_FIXED_INPUT_EXPORT')
    return saved


def algebra_gate(arm,tail,config,old_rows):
    """Exact rational identity of the binary64 lambda; also report runtime FP."""
    from . import c_pr_aware_df_stage_b as b
    legacy=b.reference._legacy('rte')
    old_tail,old_config,_=b.configure(arm,Fraction(1,40),16,0)
    new_tau=Fraction(tail.lambda_r)*Fraction(1,10)/64
    old_tau=Fraction(old_tail.lambda_r)*Fraction(1,40)/16
    if new_tau!=old_tau or tail.lambda_r!=old_tail.lambda_r:raise ValueError('TAU_LAMBDA_IDENTITY')
    rows=[]
    for m in range(13):
        n=1<<m
        a=legacy.finite_rte_attenuation(config,tail_evolutions=n)
        old=legacy.finite_rte_attenuation(old_config,tail_evolutions=4*n)
        observed=old_rows[m]['reference']['attenuation']
        if (64*n!=16*4*n or abs(a-old)>ALLOWANCES['algebra_absolute']
                or abs(a-observed)>ALLOWANCES['algebra_absolute']):raise ValueError('ATTENUATION_IDENTITY')
        rows.append(dict(m=m,new_A=a,old_A=old,saved_A=observed,
            difference=a-old,new_random_events=64*n,old_random_events=16*4*n))
    return dict(status='PASS',lambda_R=tail.lambda_r,tau_rational=str(new_tau),
        mathematical_tau_equal=True,new_tau=config.dimensionless_step_time,
        old_tau=old_config.dimensionless_step_time,
        new_tau_hex=float(config.dimensionless_step_time).hex(),old_tau_hex=float(old_config.dimensionless_step_time).hex(),
        binary64_tau_difference=config.dimensionless_step_time-old_config.dimensionless_step_time,
        normalization='K0: A=(1+tau^2)^(-r*n/2)',rounds=rows,
        PF_RTE_errors_assumed_equal=False)


def catalog_gate(atoms,saved):
    """Normalize only JSON sequence containers; preserve every value/order."""
    from .pr_dictionary_validation.prototype import atom_record
    def normalized(row):
        row=dict(row)
        for key in ('origins','word'):
            value=row[key]
            if not isinstance(value,(tuple,list)):
                raise ValueError('ATOM_CATALOG_SEQUENCE_SCHEMA')
            row[key]=tuple(value)
        return row
    current=[normalized(atom_record(x)) for x in atoms]
    if current!=[normalized(x) for x in saved]:raise ValueError('ATOM_ID_SIGN_BASIS_ORDER')
    return dict(status='PASS',atoms=len(current),JSON_sequence_normalization_only=True,
        coefficient_sign_basis_frame_and_order_preserved=True)


def stage_a(root,c,out,call):
    import numpy as np
    from scipy.linalg import expm
    from threadpoolctl import threadpool_limits
    from . import c_pr_aware_df_stage_b as b
    from . import c_pr_aware_df_stage_b_runner as loader
    from . import c_pr_aware_df_m2 as m2
    from . import c1_molecule_extension as molecule
    from . import c1_stage3_feasibility as feasibility
    saved=call('evidence_load','A/B1/B2/C1/M1/M2 saved metadata and identity',lambda:evidence(root,c))
    p=call('input_load','saved h/g/scalar',lambda:loader.load(root/c['input'],('h','g','constant')))
    data=call('fit_load','Proposed J/U only',lambda:loader.load(root/c['exports']['proposed'],('J','U')))
    m2.validate_arrays(p,data,'proposed')
    arm=call('native_model','Proposed original kappa/scalar LD2',lambda:b.prepare_arm(**p,**data,
        role='proposed',source_identity=c['saved_inputs'][c['exports']['proposed']],
        input_identity=c['original_input_sha256'],ld=2))
    sector=call('sector_reference_load','same-JU saved nine sector factors/H/mapping',
        lambda:loader.load(root/c['references']['proposed'],('generators','H','full_input_indices')))
    ctx=call('sector_context','sector36 CAR bilinears',lambda:molecule.SectorCAR(4))
    if (sector['generators'].shape!=(9,36,36) or sector['H'].shape!=(36,36)
            or not np.array_equal(np.sort(sector['full_input_indices']),b.stage1.sector_indices(4,2,2))):
        raise ValueError('SECTOR_REFERENCE_SCHEMA')
    checks=[]
    for i in range(9):
        def check(i=i):
            g=sector['generators'][i]
            b.component_check(g,g.conj().T,1e-10,label='saved factor Hermitian')
            return b.component_check(b.candidate_factor(ctx,arm,i),g,1e-10,label='same JU general-K/saved CAR')
        checks.append(call('general_factor_gate',str(i),check))
    b.component_check(sum(sector['generators']),sector['H'],1e-10,label='saved H sum')
    io.save(out/'factor_identity.json',checks)
    ground=call('ground_load','saved own-DF state, no eigensolve',
        lambda:loader.load(root/c['grounds']['proposed'],('energy','sector_state')))
    def ground_gate():
        psi=ground['sector_state'];energy=float(ground['energy']);old=saved['proposed_ground']
        if psi.shape!=(36,) or not np.isfinite(psi).all():raise ValueError('SAVED_GROUND_SCHEMA')
        residual=float(np.linalg.norm(sector['H']@psi-energy*psi))
        norm=float(abs(np.vdot(psi,psi)-1))
        if (residual>1e-10 or norm>1e-12 or abs(energy-old['energy_Ha'])>1e-12
                or old['alias_margin']<=0):raise ValueError('GROUND_ALIAS')
        return dict(status='PASS',residual=residual,normalization_error=norm,energy_Ha=energy,
            alias_margin=old['alias_margin'],saved_lowest_eigenpair=True,new_eigensolve=False)
    io.save(out/'ground_gate.json',call('ground_gate','saved own-DF/physical alias evidence',ground_gate))
    tail,config,distribution=call('law_prepare','h1/10 r64 K0',lambda:b.configure(arm,Fraction(1,10),64,0))
    def law_gate():
        report=b.tail_gate(ctx,arm,sector['generators'],tail);old=saved['proposed_tail']
        if (len(arm.plan.atoms)!=216 or report['identity']!=old['identity']
                or [x.probability for x in tail.components]!=[x['probability'] for x in old['components']]
                or not math.isclose(tail.lambda_r,old['lambda_R'],rel_tol=1e-12,abs_tol=1e-14)):
            raise ValueError('SAVED_216_ATOM_LAW')
        return dict(report,catalog_identity=catalog_gate(arm.plan.atoms,old['atoms_catalog']))
    io.save(out/'tail_gate.json',call('tail_gate','216 signed atoms/identity/mass/probabilities',law_gate))
    old_rows=saved['B1']['arms']['proposed']['rounds']
    algebra=call('algebra_gate','exact sr64 relation and all13 runtime attenuations',lambda:algebra_gate(arm,tail,config,old_rows))
    io.save(out/'algebra.json',algebra)
    kernel=call('kernel_reference','sector36 PF and corrected finite-RTE kernels',
        lambda:b.kernels(arm,sector['generators'],Fraction(1,10),config))
    loader.save_arrays(out/'sector_reference.npz',**sector,pf=kernel['pf'],corrected=kernel['corrected'])
    target=accuracy.AccuracyTarget(**TARGET);plan=integrated.resource_plan(task_count=13,full_dimension=36)
    io.save(out/'A_resource_plan.json',plan)
    def task(m):
        elapsed=Fraction(1,10)*(1<<m)
        exact=call('exact_evolution',f'm{m}',lambda:expm(-1j*float(elapsed)*sector['H']))
        row=call('round_reference',f'm{m}',lambda:b.reference.reference_round(kernel,(sector['H'],elapsed),
            repetitions=1<<m,state=ground['sector_state'],exact_operator=exact))
        signal=row.pop('raw_signal',None);row['raw_signal_parts']=None if signal is None else [signal.real,signal.imag]
        row['evidence_kind']='binary64_sector_operator_numerical_pilot'
        decision=call('round_classification',f'm{m}',lambda:feasibility.classify_round(row,target))
        value=dict(m=m,reference=row,feasibility=decision,synthesis_error=None,implemented_law_error=None,
            saved_s4_r16_reference=old_rows[m]['reference'],CA_achieved=None)
        io.save(out/f'm{m:02d}.json',value);return m,value
    rows={};failures=[]
    with threadpool_limits(limits=plan['BLAS_threads_per_worker']):
        with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
            for f in as_completed([pool.submit(task,m) for m in range(13)]):
                try:m,value=f.result();rows[m]=value
                except BaseException as error:call.stop.set();failures.append(error)
    if failures:raise failures[0]
    rows=[rows[m] for m in range(13)]
    df=saved['B1']['arms']['proposed']['saved_DF_bound_Ha']
    summary=call('A_summary','all13 registered B1 phase-room and saved DF reserve',
        lambda:feasibility.summarize([x['feasibility'] for x in rows],df,target))
    result=dict(status=summary['status'],reason=summary['reason'],rounds=rows,saved_DF_bound_Ha=df,
        algebra=algebra,synthesis_bound=None,implemented_law_bound=None,CA_achieved=None)
    io.save(out/'result.json',result)
    return dict(result=result,saved=saved,p=p,arm=arm,tail=tail,config=config,distribution=distribution,
        sector=sector,ground=ground)


def sample_trajectory(arm,tail,distribution,config,sample,source_digest):
    from . import c_pr_aware_df_stage_b as b
    from .c1_stage3_qualification import trajectory_key
    if type(sample)is not int or not 0<=sample<32:raise ValueError('M3_SAMPLE_INDEX')
    master=(SAMPLING['master'],SAMPLING['cohort'],source_digest)
    seed=b.connection.trajectory_seed(master,arm.role,arm.model.source_identity,-1,SAMPLING['axis'],sample,0)
    events=b.independent_occurrence(arm,tail,distribution,config,master=master,
        round_index=-1,axis=SAMPLING['axis'],shot=sample,occurrence=0)
    if (config.rte_steps!=64 or len(events)!=64 or any(x.taylor_order!=0 for x in events)):
        raise ValueError('M3_REGISTERED_K0_64_EVENTS')
    return events,dict(arm=arm.role,sample=sample,seed=seed,cohort=SAMPLING['cohort'],
        SOURCE_sha256=source_digest,final_JU_sha256=arm.model.source_identity,microsteps=64,
        trajectory_key=trajectory_key(events),quantum_shots=0,reused_as_coherent_repetitions=False,
        old_cohort_pooled=False,paired=False)


def delta_cost_comparison(new,old):
    return dict(status='INDEPENDENT_COHORT_DESCRIPTIVE_ONLY',old_kernels_per_delta=4,new_kernels_per_delta=1,
        old_delta_mean={k:4*old['sample_mean'][k] for k in accuracy.METRICS},
        new_delta_mean=new['sample_mean'],new_over_old_delta={k:new['sample_mean'][k]/(4*old['sample_mean'][k]) for k in accuracy.METRICS},
        difference_SE={k:math.hypot(new['sample_mean_standard_error'][k],4*old['sample_mean_standard_error'][k]) for k in accuracy.METRICS},
        paired=False,ideal_law_mean=None,cost_support=None,guaranteed_advantage=None)


def stage_b(root,c,out,call,a):
    import numpy as np
    from qiskit import qpy
    from threadpoolctl import threadpool_limits
    from . import c_pr_aware_df_stage_b as b
    from . import c_pr_aware_df_stage_b_runner as loader
    from . import c_pr_aware_df_stage_b2 as b2
    from .c_pr_aware_df_stage_b2_compat import SharedQiskitCalls
    from . import c_pr_aware_df_m2 as m2
    from .c1_h4_strong_baseline import StrongBuilder
    from .c1_h4_native_fair import basis_request,compile_checked
    from .pr_dictionary_validation.fixture import CompilerProfile
    from .pr_dictionary_validation.f2_execution import circuit_metrics
    for leaf in ('circuits','samples','references'):(out/leaf).mkdir()
    arm=a['arm'];h=Fraction(1,10);shared=SharedQiskitCalls(call)
    car=call('car_context','full256 CAR',lambda:b.stage1.car_bilinears(8));G=[]
    for i in range(9):
        G.append(call('raw_factor',str(i),lambda i=i:b.stage1.raw_base(
            b.stage1.corrected_kappa(a['p']['h'],a['p']['g']),a['p']['constant'],car)
            if i==0 else b.stage1.raw_frame(arm.J[i-1],arm.U[i-1],car)[0]))
    ix,reports=call('full_sector_gate','all nine fullCAR/saved sector factors',lambda:m2.sector_match(G,a['sector']))
    io.save(out/'full_sector_gate.json',reports)
    psi=np.zeros(256,complex);psi[ix]=a['ground']['sector_state']
    ctx=call('reference_context','full256 immutable atoms/head exponentials',lambda:b2.reference_context(arm,G,car,h))
    io.save(out/'law_gate.json',call('full_law_gate','fullCAR signed tail and identity',
        lambda:b2.law_gate(arm,a['tail'],ctx,a['saved']['proposed_tail'])))
    builder=StrongBuilder(relative_fusion=True,call=call)
    for block in arm.model.blocks:
        call('gaussian_build',block.block_id,lambda block=block:basis_request(builder,arm.model,block))
    for order in (tuple(range(3)),tuple(reversed(range(3)))):
        for i,j in zip(order,order[1:]):builder.transition(arm.model,arm.model.blocks[i],arm.model.blocks[j])
    io.save(out/'basis.json',dict(bases=builder.basis_reports,head_relative=builder.fusion_records,
        cross_head_tail_fusion=False,policy=b.NATIVE_POLICY))
    shells={}
    for label in ('head','X','Y'):
        circuits={}
        for kind in ('raw','compiled'):
            def load_qpy():
                with (root/c['shells'][label][kind]).open('rb') as f:qs=qpy.load(f)
                if len(qs)!=1 or qs[0].num_qubits!=9:raise ValueError('SHELL_SCHEMA')
                return qs[0]
            circuits[kind]=call('shell_load',label+':'+kind,load_qpy)
        shells[label],report=call('shell_assessment',label,lambda label=label:
            b2.shell_assessment(circuits['raw'],circuits['compiled'],label))
        report['metrics']=a['saved']['shell_reports'][label]['metrics'];io.save(out/(label+'_shell.json'),report)
    profile=CompilerProfile();profile.validate()
    def qsave(label,kind,qc):
        with (out/'circuits'/f'{label}_{kind}.qpy').open('xb') as f:qpy.dump(qc,f)
    def task(sample):
        label=f'proposed_{sample:02d}'
        events,record=call('trajectory_sample',label,lambda:sample_trajectory(arm,a['tail'],a['distribution'],a['config'],sample,c['source_manifest_sha256']))
        record['events']=[x.to_dict() for x in events];io.save(out/'samples'/(label+'_trajectory.json'),record)
        raw=shared('kernel_build',label,lambda:b.build_kernel(arm,h,events,a['config'],builder=builder));qsave(label,'raw',raw)
        ref=call('trajectory_reference',label,lambda:b2.trajectory_reference(arm,ctx,h,events))
        loader.save_arrays(out/'references'/(label+'.npz'),system_reference=ref)
        R,target,report=call('raw_operator_gate',label,lambda:b2.raw_gate(raw,ref));io.save(out/'samples'/(label+'_raw_gate.json'),report)
        compiled=shared('compile',label,lambda:compile_checked(raw,profile));qsave(label,'compiled',compiled)
        C,report=call('compiled_operator_gate',label,lambda:b2.compiled_gate(compiled,R,target));io.save(out/'samples'/(label+'_compiled_gate.json'),report)
        adjoints,report=call('adjoint_pair_gate',label,lambda:b2.adjoint_pair_gate(raw,compiled,R,C,target))
        for kind,qc in adjoints.items():qsave(label,kind+'_adjoint',qc)
        io.save(out/'samples'/(label+'_adjoint.json'),report)
        io.save(out/'samples'/(label+'_XY.json'),call('XY_gate',label,lambda:b2.XY_gate(R,C,ref,psi,shells)))
        metrics=call('kernel_metrics',label,lambda:circuit_metrics(compiled));io.save(out/'samples'/(label+'_metrics.json'),metrics)
        return sample,metrics
    plan=integrated.resource_plan(task_count=32,full_dimension=256)
    plan.update(Qiskit='one process-local build/compile/actual-adjoint-build mutex',
        matrices='immutable CAR/head cache; full512 all entries; no full512 SVD',
        artificial_CPU_cap=None,artificial_RAM_cap=None,artificial_worker_cap=None,wall_cap=None)
    io.save(out/'resource_plan.json',plan);rows={};failures=[]
    with threadpool_limits(limits=plan['BLAS_threads_per_worker']),shared.adjoint_builds():
        with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
            for f in as_completed([pool.submit(task,i) for i in range(32)]):
                try:
                    i,row=f.result();rows[i]=row
                    if len(rows)%8==0:
                        observation=dict(completed_trajectories=len(rows),monotonic=time.monotonic(),
                            process_CPU_seconds=time.process_time(),peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                            memory_pressure=Path('/proc/pressure/memory').read_text() if Path('/proc/pressure/memory').exists() else None)
                        with (out/'resource_observations.jsonl').open('a') as f:f.write(json.dumps(observation)+'\n')
                except BaseException as error:call.stop.set();failures.append(error)
    if failures:raise failures[0]
    metrics=[rows[i] for i in range(32)]
    calibration=call('calibration_summary','fixed32 nonpaired M3 cohort only',lambda:b2.calibration(metrics))
    comparison=call('delta_cost_comparison','1 new kernel vs4 old complete kernels perdelta',
        lambda:delta_cost_comparison(calibration,a['saved']['B2']['arms']['proposed']))
    result=dict(status='M3_NATIVE_32_COMPLETE',calibration=calibration,delta_comparison=comparison,
        samples=32,events_per_sample=64,compile_count=32,QPY_count=128,
        Qiskit_lock_wait_seconds=shared.wait_seconds,completed_build_compile=shared.completed,
        global_synthesis_bound=None,implemented_law_error=None,cost_support=None)
    io.save(out/'result.json',result)
    return dict(result=result,metrics=metrics,calibration=calibration)


def serial_cost(packet,kernel,m,axis):
    from .c1_stage3_connection import compose_serial_cost
    shell={k:math.fsum(packet['shells'][x]['metrics'][k] for x in ('head',axis)) for k in accuracy.METRICS}
    record=dict(axis=axis,mean=shell,lower=shell,upper=shell,policy='separate_native_readout',simultaneous_interval_qualified=True)
    return compose_serial_cost(kernel,record,m=m,substeps=1,target_id=packet['target_id'])


def calibration_uncertainty(packet,plans,totals):
    if totals.get('totals') is None:return dict(status='UNRESOLVED_ACCURACY',weighted_SE=None)
    W=sum(2*p['shots']*(1<<p['m']) for p in plans)
    return dict(status='DESCRIPTIVE_SHARED_CALIBRATION_ONLY',kernel_reuse_weight=W,
        weighted_SE={k:W*packet['saved_calibration']['sample_mean_standard_error'][k] for k in accuracy.METRICS},
        samples=32,independent_round_calibrations=1,paired_with_old=False,
        guaranteed_CI=None,ideal_law_cost_transfer_qualified=False)


def stage_c(root,c,out,call,a,b):
    from . import c_pr_aware_df_stage_c as old
    from . import c_pr_aware_df_m2 as m2
    target=call('target_validation','same C1 target and13round horizon',lambda:accuracy.AccuracyTarget(**TARGET))
    if target.horizon()['maximum_round_index_M']!=12:raise ValueError('REGISTERED_HORIZON')
    packet=dict(arm='proposed',target_id='proposed:'+c['saved_inputs'][c['exports']['proposed']],
        rounds=a['result']['rounds'],df_bound_Ha=a['result']['saved_DF_bound_Ha'],
        kernel_samples=b['metrics'],saved_calibration=b['calibration'],shells=a['saved']['shell_reports'],error_evidence=None)
    kernel=call('cost_calibration','same32 M3 samples, unknown all-law support',lambda:old.cost_calibration(packet))
    costs=[call('serial_cost',f'm{m}:{axis}',lambda m=m,axis=axis:serial_cost(packet,kernel,m,axis))
        for m in range(13) for axis in ('X','Y')]
    io.save(out/'serial_costs26.json',costs);results={}
    for lane in old.LANES:
        q=call('signal_qualification',lane,lambda lane=lane:old.qualification(packet,lane));plans=[]
        for row in packet['rounds']:
            p=call('round_accuracy',lane+':m'+str(row['m']),lambda row=row:old.assess_round(target,packet,q,row))
            plans.append(p);io.save(out/(lane+f'_m{row["m"]:02d}.json'),p)
        energy=call('energy_assessment',lane,lambda:old.energy(target,packet,plans))
        totals=call('total_aggregation',lane,lambda:accuracy.aggregate_totals(energy,plans,costs,cost_scope=old.COST_SCOPE))
        uncertainty=call('calibration_uncertainty',lane,lambda:calibration_uncertainty(packet,plans,totals))
        value=dict(arm='proposed',lane=lane,target_id=packet['target_id'],qualification=q,rounds=plans,
            conditional_energy_assessment=energy,resources=totals,calibration_uncertainty=uncertainty,
            actual_CA_achieved=None,rigorous_shots=None,guaranteed_total_resources=None)
        results[lane]=value;io.save(out/(lane+'.json'),value)
    comparisons={}
    for lane in old.LANES[1:]:
        for label,right in [('old_PR_s4_r16',a['saved']['C1']['assumed_scenarios'][lane]['proposed']['resources']),
                ('deterministic_s1',a['saved']['M2']['candidates']['proposed']['cost'])]:
            key=lane+':M3/'+label
            comparisons[key]=call('resource_comparison',key,lambda right=right,key=key:m2.numerical_compare(
                results[lane]['resources'],right,key,qualification='independent finite calibrations;scenario assumptions vs numerical deterministic pilot;not certified CA or mean advantage'))
    result=call('C_summary','evidence null and exactly2 old assumed scenarios',lambda:dict(
        status='M3_CONDITIONAL_RESOURCES_COMPLETE',lanes=results,comparisons=comparisons,
        actual_CA_achieved=None,all_law_synthesis=None,implemented_law_error=None,cost_support=None))
    io.save(out/'result.json',result);return result


def execute(root,c,out,call):
    root,out=Path(root),Path(out)
    for stage in ('A','B','C'):(out/stage).mkdir()
    tick=time.monotonic();a=stage_a(root,c,out/'A',call)
    timing=dict(A_seconds=time.monotonic()-tick)
    state=a['result']['status']
    if state!='POTENTIALLY_FEASIBLE':
        result=dict(status='M3_STOP_AFTER_A_'+state,stages=dict(A=state,B='NOT_RUN_A_GATE',C='NOT_RUN_A_GATE'),A=a['result'])
    else:
        tick=time.monotonic();b=stage_b(root,c,out/'B',call,a);timing['B_seconds']=time.monotonic()-tick
        tick=time.monotonic();cc=stage_c(root,c,out/'C',call,a,b);timing['C_seconds']=time.monotonic()-tick
        result=dict(status='M3_A_B_C_COMPLETE',stages=dict(A=state,B=b['result']['status'],C=cc['status']),
            A=a['result'],B=b['result'],C=cc)
    value=call('result_summary','fixedM3 only;mandatory STOP',lambda:dict(result,timing=timing,
        new_fit=0,optimizer=0,quantum_shots=0,retry=0,resume=0,mandatory_stop=True))
    io.save(out/'result.json',value);return value


def run(root,path):
    root=Path(root).resolve();c=preflight(root,path);out=root/FRESH;out.mkdir(parents=True)
    call=integrated.Calls(out,LIMITS,MAX_CALLS);start=time.monotonic();cpu=time.process_time()
    try:
        io.save(out/'contract.json',c);io.save(out/'SOURCE.json',dict(io.read(root/c['source_manifest']),
            execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()))
        return execute(root,c,out,call)
    except BaseException as error:
        call.stop.set();io.save(out/'failure.json',dict(error=str(error),first_failure=call.first_failure));raise
    finally:
        io.save(out/'STOP.json',dict(failed=call.stop.is_set(),scientific_units=sum(call.counts.values()),
            call_counts=dict(call.counts),retry=0,resume=0,mandatory_stop=True))
        io.save(out/'resources.json',dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
