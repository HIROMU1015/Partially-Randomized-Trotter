"""Saved CDF7 native32: reuse full CAR/P0/actual-adjoint/X-Y/calibration.

The saved119-call oracle, PF/RTE, shots and affine prices are inputs, never
recomputed. Rank8 runners remain immutable. Scientific execution is opt-in;
permission is given in chat, without an approval-proof mechanism.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from fractions import Fraction
from pathlib import Path
import importlib.metadata
import json,math,resource,subprocess,time
from . import c_cdf7_pr as pr
from . import c_pr_aware_df_stage_b2 as b2

io,integrated=pr.io,pr.integrated
VERSION='saved_CDF7_fullCAR_native32_P0_v1'
SCOPE='C_H4_CDF7_NATIVE32_CONDITIONAL_COST_V1'
FAMILY='artifacts/research_c_cdf7_native32/2026-10-30'
FRESH=FAMILY+'/run01'
PR_SOURCE='3e75b16901f0f4a9966d2bd5b9ff6bde8e75f472'
PR_CONTRACT_SHA='55032e96e745095611b52aebe300056f0ce12231742dce2f6007ee5d0e427c81'
SAMPLING=dict(samples=32,events_per_sample=128,master=2026103007,
    cohort='CDF7_s1_r128_native32_v1',axis='CDF7_NATIVE_KERNEL',
    PRNG='numpy.random.PCG64',paired=False,adaptive=False,old_cohort_pooled=False,
    source_key='(master,cohort,SOURCE_digest),baseline,JU_hash,-1,axis,sample,0',
    quantum_occurrence_policy='fresh independent trajectory for every coherent occurrence;not a quantum execution')
POLICY=dict(pr.POLICY,system_qubits=8,ancilla_qubits=1,controlled_dimension=512,
    native_policy=b2.b.NATIVE_POLICY,head_tail_cross_fusion=False,
    calibration_scope='complete controlled one-h-step kernel,finite32 cohort',
    law='existing PCG64 binary64 finite-RTE implementation;ideal-law distance unqualified',
    PF_RTE_shots_recomputed=False,quantum_shots=0)
ALLOWANCES=b2.ALLOWANCES
LOAD_LIMITS=dict(evidence_load=1,input_load=1,fit_load=1,saved_reference_load=1,
    ground_load=1,structure_gate=1,native_model=1,law_prepare=1,tail_gate=1)
NATIVE_LIMITS=dict(pr.old.NATIVE_LIMITS,raw_factor=8,gaussian_build=8)
LIMITS={**LOAD_LIMITS,**NATIVE_LIMITS}
MAX_CALLS=sum(LIMITS.values())


def preflight(root,path):
    root=Path(root).resolve();c=io.read(path)
    expected=dict(scope=SCOPE,implementation=VERSION,policy=POLICY,
        allowances=ALLOWANCES,sampling=SAMPLING,compiler=b2.profile_dict(),
        call_limits=LIMITS,maximum_logical_units=MAX_CALLS,fresh_output=FRESH,
        input_sha256=pr.fit.io.P_H4_SHA256,export=pr.EXPORT,export_sha256=pr.EXPORT_SHA,
        PR_SOURCE=PR_SOURCE,PR_contract_sha256=PR_CONTRACT_SHA,new_fit=0,optimizer=0,
        retry=0,resume=0,PF_RTE_shots_recomputed=False,QPY_count=128)
    for k,v in expected.items():
        if c[k]!=v:raise ValueError('CDF7_NATIVE_SCOPE:'+k)
    required={c['input'],c['export'],c['saved_reference'],c['saved_ground'],
        *c['records'].values(),*(p for kinds in c['shells'].values() for p in kinds.values())}
    if (not required<=c['saved_inputs'].keys() or c['saved_inputs'][pr.EXPORT]!=pr.EXPORT_SHA
            or set(c['records'])!={'PR','STOP','SOURCE','contract','tail'}
            or set(c['shells'])!={'head','X','Y'}
            or any(set(x)!={'raw','compiled'} for x in c['shells'].values())):
        raise ValueError('NATIVE_UNPINNED_INPUT_OR_SCHEMA')
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('NATIVE_SOURCE_MANIFEST')
    for files in (io.read(root/c['source_manifest'])['files'],c['saved_inputs']):
        for p,h in files.items():
            f=(root/p).resolve()
            if not f.is_relative_to(root) or io.sha(f)!=h:raise ValueError('NATIVE_SOURCE_INPUT:'+p)
    for p,v in c['packages'].items():
        if importlib.metadata.version(p)!=v:raise ValueError('NATIVE_ENV:'+p)
    if (root/FRESH).exists():raise ValueError('CDF7_NATIVE_FRESH_NO_RETRY_RESUME')
    return c


def evidence(root,c):
    v={k:io.read(Path(root)/p) for k,p in c['records'].items()}
    r,s=v['PR'],v['STOP'];previous=v['contract']
    if (r['status']!='CDF7_PR_REFERENCE_RPE_PRICE_COMPLETE' or s['failed']
            or s['scientific_units']!=119 or s['call_counts']!=pr.LIMITS or s['retry'] or s['resume']
            or not s['mandatory_stop'] or v['SOURCE']['execution_commit']!=PR_SOURCE
            or previous['policy']!=pr.POLICY or previous['target']!=pr.TARGET
            or previous['export_sha256']!=c['export_sha256']
            or previous['original_input_sha256']!=c['input_sha256']
            or io.sha(Path(root)/c['records']['contract'])!=c['PR_contract_sha256']):
        raise ValueError('CDF7_SAVED_PR_IDENTITY')
    saved=pr.evidence(root,previous)
    recommendation=pr.native_recommendation(dict(result=r['reference']),r['accuracy'],r['price'])
    if (recommendation!=r['native_recommendation']
            or recommendation['status']!='ELIGIBLE_FOR_SEPARATE_NATIVE_PROPOSAL'
            or r['reference']['ground']['status']!='PASS'
            or [x['m'] for x in r['reference']['rounds']]!=list(range(13))
            or any(x['feasibility']['status']!='POTENTIALLY_FEASIBLE' for x in r['reference']['rounds'])
            or v['tail']!=r['reference']['tail']):
        raise ValueError('CDF7_REGISTERED_NATIVE_ELIGIBILITY')
    return dict(PR=r,saved=saved,tail=v['tail'])


def structure(p,d,sector,ground,record):
    import numpy as np
    pr.validate_arrays(p,d)
    if (sector['generators'].shape!=(8,36,36) or sector['H'].shape!=(36,36)
            or sector['full_input_indices'].shape!=(36,) or ground['sector_state'].shape!=(36,)
            or ground['energy'].shape!=() or not all(np.isfinite(x).all() for x in [*sector.values(),*ground.values()])
            or abs(float(ground['energy'])-record['energy_Ha'])>1e-12
            or abs(np.vdot(ground['sector_state'],ground['sector_state'])-1)>ALLOWANCES['normalization']):
        raise ValueError('CDF7_SAVED_SECTOR_OR_GROUND')
    return dict(status='PASS',rank=7,factors=8,oracle='saved119-call own-ground;no new eigensolve')


def load(root,c,out,call):
    from . import c_pr_aware_df_stage_b as b,c_pr_aware_df_stage_b_runner as loader
    saved=call('evidence_load','saved119-call eligibility and comparisons',lambda:evidence(root,c))
    p=call('input_load','same original h/g/scalar',lambda:loader.load(Path(root)/c['input'],('h','g','constant')))
    d=call('fit_load','same final seven-frame J/U',lambda:loader.load(Path(root)/c['export'],('J','U')))
    sector=call('saved_reference_load','saved generators/H/mapping only;no PF decode',lambda:loader.load(
        Path(root)/c['saved_reference'],('generators','H','full_input_indices')))
    ground=call('ground_load','saved own-DF oracle',lambda:loader.load(Path(root)/c['saved_ground'],('energy','sector_state')))
    io.save(Path(out)/'structure.json',call('structure_gate','saved sector/oracle identities',lambda:structure(p,d,sector,ground,saved['PR']['reference']['ground'])))
    arm=call('native_model','unchanged spin-kappa/general-K',lambda:b.prepare_arm(**p,**d,role='baseline',
        source_identity=c['export_sha256'],input_identity=c['input_sha256'],ld=2))
    tail,config,dist=call('law_prepare','same r128/K0,h.1',lambda:b.configure(arm,Fraction(1,10),128,0))
    def tail_check():
        old=saved['tail'];components=pr.with_components({},tail)['components']
        if (config.rte_steps!=128 or config.finite_taylor_order!=0 or len(arm.model.blocks)!=8
                or components!=old['components'] or arm.plan.extracted_identity!=old['identity']
                or len(arm.plan.atoms)!=old['atoms']
                or not math.isclose(tail.lambda_r,old['lambda_R'],rel_tol=1e-12,abs_tol=1e-14)):
            raise ValueError('CDF7_SAVED_LAW_METADATA')
        return dict(status='PASS',lambda_R=tail.lambda_r,atoms=len(components),reused_saved_shots=True)
    io.save(Path(out)/'tail_metadata_gate.json',call('tail_gate','same components/sign/probabilities',tail_check))
    return dict(p=p,arm=arm,sector=sector,ground=ground,tail=tail,config=config,distribution=dist,
        tail_info=saved['tail'],saved=saved['saved'],lanes=saved['PR']['accuracy'],prices=saved['PR']['price'])


def sector_match(G,sector):
    import numpy as np
    from .c1_stage1 import sector_indices
    b=b2.b;ix=sector['full_input_indices']
    if (len(G)!=8 or sector['generators'].shape!=(8,36,36)
            or not np.array_equal(np.sort(ix),sector_indices(4,2,2))):
        raise ValueError('CDF7_EIGHT_FACTOR_MAPPING')
    checks=[b.component_check(x[np.ix_(ix,ix)],y,1e-10,label='CDF7 full/sector factor'+str(i))
        for i,(x,y) in enumerate(zip(G,sector['generators'],strict=True))]
    checks.append(b.component_check(sum(sector['generators']),sector['H'],1e-10,label='saved CDF7 sum'))
    return ix,checks


def sample(arm,tail,dist,config,index,source):
    from .c1_stage3_qualification import trajectory_key
    b=b2.b;s=SAMPLING
    if (arm.role!='baseline' or len(arm.model.blocks)!=8 or type(index)is not int
            or not 0<=index<32 or config.rte_steps!=128 or config.finite_taylor_order!=0):
        raise ValueError('CDF7_NATIVE_SAMPLE_SCOPE')
    master=(s['master'],s['cohort'],source)
    seed=b.connection.trajectory_seed(master,arm.role,arm.model.source_identity,-1,s['axis'],index,0)
    events=b.independent_occurrence(arm,tail,dist,config,master=master,round_index=-1,axis=s['axis'],shot=index,occurrence=0)
    if len(events)!=128 or any(e.taylor_order!=0 for e in events):raise ValueError('CDF7_NATIVE_EVENTS')
    return events,dict(sample=index,seed=seed,cohort=s['cohort'],SOURCE_sha256=source,
        role=arm.role,final_JU_sha256=arm.model.source_identity,microsteps=128,
        trajectory_key=trajectory_key(events),events=[e.to_dict() for e in events],
        quantum_shots=0,paired=False,old_cohort_pooled=False,reused_as_coherent_repetitions=False)


def calibrated_result(cal,a):
    metrics=pr.old.old.METRICS;values={}
    for lane in ('ideal_exact_native','assumed_raw_signal_budget'):
        affine=a['prices']['affine'][lane];W=affine['W']
        if W is None:values[lane]=dict(status='UNQUALIFIED',conditional_totals=None);continue
        totals={k:W*cal['sample_mean'][k]+affine['S'][k] for k in metrics}
        comparison={}
        for name,tag in [('CDF8_gamma0','CDF8_native'),('M4_r128','M4_native')]:
            old=a['saved'][tag]['lanes'][lane]['conditional_totals']
            comparison[name]={k:dict(ratio=totals[k]/old[k] if old[k]>0 else None,
                difference=totals[k]-old[k],mean_below_break_even=cal['sample_mean'][k]<a['prices']['comparisons'][lane+':'+name]['thresholds'][k]['mu_break_even']) for k in metrics}
        values[lane]=dict(conditional_totals=totals,weighted_calibration_SE={k:W*cal['sample_mean_standard_error'][k] for k in metrics},
            comparisons=comparison,independent_round_calibrations=1,actual_CA_achieved=None,guaranteed_total_resources=None)
    return dict(status='CDF7_NATIVE32_CONDITIONAL_COMPARISON_COMPLETE',calibration=cal,lanes=values,
        cost_support=None,all_law_synthesis=None,implemented_law_error=None,actual_CA_achieved=None,
        ideal_law_mean_advantage=None,guaranteed_advantage=None,new_fit=0,optimizer=0,
        PF_RTE_shots_recomputed=False,retry=0,resume=0,quantum_shots=0,mandatory_stop=True)


def native(root,c,out,call,a):
    import numpy as np
    from qiskit import qpy
    from threadpoolctl import threadpool_limits
    from . import c_pr_aware_df_stage_b as b,c_pr_aware_df_stage_b_runner as loader
    from .c_pr_aware_df_stage_b2_compat import SharedQiskitCalls
    from .c1_h4_strong_baseline import StrongBuilder
    from .c1_h4_native_fair import basis_request,compile_checked
    from .pr_dictionary_validation.fixture import CompilerProfile
    from .pr_dictionary_validation.f2_execution import circuit_metrics
    out=Path(out)
    for leaf in ('circuits','samples','references'):(out/leaf).mkdir()
    arm=a['arm'];h=Fraction(1,10);shared=SharedQiskitCalls(call)
    car=call('car_context','full256 CAR',lambda:b.stage1.car_bilinears(8))
    G=[call('raw_factor',str(i),lambda i=i:b.stage1.raw_base(b.stage1.corrected_kappa(a['p']['h'],a['p']['g']),a['p']['constant'],car)
        if i==0 else b.stage1.raw_frame(arm.J[i-1],arm.U[i-1],car)[0]) for i in range(8)]
    ix,reports=call('full_sector_gate','all eight full/sector factors',lambda:sector_match(G,a['sector']));io.save(out/'sector_gate.json',reports)
    psi=np.zeros(256,complex);psi[ix]=a['ground']['sector_state']
    ctx=call('reference_context','independent full CAR atom/head cache',lambda:b2.reference_context(arm,G,car,h))
    io.save(out/'law_gate.json',call('full_law_gate','signed tail+identity full256',lambda:b2.law_gate(arm,a['tail'],ctx,a['tail_info'])))
    builder=StrongBuilder(relative_fusion=True,call=call)
    for block in arm.model.blocks:call('gaussian_build',block.block_id,lambda block=block:basis_request(builder,arm.model,block))
    for order in (tuple(range(3)),tuple(reversed(range(3)))):
        for i,j in zip(order,order[1:]):builder.transition(arm.model,arm.model.blocks[i],arm.model.blocks[j])
    io.save(out/'gaussian_precision.json',builder.basis_reports)
    io.save(out/'relative_head_fusion.json',builder.fusion_records)
    shells={}
    for label in ('head','X','Y'):
        qs={}
        for kind in ('raw','compiled'):
            def shell_load():
                with (Path(root)/c['shells'][label][kind]).open('rb') as f:items=qpy.load(f)
                if len(items)!=1 or items[0].num_qubits!=9:raise ValueError('CDF7_SHELL_SCHEMA')
                return items[0]
            qs[kind]=call('shell_load',label+kind,shell_load)
        def assess_shell():
            matrices,report=b2.shell_assessment(qs['raw'],qs['compiled'],label)
            measured=circuit_metrics(qs['compiled'])
            if any(measured[k]!=a['saved']['shell_reports'][label]['metrics'][k] for k in pr.old.old.METRICS):
                raise ValueError('CDF7_SAVED_SHELL_METRICS')
            return matrices,dict(report,metrics=measured)
        shells[label],report=call('shell_assessment',label,assess_shell);io.save(out/(label+'_shell.json'),report)
    profile=CompilerProfile();profile.validate()
    def qsave(label,kind,qc):
        with (out/'circuits'/(label+'_'+kind+'.qpy')).open('xb') as f:qpy.dump(qc,f)
    def task(i):
        label=f'CDF7_{i:02d}'
        events,record=call('trajectory_sample',label,lambda:sample(arm,a['tail'],a['distribution'],a['config'],i,c['source_manifest_sha256']))
        io.save(out/'samples'/(label+'_trajectory.json'),record)
        raw=shared('kernel_build',label,lambda:b.build_kernel(arm,h,events,a['config'],builder=builder));qsave(label,'raw',raw)
        ref=call('trajectory_reference',label,lambda:b2.trajectory_reference(arm,ctx,h,events));loader.save_arrays(out/'references'/(label+'.npz'),system_reference=ref)
        R,target,report=call('raw_operator_gate',label,lambda:b2.raw_gate(raw,ref));io.save(out/'samples'/(label+'_raw.json'),report)
        compiled=shared('compile',label,lambda:compile_checked(raw,profile));qsave(label,'compiled',compiled)
        C,report=call('compiled_operator_gate',label,lambda:b2.compiled_gate(compiled,R,target));io.save(out/'samples'/(label+'_compiled.json'),report)
        adjoints,report=call('adjoint_pair_gate',label,lambda:b2.adjoint_pair_gate(raw,compiled,R,C,target))
        for kind,qc in adjoints.items():qsave(label,kind+'_adjoint',qc)
        io.save(out/'samples'/(label+'_adjoint.json'),report)
        io.save(out/'samples'/(label+'_XY.json'),call('XY_gate',label,lambda:b2.XY_gate(R,C,ref,psi,shells)))
        metrics=call('kernel_metrics',label,lambda:circuit_metrics(compiled));io.save(out/'samples'/(label+'_metrics.json'),metrics)
        return i,metrics
    plan=integrated.resource_plan(task_count=32,full_dimension=256);io.save(out/'resource_plan.json',plan)
    rows={};errors=[];tick=time.monotonic()
    with threadpool_limits(limits=plan['BLAS_threads_per_worker']),shared.adjoint_builds():
        with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
            for f in as_completed([pool.submit(task,i) for i in range(32)]):
                try:
                    i,v=f.result();rows[i]=v
                    with (out/'progress.jsonl').open('a') as stream:stream.write(json.dumps(dict(completed=len(rows),elapsed_seconds=time.monotonic()-tick))+'\n')
                except BaseException as error:call.stop.set();errors.append(error)
    if errors:raise errors[0]
    cal=call('calibration_summary','finite32 descriptive mean/SE,not support',lambda:b2.calibration([rows[i] for i in range(32)]))
    result=call('native_result','saved W/S;two targets;shared calibration uncertainty',lambda:calibrated_result(cal,a))
    result.update(QPY_count=128,completed_build_compile=shared.completed,Qiskit_lock_wait_seconds=shared.wait_seconds)
    io.save(out/'result.json',result);return result


def execute(root,c,out,call):
    a=load(root,c,out,call)
    return native(root,c,out,call,a)


def run(root,path):
    root=Path(root).resolve();c=preflight(root,path);out=root/FRESH;out.mkdir(parents=True)
    call=integrated.Calls(out,LIMITS,MAX_CALLS);tick=time.monotonic();cpu=time.process_time()
    try:
        io.save(out/'contract.json',c);io.save(out/'SOURCE.json',dict(io.read(root/c['source_manifest']),
            execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()))
        return execute(root,c,out,call)
    except BaseException as error:
        call.stop.set();io.save(out/'failure.json',dict(error=str(error),first_failure=call.first_failure));raise
    finally:
        io.save(out/'STOP.json',dict(failed=call.stop.is_set(),scientific_units=sum(call.counts.values()),call_counts=dict(call.counts),retry=0,resume=0,mandatory_stop=True))
        io.save(out/'resources.json',dict(wall_seconds=time.monotonic()-tick,CPU_seconds=time.process_time()-cpu,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
