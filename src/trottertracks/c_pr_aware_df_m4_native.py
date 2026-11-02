"""Prepared, separately unapproved r128 native calibration. Not called by M4.

Only orchestration/event-count binding is new. All sampling, native, CAR,
P0, full512/adjoint/XY and calibration algorithms are the existing B2/M3 ones.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from fractions import Fraction
import importlib.metadata
from pathlib import Path
import resource
import subprocess
import time

from . import c_pr_aware_df_m3 as m3
from . import c_pr_aware_df_m4 as m4
io=m4.io
VERSION='M4_future_r128_native_fixed32_v1'
SCOPE='C_PR_AWARE_DF_M4_R128_NATIVE_CALIBRATION_DRAFT_V1'
FRESH='artifacts/research_c_pr_aware_df_M4_native/2026-10-28/run01'
SAMPLING=dict(master=2026102805,cohort='M4_r128_native_fixed32_v1',axis='M4_R128_NATIVE_ONLY',
    samples=32,events_per_sample=128,PRNG='numpy.random.PCG64',paired=False,adaptive=False)
ATTACH=dict(evidence_load=1,input_load=1,fit_load=1,native_model=1,law_prepare=1,sector_reference_load=1,ground_load=1)
CORE={k:v for k,v in m3.B_LIMITS.items() if k!='delta_cost_comparison'}
LIMITS=dict(**ATTACH,**CORE,native_result=1)
MAX_CALLS=sum(LIMITS.values())


def preflight(root,path):
    root=Path(root).resolve();c=io.read(path)
    if (c['scope']!=SCOPE or c['implementation']!=VERSION or c['sampling']!=SAMPLING
            or c['call_limits']!=LIMITS or c['maximum_logical_units']!=MAX_CALLS
            or c['fresh_output']!=FRESH or c['retry'] or c['resume']):raise ValueError('NATIVE128_SCOPE')
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('NATIVE128_SOURCE')
    for files in (io.read(root/c['source_manifest'])['files'],c['saved_inputs']):
        for p,h in files.items():
            file=(root/p).resolve()
            if not file.is_relative_to(root) or io.sha(file)!=h:raise ValueError('NATIVE128_SOURCE_INPUT:'+p)
    for p,v in c['packages'].items():
        if importlib.metadata.version(p)!=v:raise ValueError('NATIVE128_ENVIRONMENT:'+p)
    if (root/FRESH).exists():raise ValueError('FRESH_NATIVE128')
    return c


def sample(arm,tail,distribution,config,index,source):
    from . import c_pr_aware_df_stage_b as b
    from .c1_stage3_qualification import trajectory_key
    if type(index)is not int or not 0<=index<32 or config.rte_steps!=128:raise ValueError('NATIVE128_SAMPLE_SCOPE')
    master=(SAMPLING['master'],SAMPLING['cohort'],source)
    seed=b.connection.trajectory_seed(master,arm.role,arm.model.source_identity,-1,SAMPLING['axis'],index,0)
    events=b.independent_occurrence(arm,tail,distribution,config,master=master,round_index=-1,axis=SAMPLING['axis'],shot=index,occurrence=0)
    if len(events)!=128 or any(e.taylor_order!=0 for e in events):raise ValueError('NATIVE128_EVENTS')
    return events,dict(sample=index,seed=seed,cohort=SAMPLING['cohort'],SOURCE_sha256=source,
        final_JU_sha256=arm.model.source_identity,microsteps=128,trajectory_key=trajectory_key(events),
        events=[e.to_dict() for e in events],quantum_shots=0,paired=False,old_cohort_pooled=False)


def attach(root,c,call):
    import numpy as np
    from . import c_pr_aware_df_stage_b as b
    from . import c_pr_aware_df_stage_b_runner as loader
    from . import c_pr_aware_df_m2 as m2
    def evidence():
        saved=m4.load_evidence(root,c);m4.identity_gate(saved,c)
        r=io.read(Path(root)/c['M4_records']['result']);stop=io.read(Path(root)/c['M4_records']['STOP'])
        source=io.read(Path(root)/c['M4_records']['SOURCE'])
        if (r['status']!='M4_R128_SHOTS_BREAKEVEN_COMPLETE' or stop['failed'] or stop['scientific_units']!=98
                or source['execution_commit']!=c['M4_scientific_SOURCE'] or r['A']['status']!='POTENTIALLY_FEASIBLE'
                or any(not r['B'][lane]['energy']['same_accuracy'] for lane in ('ideal_exact_native','assumed_raw_signal_budget'))):
            raise ValueError('M4_NATIVE_PREREQUISITES')
        saved['M4']=r;return saved
    saved=call('evidence_load','saved M4 feasibility/shots/breakeven,not recomputed',evidence)
    p=call('input_load','saved h/g/scalar',lambda:loader.load(Path(root)/c['input'],('h','g','constant')))
    d=call('fit_load','same Proposed J/U',lambda:loader.load(Path(root)/c['exports']['proposed'],('J','U')))
    def model():
        m2.validate_arrays(p,d,'proposed')
        return b.prepare_arm(**p,**d,role='proposed',source_identity=c['saved_inputs'][c['exports']['proposed']],input_identity=c['original_input_sha256'],ld=2)
    arm=call('native_model','original conventions,no fit',model)
    tail,config,dist=call('law_prepare','r128 only,no PF/RTE re-evaluation',lambda:b.configure(arm,Fraction(1,10),128,0))
    sector=call('sector_reference_load','saved nine generators/H/mapping',lambda:loader.load(Path(root)/c['M3_sector'],('generators','H','full_input_indices')))
    ground=call('ground_load','saved own-DF ground,no eigensolve',lambda:loader.load(Path(root)/c['grounds']['proposed'],('energy','sector_state')))
    if ground['sector_state'].shape!=(36,) or not np.isfinite(ground['sector_state']).all():raise ValueError('GROUND_SCHEMA')
    return dict(p=p,arm=arm,tail=tail,config=config,distribution=dist,sector=sector,ground=ground,saved=saved)


def calibrated_result(cal,saved):
    lanes={}
    for lane in ('ideal_exact_native','assumed_raw_signal_budget'):
        affine=saved['M4']['C']['affine'][lane];W=affine['W']
        if W is None:raise ValueError('NATIVE128_UNQUALIFIED_AFFINE')
        totals={k:W*cal['sample_mean'][k]+affine['S'][k] for k in m4.METRICS}
        lanes[lane]=dict(conditional_totals=totals,weighted_calibration_SE={k:W*cal['sample_mean_standard_error'][k] for k in m4.METRICS},
            threshold_comparisons={target:{k:dict(mean=cal['sample_mean'][k],threshold=saved['M4']['C']['comparisons'][lane+':'+target]['thresholds'][k]['mu_break_even'],
                mean_below_threshold=cal['sample_mean'][k]<saved['M4']['C']['comparisons'][lane+':'+target]['thresholds'][k]['mu_break_even']) for k in m4.METRICS} for target in ('M3_r64','M2_deterministic','old_PR_s4')},
            independent_round_calibrations=1,actual_CA_achieved=None,ideal_law_mean_advantage=None,guaranteed_total_resources=None)
    return dict(status='NATIVE128_FINITE_COHORT_COMPLETE',calibration=cal,lanes=lanes,cost_support=None,
        all_law_synthesis=None,implemented_law_error=None,actual_CA_achieved=None,mandatory_stop=True)


def execute(root,c,out,call):
    import numpy as np
    from qiskit import qpy
    from threadpoolctl import threadpool_limits
    from . import c_pr_aware_df_stage_b as b
    from . import c_pr_aware_df_stage_b2 as b2
    from . import c_pr_aware_df_stage_b_runner as loader
    from . import c_pr_aware_df_m2 as m2
    from .c_pr_aware_df_stage_b2_compat import SharedQiskitCalls
    from .c1_h4_strong_baseline import StrongBuilder
    from .c1_h4_native_fair import basis_request,compile_checked
    from .pr_dictionary_validation.fixture import CompilerProfile
    from .pr_dictionary_validation.f2_execution import circuit_metrics
    out=Path(out)
    for leaf in ('circuits','samples','references'):(out/leaf).mkdir()
    a=attach(root,c,call);arm=a['arm'];h=Fraction(1,10);shared=SharedQiskitCalls(call)
    car=call('car_context','full256CAR',lambda:b.stage1.car_bilinears(8));G=[]
    for i in range(9):G.append(call('raw_factor',str(i),lambda i=i:b.stage1.raw_base(b.stage1.corrected_kappa(a['p']['h'],a['p']['g']),a['p']['constant'],car) if i==0 else b.stage1.raw_frame(arm.J[i-1],arm.U[i-1],car)[0]))
    ix,reports=call('full_sector_gate','all nine full/sector factors',lambda:m2.sector_match(G,a['sector']));io.save(out/'sector_gate.json',reports)
    psi=np.zeros(256,complex);psi[ix]=a['ground']['sector_state']
    ctx=call('reference_context','cached independent fullCAR atoms/head',lambda:b2.reference_context(arm,G,car,h))
    io.save(out/'law_gate.json',call('full_law_gate','signed tail+identity',lambda:b2.law_gate(arm,a['tail'],ctx,a['saved']['proposed_tail'])))
    builder=StrongBuilder(relative_fusion=True,call=call)
    for block in arm.model.blocks:call('gaussian_build',block.block_id,lambda block=block:basis_request(builder,arm.model,block))
    for order in (tuple(range(3)),tuple(reversed(range(3)))):
        for i,j in zip(order,order[1:]):builder.transition(arm.model,arm.model.blocks[i],arm.model.blocks[j])
    shells={}
    for label in ('head','X','Y'):
        qs={}
        for kind in ('raw','compiled'):
            def load():
                with (Path(root)/c['shells'][label][kind]).open('rb') as f:items=qpy.load(f)
                if len(items)!=1 or items[0].num_qubits!=9:raise ValueError('SHELL_SCHEMA')
                return items[0]
            qs[kind]=call('shell_load',label+kind,load)
        shells[label],report=call('shell_assessment',label,lambda label=label:b2.shell_assessment(qs['raw'],qs['compiled'],label))
        io.save(out/(label+'_shell.json'),report)
    profile=CompilerProfile();profile.validate()
    def qsave(label,kind,qc):
        with (out/'circuits'/(label+'_'+kind+'.qpy')).open('xb') as f:qpy.dump(qc,f)
    def task(i):
        label=f'proposed_{i:02d}'
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
    plan=m4.integrated.resource_plan(task_count=32,full_dimension=256);io.save(out/'resource_plan.json',plan)
    rows={};errors=[]
    with threadpool_limits(limits=plan['BLAS_threads_per_worker']),shared.adjoint_builds():
        with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
            for f in as_completed([pool.submit(task,i) for i in range(32)]):
                try:i,v=f.result();rows[i]=v
                except BaseException as error:call.stop.set();errors.append(error)
    if errors:raise errors[0]
    cal=call('calibration_summary','32fixed samples,not all-law support',lambda:b2.calibration([rows[i] for i in range(32)]))
    r=call('native_result','savedM4 W/S and thresholds,no newshots',lambda:calibrated_result(cal,a['saved']))
    io.save(out/'result.json',r);return r


def run(root,path):
    root=Path(root).resolve();c=preflight(root,path);out=root/FRESH;out.mkdir(parents=True)
    call=m4.integrated.Calls(out,LIMITS,MAX_CALLS);start=time.monotonic();cpu=time.process_time()
    try:
        io.save(out/'contract.json',c);io.save(out/'SOURCE.json',dict(io.read(root/c['source_manifest']),execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()))
        return execute(root,c,out,call)
    except BaseException as error:
        call.stop.set();io.save(out/'failure.json',dict(error=str(error),first_failure=call.first_failure));raise
    finally:
        io.save(out/'STOP.json',dict(failed=call.stop.is_set(),scientific_units=sum(call.counts.values()),call_counts=dict(call.counts),retry=0,resume=0,mandatory_stop=True))
        io.save(out/'resources.json',dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
