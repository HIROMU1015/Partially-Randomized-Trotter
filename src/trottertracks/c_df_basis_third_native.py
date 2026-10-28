"""Gamma0-specific native orchestration using unchanged B1/B2/CAR/P0 helpers.

Derived from the P4 native orchestration; only candidate/seed/file identities and
registered descriptor capture differ. No fitter, sampler or compiler is new.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from fractions import Fraction
import json,resource,time
from pathlib import Path
from . import c_df_basis_third as program
from . import c_pr_aware_df_m4 as m4
io=program.io


def sample(arm,tail,distribution,config,index,source):
    from . import c_pr_aware_df_stage_b as b
    from .c1_stage3_qualification import trajectory_key
    s=program.SAMPLING
    if arm.role!=program.ROLE or type(index)is not int or not 0<=index<32 or config.rte_steps!=128:
        raise ValueError('GAMMA0_SAMPLE_SCOPE')
    master=(s['master'],s['cohort'],source)
    seed=b.connection.trajectory_seed(master,arm.role,arm.model.source_identity,-1,s['axis'],index,0)
    events=b.independent_occurrence(arm,tail,distribution,config,master=master,round_index=-1,axis=s['axis'],shot=index,occurrence=0)
    if len(events)!=128 or any(e.taylor_order!=0 for e in events):raise ValueError('GAMMA0_EVENTS')
    return events,dict(sample=index,seed=seed,cohort=s['cohort'],SOURCE_sha256=source,
        role=arm.role,final_JU_sha256=arm.model.source_identity,microsteps=128,
        trajectory_key=trajectory_key(events),events=[e.to_dict() for e in events],
        quantum_shots=0,paired=False,old_cohort_pooled=False)


def execute(root,c,out,call,a,lanes,prices):
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
    arm=a['arm'];h=Fraction(1,10);shared=SharedQiskitCalls(call)
    car=call('car_context','full256CAR',lambda:b.stage1.car_bilinears(8));G=[]
    for i in range(9):G.append(call('raw_factor',str(i),lambda i=i:b.stage1.raw_base(b.stage1.corrected_kappa(a['p']['h'],a['p']['g']),a['p']['constant'],car) if i==0 else b.stage1.raw_frame(arm.J[i-1],arm.U[i-1],car)[0]))
    ix,reports=call('full_sector_gate','all nine full/sector factors',lambda:m2.sector_match(G,a['sector']));io.save(out/'sector_gate.json',reports)
    psi=np.zeros(256,complex);psi[ix]=a['ground']['sector_state']
    ctx=call('reference_context','cached independent fullCAR atoms/head',lambda:b2.reference_context(arm,G,car,h))
    io.save(out/'law_gate.json',call('full_law_gate','signed tail+identity',lambda:b2.law_gate(arm,a['tail'],ctx,a['tail_info'])))
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
        def assess_shell():
            matrices,report=b2.shell_assessment(qs['raw'],qs['compiled'],label)
            measured=circuit_metrics(qs['compiled'])
            if any(measured[k]!=a['saved']['shell_reports'][label]['metrics'][k] for k in m4.METRICS):
                raise ValueError('GAMMA0_SAVED_SHELL_METRICS')
            return matrices,dict(report,metrics=measured)
        shells[label],report=call('shell_assessment',label,assess_shell)
        io.save(out/(label+'_shell.json'),report)
    profile=CompilerProfile();profile.validate()
    def qsave(label,kind,qc):
        with (out/'circuits'/(label+'_'+kind+'.qpy')).open('xb') as f:qpy.dump(qc,f)
    def task(i):
        label=f'gamma0_{i:02d}'
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
        from .c_df_basis_surrogate_v2 import trajectory_features
        desc=call('trajectory_features',label,lambda:trajectory_features(record,arm.plan.atoms,a['distances']))
        return i,dict(metrics=metrics,descriptor=dict(sample=i,seed=record['seed'],features=desc,metrics=metrics))
    plan=m4.integrated.resource_plan(task_count=32,full_dimension=256);io.save(out/'resource_plan.json',plan)
    rows={};errors=[];tick=time.monotonic()
    with threadpool_limits(limits=plan['BLAS_threads_per_worker']),shared.adjoint_builds():
        with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
            for f in as_completed([pool.submit(task,i) for i in range(32)]):
                try:
                    i,v=f.result();rows[i]=v
                    if len(rows)%8==0:
                        with (out/'progress.jsonl').open('a') as stream:
                            stream.write(json.dumps(dict(completed=len(rows),elapsed_seconds=time.monotonic()-tick,CPU_seconds=time.process_time(),peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))+'\n')
                except BaseException as error:call.stop.set();errors.append(error)
    if errors:raise errors[0]
    cal=call('calibration_summary','32fixed samples,not all-law support',lambda:b2.calibration([rows[i]['metrics'] for i in range(32)]))
    r=call('native_result','new W/S;fixed32 sharedSE;4 saved comparisons',lambda:program.calibrated_result(cal,a,lanes,prices))
    r['descriptor_rows']=[rows[i]['descriptor'] for i in range(32)]
    r['Qiskit_lock_wait_seconds']=shared.wait_seconds
    r['completed_build_compile']=shared.completed
    io.save(out/'result.json',r);return r
