"""Candidate-specific fixed M4 policy, existing RTE/RPE/native primitives."""
from fractions import Fraction
from pathlib import Path
import math,time,resource,json
from concurrent.futures import ThreadPoolExecutor,as_completed
from . import c_pr_aware_df_m3 as m3
from . import c_pr_aware_df_m4 as m4
from . import c_pr_aware_df_stage_c as accounting
from . import c_pr_aware_df_stage_b as b
from . import c_pr_aware_df_stage_b2 as b2
from . import c_pr_aware_df_stage_b_runner as loader
from .c1_stage3_integrated import Calls
io=m4.io
LIMITS=dict(model_prepare=1,sector_factor_gate=9,law_prepare=1,sector_law_gate=1,
    ground_reference=2,kernel_reference=1,exact_evolution=13,round_reference=13,
    round_classification=13,feasibility_summary=1,signal_qualification=3,round_accuracy=39,
    energy_assessment=3,shots_summary=3,shell_metrics_load=1,
    **{k:v for k,v in m3.B_LIMITS.items() if k!='delta_cost_comparison'},
    cost_calibration=1,serial_cost=26,total_aggregation=3,calibration_uncertainty=3,
    resource_comparison=1)
MAX_CALLS=sum(LIMITS.values())
SAMPLING=dict(master=2026110201,cohort='integrated_best_fixed32_v1',axis='INTEGRATED_NATIVE_ONLY',samples=32,
    events_per_sample=128,PRNG='numpy.random.PCG64',paired=False,adaptive=False)


def native(root,c,out,call,a,*,sampling=None,prepared=None,backend="threads"):
    if backend == 'process':
        from .c_native_process import native
        return native(root,c,out,call,a,sampling=sampling,prepared=prepared)
    if backend != 'threads':raise ValueError('NATIVE_BACKEND')
    tick = time.monotonic(); cpu_tick = time.process_time()
    import numpy as np
    from qiskit import qpy
    from threadpoolctl import threadpool_limits
    from .c_pr_aware_df_stage_b2_compat import SharedQiskitCalls
    from .c1_h4_strong_baseline import StrongBuilder
    from .c1_h4_native_fair import basis_request,compile_checked
    from .pr_dictionary_validation.fixture import CompilerProfile
    from .pr_dictionary_validation.f2_execution import circuit_metrics
    from .c_pr_aware_df_m2 import sector_match
    arm=a['arm'];h=Fraction(1,10);shared=SharedQiskitCalls(call)
    policy=dict(SAMPLING if sampling is None else sampling)
    steps=policy['events_per_sample']
    if (policy['samples']!=32 or type(steps) is not int or steps<1
            or (sampling is not None and a['config'].rte_steps!=steps)):
        raise ValueError('NATIVE_SAMPLING_CONFIG')
    for leaf in ('circuits','samples','references'):(out/leaf).mkdir()
    if prepared:
        if prepared['arm'] is not arm:raise ValueError('NATIVE_PREPARED_ARM_IDENTITY')
        ctx,psi,builder,shells=(prepared[k] for k in ('ctx','psi','builder','shells'))
    else:
        car=call('car_context','full256 CAR',lambda:b.stage1.car_bilinears(8))
        G=[call('raw_factor',str(i),lambda i=i:b.stage1.raw_base(b.stage1.corrected_kappa(a['p']['h'],a['p']['g']),a['p']['constant'],car)
            if i==0 else b.stage1.raw_frame(arm.J[i-1],arm.U[i-1],car)[0]) for i in range(9)]
        ix,report=call('full_sector_gate','all9 full/sector',lambda:sector_match(G,a['sector']));io.save(out/'full_sector.json',report)
        psi=np.zeros(256,complex);psi[ix]=a['ground']
        ctx=call('reference_context','immutable CAR atom/head cache',lambda:b2.reference_context(arm,G,car,h))
        saved=dict(lambda_R=a['tail'].lambda_r,identity=arm.plan.extracted_identity,atoms=len(arm.plan.atoms),
            components=[dict(probability=x.probability) for x in a['tail'].components])
        io.save(out/'full_law.json',call('full_law_gate','new candidate signed tail',lambda:b2.law_gate(arm,a['tail'],ctx,saved)))
        builder=StrongBuilder(relative_fusion=True,call=call)
        for block in arm.model.blocks:call('gaussian_build',block.block_id,lambda block=block:basis_request(builder,arm.model,block))
        for order in (tuple(range(3)),tuple(reversed(range(3)))):
            for i,j in zip(order,order[1:]):builder.transition(arm.model,arm.model.blocks[i],arm.model.blocks[j])
        shells={}
        for label in ('head','X','Y'):
            qs={}
            for kind in ('raw','compiled'):
                def load(label=label,kind=kind):
                    with (root/c['shells'][label][kind]).open('rb') as f:items=qpy.load(f)
                    if len(items)!=1 or items[0].num_qubits!=9:raise ValueError('SHELL_SCHEMA')
                    return items[0]
                qs[kind]=call('shell_load',label+kind,load)
            shells[label],report=call('shell_assessment',label,lambda label=label:b2.shell_assessment(qs['raw'],qs['compiled'],label))
            io.save(out/(label+'_shell.json'),report)
        if prepared is not None:prepared.update(arm=arm,ctx=ctx,psi=psi,builder=builder,shells=shells,origin=str(out))
    profile=CompilerProfile();profile.validate()
    from .c_native_sample import run_sample
    def task(i):
        return run_sample(i,c,out,call,a,policy,ctx,psi,builder,shells,shared)
    plan=m4.integrated.resource_plan(task_count=32,full_dimension=256);io.save(out/'native_resource_plan.json',plan)
    rows={};errors=[]
    with threadpool_limits(limits=plan['BLAS_threads_per_worker']),shared.adjoint_builds():
        with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
            for f in as_completed([pool.submit(task,i) for i in range(32)]):
                try:
                    i,v=f.result();rows[i]=v
                    with (out/'native_progress.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(completed=len(rows),CPU_seconds=time.process_time(),peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))+'\n')
                except BaseException as e:call.stop.set();errors.append(e)
    if errors:raise errors[0]
    samples=[rows[i] for i in range(32)]
    calibration=call('calibration_summary','fixed32 new-candidate cohort',lambda:b2.calibration(samples))
    from .c_full_unitary import VERSION as evaluator
    io.save(out/'native_timing.json',dict(samples=len(samples),events_per_sample=steps,
        cohort=policy['cohort'],wall_seconds=time.monotonic()-tick,
        CPU_seconds=time.process_time()-cpu_tick,resource_plan=plan,
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        full_operator_evaluator=evaluator,full_input_output_dimension=512,
        actual_inverse_circuits_independently_evaluated=True))
    return samples,calibration


def run(root,c,out,p,ctx,physical,selected):
    import numpy as np
    from scipy.linalg import expm
    from threadpoolctl import threadpool_limits
    from .c_rc2_pr_stage_b import distribution_gate
    from .c1_stage3_accuracy import AccuracyTarget
    from . import c1_stage3_feasibility as feasibility
    out=Path(out);out.mkdir();call=Calls(out,LIMITS,MAX_CALLS);start=time.monotonic()
    try:
        J,U=selected['d']['J'],selected['d']['U'];G=selected['G'];H=selected['H'];df=selected['row']
        arm=call('model_prepare','new selected J/U only',lambda:b.prepare_arm(**p,J=J,U=U,role='proposed',
            source_identity=df['candidate_sha256'],input_identity=c['saved_inputs'][c['input_paths']['input']],ld=2))
        for i in range(9):call('sector_factor_gate',str(i),lambda i=i:b.component_check(b.candidate_factor(ctx,arm,i),G[i],1e-10,label='new factor generalK/CAR'))
        tail,config,dist=call('law_prepare','new lambda/probabilities,r128 K0',lambda:b.configure(arm,Fraction(1,10),128,0))
        def law():
            row=b.tail_gate(ctx,arm,G,tail);distribution_gate(tail,config,dist)
            if not math.isclose(row['lambda_R'],df['exact_tail_lambda_Ha'],rel_tol=1e-12,abs_tol=1e-14):raise ValueError('LAMBDA_CONVENTION')
            return row
        io.save(out/'law.json',call('sector_law_gate','regenerated law and independent CAR',law))
        def ground(label,H):
            pair=loader.ground(H,out,label);row=loader.ground_gate(H,pair);io.save(out/(label+'_ground.json'),row);return pair
        call('ground_reference','physical',lambda:ground('physical',physical))
        eig=call('ground_reference','candidate',lambda:ground('candidate',H));psi=eig[1][:,0]
        kernels=call('kernel_reference','new PF and corrected RTE',lambda:b.kernels(arm,G,Fraction(1,10),config))
        loader.save_arrays(out/'sector_reference.npz',generators=np.asarray(G),H=H,pf=kernels['pf'],corrected=kernels['corrected'])
        target=AccuracyTarget(**m4.TARGET);plan=m4.integrated.resource_plan(task_count=13,full_dimension=36)
        def task(m):
            elapsed=Fraction(1,10)*(1<<m)
            exact=call('exact_evolution',str(m),lambda:expm(-1j*float(elapsed)*H))
            row=call('round_reference',str(m),lambda:b.reference.reference_round(kernels,(H,elapsed),repetitions=1<<m,state=psi,exact_operator=exact))
            signal=row.pop('raw_signal',None);row['raw_signal_parts']=None if signal is None else [float(signal.real),float(signal.imag)]
            decision=call('round_classification',str(m),lambda:feasibility.classify_round(row,target))
            value=dict(m=m,reference=row,feasibility=decision);io.save(out/f'm{m:02d}.json',value);return m,value
        with threadpool_limits(limits=plan['BLAS_threads_per_worker']):
            with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
                rounds=dict(pool.map(task,range(13)))
        rounds=[rounds[i] for i in range(13)]
        feasibility_row=call('feasibility_summary','all13',lambda:feasibility.summarize([r['feasibility'] for r in rounds],df['sector_error']+df['numerical_guard'],target))
        packet=dict(target_id='integrated:'+df['candidate_sha256'],rounds=rounds,df_bound_Ha=df['sector_error']+df['numerical_guard'],error_evidence=None)
        shell_reports=call('shell_metrics_load','common saved ancilla-only shells',lambda:io.read(root/c['reference_records']['shells']))
        packet['shells']=shell_reports;lanes={}
        for lane in accounting.LANES:
            q=call('signal_qualification',lane,lambda lane=lane:accounting.qualification(packet,lane))
            plans=[call('round_accuracy',lane+str(row['m']),lambda row=row:accounting.assess_round(target,packet,q,row)) for row in rounds]
            energy=call('energy_assessment',lane,lambda:accounting.energy(target,packet,plans))
            shots=call('shots_summary',lane,lambda:None if not energy['same_accuracy'] else sum(2*x['shots'] for x in plans))
            lanes[lane]=dict(qualification=q,rounds=plans,energy=energy,total_shots=shots)
        result=dict(status='PERFORMANCE_REFERENCES_COMPLETE',feasibility=feasibility_row,rounds=rounds,lanes=lanes,
            calibration=None,total_resources=None,actual_CA_achieved=None,whole_law_synthesis=None,implemented_law_error=None,cost_support=None)
        if not lanes['ideal_exact_native']['energy']['same_accuracy']:
            result['status']='PERFORMANCE_INFEASIBLE_NO_NATIVE'
        else:
            sector=dict(generators=np.asarray(G),H=H,full_input_indices=b.stage1.bit_reverse_indices(8)[ctx.sector.basis_indices])
            a=dict(arm=arm,p=p,tail=tail,config=config,dist=dist,sector=sector,ground=psi)
            samples,cal=native(root,c,out,call,a);packet.update(kernel_samples=samples,saved_calibration=cal)
            kernel=call('cost_calibration','new candidate32,unknown support',lambda:accounting.cost_calibration(packet))
            costs=[call('serial_cost',f'{m}{axis}',lambda m=m,axis=axis:m3.serial_cost(packet,kernel,m,axis)) for m in range(13) for axis in ('X','Y')]
            for lane,x in lanes.items():
                totals=call('total_aggregation',lane,lambda x=x:m4.accuracy.aggregate_totals(x['energy'],x['rounds'],costs,cost_scope=accounting.COST_SCOPE))
                x['conditional_totals']=totals
                x['weighted_SE']=call('calibration_uncertainty',lane,lambda x=x,totals=totals:m3.calibration_uncertainty(packet,x['rounds'],totals))
            result.update(status='PERFORMANCE_NATIVE_COMPLETE',calibration=cal)
        def compare():
            # Saved reference results are quoted separately; no old price is used for the new arm.
            refs={k:io.read(root/v) for k,v in c['reference_records'].items() if k!='shells'}
            summary={}
            if 'old_PR' in refs:
                old=refs['old_PR']
                summary['old_PR']=dict(calibration=old['calibration'],lanes=old['lanes'])
                for lane,x in lanes.items():
                    new=x.get('conditional_totals',{}).get('totals')
                    if new is not None and lane in old['lanes']:
                        totals=old['lanes'][lane]['conditional_totals']
                        x['new_over_old_PR']={k:new[k]/totals[k] if totals[k] else None for k in m4.METRICS}
                        x['comparison_kind']='conditional finite-cohort point estimates;no mean/support guarantee'
            if 'strong_D' in refs:
                old=refs['strong_D']['candidates']['proposed']
                summary['strong_D']=dict(cost=old['cost'],status=old['status'],metrics=old['metrics'])
                for x in lanes.values():
                    new=x.get('conditional_totals',{}).get('totals')
                    if new is not None:
                        x['new_over_strong_D_benchmark']={k:new[k]/old['cost']['totals'][k] for k in m4.METRICS}
                summary['strong_D']['comparison_limit']='same CA/time/failure/serial metric/state-preparation policy,but D numerical synthesis versus PR assumed all-law bounds;descriptive benchmark,not guaranteed superiority'
            if 'RC2_native' in refs:
                summary['RC2_native']={k:refs['RC2_native'][k] for k in ('arms','conditional_interrogation_totals','qualification')}
                summary['RC2_native']['comparison_limit']='different finite optimization endpoints;no paired-converged optimizer or guaranteed resource ranking'
            return summary
        result['saved_references']=call('resource_comparison','fixed reference records,qualification distinctions retained',compare)
        io.save(out/'result.json',result);return result
    except BaseException as e:
        call.stop.set();io.save(out/'failure.json',dict(error=repr(e),first_failure=call.first_failure));raise
    finally:
        io.save(out/'STOP.json',dict(failed=call.stop.is_set(),scientific_units=sum(call.counts.values()),call_counts=dict(call.counts),retry=0,resume=0,mandatory_stop=True))
        io.save(out/'resources.json',dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time(),peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
