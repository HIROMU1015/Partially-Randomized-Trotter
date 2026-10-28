"""Tail-P4 third-arm orchestration; existing PF/RTE/RPE/native engines only.

Imports and preflight are passive. A single fixed program gates a new finite
native cohort on this candidate's own feasibility, shots and break-even.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
import importlib.metadata
import math
from pathlib import Path
import resource
import subprocess
import time

from . import c_pr_aware_df_m4 as old
io,accuracy,integrated=old.io,old.accuracy,old.integrated
VERSION='tail_P4_third_arm_PR_and_gated_fixed32_native_v1'
SCOPE='C_DF_TAIL_P4_PR_PERFORMANCE_V1'
ROLE='tail_p4'
FAMILY='artifacts/research_c_df_structural_pr/2026-10-28'
FRESH=FAMILY+'/run01'
TARGET=old.TARGET
POLICY={**old.POLICY,'arm':ROLE,'PF_reuse':'NONE;newJU ownPF/RTE/ground',
    'support':'head_full_tail_P4','controlled_dimension':512}
ALLOWANCES={**old.ALLOWANCES,'signal_abs':2e-10,'unitarity_abs':1e-10,
    'ground_residual':1e-10,'alias_window_Ha':[-10.,10.],'global_phase_fit':False}
SAMPLING=dict(samples=32,events_per_sample=128,master=2026102806,
    cohort='tail_P4_s1_r128_native_fixed32_v1',axis='TAIL_P4_NATIVE_ONLY',
    PRNG='numpy.random.PCG64',paired=False,adaptive=False,old_cohort_pooled=False,
    source_key='(master,cohort,SOURCE_digest),tail_p4,JU_hash,-1,axis,sample,0')
REFERENCE_LIMITS=dict(evidence_load=1,input_load=1,fit_load=1,structure_gate=1,
    native_model=1,sector_context=1,sector_generators=1,factor_representation=9,
    arm_sum=1,own_ground=1,ground_alias=1,law_prepare=1,tail_gate=1,
    target_validation=1,kernel_reference=1,exact_evolution=13,round_reference=13,
    round_classification=13,A_summary=1)
ACCURACY_LIMITS=dict(signal_qualification=3,round_accuracy=39,energy_assessment=3,shots_summary=3)
PRICE_LIMITS=dict(affine_accounting=3,threshold_comparison=8,native_GO=1)
NATIVE_LIMITS=dict(car_context=1,raw_factor=9,full_sector_gate=1,reference_context=1,
    full_law_gate=1,gaussian_build=9,relative_gaussian=4,shell_load=6,shell_assessment=3,
    trajectory_sample=32,kernel_build=32,trajectory_reference=32,raw_operator_gate=32,
    compile=32,compiled_operator_gate=32,adjoint_pair_gate=32,XY_gate=32,kernel_metrics=32,
    calibration_summary=1,native_result=1)
LIMITS={}
for group in (REFERENCE_LIMITS,ACCURACY_LIMITS,PRICE_LIMITS,NATIVE_LIMITS,dict(result_summary=1)):
    for k,v in group.items():LIMITS[k]=LIMITS.get(k,0)+v
MAX_CALLS=sum(LIMITS.values())


def preflight(root,path):
    from .c_pr_aware_df_stage_c import SCENARIOS
    from .c_pr_aware_df_stage_b2 import profile_dict
    root=Path(root).resolve();c=io.read(path)
    for k,v in dict(scope=SCOPE,implementation=VERSION,policy=POLICY,target=TARGET,
            allowances=ALLOWANCES,sampling=SAMPLING,scenarios=list(SCENARIOS),
            compiler=profile_dict(),call_limits=LIMITS,maximum_logical_units=MAX_CALLS,
            fresh_output=FRESH,new_fit=0,retry=0,resume=0).items():
        if c[k]!=v:raise ValueError('P4_CONTRACT:'+k)
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('P4_SOURCE')
    for files in (io.read(root/c['source_manifest'])['files'],c['saved_inputs']):
        for p,h in files.items():
            f=(root/p).resolve()
            if not f.is_relative_to(root) or io.sha(f)!=h:raise ValueError('P4_SOURCE_INPUT:'+p)
    for p,v in c['packages'].items():
        if importlib.metadata.version(p)!=v:raise ValueError('P4_ENV:'+p)
    if (root/FRESH).exists():raise ValueError('P4_FRESH_NO_RETRY_RESUME')
    return c


def prepare_arm(h,g,constant,J,U,*,source_identity,input_identity):
    """Same B1 construction, separate third role/NativeModel/basis/atom IDs.

    B1's paired-role gate stays unchanged. Only its lower-level general-K,
    spin-adapted one-body basis, NativeModel and split functions are reused.
    """
    import numpy as np
    from . import c_pr_aware_df_stage_b as b
    from .c1_h4_strong_baseline import spin_adapted_eigh
    from trotterlib.df_trotter.ops import one_body_diagonal_primitives
    from .c_pr_aware_df_m2 import validate_arrays
    validate_arrays(dict(h=h,g=g,constant=np.asarray(constant)),dict(J=J,U=U),ROLE)
    model=b.general.from_ffsim_number_arrays(one_body_tensor=b.stage1.corrected_kappa(h,g),
        diag_coulomb_mats=J,orbital_rotations=U,constant=float(constant),z_representation=False,
        frame_ids=[f'F{i}' for i in range(8)],group_ids=[f'G{i}' for i in range(8)],
        source_identity=source_identity,input_identity=input_identity,
        one_body_role='ffsim_corrected_kappa',enable_general_k=True)
    u,eps,_=spin_adapted_eigh(model.one_body_tensor)
    blocks=[b.native.NativeBlock('one_body',b.native._rotation(u),one_body_diagonal_primitives(eps,1.))]
    blocks.extend(b.native.NativeBlock(f.frame_id,f.orbital_rotation,b.general.diagonal_primitives(f,1.)) for f in model.frames)
    nm=b.native.NativeModel('CDF_'+ROLE,tuple(blocks),float(constant),source_identity,input_identity)
    return b.Arm(ROLE,nm,model,b.connection.split_native_model(nm,2),np.asarray(J),np.asarray(U))


def evidence(root,c):
    v={k:io.read(Path(root)/p) for k,p in c['records'].items()}
    p,stop=v['P4'],v['P4_STOP'];row=v['P4_sector']
    if (p['status']!='DF_ELIGIBLE_NOT_PERFORMANCE_GO' or not p['optimizer_success']
            or not p['final_DF_budget_PASS'] or stop['failed'] or stop['scientific_units']!=21
            or stop['retry'] or stop['resume'] or v['P4_SOURCE']['execution_commit']!=c['P4_scientific_SOURCE']
            or row['sector_error']+row['numerical_guard']>TARGET['energy_ha']/8):raise ValueError('P4_DF_PROVENANCE')
    if (v['physical_ground']['status']!='PASS' or v['physical_ground']['alias_margin']<=0
            or v['M4_STOP']['failed'] or v['M4_STOP']['scientific_units']!=98
            or v['native_STOP']['failed'] or v['native_STOP']['scientific_units']!=332
            or v['M4']['status']!='M4_R128_SHOTS_BREAKEVEN_COMPLETE'
            or v['native']['status']!='NATIVE128_FINITE_COHORT_COMPLETE'):
        raise ValueError('P4_COMPARATOR_PROVENANCE')
    return v


def tail_record(ctx,arm,G,tail,config,distribution,saved_mass):
    from . import c_pr_aware_df_stage_b as b
    from .pr_dictionary_validation.prototype import atom_record
    report=b.tail_gate(ctx,arm,G,tail)
    if not math.isclose(report['lambda_R'],saved_mass,rel_tol=1e-12,abs_tol=1e-14):raise ValueError('P4_SAVED_MASS')
    if len(arm.plan.atoms)!=144 or any(not x.frame_id.startswith('CDF_tail_p4:') for x in arm.plan.atoms):raise ValueError('P4_144_ATOM_ROLE')
    if config.rte_steps!=128 or config.finite_taylor_order!=0:raise ValueError('P4_RTE_SCOPE')
    return dict(report,atoms_catalog=[atom_record(x) for x in arm.plan.atoms],
        components=[dict(component_id=x.component_id,probability=x.probability,
            coefficient_abs=x.coefficient_abs,coefficient_sign=x.coefficient_sign,
            basis_id=x.basis_id,df_fragment_id=x.df_fragment_id) for x in tail.components],
        configuration=dict(h=config.evolution_time,r=128,K=0,tau=config.dimensionless_step_time),
        finite_distribution=dict(orders=distribution.orders,order_probabilities=distribution.order_probabilities))


def stage_reference(root,c,out,call):
    import numpy as np
    from scipy.linalg import expm
    from threadpoolctl import threadpool_limits
    from . import c_pr_aware_df_stage_b as b,c_pr_aware_df_stage_b_runner as loader
    from . import c1_molecule_extension as molecule,c1_stage3_feasibility as feasibility
    from .c_df_structural_support import Support
    saved=call('evidence_load','P4 DF and fixed saved comparators',lambda:evidence(root,c))
    p=call('input_load','same original h/g/scalar',lambda:loader.load(root/c['input'],('h','g','constant')))
    d=call('fit_load','immutable tail-P4 final J/U',lambda:loader.load(root/c['export'],('J','U')))
    io.save(out/'structure.json',call('structure_gate','fixed head-full/tail-P4',lambda:Support().assess(d['J'])))
    arm=call('native_model','third role,original kappa/scalar/frame order',lambda:prepare_arm(**p,**d,
        source_identity=c['saved_inputs'][c['export']],input_identity=c['original_input_sha256']))
    ctx=call('sector_context','sector36 raw CAR',lambda:molecule.SectorCAR(4))
    G=call('sector_generators','new candidate base+8 frames',lambda:ctx.generators(**p,**d))
    reports=[call('factor_representation',str(i),lambda i=i:b.component_check(
        b.candidate_factor(ctx,arm,i),G[i],1e-10,label='tail_p4 general-K/rawCAR factor')) for i in range(9)]
    io.save(out/'factors.json',reports)
    def summed():
        H=sum(G);b.component_check(H,H.conj().T,1e-10,label='newDF Hermitian');return H
    H=call('arm_sum','new H,not old Proposed',summed)
    pair=call('own_ground','new own-DF eigensolve',lambda:loader.ground(H,out,ROLE))
    ground_gate=call('ground_alias','new eigenpair and saved physical alias',lambda:loader.ground_gate(H,pair))
    io.save(out/'ground_gate.json',ground_gate)
    ground=dict(energy=pair[0][0],sector_state=pair[1][:,0])
    tail,config,dist=call('law_prepare','new144 signed atoms,r128/K0',lambda:b.configure(arm,Fraction(1,10),128,0))
    tail_info=call('tail_gate','new atom/probability/sign/identity/mass',lambda:tail_record(ctx,arm,G,tail,config,dist,saved['P4']['final_diagnostic']['exact_tail_mass_Ha']))
    io.save(out/'tail.json',tail_info)
    target=call('target_validation','same CA/RPE and13 rounds',lambda:accuracy.AccuracyTarget(**TARGET))
    if target.horizon()['maximum_round_index_M']!=12:raise ValueError('P4_HORIZON')
    kernels=call('kernel_reference','newPF+correctedfiniteRTE',lambda:b.kernels(arm,G,Fraction(1,10),config))
    sector=dict(generators=np.asarray(G),H=H,full_input_indices=b.stage1.bit_reverse_indices(8)[ctx.sector.basis_indices])
    loader.save_arrays(out/'sector_reference.npz',**sector,pf=kernels['pf'],corrected=kernels['corrected'])
    plan=integrated.resource_plan(task_count=13,full_dimension=36);io.save(out/'resource_plan.json',plan)
    def task(m):
        elapsed=Fraction(1,10)*(1<<m)
        exact=call('exact_evolution',f'm{m}',lambda:expm(-1j*float(elapsed)*H))
        row=call('round_reference',f'm{m}',lambda:b.reference.reference_round(kernels,(H,elapsed),repetitions=1<<m,state=ground['sector_state'],exact_operator=exact))
        z=row.pop('raw_signal',None);row['raw_signal_parts']=None if z is None else [float(z.real),float(z.imag)]
        decision=call('round_classification',f'm{m}',lambda:feasibility.classify_round(row,target))
        value=dict(m=m,reference=row,feasibility=decision,evidence_kind='binary64_sector_operator_pilot',synthesis_error=None,implemented_law_error=None)
        io.save(out/f'm{m:02d}.json',value);return m,value
    rows={};errors=[]
    with threadpool_limits(limits=plan['BLAS_threads_per_worker']):
        with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
            for f in as_completed([pool.submit(task,m) for m in range(13)]):
                try:m,row=f.result();rows[m]=row
                except BaseException as error:call.stop.set();errors.append(error)
    if errors:raise errors[0]
    rows=[rows[m] for m in range(13)];df=saved['P4_sector']['sector_error']+saved['P4_sector']['numerical_guard']
    summary=call('A_summary','all13 sufficient conditions and savedDF reserve',lambda:feasibility.summarize([x['feasibility'] for x in rows],df,target))
    result=dict(**summary,rounds=rows,saved_DF_bound_Ha=df,ground=ground_gate,tail=tail_info,
        actual_CA_achieved=None,all_law_synthesis=None,implemented_law_error=None)
    io.save(out/'result.json',result)
    packet=dict(arm=ROLE,target_id=ROLE+':'+arm.model.source_identity,rounds=rows,df_bound_Ha=df,error_evidence=None)
    return dict(result=result,p=p,arm=arm,tail=tail,config=config,distribution=dist,sector=sector,
        ground=ground,saved=saved,packet=packet,target=target,tail_info=tail_info)


def stage_accuracy(out,call,a):
    from . import c_pr_aware_df_stage_c as rpe
    lanes={}
    for lane in rpe.LANES:
        q=call('signal_qualification',lane,lambda lane=lane:rpe.qualification(a['packet'],lane))
        plans=[call('round_accuracy',lane+f':m{x["m"]}',lambda x=x:rpe.assess_round(a['target'],a['packet'],q,x)) for x in a['packet']['rounds']]
        energy=call('energy_assessment',lane,lambda:rpe.energy(a['target'],a['packet'],plans))
        oldplans=None if lane=='evidence' else a['saved']['M4']['B'][lane]['rounds']
        summary=call('shots_summary',lane,lambda:old.shots_summary(plans,energy,oldplans))
        # Legacy helper's old_M3 field names now explicitly refer to saved M4.
        value=dict(lane=lane,qualification=q,rounds=plans,energy=energy,shots=summary,
            legacy_old_M3_fields_reference='saved M4 r128,not M3',actual_CA_achieved=None)
        io.save(out/(lane+'.json'),value);lanes[lane]=value
    return lanes


def targets(saved,lane):
    return dict(M4_r128=saved['native']['lanes'][lane]['conditional_totals'],
        M3_r64=saved['M3']['C']['lanes'][lane]['resources']['totals'],
        old_PR_s4=saved['C1']['assumed_scenarios'][lane]['proposed']['resources']['totals'],
        M2_old_JU_deterministic=saved['M2']['candidates']['proposed']['cost']['totals'])


def threshold(affine,cost,label):
    if affine['W'] is None:return dict(status='UNQUALIFIED',thresholds=None)
    W=affine['W'];rows={}
    if type(W)is not int or W<=0:raise ValueError('P4_POSITIVE_WEIGHT')
    for k in old.METRICS:
        accuracy._finite(cost[k],'target '+k);mu=(cost[k]-affine['S'][k])/W
        if not math.isfinite(mu):raise ValueError('P4_NONFINITE_THRESHOLD')
        rows[k]=dict(mu_break_even=mu,positive=mu>0,equality_means_tie=True)
    return dict(status='CONDITIONAL_POINT_BREAKEVEN',label=label,thresholds=rows,
        strict_condition='nonnegative mean price < threshold',actual_new_price=None)


def native_go(a,lanes,prices):
    principal=lanes['ideal_exact_native'];test=prices['comparisons']['ideal_exact_native:M4_r128']['thresholds']
    mu=None if test is None else test['rz_count']['mu_break_even']
    checks=dict(correctness=True,ground_alias=a['result']['ground']['status']=='PASS',
        all13=a['result']['status']=='POTENTIALLY_FEASIBLE',
        shots_energy=principal['energy']['same_accuracy'] and principal['shots']['total_shots'] is not None,
        positive_finite_RZ=mu is not None and math.isfinite(mu) and mu>0)
    return dict(status='GO_FIXED32' if all(checks.values()) else 'NOT_RUN_A_GATE',checks=checks,RZ_break_even=mu)


def stage_price(out,call,a,lanes):
    affine={lane:call('affine_accounting',lane,lambda lane=lane:old.affine_accounting(lanes[lane],a['saved']['shell_reports'])) for lane in lanes}
    comparisons={}
    for lane in ('ideal_exact_native','assumed_raw_signal_budget'):
        for name,cost in targets(a['saved'],lane).items():
            label=lane+':'+name
            comparisons[label]=call('threshold_comparison',label,lambda cost=cost,label=label:threshold(affine[lane],cost,label))
    result=dict(affine=affine,comparisons=comparisons,native_price=None,
        legacy_old_M3_fields_reference='saved M4 r128',cost_support=None)
    io.save(out/'result.json',result)
    return result


def calibrated_result(cal,a,lanes,prices):
    values={}
    for lane in ('ideal_exact_native','assumed_raw_signal_budget'):
        affine=prices['affine'][lane];W=affine['W']
        if W is None:
            values[lane]=dict(status='UNQUALIFIED',conditional_totals=None);continue
        totals={k:W*cal['sample_mean'][k]+affine['S'][k] for k in old.METRICS}
        comparison={name:{k:dict(ratio=totals[k]/cost[k] if cost[k]>0 else None,
            difference=totals[k]-cost[k],mean_below_break_even=cal['sample_mean'][k]<prices['comparisons'][lane+':'+name]['thresholds'][k]['mu_break_even']) for k in old.METRICS} for name,cost in targets(a['saved'],lane).items()}
        values[lane]=dict(conditional_totals=totals,weighted_calibration_SE={k:W*cal['sample_mean_standard_error'][k] for k in old.METRICS},
            comparisons=comparison,independent_round_calibrations=1,actual_CA_achieved=None,guaranteed_total_resources=None)
    oldcal=a['saved']['native']['calibration']
    return dict(status='P4_FIXED32_CONDITIONAL_COMPARISON_COMPLETE',calibration=cal,lanes=values,
        kernel_comparison_M4={k:dict(ratio=cal['sample_mean'][k]/oldcal['sample_mean'][k],difference=cal['sample_mean'][k]-oldcal['sample_mean'][k]) for k in old.METRICS},
        cost_support=None,all_law_synthesis=None,implemented_law_error=None,actual_CA_achieved=None,ideal_law_mean_advantage=None)


def execute(root,c,out,call):
    from .c_df_structural_pr_native import execute as native
    for name in ('reference','accuracy','price','native'):(out/name).mkdir()
    a=stage_reference(root,c,out/'reference',call);lanes={};price=None;native_result=None
    if a['result']['status']=='POTENTIALLY_FEASIBLE':
        lanes=stage_accuracy(out/'accuracy',call,a)
        price=stage_price(out/'price',call,a,lanes)
        go=call('native_GO','fixed6 criteria,no price-based adaptive stopping',lambda:native_go(a,lanes,price))
        io.save(out/'native_GO.json',go)
        if go['status']=='GO_FIXED32':native_result=native(root,c,out/'native',call,a,lanes,price)
    else:
        go=dict(status='NOT_RUN_A_FEASIBILITY',reason=a['result']['reason']);io.save(out/'native_GO.json',go)
    result=call('result_summary','fixed P4 program complete;mandatory STOP',lambda:dict(
        status='TAIL_P4_PROGRAM_COMPLETE' if native_result else 'TAIL_P4_STOP_AFTER_A',
        reference=a['result'],accuracy=lanes,price=price,native=native_result,
        stages=dict(A=a['result']['status'],B=go['status']),new_fit=0,retry=0,resume=0,quantum_shots=0,mandatory_stop=True))
    io.save(out/'result.json',result);return result


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
        io.save(out/'STOP.json',dict(failed=call.stop.is_set(),scientific_units=sum(call.counts.values()),call_counts=dict(call.counts),retry=0,resume=0,mandatory_stop=True))
        io.save(out/'resources.json',dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
