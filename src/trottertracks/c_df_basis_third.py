"""Saved gamma=0 third DF: existing reference/RPE/native, plus DF holdout.

Import/preflight are passive. Real arrays and old trajectory reanalysis enter
only through the separately permitted scientific runner. No optimizer entry.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path
import importlib.metadata
import math
import resource
import subprocess
import time

from . import c_df_structural_pr as p4
from . import c_df_basis_transfer as transfer
from . import c_df_basis_surrogate_v2 as descriptors

io, old, integrated = p4.io, p4.old, p4.integrated
VERSION = 'saved_gamma0_third_DF_PR_native_transfer_v1'
SCOPE = 'C_DF_BASIS_SWITCH_THIRD_GAMMA0_V1'
ROLE = 'baseline'
FRESH = 'artifacts/research_c_df_basis_third/2026-10-29/run01'
EXPORT = 'artifacts/research_c_pr_aware_df/2026-10-27/run01/baseline/final.npz'
EXPORT_SHA = 'f3a3141059e29f79f39c3f6d3721562abfa7221ad97ce06a118331f6583e1a02'
RECORD_NAMES = ('StageA','StageA_STOP','StageA_SOURCE','StageA_contract','sector','diagnostic',
                'optimizer','fit_STOP','physical_ground','shell_reports','M4','M4_native',
                'M4_STOP','M4_SOURCE','P4','P4_STOP','P4_SOURCE')
TARGET, ALLOWANCES = p4.TARGET, p4.ALLOWANCES
POLICY = dict(p4.POLICY,arm=ROLE,support='general_symmetric_J',
              DF_candidate=transfer.HOLDOUT,PF_reuse='NONE;own gamma0 PF/RTE/ground')
SAMPLING = dict(p4.SAMPLING,master=2026102907,
                cohort='StageA_gamma0_s1_r128_fixed32_v1',axis='GAMMA0_THIRD_DF_NATIVE_ONLY',
                source_key='(master,cohort,SOURCE_digest),baseline,JU_hash,-1,axis,sample,0')
TRAINING_LIMITS = dict(training_rows_load=1,training_identity=1,transfer_training=6)
REFERENCE_LIMITS = dict(p4.REFERENCE_LIMITS,law_descriptors=1)
ACCURACY_LIMITS = dict(p4.ACCURACY_LIMITS)
PRICE_LIMITS = dict(affine_accounting=3,threshold_comparison=4,native_GO=1)
NATIVE_LIMITS = dict(p4.NATIVE_LIMITS,trajectory_features=32)
HOLDOUT_LIMITS = dict(DF_holdout=6,transfer_qualification=1)
STAGES = dict(T=TRAINING_LIMITS,A=REFERENCE_LIMITS,RPE=ACCURACY_LIMITS,
              price=PRICE_LIMITS,B=NATIVE_LIMITS,C=HOLDOUT_LIMITS,finish=dict(result_summary=1))
LIMITS = {k:v for stage in STAGES.values() for k,v in stage.items()}
MAX_CALLS = sum(LIMITS.values())


def preflight(root, path):
    """Hashes/schema/packages only. Does not decode any molecular/trajectory data."""
    from .c_pr_aware_df_stage_b2 import profile_dict
    from .c_pr_aware_df_stage_c import SCENARIOS
    root=Path(root).resolve();c=io.read(path)
    expected=dict(scope=SCOPE,implementation=VERSION,policy=POLICY,target=TARGET,
        allowances=ALLOWANCES,sampling=SAMPLING,compiler=profile_dict(),scenarios=list(SCENARIOS),
        stage_call_limits=STAGES,call_limits=LIMITS,maximum_logical_units=MAX_CALLS,
        fresh_output=FRESH,export=EXPORT,export_sha256=EXPORT_SHA,new_fit=0,retry=0,resume=0,
        transfer_models={k:list(v) for k,v in transfer.MODELS.items()},
        transfer_policy='global_intercept_raw_features_OLS_rcond1e-12_frozen_before_third_prices')
    for k,v in expected.items():
        if c[k]!=v: raise ValueError('THIRD_DF_FIXED_SCOPE:'+k)
    if (set(c['records'])!=set(RECORD_NAMES) or set(c['training_records'])!={'result','STOP','SOURCE'}
            or set(c['training_rows'])!=set(transfer.TRAIN)
            or any(len(paths)!=32 or len(set(paths))!=32 for paths in c['training_rows'].values())):
        raise ValueError('THIRD_DF_SCHEMA')
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']: raise ValueError('SOURCE_MANIFEST')
    required={c['input'],EXPORT,*c['records'].values(),*c['training_records'].values()}
    required.update(p for paths in c['shells'].values() for p in paths.values())
    required.update(p for rows in c['training_rows'].values() for p in rows)
    if not required<=c['saved_inputs'].keys() or c['saved_inputs'][EXPORT]!=EXPORT_SHA:
        raise ValueError('UNPINNED_THIRD_INPUT')
    for group in (io.read(root/c['source_manifest'])['files'],c['saved_inputs']):
        for p,h in group.items():
            f=(root/p).resolve()
            if not f.is_relative_to(root) or io.sha(f)!=h:raise ValueError('SOURCE_INPUT:'+p)
    for package,want in c['packages'].items():
        if importlib.metadata.version(package)!=want:raise ValueError('ENVIRONMENT:'+package)
    if (root/FRESH).exists():raise ValueError('FRESH_THIRD_NO_RETRY_RESUME')
    return c


def training(root,c,out,call):
    rows=call('training_rows_load','saved run02 descriptors/metrics only',lambda:{
        key:[io.read(root/p) for p in paths] for key,paths in c['training_rows'].items()})
    def gate():
        r={k:io.read(root/p) for k,p in c['training_records'].items()}
        if (r['STOP']['failed'] or r['STOP']['scientific_units']!=148
                or r['STOP']['retry'] or r['STOP']['resume'] or not r['STOP']['mandatory_stop']
                or r['SOURCE']['execution_commit']!='0301286493ebf2b95a1786c2bd78756015c1015d'
                or r['result']['qualification']['status']!='SURROGATE_NOT_QUALIFIED'
                or set(rows)!=set(transfer.TRAIN)):
            raise ValueError('FROZEN_TWO_DF_TRAINING_PROVENANCE')
        return dict(status='PASS',training_DFs=2,heldout_prices_accessed=False)
    io.save(out/'identity.json',call('training_identity','no heldout labels/prices',gate))
    models={metric:call('transfer_training',metric,lambda metric=metric:transfer.freeze_training(rows,metric))
            for metric in old.METRICS}
    io.save(out/'frozen_models.json',models)
    # Persist this file before reading the third J/U or native price.
    return models


def evidence(root,c):
    v={k:io.read(root/p) for k,p in c['records'].items()}
    if (v['StageA']['status']!='STAGE_A_PROXY_AND_DF_PASS' or v['StageA_STOP']['failed']
            or v['StageA_STOP']['scientific_units']!=30
            or v['StageA_SOURCE']['execution_commit']!='e1085c95f5c01b68566e17d543ace5832cb94061'
            or not v['optimizer']['success'] or v['fit_STOP']['failed']
            or v['physical_ground']['status']!='PASS'
            or v['sector']['sector_error']+v['sector']['numerical_guard']>TARGET['energy_ha']/8
            or v['M4_STOP']['failed'] or v['M4_STOP']['scientific_units']!=332
            or v['P4_STOP']['failed'] or v['P4_STOP']['scientific_units']!=449
            or v['M4_SOURCE']['execution_commit']!='dcf7ab6da2ea540ae54357b82ac3ea7a02df99ad'
            or v['P4_SOURCE']['execution_commit']!='4ac0e5f96a5ad5009f6273d0fc6ff5657ff80aff'):
        raise ValueError('GAMMA0_SAVED_DF_COMPARATOR_PROVENANCE')
    if (v['StageA_contract']['conditions']['gamma_baseline']!=0
            or v['StageA_contract']['conditions']['rank']!=8
            or v['StageA_contract']['original_input']['sha256']!=c['original_input_sha256']
            or v['M4']['status']!='M4_R128_SHOTS_BREAKEVEN_COMPLETE'
            or v['M4_native']['status']!='NATIVE128_FINITE_COHORT_COMPLETE'
            or v['P4']['status']!='TAIL_P4_PROGRAM_COMPLETE'):
        raise ValueError('GAMMA0_OR_FIXED_COMPARATOR_STATUS')
    return v


def stage_reference(root,c,out,call):
    import numpy as np
    from scipy.linalg import expm
    from threadpoolctl import threadpool_limits
    from . import c_pr_aware_df_stage_b as b,c_pr_aware_df_stage_b_runner as loader
    from . import c1_molecule_extension as molecule,c1_stage3_feasibility as feasibility
    from .c_pr_aware_df_m2 import validate_arrays
    saved=call('evidence_load','same gamma0 J/U;old PF/shots NOT transferred',lambda:evidence(root,c))
    p=call('input_load','original Stage A h/g/scalar',lambda:loader.load(root/c['input'],('h','g','constant')))
    d=call('fit_load','saved gamma0 final J/U only',lambda:loader.load(root/EXPORT,('J','U')))
    call('structure_gate','general symmetric J,rank8,orthogonal U',lambda:validate_arrays(p,d,ROLE))
    arm=call('native_model','original g-kappa/scalar/frame order',lambda:b.prepare_arm(**p,**d,
        role=ROLE,source_identity=EXPORT_SHA,input_identity=c['original_input_sha256'],ld=2))
    ctx=call('sector_context','own sector36 CAR',lambda:molecule.SectorCAR(4))
    G=call('sector_generators','base+8 candidate factors',lambda:ctx.generators(**p,**d))
    reports=[call('factor_representation',str(i),lambda i=i:b.component_check(
        b.candidate_factor(ctx,arm,i),G[i],1e-10,label='gamma0 general-K/raw CAR')) for i in range(9)]
    io.save(out/'factors.json',reports)
    def summed():
        H=sum(G);b.component_check(H,H.conj().T,1e-10,label='gamma0 Hermitian');return H
    H=call('arm_sum','own DF Hamiltonian',summed)
    pair=call('own_ground','own DF eigensolve',lambda:loader.ground(H,out,ROLE))
    ground_gate=call('ground_alias','own eigenpair,registered energy window',lambda:loader.ground_gate(H,pair))
    io.save(out/'ground_gate.json',ground_gate)
    ground=dict(energy=pair[0][0],sector_state=pair[1][:,0])
    tail,config,dist=call('law_prepare','own law h=.1/r128/K0',lambda:b.configure(arm,Fraction(1,10),128,0))
    def tail_gate():
        from .pr_dictionary_validation.prototype import atom_record
        report=b.tail_gate(ctx,arm,G,tail)
        if (not math.isclose(report['lambda_R'],saved['diagnostic']['exact_tail_mass_Ha'],rel_tol=1e-12,abs_tol=1e-14)
                or config.rte_steps!=128 or config.finite_taylor_order!=0
                or any(not x.frame_id.startswith('CDF_baseline:') for x in arm.plan.atoms)):
            raise ValueError('GAMMA0_OWN_TAIL_IDENTITY')
        return dict(report,atoms_catalog=[atom_record(x) for x in arm.plan.atoms])
    tail_info=call('tail_gate','signed atoms+identity+saved mass',tail_gate);io.save(out/'tail.json',tail_info)
    desc,distances=call('law_descriptors','own exact basis keys/q/expected switches',lambda:(
        descriptors.law_features(arm.plan.atoms),descriptors.geometry(arm.plan.atoms,arm.U)))
    io.save(out/'law_descriptors.json',desc)
    target=call('target_validation','same CA/failure/phase13',lambda:old.accuracy.AccuracyTarget(**TARGET))
    if target.horizon()['maximum_round_index_M']!=12:raise ValueError('GAMMA0_HORIZON')
    kernels=call('kernel_reference','own PF+corrected finite RTE',lambda:b.kernels(arm,G,Fraction(1,10),config))
    sector=dict(generators=np.asarray(G),H=H,full_input_indices=b.stage1.bit_reverse_indices(8)[ctx.sector.basis_indices])
    loader.save_arrays(out/'sector_reference.npz',**sector,pf=kernels['pf'],corrected=kernels['corrected'])
    plan=integrated.resource_plan(task_count=13,full_dimension=36);io.save(out/'resource_plan.json',plan)
    def task(m):
        elapsed=Fraction(1,10)*(1<<m)
        exact=call('exact_evolution',f'm{m}',lambda:expm(-1j*float(elapsed)*H))
        row=call('round_reference',f'm{m}',lambda:b.reference.reference_round(kernels,(H,elapsed),
            repetitions=1<<m,state=ground['sector_state'],exact_operator=exact))
        z=row.pop('raw_signal',None);row['raw_signal_parts']=None if z is None else [float(z.real),float(z.imag)]
        decision=call('round_classification',f'm{m}',lambda:feasibility.classify_round(row,target))
        value=dict(m=m,reference=row,feasibility=decision,evidence_kind='binary64_sector_operator_pilot',
                   synthesis_error=None,implemented_law_error=None)
        io.save(out/f'm{m:02d}.json',value);return m,value
    rows={};errors=[]
    with threadpool_limits(limits=plan['BLAS_threads_per_worker']):
        with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
            for f in as_completed([pool.submit(task,m) for m in range(13)]):
                try:m,row=f.result();rows[m]=row
                except BaseException as error:call.stop.set();errors.append(error)
    if errors:raise errors[0]
    rows=[rows[m] for m in range(13)];df=saved['sector']['sector_error']+saved['sector']['numerical_guard']
    summary=call('A_summary','13 sufficient conditions/saved DF reserve',lambda:feasibility.summarize([x['feasibility'] for x in rows],df,target))
    result=dict(**summary,rounds=rows,saved_DF_bound_Ha=df,ground=ground_gate,tail=tail_info,actual_CA_achieved=None)
    io.save(out/'result.json',result)
    packet=dict(arm=ROLE,target_id=transfer.HOLDOUT+':'+EXPORT_SHA,rounds=rows,df_bound_Ha=df,error_evidence=None)
    return dict(result=result,p=p,arm=arm,tail=tail,config=config,distribution=dist,sector=sector,ground=ground,
        saved=saved,packet=packet,target=target,tail_info=tail_info,distances=distances)


# The unchanged accuracy helper needs only own packet/target and saved M4 shot
# values for explicitly named descriptive comparisons; never their PF/ground.
stage_accuracy = p4.stage_accuracy


def targets(saved,lane):
    return dict(M4_r128=saved['M4_native']['lanes'][lane]['conditional_totals'],
                tail_P4=saved['P4']['native']['lanes'][lane]['conditional_totals'])


def stage_price(out,call,a,lanes):
    affine={lane:call('affine_accounting',lane,lambda lane=lane:old.affine_accounting(lanes[lane],a['saved']['shell_reports'])) for lane in lanes}
    comparisons={}
    for lane in ('ideal_exact_native','assumed_raw_signal_budget'):
        for name,cost in targets(a['saved'],lane).items():
            label=lane+':'+name
            comparisons[label]=call('threshold_comparison',label,lambda cost=cost,label=label:p4.threshold(affine[lane],cost,label))
    result=dict(affine=affine,comparisons=comparisons,native_price=None,cost_support=None)
    io.save(out/'result.json',result);return result


def native_go(a,lanes,price):
    # Measure the registered validation cohort even if the break-even price is
    # nonpositive. Never select this cohort using its unseen prices.
    lane=lanes['ideal_exact_native'];W=price['affine']['ideal_exact_native']['W']
    checks=dict(all13=a['result']['status']=='POTENTIALLY_FEASIBLE',
        ground=a['result']['ground']['status']=='PASS',energy=lane['energy']['same_accuracy'],
        shots=type(lane['shots']['total_shots'])is int and lane['shots']['total_shots']>0,
        positive_weight=type(W)is int and W>0)
    return dict(status='GO_FIXED32' if all(checks.values()) else 'NOT_RUN_A_GATE',checks=checks,
                price_selection=False)


def calibrated_result(cal,a,lanes,prices):
    values={}
    for lane in ('ideal_exact_native','assumed_raw_signal_budget'):
        affine=prices['affine'][lane];W=affine['W']
        if W is None:values[lane]=dict(status='UNQUALIFIED',conditional_totals=None);continue
        totals={k:W*cal['sample_mean'][k]+affine['S'][k] for k in old.METRICS}
        comparison={name:{k:dict(ratio=totals[k]/cost[k] if cost[k]>0 else None,
            difference=totals[k]-cost[k]) for k in old.METRICS} for name,cost in targets(a['saved'],lane).items()}
        values[lane]=dict(conditional_totals=totals,comparisons=comparison,
                         weighted_calibration_SE={k:W*cal['sample_mean_standard_error'][k] for k in old.METRICS})
    old_cal=dict(M4_r128=a['saved']['M4_native']['calibration'],tail_P4=a['saved']['P4']['native']['calibration'])
    kernel_comparisons={name:{k:dict(difference=cal['sample_mean'][k]-other['sample_mean'][k],
        ratio=cal['sample_mean'][k]/other['sample_mean'][k] if other['sample_mean'][k]>0 else None)
        for k in old.METRICS} for name,other in old_cal.items()}
    return dict(status='GAMMA0_FIXED32_CONDITIONAL_PILOT_COMPLETE',calibration=cal,lanes=values,
                kernel_comparisons=kernel_comparisons,
                cost_support=None,all_law_synthesis=None,implemented_law_error=None,
                actual_CA_achieved=None,guaranteed_total_resources=None,causal_effect=None)


def execute(root,c,out,call):
    from .c_df_basis_third_native import execute as native
    for name in ('training','reference','accuracy','price','native','holdout'):(out/name).mkdir()
    frozen=training(root,c,out/'training',call)
    a=stage_reference(root,c,out/'reference',call);lanes={};prices=None;cal=None;holdout={};q=None
    if a['result']['status']=='POTENTIALLY_FEASIBLE':
        lanes=stage_accuracy(out/'accuracy',call,a);prices=stage_price(out/'price',call,a,lanes)
        go=call('native_GO','feasibility/shots only;no price-adaptive selection',lambda:native_go(a,lanes,prices))
        io.save(out/'native_GO.json',go)
        if go['status']=='GO_FIXED32':
            cal=native(root,c,out/'native',call,a,lanes,prices)
            for metric in old.METRICS:
                holdout[metric]=call('DF_holdout',metric,lambda metric=metric:transfer.evaluate(frozen[metric],cal['descriptor_rows'],cohort=transfer.HOLDOUT))
            q=call('transfer_qualification','registered A vs mean/B RZ diagnostic only',lambda:transfer.qualification(holdout['rz_count']))
            io.save(out/'holdout/result.json',dict(metrics=holdout,qualification=q))
    result=call('result_summary','registered thirdDF only;mandatory STOP',lambda:dict(
        status='THIRD_DF_FINITE_PILOT_COMPLETE' if cal else 'THIRD_DF_STOP_AFTER_A',
        reference=a['result'],accuracy=lanes,price=prices,native=cal,transfer=q,
        new_fit=0,optimizer=0,retry=0,resume=0,mandatory_stop=True))
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
        io.save(out/'STOP.json',dict(failed=call.stop.is_set(),scientific_units=sum(call.counts.values()),
            call_counts=dict(call.counts),retry=0,resume=0,mandatory_stop=True))
        io.save(out/'resources.json',dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
