"""One saved CDF7 candidate: existing CAR/PF/RTE/RPE/break-even engines.

No fit, sampling, circuit construction, compilation or cost calibration entry.
Default preflight checks bytes/metadata/packages only. Real molecular execution
requires separate user permission; the contract is scientific scope, not proof
of permission. Historical rank8 and one-fit SOURCE/results remain untouched.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from fractions import Fraction
from pathlib import Path
import importlib.metadata
import math,resource,subprocess,time

from . import c_df_structural_pr as old
from . import c_cdf7_onefit as fit
from .c_df_basis_third_v2 import with_components

io,integrated=old.io,old.integrated
VERSION='saved_CDF7_reference_RPE_price_no_native_v1'
SCOPE='C_H4_CDF7_PR_REFERENCE_RPE_PRICE_V1'
FRESH='artifacts/research_c_cdf7_PR/2026-10-30/run01'
EXPORT='artifacts/research_c_cdf7_onefit/2026-10-30/run01/fit/final.npz'
EXPORT_SHA='4e4939d1380386ebb98c3b39a13954cbfc3148e35fb8e2cfd3cf536bd0b57ff4'
FIT_SOURCE='c13d6f82a22d497b3334eaaa40d8430a5010f70f'
TARGET,ALLOWANCES=old.TARGET,old.ALLOWANCES
POLICY=dict(atoms=4,distance_A=1.,basis='STO-3G',rank=7,Nalpha=2,Nbeta=2,
    sector_dimension=36,frame_order=list(range(7)),tail_frames=list(range(2,7)),ld=2,
    PF='partial ordered symmetric S2',delta='1/10',h='1/10',s=1,r=128,K=0,
    rounds=list(range(13)),representation='number_general_symmetric_J',
    U='real_orthogonal',original_g_kappa=True,scalar='nuclear once',
    oracle='own_DF_ground_state_oracle',state_preparation='EXCLUDED',
    law='ideal_finite_RTE_IID',DF_error_recomputed=False)
REFERENCE=dict(old.REFERENCE_LIMITS,factor_representation=8)
RPE=dict(old.ACCURACY_LIMITS)
PRICE=dict(affine_accounting=3,threshold_comparison=4,native_GO=1)
STAGES=dict(reference=REFERENCE,RPE=RPE,price=PRICE,finish=dict(result_summary=1))
LIMITS={k:v for group in STAGES.values() for k,v in group.items()}
MAX_CALLS=sum(LIMITS.values())
RECORDS=('fit_result','fit_STOP','fit_SOURCE','fit_contract','fit_sector','fit_optimizer',
    'physical_ground','physical_contract','shell_reports','M4','M4_native','M4_STOP','M4_SOURCE',
    'M4_contract','CDF8_native','CDF8_STOP','CDF8_SOURCE','CDF8_contract')
COMPARATORS=('CDF8_gamma0','M4_r128')


def preflight(root,path):
    from .c_pr_aware_df_stage_c import SCENARIOS
    root=Path(root).resolve();c=io.read(path)
    expected=dict(scope=SCOPE,implementation=VERSION,policy=POLICY,target=TARGET,
        allowances=ALLOWANCES,stage_call_limits=STAGES,call_limits=LIMITS,maximum_logical_units=119,
        export=EXPORT,export_sha256=EXPORT_SHA,fit_SOURCE=FIT_SOURCE,fresh_output=FRESH,
        comparators=list(COMPARATORS),scenarios=list(SCENARIOS),new_fit=0,optimizer=0,
        retry=0,resume=0,DF_error_recalculation=False,native_build_compile=0,sampling=0,
        actual_CDF7_native_price=None,cost_support=None)
    for k,v in expected.items():
        if c[k]!=v:raise ValueError('CDF7_PR_SCOPE:'+k)
    if set(c['records'])!=set(RECORDS):raise ValueError('EVIDENCE_RECORD_SCHEMA')
    required={c['input'],EXPORT,*c['records'].values()}
    if not required<=c['saved_inputs'].keys() or c['saved_inputs'][EXPORT]!=EXPORT_SHA:
        raise ValueError('UNPINNED_SAVED_INPUT')
    if c['original_input_sha256']!=fit.io.P_H4_SHA256:raise ValueError('ORIGINAL_HAMILTONIAN_IDENTITY')
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    for group in (io.read(root/c['source_manifest'])['files'],c['saved_inputs']):
        for p,h in group.items():
            f=(root/p).resolve()
            if not f.is_relative_to(root) or io.sha(f)!=h:raise ValueError('SOURCE_OR_INPUT:'+p)
    for p,v in c['packages'].items():
        if importlib.metadata.version(p)!=v:raise ValueError('ENVIRONMENT:'+p)
    if (root/FRESH).exists():raise ValueError('FRESH_CDF7_NO_RETRY_RESUME')
    return c


def evidence(root,c):
    v={k:io.read(Path(root)/p) for k,p in c['records'].items()}
    r,s=v['fit_result'],v['fit_STOP'];p=v['fit_contract'];row=v['fit_sector']
    if (r['status']!='CDF7_DF_ELIGIBLE' or not r['optimizer_success'] or r['DF_budget_status']!='PASS'
            or s['failed'] or s['scientific_units']!=25 or s['retry'] or s['resume'] or not s['mandatory_stop']
            or v['fit_SOURCE']['execution_commit']!=FIT_SOURCE or not v['fit_optimizer']['success']
            or p['conditions']!=fit.CONDITIONS or p['original_input_sha256']!=c['original_input_sha256']
            or row['actual_rank']!=7 or row['guard_policy']!=fit.molecule.GUARD_POLICY):
        raise ValueError('CDF7_SAVED_FIT_PROVENANCE')
    if any(row[k]!=r['sector']['final'][k] for k in ('actual_rank','sector_error','numerical_guard','guard_policy')):
        raise ValueError('CDF7_SAVED_SECTOR_IDENTITY')
    bound=row['sector_error']+row['numerical_guard']
    if not math.isfinite(bound) or not 0<=bound<=TARGET['energy_ha']/8:raise ValueError('SAVED_DF_RESERVE')
    mass=r['tail_auxiliary']['lambda_R']
    if not math.isfinite(mass) or mass<=0:raise ValueError('REGISTERED_NONEMPTY_TAIL')
    if (v['physical_ground']['status']!='PASS' or v['physical_ground']['alias_margin']<=0
            or v['physical_contract']['original_input']['sha256']!=c['original_input_sha256']):
        raise ValueError('SAVED_PHYSICAL_GROUND_IDENTITY')
    for tag,units,source,status in [
        ('M4',332,'dcf7ab6da2ea540ae54357b82ac3ea7a02df99ad','NATIVE128_FINITE_COHORT_COMPLETE'),
        ('CDF8',493,'30436b142d7fa79c841b5977d1b4c0fbfbd1bb4c','GAMMA0_FIXED32_CONDITIONAL_PILOT_COMPLETE')]:
        stop=v[tag+'_STOP'];contract=v[tag+'_contract']
        if (stop['failed'] or stop['scientific_units']!=units or stop['retry'] or stop['resume']
                or v[tag+'_SOURCE']['execution_commit']!=source or v[tag+'_native']['status']!=status
                or contract['original_input_sha256']!=c['original_input_sha256']):
            raise ValueError('SAVED_COMPARATOR_IDENTITY:'+tag)
        q=contract['policy']
        if (q['rank']!=8 or contract['target']!=TARGET
                or any(q[k]!=POLICY[k] for k in ('ld','delta','h','s','r','K','rounds','Nalpha','Nbeta','sector_dimension'))):
            raise ValueError('COMPARATOR_SCIENTIFIC_CONDITIONS:'+tag)
    if (v['M4']['status']!='M4_R128_SHOTS_BREAKEVEN_COMPLETE'
            or set(v['shell_reports'])!={'head','X','Y'}):raise ValueError('SAVED_RPE_SHELL_SCHEMA')
    return v


def validate_arrays(packet,d):
    import numpy as np
    fit.structure(d['J'],d['U'])
    if (set(packet)!= {'h','g','constant'} or packet['h'].shape!=(4,4)
            or packet['g'].shape!=(4,)*4 or packet['constant'].shape!=()
            or any(a.dtype!=np.float64 or not np.isfinite(a).all() for a in packet.values())):
        raise ValueError('ORIGINAL_PACKET_SHAPE_DTYPE_FINITE')
    return dict(status='PASS',rank=7,frame_order=list(range(7)),ld=2)


def stage_reference(root,c,out,call):
    import numpy as np
    from scipy.linalg import expm
    from threadpoolctl import threadpool_limits
    from . import c_pr_aware_df_stage_b as b,c_pr_aware_df_stage_b_runner as loader
    from . import c1_molecule_extension as molecule,c1_stage3_feasibility as feasible
    from .pr_dictionary_validation.prototype import atom_record
    root,out=Path(root),Path(out)
    saved=call('evidence_load','saved CDF7 DF;no old PF transferred',lambda:evidence(root,c))
    p=call('input_load','same saved original h/g/scalar',lambda:loader.load(root/c['input'],('h','g','constant')))
    d=call('fit_load','qualified seven-frame export only',lambda:loader.load(root/c['export'],('J','U')))
    io.save(out/'structure.json',call('structure_gate','actual seven frames',lambda:validate_arrays(p,d)))
    arm=call('native_model','general-K data only;no circuit',lambda:b.prepare_arm(**p,**d,role='baseline',
        source_identity=c['export_sha256'],input_identity=c['original_input_sha256'],ld=2))
    ctx=call('sector_context','own sector36',lambda:molecule.SectorCAR(4))
    G=call('sector_generators','base plus seven raw factors',lambda:ctx.generators(**p,**d))
    if ctx.d!=36 or len(G)!=8:raise ValueError('EIGHT_FACTORS_SECTOR36')
    factors=[call('factor_representation',str(i),lambda i=i:b.component_check(
        b.candidate_factor(ctx,arm,i),G[i],1e-10,label='CDF7 general-K/raw CAR')) for i in range(8)]
    io.save(out/'factors.json',factors)
    def summed():
        H=sum(G);b.component_check(H,H.conj().T,1e-10,label='CDF7 Hermitian');return H
    H=call('arm_sum','own DF Hamiltonian',summed)
    pair=call('own_ground','new own-DF oracle eigenpair',lambda:loader.ground(H,out,'CDF7'))
    gate=call('ground_alias','own eigenpair and declared branch',lambda:loader.ground_gate(H,pair))
    io.save(out/'ground_gate.json',gate)
    ground=dict(energy=pair[0][0],sector_state=pair[1][:,0])
    tail,config,dist=call('law_prepare','own h.1/r128/K0 law',lambda:b.configure(arm,Fraction(1,10),128,0))
    def check_tail():
        if config is None or config.rte_steps!=128 or config.finite_taylor_order!=0:
            raise ValueError('CDF7_REGISTERED_RTE_SCOPE')
        report=b.tail_gate(ctx,arm,G,tail)
        if (not math.isclose(report['lambda_R'],saved['fit_result']['tail_auxiliary']['lambda_R'],rel_tol=1e-12,abs_tol=1e-14)
                or any(a.frame_id not in {'CDF_baseline:F'+str(i) for i in range(2,7)} for a in arm.plan.atoms)):
            raise ValueError('CDF7_TAIL_FRAME_OR_MASS')
        return with_components(dict(report,atoms_catalog=[atom_record(x) for x in arm.plan.atoms]),tail)
    info=call('tail_gate','signed atoms,identity,mass,component records',check_tail);io.save(out/'tail.json',info)
    target=call('target_validation','same CA/phase/failure horizon',lambda:old.accuracy.AccuracyTarget(**TARGET))
    if target.horizon()['maximum_round_index_M']!=12:raise ValueError('CDF7_HORIZON')
    kernels=call('kernel_reference','own partial PF and corrected RTE',lambda:b.kernels(arm,G,Fraction(1,10),config))
    loader.save_arrays(out/'sector_reference.npz',generators=np.asarray(G),H=H,
        full_input_indices=b.stage1.bit_reverse_indices(8)[ctx.sector.basis_indices],pf=kernels['pf'],corrected=kernels['corrected'])
    plan=integrated.resource_plan(task_count=13,full_dimension=36);io.save(out/'resource_plan.json',plan)
    def task(m):
        elapsed=Fraction(1,10)*(1<<m)
        exact=call('exact_evolution',str(m),lambda:expm(-1j*float(elapsed)*H))
        row=call('round_reference',str(m),lambda:b.reference.reference_round(kernels,(H,elapsed),
            repetitions=1<<m,state=ground['sector_state'],exact_operator=exact))
        z=row.pop('raw_signal',None);row['raw_signal_parts']=None if z is None else [float(z.real),float(z.imag)]
        decision=call('round_classification',str(m),lambda:feasible.classify_round(row,target))
        value=dict(m=m,reference=row,feasibility=decision,evidence_kind='binary64_sector_operator_pilot',
            synthesis_error=None,implemented_law_error=None)
        io.save(out/f'm{m:02d}.json',value);return m,value
    rows={};errors=[]
    with threadpool_limits(limits=plan['BLAS_threads_per_worker']):
        with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
            for f in as_completed([pool.submit(task,m) for m in range(13)]):
                try:m,v=f.result();rows[m]=v
                except BaseException as e:call.stop.set();errors.append(e)
    if errors:raise errors[0]
    rows=[rows[m] for m in range(13)];df=saved['fit_sector']['sector_error']+saved['fit_sector']['numerical_guard']
    summary=call('A_summary','all thirteen sufficient conditions',lambda:feasible.summarize([x['feasibility'] for x in rows],df,target))
    result=dict(**summary,rounds=rows,saved_DF_bound_Ha=df,ground=gate,tail=info,actual_CA_achieved=None)
    io.save(out/'result.json',result)
    packet=dict(arm='CDF7',target_id='CDF7:'+c['export_sha256'],rounds=rows,df_bound_Ha=df,error_evidence=None)
    return dict(result=result,packet=packet,target=target,saved=saved,arm=arm,tail=tail,tail_info=info,
        generators=G,context=ctx)


stage_accuracy=old.stage_accuracy


def stage_price(out,call,a,lanes):
    affine={lane:call('affine_accounting',lane,lambda lane=lane:old.old.affine_accounting(
        lanes[lane],a['saved']['shell_reports'])) for lane in lanes}
    comparisons={}
    for lane in ('ideal_exact_native','assumed_raw_signal_budget'):
        costs=dict(CDF8_gamma0=a['saved']['CDF8_native']['lanes'][lane]['conditional_totals'],
                   M4_r128=a['saved']['M4_native']['lanes'][lane]['conditional_totals'])
        for name,cost in costs.items():
            label=lane+':'+name
            comparisons[label]=call('threshold_comparison',label,lambda cost=cost,label=label:old.threshold(affine[lane],cost,label))
    result=dict(affine=affine,comparisons=comparisons,native_price=None,cost_support=None,actual_CA_achieved=None)
    io.save(Path(out)/'result.json',result);return result


def native_recommendation(a,lanes,price):
    lane=lanes['ideal_exact_native'];W=price['affine']['ideal_exact_native']['W']
    positive={}
    for name in COMPARATORS:
        rows=price['comparisons']['ideal_exact_native:'+name]['thresholds']
        mu=None if rows is None else rows['rz_count']['mu_break_even']
        positive[name]=mu is not None and math.isfinite(mu) and mu>0
    checks=dict(all13=a['result']['status']=='POTENTIALLY_FEASIBLE',ground=a['result']['ground']['status']=='PASS',
        conditional_energy=lane['energy']['same_accuracy'],
        shots=type(lane['shots']['total_shots'])is int and lane['shots']['total_shots']>0,
        positive_W=type(W)is int and W>0,positive_RZ_both=all(positive.values()))
    return dict(status='ELIGIBLE_FOR_SEPARATE_NATIVE_PROPOSAL' if all(checks.values()) else 'NOT_RECOMMENDED_UNDER_REGISTERED_CONDITIONS',
        checks=checks,positive_RZ_targets=positive,native_execution_authorized=False,
        native_price=None,resource_superiority=None)


def execute(root,c,out,call):
    for name in ('reference','accuracy','price'):(out/name).mkdir()
    a=stage_reference(root,c,out/'reference',call);lanes={};price=None
    recommendation=dict(status='NOT_RUN_PF_RTE_SUFFICIENT_BOUND',native_execution_authorized=False)
    if a['result']['status']=='POTENTIALLY_FEASIBLE':
        lanes=stage_accuracy(out/'accuracy',call,a);price=stage_price(out/'price',call,a,lanes)
        recommendation=call('native_GO','recommendation only;never executes native',lambda:native_recommendation(a,lanes,price))
    io.save(out/'native_recommendation.json',recommendation)
    result=call('result_summary','complete registered scope and stop',lambda:dict(
        status='CDF7_PR_REFERENCE_RPE_PRICE_COMPLETE' if lanes else 'CDF7_PR_STOP_AFTER_REFERENCE',
        reference=a['result'],accuracy=lanes,price=price,native_recommendation=recommendation,
        new_fit=0,optimizer=0,DF_error_recomputed=False,native_build_compile=0,sampling=0,
        actual_CA_achieved=None,native_price=None,total_resources=None,retry=0,resume=0,mandatory_stop=True))
    io.save(out/'result.json',result);return result


def run(root,path):
    root=Path(root).resolve();c=preflight(root,path);out=root/FRESH;out.mkdir(parents=True)
    call=integrated.Calls(out,LIMITS,MAX_CALLS);tick=time.monotonic();cpu=time.process_time()
    try:
        io.save(out/'contract.json',c);io.save(out/'SOURCE.json',dict(io.read(root/c['source_manifest']),
            execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()))
        return execute(root,c,out,call)
    except BaseException as e:
        call.stop.set();io.save(out/'failure.json',dict(error=str(e),first_failure=call.first_failure));raise
    finally:
        io.save(out/'STOP.json',dict(failed=call.stop.is_set(),scientific_units=sum(call.counts.values()),
            call_counts=dict(call.counts),retry=0,resume=0,mandatory_stop=True))
        io.save(out/'resources.json',dict(wall_seconds=time.monotonic()-tick,CPU_seconds=time.process_time()-cpu,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
