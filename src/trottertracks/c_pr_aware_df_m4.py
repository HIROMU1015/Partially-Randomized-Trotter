"""Saved-PF M4 r128: existing RTE/accuracy, unknown-price affine accounting.

No native sampling/build/compile entry. No old runner is restarted. Imports
and passive preflight read no molecular arrays. New science is sector36 only.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from fractions import Fraction
import importlib.metadata
import math
from pathlib import Path
import resource
import subprocess
import time

from . import c1_stage1_execution as io
from . import c1_stage3_accuracy as accuracy
from . import c1_stage3_integrated as integrated

VERSION='M4_saved_PF_r128_RTE_shots_affine_price_v1'
SCOPE='C_PR_AWARE_DF_M4_S1_R128_BREAKEVEN_V1'
BASE='e61e79f68a29d0ff52a9d0a9448a569fc98e0e86'
FAMILY='artifacts/research_c_pr_aware_df_M4/2026-10-28'
FRESH=FAMILY+'/run01'
TARGET=dict(energy_ha=.00159,df_reserve_fraction=1/8,base_time=.1,
    beta_total=math.pi/6,beta_stat=math.pi/12,alpha_axis=.01/26,alpha_total=.01)
POLICY=dict(atoms=4,distance_A=1.,basis='STO-3G',rank=8,arm='proposed',Nalpha=2,Nbeta=2,
    sector_dimension=36,system_modes=8,frame_order=list(range(8)),ld=2,
    PF='partial ordered symmetric S2',delta='1/10',s=1,h='1/10',r=128,K=0,
    rounds=list(range(13)),original_g_kappa=True,scalar='nuclear once',
    oracle='own_DF_ground_state_oracle',state_preparation='EXCLUDED',
    law='ideal_finite_RTE_IID',PF_reuse='M3 r64 same h/s/schedule/JU;no PF or exact-evolution recomputation')
ALLOWANCES=dict(component_abs=1e-10,normalization=1e-12,mass_relative=1e-12,
    mass_absolute=1e-14,algebra_absolute=1e-12,rtol=0,strict_enclosure=None)
A_LIMITS=dict(evidence_load=1,identity_gate=1,target_validation=1,input_load=1,fit_load=1,
    native_model=1,sector_load=1,sector_schema_gate=1,law_prepare=1,law_identity=1,tau_gate=1,
    corrected_kernel=1,round_reference=13,round_classification=13,A_summary=1)
B_LIMITS=dict(signal_qualification=3,round_accuracy=39,energy_assessment=3,shots_summary=3)
C_LIMITS=dict(affine_accounting=3,threshold_comparison=6,comparison_summary=1)
LIMITS=dict(**A_LIMITS,**B_LIMITS,**C_LIMITS,result_summary=1)
MAX_CALLS=sum(LIMITS.values())
METRICS=accuracy.METRICS
GROUPS={'m0_to9':tuple(range(10)),'m10_to12':(10,11,12),'m12':(12,)}


def preflight(root,path):
    from .c_pr_aware_df_stage_c import SCENARIOS
    root=Path(root).resolve();c=io.read(path)
    for key,want in dict(scope=SCOPE,implementation=VERSION,base_commit=BASE,policy=POLICY,
            target=TARGET,allowances=ALLOWANCES,scenarios=list(SCENARIOS),
            stage_call_limits=dict(A=A_LIMITS,B=B_LIMITS,C=C_LIMITS),call_limits=LIMITS,
            maximum_logical_units=MAX_CALLS,fresh_output=FRESH,retry=0,resume=0,
            new_fit=0,trajectory_sampling=0,build_compile=0).items():
        if c[key]!=want:raise ValueError('M4_FIXED_SCOPE:'+key)
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    for files in (io.read(root/c['source_manifest'])['files'],c['saved_inputs']):
        for p,want in files.items():
            file=(root/p).resolve()
            if not file.is_relative_to(root) or io.sha(file)!=want:raise ValueError('SOURCE_OR_INPUT:'+p)
    for package,want in c['packages'].items():
        if importlib.metadata.version(package)!=want:raise ValueError('ENVIRONMENT:'+package)
    if (root/FRESH).exists():raise ValueError('FRESH_M4_NO_RETRY_RESUME')
    return c


def load_evidence(root,c):
    from . import c_pr_aware_df_m3 as m3
    saved=m3.evidence(root,c)
    saved.update({k:io.read(Path(root)/p) for k,p in c['M3_records'].items()})
    return saved


def identity_gate(saved,c):
    """Saved PF identity by inputs, unchanged reference SOURCE, policy, STOP."""
    from . import c_pr_aware_df_m3 as m3
    from . import c_pr_aware_df_stage_b_runner as b1
    old=saved['M3_contract'];r=saved['M3'];stop=saved['M3_STOP']
    if (r['status']!='M3_A_B_C_COMPLETE' or stop['failed'] or stop['scientific_units']!=471
            or stop['call_counts']!=m3.LIMITS or stop['retry'] or stop['resume']
            or saved['M3_SOURCE']['execution_commit']!='5354edecc6aa995082440dd14be0310bdd9902e1'
            or old['source_manifest_sha256']!='b375bb1f9f979a70de7577729a28f3d7755a66e6643e7b46366d74acc830d857'):
        raise ValueError('M3_SOURCE_STOP')
    if (old['policy']!=m3.POLICY or old['target']!=TARGET or old['exports']!=c['exports']
            or old['input']!=c['input'] or old['grounds']!=c['grounds']
            or old['original_input_sha256']!=c['original_input_sha256']):raise ValueError('PF_REUSE_MODEL_IDENTITY')
    for key in ('atoms','distance_A','basis','rank','arm','Nalpha','Nbeta','sector_dimension',
            'system_modes','frame_order','ld','PF','delta','s','h','K','rounds','original_g_kappa','scalar','oracle','state_preparation'):
        if old['policy'][key]!=POLICY[key]:raise ValueError('PF_REUSE_CONDITIONS:'+key)
    for p in (c['input'],c['exports']['proposed']):
        if old['saved_inputs'][p]!=c['saved_inputs'][p]:raise ValueError('PF_REUSE_INPUT_HASH')
    # The old manifest is evidence, not a new program to run. Every old
    # scientific SOURCE file is unchanged at this new experiment's start.
    if any(c['pinned_M3_SOURCE'][p]!=h for p,h in saved['M3_SOURCE']['files'].items()):
        raise ValueError('PF_REUSE_REFERENCE_SOURCE')
    a=r['A']
    if (a['status']!='POTENTIALLY_FEASIBLE' or len(a['rounds'])!=13
            or [x['m'] for x in a['rounds']]!=list(range(13))
            or saved['M3_ground']['status']!='PASS' or saved['M3_ground']['alias_margin']<=0
            or len(saved['M3_factors'])!=9 or saved['M3_tail']['status']!='PASS'
            or saved['M3_tail']['catalog_identity']['status']!='PASS'):
        raise ValueError('SAVED_PF_GROUND_TAIL_EVIDENCE')
    for x in a['rounds']:
        accuracy._finite(x['reference']['pf_operator_error'],'saved PF error')
    df=a['saved_DF_bound_Ha'];accuracy._finite(df,'saved DF')
    if df>TARGET['energy_ha']/8:raise ValueError('SAVED_DF_RESERVE')
    for lane in ('ideal_exact_native','assumed_raw_signal_budget'):
        if not r['C']['lanes'][lane]['conditional_energy_assessment']['same_accuracy']:
            raise ValueError('SAVED_TARGET_ACCURACY')
    return dict(status='PASS',same_JU_h_s_ld_schedule=True,original_kappa_scalar=True,
        saved_ground_alias=True,reference_SOURCE_unchanged=True,PF_error_definition='norm2(PF(h)^2^m-exact(t=.1*2^m))',
        reused_PF_kernel=True,reused_PF_rounds=13,new_PF_kernel=0,new_exact_evolution=0,new_ground_eigensolve=0,
        r_changes_corrected_RTE_only=True)


def sector_schema_gate(values):
    import numpy as np
    from .c1_stage1 import sector_indices
    G,H,pf,ix=(values[k] for k in ('generators','H','pf','full_input_indices'))
    if G.shape!=(9,36,36) or H.shape!=(36,36) or pf.shape!=(36,36):raise ValueError('SECTOR_PF_SHAPE')
    if any(not np.isfinite(x).all() for x in (G,H,pf)) or not np.array_equal(np.sort(ix),sector_indices(4,2,2)):
        raise ValueError('SECTOR_PF_FINITE_MAPPING')
    from .c1_stage3_molecular import component_check
    return component_check(sum(G),H,1e-10,label='unchanged M3 sector sum')


def corrected_kernel(generators,config,*,ld,identity_tail,saved_pf):
    """Reuse existing head-exponential and finite-RTE functions, omitting PF.

    This is the RTE branch of reference_kernels verbatim in meaning. Its PF
    tail exponential is intentionally absent: r does not enter that branch.
    """
    import numpy as np
    from . import c1_stage3_reference as ref
    h=Fraction(1,10)
    if config.rte_steps!=128 or config.evolution_time!=float(h):raise ValueError('R128_H_CONFIG')
    G=tuple(np.asarray(g,dtype=complex) for g in generators);cache={}
    forward=tuple((i,h/2) for i in range(ld+1))
    left=ref._ordered_exponentials(G,forward,cache)
    right=ref._ordered_exponentials(G,tuple(reversed(forward)),cache)
    tail=sum(G[ld+1:],np.zeros_like(G[0]))-identity_tail*np.eye(len(G[0]),dtype=complex)
    corrected=ref._legacy('rte').finite_rte_corrected_operator(tail/config.lambda_r,config)
    matrix=np.exp(-1j*float(h)*identity_tail)*right@corrected@left
    if not np.isfinite(matrix).all():raise ValueError('NONFINITE_CORRECTED_KERNEL')
    return dict(pf=np.array(saved_pf,copy=True),corrected=matrix,config=config,step_time=float(h))


def rte_round(kernel,m,saved_pf_error):
    """Same existing two powers/norm2/A, without exact exp or PF error norm."""
    import numpy as np
    from . import c1_stage3_reference as ref
    if type(m)is not int or not 0<=m<=12:raise ValueError('M4_ROUND_INDEX')
    n=1<<m;A=ref._legacy('rte').finite_rte_attenuation(kernel['config'],tail_evolutions=n)
    if not math.isfinite(A) or not 0<=A<=1:raise ValueError('ATTENUATION_RANGE')
    if A==0:return dict(pf_operator_error=saved_pf_error,rte_corrected_operator_error=None,attenuation=0.)
    pf=np.linalg.matrix_power(kernel['pf'],n)
    corrected=np.linalg.matrix_power(kernel['corrected'],n)
    error=float(np.linalg.norm(corrected-pf,2))
    if not math.isfinite(error):raise ValueError('NONFINITE_RTE_ERROR')
    return dict(pf_operator_error=saved_pf_error,rte_corrected_operator_error=error,attenuation=A,
        evidence_kind='binary64_sector_operator_numerical_pilot',PF_reused=True)


def tau_gate(tail,config,saved):
    old=saved['M3']['A']['algebra'];exact=Fraction(tail.lambda_r)*Fraction(1,10)/128
    if (exact!=Fraction(old['tau_rational'])/2 or config.dimensionless_step_time!=old['new_tau']/2
            or tail.lambda_r!=old['lambda_R']):raise ValueError('R128_TAU_HALF_IDENTITY')
    return dict(status='PASS',lambda_R=tail.lambda_r,tau_rational=str(exact),
        tau128=config.dimensionless_step_time,tau64=old['new_tau'],
        mathematical_half=True,binary64_half=True,r64_events_per_delta=64,r128_events_per_delta=128,
        normalization='A=(1+tau^2)^(-r*2^m/2)',A128_not_reused_from64=True)


def prepare_and_A(root,c,out,call):
    from threadpoolctl import threadpool_limits
    from . import c_pr_aware_df_m2 as m2
    from . import c_pr_aware_df_m3 as m3
    from . import c_pr_aware_df_stage_b as b
    from . import c_pr_aware_df_stage_b_runner as loader
    from . import c1_stage3_feasibility as feasibility
    saved=call('evidence_load','saved M3/M2/C1 and scientific identities',lambda:load_evidence(root,c))
    io.save(out/'identity.json',call('identity_gate','same PF input/head/tail/time/SOURCE',lambda:identity_gate(saved,c)))
    target=call('target_validation','registered CA/13round/phase/failure',lambda:accuracy.AccuracyTarget(**TARGET))
    if target.horizon()['maximum_round_index_M']!=12:raise ValueError('REGISTERED_HORIZON')
    p=call('input_load','original saved h/g/scalar',lambda:loader.load(Path(root)/c['input'],('h','g','constant')))
    data=call('fit_load','same saved Proposed J/U',lambda:loader.load(Path(root)/c['exports']['proposed'],('J','U')))
    def prepare():
        m2.validate_arrays(p,data,'proposed')
        return b.prepare_arm(**p,**data,role='proposed',source_identity=c['saved_inputs'][c['exports']['proposed']],
            input_identity=c['original_input_sha256'],ld=2)
    arm=call('native_model','model only;no circuit;original kappa/scalar/frame order',prepare)
    sector=call('sector_load','saved M3 PF/generators/H/mapping only',lambda:loader.load(Path(root)/c['M3_sector'],('generators','H','pf','full_input_indices')))
    io.save(out/'sector_gate.json',call('sector_schema_gate','saved PF/sector schema',lambda:sector_schema_gate(sector)))
    tail,config,distribution=call('law_prepare','h=.1/r128/K0',lambda:b.configure(arm,Fraction(1,10),128,0))
    def law_identity():
        old=saved['proposed_tail']
        m3.catalog_gate(arm.plan.atoms,old['atoms_catalog'])
        if (len(arm.plan.atoms)!=216 or tail.lambda_r!=saved['M3']['A']['algebra']['lambda_R']
                or arm.plan.extracted_identity!=old['identity']
                or [x.probability for x in tail.components]!=[x['probability'] for x in old['components']]):
            raise ValueError('M4_216_ATOM_LAMBDA_IDENTITY')
        return dict(status='PASS',lambda_R=tail.lambda_r,atoms=216,identity=arm.plan.extracted_identity,
            order_coefficient_sign_basis_probability_unchanged=True,JSON_roundtrip_supported=True)
    io.save(out/'law_gate.json',call('law_identity','exact216catalog and probability column',law_identity))
    tau=call('tau_gate','new tau = saved tau/2',lambda:tau_gate(tail,config,saved));io.save(out/'tau.json',tau)
    kernel=call('corrected_kernel','r128 only;reuse M3 PF,omit tail PF exponential',lambda:corrected_kernel(
        sector['generators'],config,ld=2,identity_tail=arm.plan.extracted_identity,saved_pf=sector['pf']))
    loader.save_arrays(out/'reference.npz',pf=kernel['pf'],corrected=kernel['corrected'])
    plan=integrated.resource_plan(task_count=13,full_dimension=36);io.save(out/'resource_plan.json',plan)
    rows={};errors=[]
    def task(m):
        old=saved['M3']['A']['rounds'][m]
        ref=call('round_reference',f'm{m}',lambda:rte_round(kernel,m,old['reference']['pf_operator_error']))
        decision=call('round_classification',f'm{m}',lambda:feasibility.classify_round(ref,target))
        value=dict(m=m,reference=ref,feasibility=decision,saved_M3_reference=old['reference'],
            synthesis_error=None,implemented_law_error=None)
        io.save(out/f'm{m:02d}.json',value);return m,value
    with threadpool_limits(limits=plan['BLAS_threads_per_worker']):
        with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
            for f in as_completed([pool.submit(task,m) for m in range(13)]):
                try:m,v=f.result();rows[m]=v
                except BaseException as error:call.stop.set();errors.append(error)
    if errors:raise errors[0]
    rows=[rows[i] for i in range(13)];df=saved['M3']['A']['saved_DF_bound_Ha']
    classification=call('A_summary','same B1 sufficient conditions;all13 preserved',lambda:feasibility.summarize([x['feasibility'] for x in rows],df,target))
    result=dict(classification,rounds=rows,tau=tau,saved_DF_bound_Ha=df,PF_reused_rounds=13,
        new_PF_evaluation=0,new_exact_evolution=0,kernel_price=None,CA_achieved=None)
    io.save(out/'result.json',result)
    packet=dict(arm='proposed',target_id='proposed:'+c['saved_inputs'][c['exports']['proposed']],
        rounds=rows,df_bound_Ha=df,error_evidence=None)
    return dict(result=result,saved=saved,target=target,packet=packet)


def shots_summary(plans,energy,old):
    if not energy['same_accuracy']:
        return dict(status='UNQUALIFIED_OR_INFEASIBLE',total_shots=None,rounds=plans,groups=None)
    if ([p['m'] for p in plans]!=list(range(13)) or [p['m'] for p in old]!=list(range(13))
            or any(type(p.get('shots'))is not int or p['shots']<1 for p in plans+old)):
        raise ValueError('INTEGER_13_ROUND_SHOTS')
    rows=[dict(m=p['m'],shots_per_axis=p['shots'],old_M3_shots_per_axis=o['shots'],
        difference_per_axis=p['shots']-o['shots'],kernel_occurrences=1<<p['m']) for p,o in zip(plans,old,strict=True)]
    groups={k:dict(new_shots=sum(2*rows[m]['shots_per_axis'] for m in group),
        old_shots=sum(2*rows[m]['old_M3_shots_per_axis'] for m in group),
        new_weight=sum(2*rows[m]['shots_per_axis']*(1<<m) for m in group),
        old_weight=sum(2*rows[m]['old_M3_shots_per_axis']*(1<<m) for m in group)) for k,group in GROUPS.items()}
    return dict(status='CONDITIONAL_SHOTS_AVAILABLE',total_shots=sum(2*p['shots'] for p in plans),
        old_total_shots=sum(2*p['shots'] for p in old),rounds=rows,groups=groups,
        kernel_price=None,total_resources=None,actual_CA_achieved=None,guaranteed_shots=None)


def stage_B(out,call,a):
    from . import c_pr_aware_df_stage_c as old
    lanes={}
    for lane in old.LANES:
        def qualification():
            q=old.qualification(a['packet'],lane)
            return dict(q,M4_kernel_price='UNKNOWN',M4_native_calibration_used=False,
                cost_proxy_assumption_scope='saved comparison targets only;no B2 price transferred to r128')
        q=call('signal_qualification',lane,qualification);plans=[]
        for row in a['packet']['rounds']:
            p=call('round_accuracy',lane+':m'+str(row['m']),lambda row=row:old.assess_round(a['target'],a['packet'],q,row))
            io.save(out/(lane+f'_m{row["m"]:02d}.json'),p);plans.append(p)
        energy=call('energy_assessment',lane,lambda:old.energy(a['target'],a['packet'],plans))
        oldplans=None if lane=='evidence' else a['saved']['M3']['C']['lanes'][lane]['rounds']
        summary=call('shots_summary',lane,lambda:shots_summary(plans,energy,oldplans))
        value=dict(lane=lane,qualification=q,rounds=plans,energy=energy,shots=summary,
            kernel_price=None,total_resources=None,actual_CA_achieved=None,guaranteed_shots=None)
        lanes[lane]=value;io.save(out/(lane+'.json'),value)
    return lanes


def affine_accounting(lane,shells):
    if not lane['energy']['same_accuracy'] or lane['shots']['total_shots'] is None:
        return dict(status='UNQUALIFIED_OR_INFEASIBLE',W=None,S=None,mu=None,total_resources=None)
    shell={axis:{k:math.fsum(shells[x]['metrics'][k] for x in ('head',axis)) for k in METRICS} for axis in ('X','Y')}
    for v in shell.values():
        for k,x in v.items():accuracy._finite(x,'shell '+k)
    def coefficients(rows,field):
        W=sum(2*r[field]*(1<<r['m']) for r in rows)
        S={k:math.fsum(r[field]*shell[axis][k] for r in rows for axis in ('X','Y')) for k in METRICS}
        return dict(W=W,S=S)
    rows=lane['shots']['rounds'];new=coefficients(rows,'shots_per_axis');old=coefficients(rows,'old_M3_shots_per_axis')
    mixed=[dict(r,shots_per_axis=r['shots_per_axis'] if r['m']>=10 else r['old_M3_shots_per_axis']) for r in rows]
    high=coefficients(mixed,'shots_per_axis')
    if type(new['W'])is not int or new['W']<=0:raise ValueError('POSITIVE_KERNEL_WEIGHT')
    return dict(status='AFFINE_COST_ONLY_UNKNOWN_PRICE',**new,mu=None,total_resources=None,
        formula='C_k(mu_k)=W*mu_k+S_k',shell_once_per_interrogation=True,substeps=1,
        old_M3_shots_accounting=old,high_m_only_shots_swap=high,
        counterfactual_kind='accounting substitution only;no extra candidate',groups=lane['shots']['groups'])


def breakeven(affine,target,existing_mean,label):
    if affine.get('W') is None or target.get('totals') is None:
        return dict(status='UNQUALIFIED_OR_INFEASIBLE',label=label,thresholds=None,actual_r128_cost=None)
    if (type(affine['W'])is not int or affine['W']<=0
            or target.get('cost_scope')!='serial_native_kernel_composition_without_state_preparation'):
        raise ValueError('AFFINE_OR_TARGET_COST_SCOPE')
    rows={}
    for k in METRICS:
        value=target['totals'][k];accuracy._finite(value,'target cost '+k)
        accuracy._finite(existing_mean[k],'existing kernel mean '+k)
        mu=(value-affine['S'][k])/affine['W']
        old=affine['old_M3_shots_accounting'];hi=affine['high_m_only_shots_swap']
        before=(value-old['S'][k])/old['W'];high=(value-hi['S'][k])/hi['W']
        if not all(math.isfinite(x) for x in (mu,before,high)):raise ValueError('NONFINITE_BREAKEVEN')
        rows[k]=dict(mu_break_even=mu,strict_advantage_condition='nonnegative mu < threshold',
            nonnegative_price_can_strictly_win=mu>0,zero_or_negative_threshold=mu<=0,
            existing_r64_mean=existing_mean[k],headroom_above_r64_kernel=mu-existing_mean[k],
            headroom_interpretation='accounting allowance for all64additional events and changed compilation;not measured event unit price',
            break_even_with_old_M3_shots=before,break_even_with_only_m10_to12_shots_changed=high,
            high_m_shots_threshold_change=high-before,all_shots_threshold_change=mu-before,
            equality_means_tie=True)
    return dict(status='CONDITIONAL_POINT_BREAKEVEN_ONLY',label=label,thresholds=rows,
        target_cost=target,actual_r128_kernel_price=None,actual_r128_total_cost=None,
        ideal_law_population_advantage=None,guaranteed_advantage=None,cost_support=None)


def stage_C(out,call,a,lanes):
    affine={lane:call('affine_accounting',lane,lambda lane=lane:affine_accounting(lanes[lane],a['saved']['shell_reports'])) for lane in lanes}
    comparisons={};mean=a['saved']['M3']['B']['calibration']['sample_mean']
    for lane in ('ideal_exact_native','assumed_raw_signal_budget'):
        targets=dict(M3_r64=a['saved']['M3']['C']['lanes'][lane]['resources'],
            M2_deterministic=a['saved']['M2']['candidates']['proposed']['cost'],
            old_PR_s4=a['saved']['C1']['assumed_scenarios'][lane]['proposed']['resources'])
        for name,target in targets.items():
            label=lane+':'+name
            value=call('threshold_comparison',label,lambda label=label,target=target:breakeven(affine[lane],target,mean,label))
            comparisons[label]=value;io.save(out/(lane+'_'+name+'.json'),value)
    result=call('comparison_summary','unknown r128 mu remains unmeasured',lambda:dict(
        status='M4_AFFINE_BREAKEVEN_COMPLETE',affine=affine,comparisons=comparisons,
        native_price=None,r128_total_resources=None,actual_CA_achieved=None,cost_support=None))
    io.save(out/'result.json',result);return result


def execute(root,c,out,call):
    out=Path(out)
    for leaf in ('A','B','C'):(out/leaf).mkdir()
    a=prepare_and_A(root,c,out/'A',call);lanes=stage_B(out/'B',call,a);cost=stage_C(out/'C',call,a,lanes)
    result=call('result_summary','M4 only,no native execution',lambda:dict(status='M4_R128_SHOTS_BREAKEVEN_COMPLETE',
        scope=SCOPE,A=a['result'],B=lanes,C=cost,new_fit=0,optimizer=0,new_ground_eigensolve=0,
        new_PF_kernel=0,new_exact_evolution=0,trajectory_sampling=0,build_compile=0,QPY=0,
        r128_kernel_price=None,r128_total_resources=None,strict_enclosure=None,actual_CA_achieved=None,
        all_law_synthesis=None,implemented_law_error=None,retry=0,resume=0,mandatory_stop=True))
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
