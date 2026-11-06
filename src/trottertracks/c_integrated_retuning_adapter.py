"""Candidate-bound orchestration over existing research-C numerical APIs.

No science runs on import. Call only after separate scientific permission and
upstream input/DF/ground qualification. Existing frozen runners are not resumed.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from fractions import Fraction
import math
from pathlib import Path

from .c_integrated_retuning_policy import R_VALUES


def _engines():
    from . import c_rte_r_retuning as ret
    from . import c1_stage3_feasibility as feasibility
    from .c_rc2_pr_stage_b import distribution_gate
    return ret, feasibility, distribution_gate


def _strong_engine():
    from .c_update064_strong_d import evaluate
    return evaluate


def _binding_engine():
    from .c_pr_aware_df_m2 import validate_arrays
    from .c1_stage1 import sector_indices
    return validate_arrays, sector_indices


def prepare_candidate(candidate_path, reference_path, physical_record, common, out, call):
    """Bind a newly qualified endpoint to existing downstream A-packet APIs.

    ``common`` supplies original p={h,g,constant}, input_identity, independent
    SectorCAR ctx, full-input sector indices, AccuracyTarget and shell records.
    Reference export must contain this endpoint's J/U/generators/H. The caller
    fixes its path/provenance in the integrated SOURCE; this function verifies
    candidate bytes, DF qualification, exact J/U, sector basis and nine factors.
    Fourteen logical calls; law/configuration belongs to reference_r, not here.
    """
    import numpy as np
    ret,_,_=_engines();b,loader,io=ret.perf.b,ret.perf.loader,ret.io
    validate_arrays,sector_indices=_binding_engine()
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    candidate=physical_record['candidate_sha256']
    d=call('selected_load',candidate,lambda:loader.load(candidate_path,('J','U')))
    sector=call('reference_load',candidate,lambda:loader.load(reference_path,('J','U','generators','H')))
    def bind():
        if io.sha(candidate_path)!=candidate:raise ValueError('CANDIDATE_BYTE_IDENTITY')
        validate_arrays(common['p'],d,candidate)
        if not all(np.array_equal(d[k],sector[k]) for k in ('J','U')):raise ValueError('MIXED_CANDIDATE_REFERENCE')
        ix=common['basis'];df=physical_record
        if (df['input_sha256']!=common['input_identity']
            or df['guard_policy']!='sector_operation_scale_guard8_proposal_v1'):
            raise ValueError('PHYSICAL_INPUT_AND_GUARD_BINDING')
        if (sector['generators'].shape!=(9,36,36) or sector['H'].shape!=(36,36)
            or not np.array_equal(np.sort(ix),sector_indices(4,2,2))):raise ValueError('SECTOR_REFERENCE_SCHEMA')
        bound=df['sector_error']+df['numerical_guard']
        if (not df['budget_PASS'] or df['actual_rank']!=8 or df['kappa']!='original_g_corrected'
            or df['scalar']!='nuclear_once' or df['budget_Ha']!=.00019875
            or not math.isfinite(bound) or min(df['sector_error'],df['numerical_guard'])<0 or bound>.00019875):
            raise ValueError('DF_QUALIFICATION_REQUIRED')
        b.component_check(sum(sector['generators']),sector['H'],1e-10,label='same-candidate sector sum')
        sector['full_input_indices']=ix
        return dict(status='PASS',candidate=candidate,input_identity=common['input_identity'])
    io.save(out/'identity.json',call('candidate_binding',candidate,bind))
    arm=call('native_model',candidate,lambda:b.prepare_arm(**common['p'],**d,role='proposed',
        source_identity=candidate,input_identity=common['input_identity'],ld=2))
    for i in range(9):
        call('general_factor_gate',str(i),lambda i=i:b.component_check(
            b.candidate_factor(common['ctx'],arm,i),sector['generators'][i],1e-10,label='GeneralK vs independent physical CAR'))
    def ground():
        pair=loader.ground(sector['H'],out,'candidate')
        try:row=loader.ground_gate(sector['H'],pair)
        except ValueError as exc:
            if str(exc)!='UNRESOLVED_GROUND_ALIAS':raise
            row=dict(status='UNRESOLVED_GROUND_ALIAS',eligible=False)
        io.save(out/'candidate_ground.json',row)
        return pair,row
    pair,row=call('candidate_ground',candidate,ground)
    if row['status']!='PASS':return None,row
    return dict(candidate_identity=candidate,p=common['p'],d=d,sector=sector,arm=arm,ctx=common['ctx'],
        pair=pair,ground=pair[1][:,0],target=common['target'],saved=dict(physical=physical_record,shells=common['shells'])),row


def strong_d(root, contract, out, call, a):
    """Existing full-frame S2 evaluator on the same qualified candidate A packet.

    The reused evaluator's historical update064 labels are presentation strings;
    its J/U, Hamiltonian, ground and candidate binding all come from ``a``.
    """
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    value=_strong_engine()(root,contract,out,call,a)
    if value['candidate']!=a['arm'].model.source_identity:raise ValueError('MIXED_D_CANDIDATE')
    return value


def reference_r(a, r, out, call, *, pf_cache=None, r_values=R_VALUES, tail_gate=None):
    """Reuse qualified candidate A-packet; compute r32 first, then r16/r64.

    Returns (native A-packet, result, accounting packet, candidate-bound PF
    cache). PF/exact-H references are made once per candidate and reused across
    r. No old-candidate scalar values, eigenpairs or prices are substituted.
    """
    if type(r) is not int or r not in r_values:raise ValueError('REGISTERED_R_ONLY')
    ret, feasibility, distribution_gate = _engines()
    b, m4, io = ret.perf.b, ret.m4, ret.io
    candidate = a['arm'].model.source_identity
    input_identity = a['arm'].model.input_identity
    if pf_cache is None and r != 32:raise ValueError('R32_REFERENCE_REQUIRED_FIRST')
    if pf_cache is not None and (pf_cache['candidate_sha256'] != candidate or pf_cache['input_identity'] != input_identity):
        raise ValueError('MIXED_PF_CANDIDATE')
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    tail, config, dist = call('law_prepare', str(r), lambda:b.configure(a['arm'],Fraction(1,10),r,0))
    def law():
        row = (b.tail_gate if tail_gate is None else tail_gate)(a['ctx'],a['arm'],a['sector']['generators'],tail)
        distribution_gate(tail,config,dist,expected_steps=r)
        return row
    io.save(out/'law.json',call('law_gate',str(r),law))
    io.save(out/'normalization.json',call('normalization_gate',str(r),lambda:ret.normalization(tail,config,r)))
    first = pf_cache is None
    if first:
        kernels = call('kernel_reference',candidate,lambda:b.kernels(a['arm'],a['sector']['generators'],Fraction(1,10),config))
    else:
        kernels = call('corrected_kernel',candidate,lambda:m4.corrected_kernel(
            a['sector']['generators'],config,ld=2,identity_tail=a['arm'].plan.extracted_identity,
            saved_pf=pf_cache['pf'],expected_steps=r))
    def task(m):
        if first:
            from scipy.linalg import expm
            H=a['sector']['H']; elapsed=Fraction(1,10)*(1<<m)
            exact=call('exact_evolution',str(m),lambda:expm(-1j*float(elapsed)*H))
            row=call('round_reference',str(m),lambda:b.reference.reference_round(
                kernels,(H,elapsed),repetitions=1<<m,state=a['ground'],exact_operator=exact))
            signal=row.pop('raw_signal',None)
            row['raw_signal_parts']=None if signal is None else [float(signal.real),float(signal.imag)]
        else:
            row=call('round_reference',str(m),lambda:m4.rte_round(kernels,m,pf_cache['pf_errors'][m],state=a['ground']))
        decision=call('round_classification',str(m),lambda:feasibility.classify_round(row,a['target']))
        value=dict(m=m,reference=row,feasibility=decision)
        io.save(out/f'm{m:02d}.json',value)
        return value
    from threadpoolctl import threadpool_limits
    plan=m4.integrated.resource_plan(task_count=13,full_dimension=36)
    io.save(out/'resource_plan.json',plan)
    with threadpool_limits(limits=plan['BLAS_threads_per_worker']):
        with ThreadPoolExecutor(max_workers=plan['workers']) as pool:rounds=list(pool.map(task,range(13)))
    df=a['saved']['physical'];bound=df['sector_error']+df['numerical_guard']
    f=call('feasibility_summary',candidate,lambda:feasibility.summarize([x['feasibility'] for x in rounds],bound,a['target']))
    packet=dict(target_id=f'integrated:{candidate}:r{r}',rounds=rounds,df_bound_Ha=bound,
                error_evidence=None,shells=a['saved']['shells'])
    lanes=ret.assess_lanes(packet,a['target'],call)
    admitted=call('native_admission',candidate,lambda:f['status']=='POTENTIALLY_FEASIBLE' and lanes['ideal_exact_native']['energy']['same_accuracy'])
    result=dict(candidate=candidate,r=r,rounds=rounds,feasibility=f,lanes=lanes,
                native_admitted=bool(admitted),actual_CA_achieved=None)
    if first:
        pf_cache=dict(candidate_sha256=candidate,input_identity=input_identity,pf=kernels['pf'],pf_errors=[x['reference']['pf_operator_error'] for x in rounds])
    return dict(a,tail=tail,config=config,dist=dist),result,packet,pf_cache


def native_cohort(root, contract, out, call, a, result, packet, *, split, cohort_key, prepared=None):
    """Existing native32 batches: one selection batch or two final batches.

    Cohort key must include geometry/start/strength/budget identity supplied by
    the integrated plan. Split/r/batch and SOURCE are additionally bound here.
    Selection data are never pooled into final data. Same-candidate immutable
    CAR/Gaussian/shell cache may be shared through ``prepared``.
    """
    if split not in ('selection','final') or not cohort_key:raise ValueError('COHORT_IDENTITY_REQUIRED')
    if not result['native_admitted']:raise ValueError('NATIVE_ADMISSION_REQUIRED')
    candidate=a['arm'].model.source_identity;r=result['r']
    if result['candidate']!=candidate or a['config'].rte_steps!=r:raise ValueError('MIXED_NATIVE_CANDIDATE_OR_R')
    result=deepcopy(result);packet=dict(packet)
    ret,_,_=_engines();perf=ret.perf;io=ret.io
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    shared={} if prepared is None else prepared
    rows=[];cal=None
    for batch in range(1 if split=='selection' else 2):
        folder=out/f'batch{batch:02d}';folder.mkdir()
        sampling=dict(master=2026110501,cohort=f'integrated:{cohort_key}:{split}:r{r}:batch{batch:02d}',
            axis=f'INTEGRATED_{split.upper()}_PRICE_ONLY',samples=32,events_per_sample=r,
            PRNG='numpy.random.PCG64',paired=False,adaptive=False)
        current,cal=perf.native(root,contract,folder,call,a,sampling=sampling,prepared=shared)
        rows.extend(current)
    if split=='final':cal=call('final_calibration',candidate,lambda:perf.b2.calibration(rows))
    packet.update(kernel_samples=rows,saved_calibration=cal)
    kernel=call('cost_calibration',candidate,lambda:perf.accounting.cost_calibration(packet,expected_n=len(rows)))
    costs=[call('serial_cost',f'{m}{axis}',lambda m=m,axis=axis:ret.m3.serial_cost(packet,kernel,m,axis))
           for m in range(13) for axis in ('X','Y')]
    for lane,value in result['lanes'].items():
        totals=call('total_aggregation',lane,lambda value=value:ret.m4.accuracy.aggregate_totals(
            value['energy'],value['rounds'],costs,cost_scope=perf.accounting.COST_SCOPE))
        value['conditional_totals']=totals
        value['weighted_SE']=call('calibration_uncertainty',lane,lambda value=value,totals=totals:ret.m3.calibration_uncertainty(packet,value['rounds'],totals,expected_n=len(rows)))
    result.update(calibration=cal,split=split,samples=len(rows))
    io.save(out/'result.json',result)
    return result,packet
