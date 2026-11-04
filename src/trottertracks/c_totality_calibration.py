"""Opt-in qualification and failure accounting; no sampler or accuracy engine.

Hypothetical scalar policies are not adopted scientific conditions. Actual H4
outcomes cannot be supplied by this module: its entry is passive only.
"""
from fractions import Fraction as F
from . import c_iid_grid_expected_bias as expected
from . import c_per_word_population_range as population
from . import c_p0_compiler_totality as compiler

BASE='c01bd2742ca88071d17db1fc3e3aec6cb9f3cb9e'
FAMILY='artifacts/research_c_stage_p_totality_calibration/2026-11-04'
CONTRACT='docs/research/research_c_stage_p_totality_calibration_contract_unapproved_20261104.json'
SCOPE='C_STAGE_P_TOTALITY_CALIBRATION_FEASIBILITY_V1'
FAIL={'RAW_BUILD_FAILURE','COMPILE_EXCEPTION','CONTROLLER_GUARD_FAILURE','INVALID_NATIVE_OUTPUT'}
UNKNOWN={'UNKNOWN','NONTERMINATION_UNRESOLVED','EXECUTION_INTERRUPTION'}


def evidence_graph(population_record,*,fixed_word_record=None):
    a=expected.nonnegative(population_record['D_norm_upper_exact'])
    M=expected.nonnegative(population_record['d_range_upper_exact'])
    if (population_record['status']!='VALIDATED_LITERAL_POPULATION_NORM'
            or M!=1+a or population_record['all_q_words_compile_success_qualified'] is not False):
        raise ValueError('FIXED_POPULATION_SCOPE')
    finite=(fixed_word_record is not None and fixed_word_record.get('status')=='STAGE_P_MISSING_OUTPUTS_COMPLETE'
            and fixed_word_record.get('missing_outputs_obtained') is True
            and fixed_word_record.get('original_41_call_run_complete') is False)
    return dict(status='NOT_READY_POPULATION_COMPILED_BIAS',
        fixed_word_compile_success=finite,fixed_word_operator_enclosure=finite,
        fixed_word_inverse_and_shell_enclosure=finite,literal_population_norm_qualified=True,
        qualification_records_require_fixed_provenance=True,
        literal_norm_upper_exact=str(a),operator_error_range_exact=str(M),
        operator_error_range_status='conditional_on_total_compiler_domain_or_separately_adopted_totalized_scope',
        all_q_words_compile_success_qualified=False,all_word_synthesis_qualified=False,
        IID_population_expected_bias_qualified=False,calibration_ready=False,
        IID_RPE_sufficient_conditions_qualified=False,actual_CA_achieved=None,
        missing=['compiler domain or adopted abort-aware estimand','fresh fixed-N enclosures and confidence policy',
                 'joint transport/RPE qualification without missing terms'],
        failed_word_discard_allowed=False,Stage_C_ready=False)


def failure_rate(outcomes,*,planned_N,delta,event='pipeline',experiment='fixed_N'):
    """Fixed-N hypothetical IID trials; duplicates of WORDS are retained.

Known failures can be counted only in an explicitly selected fixed-N experiment.
Unresolved/hanging trials cannot silently become success, failure or discarded.
Zero failure uses the existing outward exact-binomial adapter; k>0 uses the
existing conservative Hoeffding radius, not an unvalidated root solver.
"""
    if type(planned_N) is not int or planned_N<1:raise ValueError('FIXED_POSITIVE_N')
    d=expected.probability(delta);rows=tuple(outcomes)
    if experiment not in ('fixed_N','stop_on_failure'):raise ValueError('EXPERIMENT_SCOPE')
    if len(rows)>planned_N or [r['draw_id'] for r in rows]!=list(range(len(rows))):
        raise ValueError('NO_DISCARD_REORDER_OR_REPLACEMENT')
    if any(r.get('retry',0)!=0 or r.get('resume',0)!=0 or r.get('resampled',False) is not False for r in rows):
        raise ValueError('NO_RETRY_RESUME_RESAMPLING')
    if any(r['outcome'] not in FAIL|UNKNOWN|{'SUCCESS'} for r in rows):raise ValueError('OUTCOME_DEFINITION')
    if len(rows)!=planned_N or any(r['outcome'] in UNKNOWN for r in rows):
        return dict(status='INCOMPLETE_NO_FIXED_N_CONFIDENCE',failure_upper=None,attempts=len(rows),planned_N=planned_N)
    if experiment=='stop_on_failure' and any(r['outcome'] in FAIL for r in rows):
        return dict(status='STOP_NO_CERTIFICATION',failure_upper=None,attempts=len(rows))
    if event=='pipeline':selected=rows;bad=lambda r:r['outcome'] in FAIL
    elif event=='raw_build':selected=rows;bad=lambda r:r['outcome']=='RAW_BUILD_FAILURE'
    elif event=='compile_given_raw_success':
        selected=tuple(r for r in rows if r['outcome']!='RAW_BUILD_FAILURE');bad=lambda r:r['outcome'] in FAIL
    else:raise ValueError('FAILURE_ESTIMAND')
    n=len(selected);k=sum(bad(r) for r in selected)
    if not n:return dict(status='NO_RAW_SUCCESS_CONDITIONAL_RATE_UNQUALIFIED',failure_upper=None)
    if k==0:
        from .c_pr_aware_df_guarantee_strategy import zero_failure_tail_limit
        upper=F(zero_failure_tail_limit(n,str(d),iid=True,fresh_holdout=True,certified_decisions=True)['tail_probability_upper_exact'])
        method='outward zero-failure binomial upper'
    else:
        upper=min(F(1),F(k,n)+expected.hoeffding_radius('1',n,str(d)))
        method='conservative one-sided Hoeffding;not exact nonzero-k Clopper-Pearson'
    return dict(status='CONDITIONAL_FIXED_N_FAILURE_BOUND',event=event,denominator=n,failures=k,
        failure_upper=str(upper),delta_compile=str(d),method=method,
        assumptions='fresh IID words and fixed deterministic word-to-outcome map;or explicitly justified IID joint law',
        scientific_policy_adopted=False,all_q_words_compile_success_qualified=False)


def compile_requests(shots_X,shots_Y,*,cache='none',deterministic_word_map=False):
    """Symbolic policy at real inputs=None; numeric evaluation uses invented tests only."""
    if shots_X is None or shots_Y is None:
        return dict(status='SHOTS_AND_CACHE_POLICY_UNFIXED',word_draws=None,
            formula='sum_m=0..12 (shots_X[m]+shots_Y[m])*2**m',actual_compile_calls=None)
    if (len(shots_X)!=13 or len(shots_Y)!=13
            or any(type(n) is not int or n<0 for n in [*shots_X,*shots_Y])):
        raise ValueError('THIRTEEN_ROUND_XY_COUNTS')
    if cache not in ('none','exact_word_source_environment'):raise ValueError('CACHE_POLICY')
    if cache!='none' and deterministic_word_map is not True:raise ValueError('CACHE_OUTCOME_IDENTITY_PREMISE')
    T=sum((x+y)*(1<<m) for m,(x,y) in enumerate(zip(shots_X,shots_Y)))
    return dict(word_draws=T,uncached_kernel_compile_calls=T,
        actual_compile_calls_interval=[int(T>0),T] if cache!='none' else [T,T],
        independent_occurrences=True,duplicate_draws_retained=True,actual_inverse_new_word_draws=0,
        inverse_policy='derive actual inverse of the same realized compiled circuit',
        shared_shell_compile_calls='separately accounted;0 only if fixed qualified shells are reused',
        cache_changes_IID_law=False,all_q_words_compile_success_qualified=False)


def confidence(events,*,p_fail_upper=None,requests=None,rpe_scope_qualified=False):
    """Union bounds only; alpha=.01 is preserved, not silently reallocated."""
    unique={}
    for row in events:
        key=row['id'];value=(row['kind'],expected.probability(row['delta']))
        if value[0] not in ('calibration','compile_confidence'):raise ValueError('CONFIDENCE_EVENT_KIND')
        if key in unique and unique[key]!=value:raise ValueError('CONFIDENCE_EVENT_ID_CONFLICT')
        unique[key]=value
    dc=sum((d for kind,d in unique.values() if kind=='calibration'),F())
    dp=sum((d for kind,d in unique.values() if kind=='compile_confidence'),F())
    if dc+dp>=1:raise ValueError('JOINT_CONFIDENCE_RANGE')
    abort=None
    if p_fail_upper is not None and requests is not None:
        p=expected.nonnegative(p_fail_upper)
        if p>1 or type(requests) is not int or requests<0:raise ValueError('OPERATIONAL_REQUESTS_OR_RATE')
        abort=min(F(1),p*requests)
    union=None if abort is None else str(min(F(1),F(1,100)+dc+dp+abort))
    return dict(measurement_alpha='1/100',reported_alpha_total='1/100',delta_cal=str(dc),delta_compile=str(dp),
        unique_confidence_events=len(unique),operational_abort_upper=None if abort is None else str(abort),
        candidate_union_under_RPE_scope_premise=union,
        unconditional_failure_upper=union if rpe_scope_qualified else None,
        RPE_abort_scope_qualified=rpe_scope_qualified,
        joint_001_policy_adopted=False,measurement_alpha_reallocated=False,
        interpretation='conditional on qualified bounds;confidence,operational abort,and measurement failure are distinct')


def calibration_gate(population_record,samples=(),*,fixed_word_record=None):
    graph=evidence_graph(population_record,fixed_word_record=fixed_word_record)
    # Current fixed evidence supplies no all-q structural argument and adopts no
    # abort extension. Retain the existing rejection instead of fabricating one.
    gate=population.calibration_inputs(samples,population_record,compiler_domain=None)
    result=expected.assess_if_adopted(None,None,dict(eligible=False),policy_adopted=False)
    return dict(graph=graph,existing_range_gate=gate,existing_accuracy_gate=result,
                expected_bias_upper=None,Stage_C_calls=0,Stage_C_ready=False)


def run_science(*,execute_science=False):
    if not execute_science:raise PermissionError('NEW_FIXED_SCIENTIFIC_PERMISSION')
    raise NotImplementedError('NOT_READY_METHOD_N_DELTAS_AND_ABORT_POLICY_UNADOPTED')


def preflight(root,path):
    from pathlib import Path
    from .c_stage_p_completion import io
    root=Path(root).resolve();c=io.read(path)
    fixed=dict(base_commit=BASE,scope=SCOPE,executable=False,method=None,samples=None,
        delta_cal=None,delta_compile=None,abort_policy=None,maximum_logical_units=None,
        measurement_alpha='1/100',joint_001_policy_adopted=False,retry=0,resume=0,mandatory_stop=True)
    from .c_per_word_range_schema import same_json
    if not same_json({k:c[k] for k in fixed},fixed):raise ValueError('NONEXECUTABLE_UNADOPTED_CALIBRATION_SCOPE')
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    for group in (io.read(root/c['source_manifest'])['files'],c['saved_inputs']):
        for p,h in group.items():
            f=(root/p).resolve()
            if not f.is_relative_to(root) or io.sha(f)!=h:raise ValueError('SOURCE_INPUT:'+p)
    if (root/c['fresh_output']).exists():raise ValueError('FRESH_NO_RETRY_RESUME')
    r=io.read(root/c['Stage_P_result']);stop=io.read(root/c['Stage_P_STOP'])
    if (r['status']!='STAGE_P_MISSING_OUTPUTS_COMPLETE' or r['original_41_call_run_complete'] is not False
            or stop['scientific_units']!=15 or stop['failed'] is not None or stop['retry'] or stop['resume']):
        raise ValueError('FIXED_STAGE_P_EVIDENCE')
    graph=evidence_graph(io.read(root/c['population']),fixed_word_record=r)
    return dict(status='PASSIVE_TECHNICAL_READY_CALIBRATION_NOT_READY',scientific_calls=0,
        arrays_decoded=0,new_H4_build_compile=0,Stage_C_calls=0,evidence=graph,
        maximum_logical_units=None,calibration_ready=False,mandatory_stop=True)
