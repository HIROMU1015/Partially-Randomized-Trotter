"""Unadopted failure-aware scope, delegating A/B/C arithmetic and identity.

No sampler, compiler callback, continuing scientific runner, or fallback circuit.
Decision functions describe a future fixed-N lane, not execution authorization.
"""
from . import c_iid_calibration_integration as integrated
from . import c_fresh_iid_calibration as fresh
from . import c_totality_calibration as accounting


def requirements():
    return dict(status='UNADOPTED_FAILURE_AWARE_SCOPE', executable=False,
        failure_policy=None, N=None, delta_compile=None, delta_cal=None,
        operational_abort_allowance=None, cache_policy=None, nontermination_policy=None,
        compiler_failure_upper=None, unconditional_RPE_failure_upper=None,
        measurement_alpha='1/100', joint_001_policy_adopted=False,
        future_runner_requirements=[
            'fixed complete planned occurrence IDs;persist original word before raw build',
            'one raw attempt and at most one P0 attempt per eligible occurrence',
            'keep failed/unknown words;no discard,redraw,retry,resume or fallback',
            'record-all continuation only for adopted known algorithmic failure outcomes',
            'SOURCE/input/identity or mathematical correctness FAIL always stops dependent work',
            'unknown/interruption/nontermination stays unqualified;no invented timeout',
            'independent compiler outcome map or justified IID joint random map',
            'cache identity includes SOURCE,candidate,law,word,angle,phase,P0,environment',
            'same realized inverse;preserve draw multiplicities despite cache hits',
            'adopt estimand and selection transport before unconditional RPE connection'],
        all_q_words_compile_success_qualified=False, Stage_C_ready=False, actual_CA_achieved=None)


def continuation(status, *, failure_policy=None, correctness='PASS'):
    if correctness not in ('PASS','FAIL','UNKNOWN'): raise ValueError('CORRECTNESS_STATUS')
    if correctness != 'PASS': return 'STOP_CORRECTNESS_OR_UNKNOWN_NO_CONFIDENCE'
    if status not in fresh.STATUSES: raise ValueError('OUTCOME_SCHEMA')
    if failure_policy not in ('stop_on_failure','record_all_fixed_N'):
        return 'NOT_READY_UNADOPTED_POLICY'
    if status == 'SUCCESS': return 'CONTINUE_ONLY_WITHIN_ADOPTED_PLANNED_N'
    if status in ('GENERATED','INCOMPLETE'): return 'STOP_UNKNOWN_NO_FIXED_N_CONFIDENCE'
    if failure_policy == 'stop_on_failure': return 'STOP_FIRST_FAILURE_NO_REPLACEMENT'
    if status in ('RAW_BUILD_FAILED','P0_COMPILE_FAILED','NATIVE_INELIGIBLE'):
        return 'RECORD_FAILURE_CONTINUATION_REQUIRES_ADOPTED_FIXED_N_RUNNER'
    return 'STOP_CERTIFICATE_OR_NUMERICAL_CORRECTNESS_NOT_COMPILER_FAILURE'


def audit(rows, *, context, planned_ids, delta_compile=None, failure_policy=None, iid_assumed=False,
          correctness='UNKNOWN'):
    """Retain every draw; reuse C's source/word and occurrence identity checks."""
    rows = list(rows)
    if correctness not in ('PASS','FAIL','UNKNOWN'): raise ValueError('CORRECTNESS_STATUS')
    values = {}
    for event in ('pipeline','raw_build','compile_given_raw_success'):
        values[event] = integrated.compiler_failure_summary(rows, context=context,
            planned_ids=planned_ids, delta_compile=delta_compile if correctness=='PASS' else None,
            failure_policy=failure_policy, iid_assumed=iid_assumed, event=event)
    after_native = sum(p.get('raw_status') == p.get('compile_status') == 'SUCCESS'
        for r in rows if r['status'] in ('PER_WORD_CERTIFICATE_FAILED','NUMERICAL_UNEVALUABLE')
        for p in r['evidence'].get('occurrences', []))
    return dict(status='FAILURE_SCOPE_AUDIT_ONLY', rates=values,
        correctness_classification=correctness,
        native_success_with_unqualified_certificate=after_native,
        failed_words_excluded=False, redraw_allowed=False, fresh_fixed_N_runner_implemented=False,
        scientific_policy_adopted=False, all_q_words_compile_success_qualified=False,
        IID_original_law_mean_qualified=False, Stage_C_ready=False, actual_CA_achieved=None)


def operational_requests(shots_X=None, shots_Y=None, *, cache=None, deterministic_word_map=False):
    if cache is None:
        return dict(status='NOT_READY_CACHE_AND_SHOTS_POLICY', requests=None,
            symbolic=accounting.compile_requests(None,None), inverse_new_word_draws=0)
    return accounting.compile_requests(shots_X,shots_Y,cache=cache,
                                       deterministic_word_map=deterministic_word_map)


def confidence(events, *, p_fail_upper=None, requests=None):
    # No RPE scope premise is established by this technical implementation.
    return accounting.confidence(events, p_fail_upper=p_fail_upper, requests=requests,
                                 rpe_scope_qualified=False)


def run_science(*, execute_science=False):
    if not execute_science: raise PermissionError('SEPARATE_FIXED_SCIENTIFIC_PERMISSION')
    raise NotImplementedError('FAILURE_POLICY_N_DELTAS_ABORT_ESTIMAND_NOT_ADOPTED')
