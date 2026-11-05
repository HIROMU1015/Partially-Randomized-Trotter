"""Execution orchestration for the fixed research-C integrated contract.

Only existing qualified numerical APIs are called. Import is passive; this file
does not launch a job or amend fixed scientific SOURCE, conditions or budgets.
Keep returned opaque state alive through selection and final evaluation: the
contract does not budget repeating candidate setup/PF/native common work.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path

SETUP = dict(selected_load=1, reference_load=1, candidate_binding=1,
             native_model=1, general_factor_gate=9, candidate_ground=1)
REFERENCE32 = dict(law_prepare=1, law_gate=1, normalization_gate=1,
    kernel_reference=1, exact_evolution=13, round_reference=13,
    round_classification=13, feasibility_summary=1, signal_qualification=3,
    round_accuracy=39, energy_assessment=3, shots_summary=3, native_admission=1)
REFERENCE_EXTRA = {k:v for k,v in REFERENCE32.items() if k not in ('kernel_reference','exact_evolution')}
REFERENCE_EXTRA['corrected_kernel'] = 1
NATIVE_COMMON = dict(car_context=1, raw_factor=9, full_sector_gate=1,
    reference_context=1, full_law_gate=1, gaussian_build=9,
    relative_gaussian=4, shell_load=6, shell_assessment=3)
NATIVE32 = {k:32 for k in ('trajectory_sample','kernel_build','trajectory_reference',
    'raw_operator_gate','compile','compiled_operator_gate','adjoint_pair_gate',
    'XY_gate','kernel_metrics')}
NATIVE32['calibration_summary'] = 1
ACCOUNTING = dict(cost_calibration=1, serial_cost=26, total_aggregation=3,
                  calibration_uncertainty=3)
E_WRAPPER = dict(native_rows_validation=1, accounting_binding=1,
                price_statistics=2, secondary_summary=1, arm_summary=1)


def _sum_limits(*parts):
    total=Counter()
    for part in parts:total.update(part)
    return dict(total)


SELECTION_LIMITS = _sum_limits(SETUP,REFERENCE32,REFERENCE_EXTRA,REFERENCE_EXTRA,
    NATIVE_COMMON,NATIVE32,NATIVE32,NATIVE32,ACCOUNTING,ACCOUNTING,ACCOUNTING)
FINAL_LIMITS = _sum_limits(NATIVE32,NATIVE32,dict(final_calibration=1),ACCOUNTING,E_WRAPPER)
assert sum(SELECTION_LIMITS.values()) == 1268
assert sum(FINAL_LIMITS.values()) == 618


def _backend():
    from . import c_integrated_retuning_adapter as adapter
    from . import c_near_r32_statistics as statistics
    from .c1_stage3_integrated import Calls
    from .c_update064_r32_downstream import D_LIMITS
    ret,_,_=adapter._engines()
    return adapter,statistics,ret.io,Calls,D_LIMITS


def runtime_contract(contract, source_digest, shell_paths):
    """Minimal runtime packet for old native APIs, without old run() guards.

    Shell paths are repository-relative raw/compiled QPY paths for head/X/Y;
    the parent preflight binds their bytes and the frozen evaluation profile.
    """
    if not isinstance(source_digest,str) or not source_digest:
        raise ValueError('SOURCE_DIGEST_REQUIRED')
    if set(shell_paths)!={'head','X','Y'} or any(set(v)!={'raw','compiled'} for v in shell_paths.values()):
        raise ValueError('THREE_RAW_COMPILED_SHELL_PAIRS_REQUIRED')
    for paths in shell_paths.values():
        for path in paths.values():
            p=Path(path)
            if p.is_absolute() or '..' in p.parts:raise ValueError('REPOSITORY_RELATIVE_SHELL_REQUIRED')
    return dict(source_manifest_sha256=source_digest,shells=deepcopy(shell_paths),
                compiler=deepcopy(contract['compiler']),target=deepcopy(contract['target']),
                allowances=deepcopy(contract['allowances']))


def cohort_key(task, candidate):
    return '/'.join(str(x) for x in (task['geometry'],task['arm'],task['start'],
        'strength_'+str(task['strength']),'budget_'+str(task['budget']),candidate))


def _call_summary(io,out,call,maximum,status):
    io.save(Path(out)/'call_summary.json',dict(status=status,maximum=maximum,
        scientific_units=sum(call.counts.values()),call_counts=dict(call.counts),
        failed=call.first_failure,retry=0,resume=0))


def _row(task,common,candidate,r,result=None):
    value=None
    if result is not None and result['native_admitted']:
        totals=result['lanes']['ideal_exact_native'].get('conditional_totals',{}).get('totals')
        if totals is not None:value=totals['rz_count']
    return dict(task,input_sha256=common['input_identity'],candidate_sha256=candidate,
        r=r,cohort_role='selection',samples=32,actual_samples=0 if value is None else 32,
        correctness='PASS',eligible=value is not None,total_RZ=value)


def candidate_selection(root,runtime,candidate_path,reference_path,physical_record,
                        common,task,out):
    """One qualified candidate's fixed 3r selection, maximum1268 calls.

    Returns a serializable report and opaque state. The parent selects strengths
    and policies across these rows, freezes the transfer recipe, and retains the
    state for selected final pairs. No candidate is replaced after an ineligible
    outcome. Correctness exceptions preserve the local journal and propagate.
    """
    adapter,statistics,io,Calls,_=_backend()
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    call=Calls(out,SELECTION_LIMITS,1268)
    candidate=physical_record['candidate_sha256']
    state=dict(candidate_sha256=candidate,input_sha256=common['input_identity'],
        task=dict(task),common=common,A=None,policies={},native_prepared={},D=None,
        D_executed=False,finals={},selection_output=str(out.resolve()))
    try:
        if not physical_record['budget_PASS']:
            status='UNQUALIFIED_DF_BUDGET';qualification=dict(status=status)
        else:
            a,qualification=adapter.prepare_candidate(candidate_path,reference_path,
                physical_record,common,out/'A',call)
            state['A']=a
            status=qualification['status']
        rows=[]
        if state['A'] is None:
            rows=[_row(task,common,candidate,r) for r in (16,32,64)]
        else:
            pf_cache=None
            for r in (32,16,64):
                a,result,packet,pf_cache=adapter.reference_r(state['A'],r,out/f'r{r}'/'B',call,pf_cache=pf_cache)
                io.save(out/f'r{r}'/'references.json',result)
                if result['native_admitted']:
                    result,packet=adapter.native_cohort(root,runtime,out/f'r{r}'/'selection',call,a,result,packet,
                        split='selection',cohort_key=cohort_key(task,candidate),prepared=state['native_prepared'])
                state['policies'][r]=dict(A=a,result=result,packet=packet)
                rows.append(_row(task,common,candidate,r,result))
            state['PF_cache']=pf_cache
            status='SELECTION_COMPLETE'
        rows.sort(key=lambda x:x['r'])
        report=dict(status=status,candidate_sha256=candidate,input_sha256=common['input_identity'],
            qualification=qualification,task=task,rows=rows,actual_CA_achieved=None)
        io.save(out/'result.json',report)
        _call_summary(io,out,call,1268,status)
        return report,state
    except BaseException:
        _call_summary(io,out,call,1268,'CORRECTNESS_FAILURE')
        raise


def candidate_d(root,runtime,state,out):
    """Unique selected candidate's same-J/U D, maximum141 calls.

    Parent deduplicates candidate identities; this function rejects a repeated
    invocation on the same retained state rather than consuming a second run.
    D admission is independent of PR admission.
    """
    if state['D_executed']:raise ValueError('D_ALREADY_PRODUCED')
    if state['A'] is None:raise ValueError('QUALIFIED_CANDIDATE_REQUIRED')
    adapter,_,io,Calls,limits=_backend()
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    call=Calls(out,limits,141)
    state['D_executed']=True
    try:
        value=adapter.strong_d(root,runtime,out/'D',call,state['A'])
        state['D']=value
        state['D_output']=str(out.resolve())
        io.save(out/'runtime_binding.json',dict(candidate_sha256=state['candidate_sha256'],
            input_sha256=state['input_sha256'],same_A_packet_as_selection=True,
            actual_ground_origin=str(Path(state['selection_output'])/'A'/'ground_candidate.npz'),
            legacy_inner_labels='update064 and ../A are historical strings in the reused fixed D engine'))
        _call_summary(io,out,call,141,value['status'])
        return value
    except BaseException:
        _call_summary(io,out,call,141,'CORRECTNESS_FAILURE')
        raise


def final_pair(root,runtime,state,r,out):
    """Selected candidate/r independent final64 plus E wrapper, max618 calls.

    Native immutable common context comes from this candidate's selection; the
    stored D result is shared across r. The parent deduplicates exact final pairs
    and reuses this return value when common/tuned select the same pair.
    """
    if r in state['finals']:raise ValueError('FINAL_PAIR_ALREADY_PRODUCED')
    policy=state['policies'][r]
    if not policy['result']['native_admitted']:raise ValueError('SELECTED_POLICY_NOT_ELIGIBLE')
    if not state['native_prepared']:raise ValueError('RETAIN_SELECTION_NATIVE_CONTEXT')
    if not state['D_executed']:raise ValueError('SAME_CANDIDATE_D_REQUIRED_FIRST')
    adapter,statistics,io,Calls,_=_backend()
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    call=Calls(out,FINAL_LIMITS,618)
    try:
        result,packet=adapter.native_cohort(root,runtime,out/'native',call,policy['A'],policy['result'],policy['packet'],
            split='final',cohort_key=cohort_key(state['task'],state['candidate_sha256']),prepared=state['native_prepared'])
        rows=packet['kernel_samples']
        call('native_rows_validation','complete independent final64',lambda:statistics.shared.validate_rows(rows,64))
        bindings=call('accounting_binding','same kernel occurrence/shell accounting',lambda:statistics.affine(packet,result))
        secondary_policy=dict(statistics.POLICY,samples=64,confidence=.95)
        summary=call('secondary_summary','individual nominal secondary PR/D;all metric summaries',
            lambda:statistics.arm_summary(rows,bindings,state['D'],calibration=result['calibration'],
                call=call,expected_n=64,statistics=secondary_policy))
        report=call('arm_summary','final64 qualification and statistics',lambda:dict(
            status='FINAL_PAIR_COMPLETE',candidate_sha256=state['candidate_sha256'],input_sha256=state['input_sha256'],
            r=r,task=state['task'],samples=64,cohort_role='final',bindings=bindings,statistics=summary,
            selection_reference=state['selection_output'],same_JU_D_reference=state['D_output'],
            actual_CA_achieved=None))
        final=dict(report=report,result=result,packet=packet,rows=rows,bindings=bindings,
                   candidate_sha256=state['candidate_sha256'],input_sha256=state['input_sha256'],r=r)
        io.save(out/'result.json',report)
        _call_summary(io,out,call,618,report['status'])
        state['finals'][r]=final
        return report,final
    except BaseException:
        _call_summary(io,out,call,618,'CORRECTNESS_FAILURE')
        raise


def compare_final(pr,entry,out,call,*,primary):
    """One registered PR-minus-ENTRY comparison; parent owns geometry counts.

    Primary=True is restricted to C0/update64 and confidence98.75%; remaining
    registered cells use descriptive nominal95% intervals without a simultaneous
    superiority claim. Ineligible/missing inputs retain an unresolved result.
    """
    _,statistics,io,_,_=_backend()
    def evaluate():
        if pr is None or entry is None:
            return dict(status='UNRESOLVED_MISSING_FINAL_PAIR',primary_family=primary,comparison=None)
        p,e=pr['report'],entry['report']
        for key in ('geometry','start','budget'):
            if p['task'][key]!=e['task'][key]:raise ValueError('MIXED_COMPARISON_GROUP')
        if p['task']['arm']!='PR' or e['task']['arm']!='ENTRY' or p['input_sha256']!=e['input_sha256']:
            raise ValueError('MIXED_COMPARISON_INPUT_OR_ARM')
        if primary and (p['task']['start']!='C0' or p['task']['budget']!=64):
            raise ValueError('PRIMARY_C0_UPDATE64_ONLY')
        lane='ideal_exact_native'
        if pr['bindings'][lane] is None or entry['bindings'][lane] is None:
            return dict(status='UNRESOLVED_FINAL_LANE_INELIGIBLE',primary_family=primary,comparison=None)
        policy=dict(statistics.POLICY,samples=64,confidence=.9875 if primary else .95)
        value=statistics.primary(pr['rows'],entry['rows'],pr['bindings'][lane],entry['bindings'][lane],
            calibrations=(pr['result']['calibration'],entry['result']['calibration']),expected_n=64,statistics=policy)
        return dict(status='REGISTERED_COMPARISON_COMPLETE',primary_family=primary,comparison=value,
            confidence=policy['confidence'],simultaneous_all_cell_superiority=None)
    result=call('main_comparison','fixed registered final PR-minus-ENTRY',evaluate)
    io.save(out,result)
    return result
