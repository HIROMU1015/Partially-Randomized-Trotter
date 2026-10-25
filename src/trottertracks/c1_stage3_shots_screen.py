"""Optimistic shots within the adopted sufficient formula, never a CA certificate.

Import/preflight do not decode saved scientific values. Real saved-data
reanalysis requires separate chat permission. No circuit or operator path.
"""
from collections import Counter
from dataclasses import asdict
from fractions import Fraction
import importlib.metadata
import math
from pathlib import Path
import resource
import sys
import time
import traceback

from . import c1_stage1_execution as io
from . import c1_stage3_accuracy as accuracy
from . import c1_stage3_feasibility as feasibility
from . import c1_stage3_integrated as integrated

SCOPE = 'C1_STAGE3_OPTIMISTIC_SHOTS_SCREEN_V1'
FAMILY = 'artifacts/research_c_c1_stage3_shots_science'
ASSUMPTION = 'Counterfactual synthesis/law signal errors zero; actual unknown errors remain unqualified'
FORMULA = 'ceil(2*log(2/alpha_axis)/[A*(1-epsPF-epsRTE)*sin(beta_stat)/sqrt(2)]^2)'


def call_limits():
    return dict(saved_evidence_load=1, optimistic_round_assessment=260,
                candidate_shots_summary=20, screening_summary=1)


def preflight(root, contract_path):
    """Bytes and contract metadata only; never parse the 260 saved cells."""
    root, contract_path = Path(root).resolve(), Path(contract_path).resolve()
    c = io.read(contract_path)
    ids = [x.id for x in integrated.candidates()]
    if (c['scope'] != SCOPE or c['call_limits'] != call_limits()
            or c['maximum_logical_scientific_units'] != sum(call_limits().values())
            or c['candidate_ids'] != ids or c['rounds'] != list(range(13))
            or c['retry'] != 0 or c['resume'] != 0 or c['new_fit'] != 0):
        raise ValueError('SHOTS_SCOPE_GRID_CALLS')
    parent = root/c['parent_contract']
    if io.sha(parent) != c['parent_contract_sha256']:
        raise ValueError('PARENT_CONTRACT')
    p = io.read(parent)
    if (c['accuracy_target'] != p['proposal']['accuracy_target']
            or c['physical_conditions'] != p['physical_conditions']
            or c['parent_proposal'] != p['proposal']
            or c['runtime_required_packages'] != p['runtime_required_packages']):
        raise ValueError('CHANGED_SCIENTIFIC_CONDITIONS')
    source = io.read(root/c['source_manifest'])
    if io.sha(root/c['source_manifest']) != c['source_manifest_sha256']:
        raise ValueError('SOURCE_MANIFEST')
    for name, want in source['files'].items():
        if io.sha(root/name) != want: raise ValueError('SOURCE: '+name)
    expected = {str(Path(c['saved_run'])/name) for name in ['result.json','prepared.json','STOP.json']}
    expected.update(str(Path(c['saved_run'])/'candidates'/(cid+'.json')) for cid in ids)
    if set(c['saved_inputs']) != expected: raise ValueError('SAVED_INPUT_SET')
    for name, want in c['saved_inputs'].items():
        if io.sha(root/name) != want: raise ValueError('SAVED_INPUT: '+name)
    versions = {name:importlib.metadata.version(name) for name in c['runtime_required_packages']}
    if versions != c['runtime_required_packages'] or sys.version_info[:2] != (3,11):
        raise ValueError('NUMERICAL_ENVIRONMENT')
    return dict(root=root, contract=c, source=source, contract_path=contract_path,runtime=versions)


def _count(unit_coordinate, attenuation, alpha):
    """Reuse the float helper where it resolves integers; extend its range.

    The fallback is the same formula on binary64 geometry/log constants,
    with exact rational products and Python integer ceil. It is not an
    interval enclosure of transcendental functions or saved evidence.
    """
    coordinate = unit_coordinate*attenuation
    log_term = math.log(2./alpha)
    if coordinate > 0 and coordinate*coordinate > 0:
        value = (2./(coordinate*coordinate))*log_term
        if math.isfinite(value) and value < 2**53:
            n = accuracy._legacy('rpe_resource_accounting')._hoeffding_shots(coordinate,alpha)
            return n, 'existing_binary64_helper'
    epsilon = Fraction.from_float(unit_coordinate)*Fraction.from_float(attenuation)
    value = 2*Fraction.from_float(log_term)/(epsilon*epsilon)
    return -(-value.numerator//value.denominator), 'rational_ceil_binary64_constants'


def assess_optimistic_round(target, row, *, target_id):
    """Counterfactual zero-extra-error estimate; actual errors stay None.

    Normalize A=1 before reusing the old radius/phase geometry, then scale
    the coordinate by the unchanged saved A. This avoids A*eps underflow.
    The no-attenuation diagnostic retains exactly the same PF/RTE errors.
    """
    common = dict(m=row['m'], saved_reference=dict(row),assumption=ASSUMPTION, actual_synthesis_signal_error=None,
                  actual_law_signal_error=None, CA_achieved=None,
                  guaranteed_actual_shots=None, universal_shots_lower_bound=False,
                  evidence_kind='optimistic_numerical_screen_not_a_guarantee')
    known = feasibility.classify_round(row,target)
    if known['status'] != 'POTENTIALLY_FEASIBLE':
        return dict(**common,status='INFEASIBLE_REGISTERED_BOUND' if known['status']=='INFEASIBLE'
                    else 'UNRESOLVED_SAVED_VALUES',reason=known.get('reason'),axes=None,
                    saved_feasibility=known['status'])
    unit = accuracy.assess_round(target,accuracy.RoundEvidence(
        m=row['m'],attenuation=1.,pf_operator_error=row['pf_operator_error'],
        rte_corrected_operator_error=row['rte_corrected_operator_error'],
        synthesis_signal_error=0.,finite_law_signal_error=0.,target_id=target_id,
        state_model='own_DF_ground_state_oracle',alias_branch_verified=True,
        iid_per_shot=True,law_model='ideal_finite_RTE_law'))
    if unit['status'] != 'CONDITIONAL_ROUND_PASS':
        return dict(**common,status='UNRESOLVED_NUMERICAL_PHASE_BOUNDARY',axes=None,
                    saved_feasibility=known['status'],reason=unit['status'])
    coordinate = unit['epsilon_coordinate']; A = row['attenuation']
    n, method = _count(coordinate,A,target.alpha_axis)
    n0, method0 = _count(coordinate,1.,target.alpha_axis)
    axis = dict(optimistic_shots_decimal=str(n),no_attenuation_shots_decimal=str(n0),
                alpha=target.alpha_axis,integer_exceeds_int64=n>2**63-1,
                arithmetic=method,no_attenuation_arithmetic=method0)
    return dict(**common,status='OPTIMISTIC_FORMULA_ESTIMATE',saved_feasibility=known['status'],
        axes={a:dict(axis) for a in ('X','Y')},beta_deterministic=unit['beta_deterministic'],
        normalized_radius_lower=unit['radius_lower'],unit_coordinate=coordinate,
        coordinate_product_binary64=A*coordinate,coordinate_underflow_binary64=A*coordinate==0,
        attenuation_penalty_log10=math.log10(n)-math.log10(n0))


def candidate_summary(candidate, rounds):
    common = dict(id=candidate['id'],candidate=candidate['candidate'],prior_feasibility=candidate['status'],
        df_error_bound_ha=candidate['df_error_bound_ha'],assumption=ASSUMPTION,
        actual_synthesis_signal_error=None,actual_law_signal_error=None,
        CA_achieved=None,guaranteed_actual_shots=None,total_resources=None,
        final_resource_ranking=None,universal_shots_lower_bound=False,
        evidence_kind='optimistic_numerical_screen_not_a_guarantee')
    if candidate['status'] != 'POTENTIALLY_FEASIBLE':
        return dict(**common,status='EXCLUDED_PRIOR_'+candidate['status'],
                    optimistic_total_shots_decimal=None,no_attenuation_total_shots_decimal=None)
    if any(x['status'] != 'OPTIMISTIC_FORMULA_ESTIMATE' for x in rounds):
        return dict(**common,status='UNRESOLVED_ROUND_SCREEN',
                    optimistic_total_shots_decimal=None,no_attenuation_total_shots_decimal=None)
    total = sum(int(x['axes'][axis]['optimistic_shots_decimal']) for x in rounds for axis in ('X','Y'))
    total0 = sum(int(x['axes'][axis]['no_attenuation_shots_decimal']) for x in rounds for axis in ('X','Y'))
    return dict(**common,status='OPTIMISTIC_FORMULA_SCREENED',
        optimistic_total_shots_decimal=str(total),no_attenuation_total_shots_decimal=str(total0),
        optimistic_total_shots_log10=math.log10(total),
        attenuation_penalty_log10=math.log10(total)-math.log10(total0),
        integer_exceeds_int64=total>2**63-1)


def load_saved(v):
    """Only future permitted science parses the saved scalar evidence."""
    run = v['root']/v['contract']['saved_run']
    result, prepared, stop = (io.read(run/name) for name in ['result.json','prepared.json','STOP.json'])
    ids = v['contract']['candidate_ids']
    if (result['status'] != 'COMPLETE_FEASIBILITY_ONLY' or result['candidate_ids'] != ids
            or result['scientific_units'] != 639 or stop['failed'] or not stop['mandatory_stop']
            or prepared['state_preparation'] != 'ORACLE_EXCLUDED'
            or prepared['sampling_law']['model'] != 'ideal_finite_RTE_IID'):
        raise ValueError('SAVED_RUN_NOT_QUALIFIED_FOR_SCREEN')
    records = [io.read(run/'candidates'/(cid+'.json')) for cid in ids]
    statuses = dict((x['id'],x['status']) for x in result['candidate_statuses'])
    for candidate, expected in zip(records,integrated.candidates()):
        if (candidate['id'] != expected.id or candidate['candidate'] != asdict(expected)
                or candidate['status'] != statuses[expected.id]
                or [x['m'] for x in candidate['rounds']] != list(range(13))
                or len(candidate['feasibility']) != 13
                or candidate['CA_achieved'] is not None
                or candidate['synthesis_signal_error'] is not None
                or candidate['df_error_bound_ha'] != prepared['df_error_bounds_ha'][expected.arm]):
            raise ValueError('SAVED_CANDIDATE_SCHEMA')
    if Counter(x['status'] for x in records) != result['candidate_status_counts']:
        raise ValueError('SAVED_STATUS_COUNTS')
    return records


def run(loader, target, output):
    """Small scalar loop. No operator, circuit, cost, sampling or refit calls."""
    output = Path(output); output.mkdir(parents=True,exist_ok=False)
    (output/'candidates').mkdir()
    calls = integrated.Calls(output,call_limits(),sum(call_limits().values()))
    start, cpu = time.monotonic(),time.process_time()
    try:
        records = calls('saved_evidence_load','saved scalar bundle',loader)
        if [c['id'] for c in records] != [c.id for c in integrated.candidates()]:
            raise ValueError('REGISTERED_CANDIDATE_ORDER')
        summaries = []
        for candidate in records:
            cid = candidate['id']; rounds = []
            if [x['m'] for x in candidate['rounds']] != list(range(13)):
                raise ValueError('REGISTERED_ROUNDS')
            for row, prior in zip(candidate['rounds'],candidate['feasibility'],strict=True):
                def assess(row=row,prior=prior):
                    if feasibility.classify_round(row,target)['status'] != prior['status']:
                        raise ValueError('SAVED_FEASIBILITY_MISMATCH')
                    return assess_optimistic_round(target,row,target_id=cid)
                screen = calls('optimistic_round_assessment',f'{cid}:m{row["m"]}',assess)
                record = dict(saved_reference=row,screen=screen)
                integrated.progress(output,cid,'optimistic_shots',record)
                rounds.append(screen)
            def summarize():
                saved = feasibility.summarize(candidate['feasibility'],candidate['df_error_bound_ha'],target)
                if saved['status'] != candidate['status']: raise ValueError('SAVED_CANDIDATE_STATUS_MISMATCH')
                return candidate_summary(candidate,rounds)
            summary = calls('candidate_shots_summary',cid,summarize)
            io.save(output/'candidates'/(cid+'.json'),dict(**summary,rounds=rounds))
            summaries.append(summary)
        def final():
            return dict(status='COMPLETE_OPTIMISTIC_FORMULA_SCREEN',scope=SCOPE,
                candidate_status_counts=dict(Counter(x['status'] for x in summaries)),
                candidates=summaries,assumption=ASSUMPTION,formula=FORMULA,
                new_fit=0,optimizer=0,retry=0,resume=0,operator=0,PF_RTE_recomputed=0,
                circuit_build=0,compile=0,quantum_shots_drawn=0,CA_achieved=None,
                guaranteed_actual_shots=None,total_resources=None,final_resource_ranking=None,
                universal_shots_lower_bound=False,mandatory_stop=True)
        result = calls('screening_summary','registered-order table; no cost ranking',final)
        result.update(scientific_units=sum(calls.counts.values()),call_counts=dict(calls.counts))
        io.save(output/'result.json',result)
        return result
    except BaseException as error:
        calls.stop.set()
        io.save(output/'failure.json',dict(error_type=type(error).__name__,error=str(error),
            traceback=traceback.format_exc(),first_failure=calls.first_failure,call_counts=dict(calls.counts)))
        raise
    finally:
        io.save(output/'resources.json',dict(wall_seconds=time.monotonic()-start,
            process_cpu_seconds=time.process_time()-cpu,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            execution='260 scalar rows and small exact-rational counts; serial loop avoids process overhead; no resource caps'))
        io.save(output/'STOP.json',dict(mandatory_stop=True,failed=calls.stop.is_set(),
            scientific_units=sum(calls.counts.values()),call_counts=dict(calls.counts),retry=0,resume=0))


def execute(root, contract_path, output):
    """Separate explicit user permission is required for saved-data reanalysis."""
    v = preflight(root,contract_path)
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(v['root']/FAMILY):
        raise ValueError('FRESH_SHOTS_OUTPUT_REQUIRED')
    target = accuracy.AccuracyTarget(**v['contract']['accuracy_target'])
    try:
        return run(lambda:load_saved(v),target,output)
    finally:
        if output.exists():
            io.save(output/'SOURCE.json',v['source']);io.save(output/'contract.json',v['contract'])
            io.save(output/'runtime.json',dict(python=sys.version,executable=sys.executable,packages=v['runtime']))
