"""Assemble this plan's saved metadata; never start a scientific worker."""
from copy import deepcopy
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parent
SOURCE = Path('/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-distributed-q3-source-28d976b99')
SHA = '28d976b99cdf8329adcc1b79327e329e947ef774'
sys.dont_write_bytecode = True
if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)
sys.path.insert(0, str(SOURCE / 'src'))
from trottertracks import rpe_method_optimizer as O


def read(path):
    path = Path(path)
    if not path.exists():
        path = path.with_suffix(path.suffix + '.gz')
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def rational(value):
    value = O.frac(value)
    return dict(numerator=value.numerator, denominator=value.denominator,
                decimal_display=float(value))


def main():
    catalog = read(BASE / 'checks/CATALOG_INVENTORY.json')
    pf = read(BASE / 'checks/PF_INVENTORY.json')
    server = read(BASE / 'checks/SERVER2_INVENTORY.json')
    result = read(BASE / 'checks/full_history/ROUND_RESULT.json')
    problem = read(BASE / 'inputs/PROBLEM.json')
    reader, _, normalized = O.load_catalog(SOURCE, problem)
    baselines = {}
    for point in catalog['minima_details']:
        method = point['method']
        baselines[method] = dict(cost=point['conditional_cost'], configuration_id=point['configuration_id'],
            N_X=point['N_X'], N_Y=point['N_Y'], configuration_source=point['configuration_source'],
            evidence=point['q3_source'], status='SAVED_FEASIBLE',
            catalog_record=point['catalog_record_locator'])
    for method in ('FULL_PF', 'TRUNCATED_PF', 'PR_DF'):
        best = result['methods'][method]['best']
        baselines[method].update(cost=best['cost'], N_X=best['N_X'], N_Y=best['N_Y'],
            configuration_id=best['configuration_id'], normal_entry_evidence_kind=best['evidence'],
            verified_by='fixed existing Executor.prior_records + own candidate prices + exact saved MSE gate')
    baselines['TRUNCATED_PF']['snapshot_evidence'] = 'inputs/pf_snapshot/TRUNCATED_PF/2f8a9c696808106e594f/'
    baselines['TRUNCATED_PF']['evidence'] = 'inputs/pf_snapshot/TRUNCATED_PF/2f8a9c696808106e594f/OUTCOME.json'
    baselines['PR_DF']['snapshot_evidence'] = 'inputs/pr_snapshot/PR_DF/99ecbaa68b07b0a9ab47/'
    baselines['PR_DF']['evidence'] = 'inputs/pr_snapshot/PR_DF/99ecbaa68b07b0a9ab47/OUTCOME.json'
    for method in ('TRUNCATED_PF', 'PR_DF'):
        baselines[method]['previous_catalog_point'] = baselines[method].pop('catalog_record')
        baselines[method]['latest_allocation_not_in_fixed_C_catalog'] = True
    baselines['PR_PAULI']['normal_entry_price_binding_missing'] = True
    rows = []
    dependencies = []
    for method, output in result['methods'].items():
        pending = output['pending_request']
        assignment = read(BASE / 'assignments' / (method + '.json'))
        request = assignment['request']
        point = next(p for p in normalized['candidates'] if p['configuration_id'] == request['configuration_id'])
        config = reader.read(point['configuration_source'])
        config = config.get('configuration', config)
        price = O.frac(pending['cost'])
        current = O.frac(baselines[method]['cost'])
        delta = current - price
        observed = read(BASE / 'dry-check/server2' / method / 'READINESS.json')
        known = any(e['configuration_id'] == request['configuration_id'] and
                    e['N_X'] == request['N_X'] and e['N_Y'] == request['N_Y']
                    for e in output['evaluated'])
        assert not known, 'Do not report an already evaluated vector as new'
        row = dict(method=method, job_id=request['request_id'], configuration_id=request['configuration_id'],
            configuration_source=point['configuration_source'], formula=point['formula'], L_D=point['L_D'],
            p=point['L_D'] if method in ('FULL_PF', 'TRUNCATED_PF') else None,
            requested_delta_pf=point['delta_pf'], h=point['h'], kappa=point.get('kappa'),
            q=[r['q_PF'] for r in config['rounds']], R=[r.get('R_outer') for r in config['rounds']],
            actual_delta_pf=[r['actual_delta_pf'] for r in config['rounds']],
            N_X=request['N_X'], N_Y=request['N_Y'], classification=pending['cls'], origin=pending['origin'],
            conditional_cost_if_Q3_passes=rational(price), decrease_RZ_if_passes=rational(delta),
            decrease_percent_if_passes=rational(100 * delta / current),
            immediately_beats_current_minimum_if_passes=delta > 0,
            status='UNEVALUATED', MSE_lower=None, MSE_upper=None,
            evaluation_contract=dict(problem_accuracy=problem['accuracy'], target=problem['target'],
                route_kind=request['route']['kind'], expected=request['expected']),
            existing_evaluation_match_in_validated_context=known,
            request='requests/' + method + '.json', assignment='assignments/' + method + '.json',
            independent_of_other_first_Q3_results=True,
            reason=('UPDATE: decrease own saved incumbent and method minimum' if pending['cls'] == 'UPDATE' else
                    'EXPLORE: existing METHOD_MINIMUM_PRIORITY_V2 grants another configuration an opportunity; this allocation does not yet beat the method minimum'),
            first_wave_scientific_execution_ready=observed['status'] == 'REAL_HOST_DRY_CHECK_PASS',
            production_dispatch_performed=False,
            host_checks={
                'server2': dict(classification='実機dry-check PASS' if observed['status'] == 'REAL_HOST_DRY_CHECK_PASS' else '実機dry-checkで不足を確認',
                    result='dry-check/server2/' + method + '/READINESS.json', status=observed['status'],
                    missing=observed.get('error')),
                'server1': dict(classification='今回は実機未確認',
                    saved_scope='Artificial mailbox PASS only; known Python 3.12.3 differs from the real route requirement; fixed 28d and route/input placement not checked on this host for this candidate',
                    instructions='LOCAL0_DRY_CHECK.md'),
                'local': dict(classification='今回は実機未確認',
                    saved_scope='Artificial mailbox PASS and older PF/Pauli metadata checks; PR-DF nested server path was missing. This candidate at SOURCE28d has not been checked on RomuTail',
                    instructions='LOCAL0_DRY_CHECK.md')})
        if method == 'PR_PAULI':
            row['first_wave_scientific_execution_ready'] = False
            row['plan_qualification'] = 'PROVISIONAL: fixed entry excludes latest 410199237 total-only catalog point for PRICE_26AXIS; its internal 437758513 baseline must not replace the global saved minimum'
        rows.append(row)
        for choice in pending['queue']:
            if choice['origin']['kind'] == 'CHAIN_LOOKAHEAD_IF_PREVIOUS_PASSES':
                dependencies.append(dict(method=method, candidate=choice['candidate'],
                    N_X=choice['vector'][::2], N_Y=choice['vector'][1::2], cost=choice['cost'],
                    origin=choice['origin'], status='CONDITIONAL_PROPOSAL_NOT_FIRST_WAVE',
                    required='Earlier Q3 result must pass before this becomes the rule-selected next candidate; HYBRID speculation is not independent work'))
    holds = [dict(job_id=j['job_id'], method=j['method'], reason='R06_API_PRESTART_FAILURE_UNKNOWN_NO_AUTOMATIC_RETRY')
             for j in server['jobs'] if j['outcome_sha256'] is None]
    holds.append(dict(job_id='f999a51b2e1fd9e317e8', method='TRUNCATED_PF',
        reason='STARTED_OUTCOME_UNKNOWN_PARTIAL_M6_NO_FINAL_RETURN', inspection='checks/B0I_UNKNOWN_INSPECTION.json'))
    assert len(holds) == 5
    selection = read(BASE / 'checks/pf_auto/SELECTION.json')
    passive = next(p for p in catalog['minima_details'] if p['method'] == 'PR_PAULI')['passive_price_reconstruction']
    stages = [json.loads(line) for line in (BASE / 'checks/full_history/ROUND_STAGES.jsonl').read_text().splitlines()]
    summary = dict(schema='RPE_DISTRIBUTED_REAL_H4_PLAN_V1', date_jst='2026-11-17',
        scope='PLAN_ONLY_AND_REAL_INPUT_DRY_CHECK_NO_SCIENTIFIC_EXECUTION',
        computation_source=SHA, normal_entry='scripts/run_rpe_schedule_search.py',
        problem='inputs/PROBLEM.json', round='inputs/FULL_HISTORY_PLAN_ROUND.json',
        plan_timing=dict(first_stage_to_last_stage_seconds=stages[-1]['epoch'] - stages[0]['epoch'],
            before_SAVE_RESULTS_seconds=result['wall_seconds'],
            SAVE_RESULTS_seconds=stages[-1]['seconds'], scientific_Q3_time_measured=False),
        catalog=dict(commit=catalog['catalog_source']['fixed_commit'], path=catalog['catalog_source']['catalog_path'], git_blob=catalog['catalog_source']['git_blob']),
        input_comparison=catalog['comparison'], current_best_saved=baselines,
        first_independent_Q3_candidates=rows, conditional_following_candidates=dependencies,
        additional_automatic_B0I_preparation=dict(selection=selection['selected'], rule=selection['rule'],
            reason=selection['reason'], status='INPUTS_AND_26_PRICES_REQUIRED_NO_SHOT_VECTOR_SELECTED',
            conditional_cost=None, expected_MSE=None, Q3_request_created=False),
        prior_reuse={m:v['prior_rounds'] for m,v in result['methods'].items()},
        prior_reused_fresh_total=sum(v['prior_rounds']['loaded'] for v in result['methods'].values()),
        prior_already_known_total=sum(v['prior_rounds']['already_known'] for v in result['methods'].values()),
        unresolved_scientific_jobs_never_requeue=holds,
        pauli_saved_unit_price_binding=dict(fixed_entry_best=result['methods']['PR_PAULI']['best']['cost'],
            authoritative_saved_minimum=baselines['PR_PAULI']['cost'], passive_existing_MODEL_quote=passive,
            needed='Bind the recovered configuration-specific 26-axis saved MODEL prices to the normal optimizer input before a production Pauli search; do not relabel the fixed entry as latest-global complete'),
        supplementary_schedule_plan='checks/WIDTH_FRONTIER.json',
        source_bound_input_snapshots='inputs/SNAPSHOT_LOCATORS.json',
        future_round_draft='inputs/NEXT_RUN_ROUND_DRAFT.json', commands='START_RESUME.md',
        local0_remote_checks='LOCAL0_DRY_CHECK.md',
        new_Q3=0, new_signals=0, new_calibrations=0, new_compiles=0, production_queue_submissions=0,
        old_acceptance_reruns=0,
        claim_scope='Conditional RAW IID model MSE and accepted saved/model price scenarios; not formal E[C], CI, or final G/winner')
    (BASE / 'RESULT.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print('RESULT.json', [(r['method'], r['job_id'], r['host_checks']['server2']['classification']) for r in rows])


if __name__ == '__main__':
    main()
