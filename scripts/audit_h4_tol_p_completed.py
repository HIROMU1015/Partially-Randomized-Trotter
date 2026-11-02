"""Prepared post-run audit; never calls P science or recomputes its statistics."""
from collections import Counter
from pathlib import Path
import h4_tol_p_contract as p
from audit_h4_tol_d_completed import owned_receipt
c = p.c


def audit(spec, grant):
    output = Path(spec['output']); run = output.parent
    native, terminal, result = (c.read(x) for x in (run/'native_exit.json', output/'terminal.json', output/'result.json'))
    p.verify_bytes(spec)
    c.require(grant['approved_by_user'] and grant['runtime_authorization'] and grant['one_shot']
        and not grant['automatic_retry'] and not grant['next_stage_authorized'], 'APPROVED_P_ONLY')
    c.require(grant['scope'] == p.SCOPE and grant['stage'] == 'P' and spec['status'] == 'SEALED_FOR_EXECUTION'
        and spec['pending'] == [] and c.sha(grant['approval_receipt']) == grant['approval_receipt_sha256'],
        'P_ONLY_APPROVAL_RECEIPT')
    c.require(c.digest(spec) == grant['contract_digest'] == result['contract_digest']
        == c.read(spec['lock'])['contract_digest'], 'SAME_ONE_SHOT_CONTRACT')
    c.require(native['exit_code'] == 0 and native['same_parent_reaped']
        and native['result_sha256'] == terminal['result_sha256'] == c.sha(output/'result.json'), 'NATIVE_EXIT_REAP_SHA')
    c.require(result['SOURCE'] == terminal['SOURCE'] == native['SOURCE'] == spec['SOURCE']
        and result['status'] == terminal['status'] == 'COMPLETE_MANDATORY_STOP'
        and terminal['own_workers_reaped'] and terminal['final_closure_checked']
        and not terminal['automatic_retry'] and not terminal['next_stage_authorized']
        and not (output/'STOP.json').exists(), 'CLEAN_FINAL_TERMINAL')
    tasks = c.read(spec['seed_registry'])['tasks']; by_id = {t['id']:t for t in tasks}
    measured = c.read(output/'paired_measurements.json')
    c.require(measured['old_cohorts_not_pooled'], 'NO_POOLING')
    rows = measured['rows']
    # validate_pairs is structural only: no cost mean, variance, bootstrap or design evaluation.
    from h4_tol_p_science import validate_pairs
    calls = validate_pairs(rows, tasks)
    receipts = []
    for phase, first in [('first_control_workers', True), ('remaining_workers', False)]:
        expected = {t['id'] for t in tasks if (t['replica'] == 0) == first}
        receipts.extend(owned_receipt(output/phase, expected))
    first_receipt = c.read(output/'first_control_workers/accepted_native_receipt.json')
    rest_receipt = c.read(output/'remaining_workers/accepted_native_receipt.json')
    c.require(min(w['started'] for w in rest_receipt['workers']) >= max(
        w.get('last_resource_observation', w['started']) for w in first_receipt['workers']),
        'FIRST_BATCH_PRECEDES_REMAINING')
    strata = {cid: Counter() for cid in p.N}
    for row in rows:
        tid = row['cell_id']+'_'+str(row['replica']); task = by_id[tid]
        work = output/('first_control_workers' if row['replica'] == 0 else 'remaining_workers')/tid
        c.require(row == c.read(work/'task_result.json'), 'NATIVE_ROW_SHA_BOUND')
        trajectory = c.read(work/'trajectory.json')
        c.require(trajectory['cell'] == task['cell'] and trajectory['seed'] == task['seed']
            and trajectory['step_seeds'] == task['step_seeds'] and not trajectory['conditional']
            and not trajectory['quantum_weights_changed'] and trajectory['cohort'] == p.COHORT
            and trajectory['event_digest'] == row['event_digest'] == c.digest(trajectory['events']), 'FROZEN_EVENT_IDENTITY')
        c.require(len(trajectory['events']) == task['cell']['q']
            and sum(map(len, trajectory['events'])) == task['cell']['R'], 'EXACT_EVENT_SLOT_COUNT')
        c.require(c.event_stratum(trajectory['events']) == row['stratum'], 'SAVED_STRATUM')
        strata[row['cell_id']][row['stratum']] += 1
        ledger = c.read(work/'calls_ledger.json')
        c.require(ledger['nonrefundable'] and ledger['reserved'] == row['calls'], 'NATIVE_CALL_LEDGER')
        for axis in c.AXES: c.require(row['axes'][axis] == c.read(work/('measured_'+axis+'.json')), 'MEASURED_AXES')
        if row['replica'] == 0:
            ctrl = c.read(work/'control.json')
            c.require(ctrl['status'] == 'PASS' and ctrl['max_error'] < ctrl['tolerance'] == 1e-9
                and ctrl['control_probes'] == 4 and ctrl['independent_event_oracles'] == 1, 'CONTROL_ORACLE')
    expected_calls = {k:p.plan()['calls'][k] for k in calls}
    c.require(calls == result['actual_worker_calls'] == expected_calls, 'EXACT_TOTAL_SCIENTIFIC_CALLS')
    c.require(c.read(output/'preparations.json')['reconstructions'] == 2, 'TWO_PREPARATIONS')
    c.require(c.read(output/'endpoint_accounting_reserved.json') == dict(passes=1, comparisons=2, nonrefundable=True),
        'SOLE_ENDPOINT_PASS')
    eps = c.read(spec['endpoints'])
    c.require(len(result['comparisons']) == 2 and result['endpoint_accounting_pass'] == 1
        and result['endpoint_comparisons'] == 2, 'TWO_COMPLETE_ENDPOINTS')
    for ep, row in zip(eps, result['comparisons']):
        c.require(all(row[k] == v for k,v in ep.items()) and row['result']['n'] == ep['n']
            and row['result']['status'] in ('REPRODUCED', 'NOT_REPRODUCED', 'INCONCLUSIVE')
            and row['unseen_stratum_cost'] is None and row['unseen_stratum_variance'] is None,
            'FIXED_SAVED_PF_AND_UNKNOWN_TAIL')
        c.require(row['strata'] == {s:strata[ep['cell']['id']][s] for s in row['strata']}, 'COMPLETE_STRATUM_COUNTS')
    c.require(not any(result[k] for k in ('old_samples_pooled','population_mean_certified','actual_power_certified',
        'candidate_reselected','shots_recalculated','unseen_strata_zero_filled','next_stage_authorized')), 'SCOPE_LIMITS')
    private = output.parent.parent
    for name in ('prior_preservation_manifest_PRIVATE.json', 'technical_seal_preservation_manifest_PRIVATE.json'):
        for row in c.read(private/name):
            c.require(c.sha(row['path']) == row['sha256'], 'PRIOR_OR_FINAL_SEAL_BYTES_PRESERVED')
    return dict(status='PASS_COMPLETE_P_MANDATORY_STOP', workers_reaped=len(receipts),
        actual_worker_calls=calls, numeric_statistics_recomputed=False, mandatory_STOP=True)
