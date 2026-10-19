"""Audit saved H2 A receipts/bytes only; no sampling, compile, or reaccounting."""
import argparse
from collections import Counter, defaultdict
import math
import os
from pathlib import Path

import h2_panels_contract as c
import prepare_h2_a_seal as seal
from prepare_h2_panels import preservation
from audit_h2_fp_completed import original_process_alive


def check_accounting(result, records, pairs, pf):
    """Check saved joins and cohort coverage without evaluating a cost/statistic."""
    account = result['accounting']; rows = account['rows']
    c.require(len(rows) == 330 and {(r['cell_id'], r['epsilon']) for r in rows} ==
              {(cid, e) for cid in records for e in c.EPS}, 'A66_FIVE_ENDPOINTS')
    c.require(account['registered_grid_only'] and not account['population_mean_certified'],
              'FINITE_SAMPLE_SCOPE')
    for row in rows:
        ep = records[row['cell_id']]['precision_rows'][str(row['epsilon'])]['1.0']
        c.require(row['eligible'] == ep['cost_acquisition_eligible']
                  and row['cohort'] == 'EXPLORATION_N8', 'SAVED_A_ELIGIBILITY_COHORT')
        if row['eligible']:
            values = pairs[row['cell_id']]
            c.require(len(values) == 8 and {v['replica'] for v in values} == set(range(8))
                      and all(v['cohort'] == 'EXPLORATION_N8' for v in values)
                      and row['n'] == 8 and row['G_RZ_exact'] is not None,
                      'FRESH_N8_WITHOUT_FP_COST_POOLING')
            c.require(row['shots'] == dict(cosine=ep['axes']['real']['shots'], sine=ep['axes']['imag']['shots'])
                      and math.isfinite(row['G_RZ_SE']) and row['G_RZ_SE'] >= 0
                      and math.isfinite(row['sample_SD']) and row['sample_SD'] >= 0,
                      'SAVED_SHOTS_AND_SAMPLE_STATISTICS')
        else:
            c.require(row['n'] is None and row['G_RZ_exact'] is None, 'INELIGIBLE_IS_NOT_ZERO_COST')
    c.require(set(account['winners']) == {str(e) for e in c.EPS}
              and all(v is None or v in rows for v in account['winners'].values()), 'SAVED_PR_WINNER_ROWS')
    comparisons = account['comparisons']
    c.require(len(comparisons) == 5 and [r['epsilon'] for r in comparisons] == c.EPS, 'FIVE_SAVED_COMPARISONS')
    for comp in comparisons:
        key = str(comp['epsilon']); pr = account['winners'][key]
        c.require(comp['PF'] == pf[key] and comp['PR'] == pr, 'FROZEN_D_COUNTERPART')
        if pr is not None and pf[key] is not None:
            c.require(comp['status'] == 'FINITE_SAMPLE_POINT_COMPARISON'
                      and math.isfinite(comp['PR_to_PF_ratio']), 'SAVED_POINT_COMPARISON_SCOPE')


def audit():
    activation = c.PRIVATE/'A_activation01'
    spec = c.read(activation/'A_spec_EXECUTION.json'); grant = c.read(activation/'A_grant_APPROVED_PRIVATE.json')
    c.authorize(spec, grant, runtime=False, interpreter=False)
    pending = c.read(spec['pending_spec'])
    c.require(c.sha(spec['pending_spec']) == '4baa3f2eabca0538b7ac01f377acf205b00b44212904040934eac717bb1a4687',
              'APPROVED_FINAL_PENDING_SHA')
    allowed = {'status','pending','pending_spec','pending_spec_sha256','worker_cpus','resource_receipt',
               'resource_receipt_sha256','seed_refresh','seed_refresh_sha256'}
    c.require({k:v for k,v in spec.items() if k not in allowed} ==
              {k:v for k,v in pending.items() if k not in allowed}, 'ACTIVATION_FIXED_FIELDS')
    for rel, expected in pending['seal_tool_hashes'].items():
        c.require(c.sha(c.ROOT/rel) == expected, 'SEALED_METADATA_TOOL_PRESERVED')
    for manifest in [*pending['preservation_manifests'], seal.STORE/'seal_preservation_manifest_PRIVATE.json',
                     activation/'activation_preservation_manifest.json']:
        for row in c.read(manifest):
            c.require(c.sha(row['path']) == row['sha256'], 'PRIOR_ARTIFACT_BYTES_PRESERVED')
    before = preservation()
    c.require(spec['dependencies'] == c.read(pending['dependency_bindings']), 'FROZEN_DEPENDENCY_BINDINGS')
    deps = {stage:seal.inspect_dependency(stage, dep) for stage,dep in spec['dependencies'].items()}
    pf = c.read(pending['saved_PF_controls'])
    seal.pf_payload(pf, deps['D'], spec['dependencies']['D']['sha256'])
    attempt = c.owned(spec['attempt']); output = attempt/'output'
    result = c.read(output/'result.json'); terminal = c.read(output/'terminal.json')
    native = c.read(attempt/'native_exit.json'); started = c.read(attempt/'native_started.json')
    marker = c.read(attempt/'ONE_SHOT.json'); done = c.read(attempt/'completion_audit.json')
    c.require(result['status'] == terminal['status'] == done['status'] == 'COMPLETE_MANDATORY_STOP'
              and result['stage'] == terminal['stage'] == done['stage'] == 'A'
              and result['scope'] == c.SCOPE and result['binding'] == c.BINDING, 'A_COMPLETION_BINDING')
    c.require(result['SOURCE'] == marker['SOURCE'] == done['SOURCE'] == spec['SOURCE']
              and marker['spec_digest'] == terminal['spec_digest'] == c.digest(spec)
              and marker['grant_digest'] == c.digest(grant), 'A_SPEC_GRANT_SOURCE_DIGEST')
    c.require(native['exit_code'] == terminal['exit_code'] == done['native_exit_code'] == 0
              and all(v['same_parent_reaped'] and v['wait4_complete'] for v in (native, done))
              and started['pid'] == native['pid'] and started['parent_pid'] == marker['parent_pid'], 'NATIVE_WAIT4_EXIT0')
    c.require(done['result_sha256'] == c.sha(output/'result.json') and
              not any(p.exists() for p in (attempt/'driver_STOP.json', output/'STOP.json')), 'RESULT_BYTES_NO_STOP')
    c.require(not result['FP_PR_cost_pooled_with_A'] and not result['automatic_retry']
              and not result['next_stage_authorized'], 'APPROVED_A_ONLY')
    reserved = {k:0 for k in spec['calls']}
    for p in sorted(output.glob('reservation_*.json')):
        r = c.read(p); c.require(r['nonrefundable'] and type(r['count']) is int and r['count'] >= 0,
                               'NONREFUNDABLE_LEDGER')
        reserved[r['category']] += r['count']
    c.require(reserved == result['calls_reserved'] == done['calls_reserved']
              and result['calls_maximum'] == spec['calls']
              and all(reserved[k] <= v for k,v in spec['calls'].items()), 'FIXED_BUDGET_AND_ACTUAL_LEDGER')
    cells = c.registrations()['A']; records = {r['cell']['id']:r for r in result['signals']}
    selected = {x['id'] for x in c.selected(result['signals'], cells)}
    signals = c.read(output/'signals.json')
    c.require([r['cell'] for r in result['signals']] == cells and signals['signals'] == result['signals']
              and result['reference'] == deps['FP']['reference'], 'EXACT_SIGNAL_GRID_REFERENCE_REUSE')
    old = {r['cell']['id']:r for r in deps['FP']['signals']}
    reused = set(records) & set(old)
    c.require(len(reused) == 2 and set(signals['reused_signal_ids']) == reused
              and all(records[k] == old[k] for k in reused), 'EXACT_TWO_FP_SIGNAL_COPIES')
    measured = result['measurements']; by_task = {r['cell_id']+'_'+str(r['replica']):r for r in measured}
    c.require(len(measured) == len(by_task) == 8*len(selected)
              and set(by_task) == {cid+'_'+str(i) for cid in selected for i in range(8)}, 'COMPLETE_EIGHT_REPLICAS')
    registry = {t['id']:t for t in c.read(spec['seed_registry'])['tasks']}
    expected_signal = {cid for cid in records if cid not in reused and records[cid].get('status') != 'DEGENERATE_EMPTY_TAIL'}
    expected_first = {cid+'_0' for cid in selected}
    expected_rest = set(by_task)-expected_first
    controls = []; trajectories = []; workers = []; pairs = defaultdict(list); strata = defaultdict(Counter)
    occurrences = 0; rte_events = 0
    for group, expected in [('signal_workers', expected_signal), ('first_control_workers', expected_first),
                            ('remaining_workers', expected_rest)]:
        directory = output/group
        if not expected:
            c.require(not directory.exists(), 'EMPTY_WAVE_NOT_EXECUTED'); continue
        receipt = c.read(directory/'accepted_native_receipt.json')
        c.require(receipt['status'] == 'PASS' and receipt['all_reaped'] and not receipt['cleanup_errors']
                  and receipt['first_error'] is None and not receipt['automatic_retry']
                  and len(receipt['workers']) == len(expected)
                  and {w['task_id'] for w in receipt['workers']} == expected, 'COMPLETE_OWNED_POOL_RECEIPT')
        for worker in receipt['workers']:
            path = c.owned(worker['directory']); value = c.read(path/'task_result.json'); ident = worker['identity']
            c.require(path.parent == directory and worker['exit_code'] == 0 and worker['reaped']
                      and c.sha(path/'task_result.json') == worker['result_sha256'], 'WORKER_EXIT_AND_BYTES')
            c.require(ident[0] == worker['pid'] and ident[2] == native['pid'] and ident[3] == os.getuid()
                      and not original_process_alive(ident[0], ident[1]) and worker['cpu'] in spec['worker_cpus'],
                      'OWNED_WORKER_IDENTITY_REAP')
            workers.append(worker)
            if group == 'signal_workers':
                c.require(value == records[value['cell']['id']], 'SAVED_FORWARD_COPY'); continue
            task = registry[worker['task_id']]; c.check_pair(value)
            c.require(value == by_task[task['id']] and value['seed'] == task['seed']
                      and value['cell_id'] == task['cell']['id'] and value['replica'] == task['replica']
                      and value['cohort'] == 'EXPLORATION_N8'
                      and value['preparation_hash'] == result['preparation_identities'][value['cell_id']],
                      'FROZEN_TRAJECTORY_SEED_PREPARATION_COHORT')
            trajectory = c.read(path/'trajectory.json'); events = trajectory['events']
            c.require(trajectory['cell'] == task['cell'] and trajectory['seed'] == task['seed']
                      and not trajectory['conditional'] and not trajectory['quantum_weights_changed']
                      and trajectory['cohort'] == 'EXPLORATION_N8'
                      and c.digest(events) == trajectory['event_digest'] == value['event_digest'],
                      'ORIGINAL_UNCONDITIONAL_PAIRED_EVENTS')
            c.require(len(events) == task['cell']['q'] and all(len(v) == task['cell']['r'] for v in events)
                      and c.stats.event_stratum(events) == value['stratum'], 'REGISTERED_EVENT_LAYOUT_STRATUM')
            occurrences += len(events); rte_events += sum(len(v) for v in events)
            for axis in ('cosine','sine'):
                c.require(c.read(path/('measured_'+axis+'.json')) == value['axes'][axis], 'SAVED_COMPILED_AXIS')
                reservation = c.read(path/('compile_'+axis+'_reserved.json'))
                c.require(reservation['fingerprint'] == value['axes'][axis]['fingerprint']
                          and reservation['event_digest'] == value['event_digest'], 'COMPILE_EVENT_FINGERPRINT')
            if group == 'first_control_workers':
                control = c.read(path/'control.json')
                c.require(control['status'] == 'PASS' and control['max_error'] < 1e-9
                          and control['control_probes'] == 4 and control['independent_event_oracles'] == 1
                          and control['full_Fock_scalar_once'] and not control['reference_rotated']
                          and control['ordinary_ancilla_branch'], 'FIRST_REPLICA_CONTROL_ORACLE')
                controls.append(control)
            pairs[value['cell_id']].append(value); strata[value['cell_id']][value['stratum']] += 1
            trajectories.append(dict(task_id=task['id'], seed=task['seed'], stratum=value['stratum'],
                event_digest=value['event_digest'], trajectory_sha256=c.sha(path/'trajectory.json')))
    c.require(reserved['base_signal'] == reserved['independent_forward_signal'] == len(expected_signal)
              and reserved['new_signed_shot_endpoints'] == 5*len(expected_signal)
              and reserved['native_evolution'] == reserved['trajectory'] == len(measured)
              and reserved['wrapper_compile'] == 2*len(measured)
              and reserved['occurrence'] == occurrences and reserved['RTE_event'] == rte_events
              and reserved['control_probe'] == 4*len(controls)
              and reserved['independent_event_oracle'] == len(controls)
              and reserved['endpoint_accounting_pass'] == 1, 'ACTUAL_SCIENTIFIC_SCOPE_COUNTS')
    primitive = c.read(output/'primitive_validation.json')
    c.require(primitive['actual_time_count']*primitive['probe_count'] == reserved['primitive_probe'] == 90
              and primitive['all_registered_times'] and primitive['max_native_expm_error'] < 1e-9
              and primitive['max_spectral_expm_error'] < 1e-9, 'SAVED_PRIMITIVE_CHECKS')
    check_accounting(result, records, pairs, pf['winners'])
    c.require(not (c.PRIVATE/'attempts/S_STATS_run01').exists() and not (c.PRIVATE/'attempts/P_run01').exists(),
              'NO_LATER_STAGE_EXECUTION')
    winner_ids = {v['cell_id'] for v in result['accounting']['winners'].values() if v is not None}
    return dict(status='COMPLETE_MANDATORY_STOP', stage='A', SOURCE=spec['SOURCE'],
        result_sha256=c.sha(output/'result.json'), native_exit_code=0, same_parent_reaped=True, wait4_complete=True,
        worker_processes_reaped=len(workers), signal_workers=len(expected_signal),
        first_control_workers=len(expected_first), remaining_workers=len(expected_rest),
        native_wall_seconds=native['wall_seconds'], native_peak_RSS_bytes=native['peak_RSS_bytes'],
        calls_reserved=reserved, calls_maximum=spec['calls'], control_configurations=len(controls),
        maximum_control_error=max((v['max_error'] for v in controls), default=0), primitive=primitive,
        signals=66, FP_signals_reused=2, new_trajectories=len(measured), new_compile_axes=2*len(measured),
        eligible_native_candidates=len(selected), accounting_rows=330, saved_PF_counterparts_unchanged=True,
        all_cohorts='EXPLORATION_N8', FP_diagnostic_costs_pooled=False,
        stratum_counts={cid:dict(v) for cid,v in sorted(strata.items())},
        winner_stratum_counts={cid:dict(strata[cid]) for cid in sorted(winner_ids)},
        trajectory_receipts=trajectories, seed_redrawn=False, scientific_computation_repeated_by_auditor=False,
        NPZ_decoded_by_auditor=False, accounting_recomputed_by_auditor=False,
        F_records_preserved=before['records_checked'], FP_private_files_preserved=204,
        original_D_private_files_preserved=541, completed_D_private_files_preserved=314,
        independent_scientific_review=False, population_mean_certified=False, automatic_retry=False,
        next_stage_authorized=False, S_STATS_P_executed=False, push_performed=False)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); target = c.owned(args.output.resolve()); result = audit(); c.save(target, result)
    print(__import__('json').dumps({k:v for k,v in result.items() if k not in ('stratum_counts','trajectory_receipts')}, indent=2))


if __name__ == '__main__':
    main()
