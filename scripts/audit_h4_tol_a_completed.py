"""Audit finished A receipts; preserve the sole accounting, never recompute it."""
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import gzip
import math
from pathlib import Path

import h4_tol_main_contract as c
import seal_h4_tol_a as a
from audit_h4_tol_d_completed import owned_receipt, files

ACT = a.PRIVATE/'activation01_20261019'
PUBLIC = c.ROOT/'artifacts/resource_applicability/track_a_h4_tol_a_completed/2026-10-19'


def audit():
    contract_path = ACT/'A_contract_EXECUTION_PRIVATE.json'
    grant_path = ACT/'A_grant_APPROVED_PRIVATE.json'
    contract, grant = c.read(contract_path), c.read(grant_path)
    sealed = c.read(a.PRIVATE/'A_contract_SEALED_PENDING_APPROVAL_PRIVATE.json')
    output = Path(contract['output']); run = output.parent
    native, terminal = c.read(run/'native_exit.json'), c.read(output/'terminal.json')
    result = c.read(output/'result.json')
    c.require(contract['stage'] == 'A' and contract['plan'] == c.plan('A')
        and contract['status'] == 'SEALED_FOR_EXECUTION' and not contract['pending'], 'FIXED_ACTIVATED_A')
    c.require(all(contract[k] == sealed[k] for k in a.IMMUTABLE)
        and contract['inputs'][:len(sealed['inputs'])] == sealed['inputs'], 'ORIGINAL_SCIENCE_RETAINED')
    c.require(contract['SOURCE'] == grant['SOURCE'] == terminal['SOURCE'] == native['SOURCE'] == a.SOURCE,
        'ACTUAL_SOURCE_BINDING')
    c.require(c.digest(contract) == grant['contract_digest'] == c.read(run/'ONE_SHOT.json')['contract_digest'],
        'ONE_SHOT_CONTRACT')
    c.require(grant['approved_by_user'] and grant['runtime_authorization'] and grant['one_shot']
        and not grant['automatic_retry'] and not grant['next_stage_authorized'], 'EXPLICIT_A_ONLY_APPROVAL')
    c.require(c.sha(grant['approval_receipt']) == grant['approval_receipt_sha256'], 'APPROVAL_BYTES')
    c.require(native['exit_code'] == 0 and native['same_parent_reaped']
        and native['result_sha256'] == c.sha(output/'result.json'), 'NATIVE_EXIT_AND_RESULT')
    c.require(terminal['status'] == 'COMPLETE_MANDATORY_STOP' and terminal['own_workers_reaped']
        and not (output/'STOP.json').exists(), 'NO_STOP_COMPLETE_TERMINAL')
    closure = a.d.verify_bytes(contract)
    c.validate_seed_registry(contract)
    c.require(c.sha(contract['accepted_D']['path']) == contract['accepted_D']['sha256'], 'ACCEPTED_D_BYTES')
    refresh = c.read(contract['seed_refresh_receipt'])
    c.require(refresh['status'] == 'COMPLETE_COLLISION_ZERO' and refresh['new_explicit_collisions'] == 0
        and not refresh['seed_redrawn'] and refresh['old_sealed_registry_sha256'] == c.sha(contract['seed_registry']),
        'ACTIVATION_SEED_REFRESH')
    old_preservation = a.preserve()
    for row in c.read(a.PRIVATE/'seal_preservation_manifest_PRIVATE.json'):
        c.require(c.sha(row['path']) == row['sha256'], 'ORIGINAL_A_SEAL_RECORD')
    for name, h in c.read(a.PUBLIC/'artifact_manifest.json')['files'].items():
        c.require(c.sha(a.PUBLIC/name) == h, 'PUBLISHED_A_SEAL_BYTES')
    c.require(result['stage'] == 'A' and result['SOURCE'] == a.SOURCE and result['binding'] == c.BINDING
        and result['status'] == 'COMPLETE_MANDATORY_STOP' and result['registered_set_complete'], 'COMPLETE_A_RESULT')
    c.require(result['call_budget'] == c.budgets()['A'] and result['endpoint_accounting_pass'] == 1
        and result['new_SCF_DF_state_reference'] == 0 and not result['next_stage_authorized'], 'SOLE_FIXED_ACCOUNTING')
    c.require(not result['saved_FP_raw_aggregation'] and not result['population_mean_certified']
        and result['strict_bias_certificate'] is None, 'STATISTICAL_SCOPE')

    signal_file = c.read(output/'signals.json'); signals = signal_file['signals']
    registered = c.registrations()['A']; by_id = {s['cell']['id']:s for s in signals}
    eligible = {x['id'] for x in c.selected(signals, registered)} - {c.FP_PR}
    fp = c.read(contract['FP_evidence'])
    saved = next(x for x in fp['signals'] if x['cell']['id'] == c.FP_PR)
    c.require(signal_file['reused_records'] == [c.FP_PR] and by_id[c.FP_PR] == saved, 'SAVED_FP_SIGNAL_UNCHANGED')
    missing = {x['id'] for x in registered} - {c.FP_PR}
    signal_receipts = owned_receipt(output/'signal_workers', missing)
    for cid in missing:
        work = output/'signal_workers'/cid
        c.require(c.read(work/'task_result.json') == by_id[cid]
            and c.read(work/'signal_reserved.json')['nonrefundable'], 'ALL_49_NEW_SIGNAL_RECEIPTS')
    primitive, preps = c.read(output/'primitive_validation.json'), c.read(output/'preparations.json')
    c.require(primitive['actual_time_count']*primitive['probe_count'] == 120
        and primitive['all_registered_times'], 'A_PRIMITIVE_120')
    c.require(preps['reconstructions'] == 3 and preps['new_reference'] == 0
        and preps['SCF_DF_state_solver'] == 0, 'SAVED_INPUT_PREPARATION_POLICY')

    tasks = [t for t in c.read(contract['seed_registry'])['tasks'] if t['cell']['id'] in eligible]
    by_task = {t['id']:t for t in tasks}
    first = {t['id'] for t in tasks if t['replica'] == 0}
    rest = set(by_task)-first
    first_receipts = owned_receipt(output/'first_control_workers', first) if first else []
    rest_receipts = owned_receipt(output/'remaining_workers', rest) if rest else []
    measured_file = c.read(output/'paired_measurements.json'); measured = measured_file['rows']
    c.require(measured_file['old_FP_cohort_not_pooled'] and len(measured) == len(tasks) == 8*len(eligible)
        and {(r['cell_id'],r['replica']) for r in measured} == {(t['cell']['id'],t['replica']) for t in tasks},
        'EXACT_NEW_COHORT_N8')
    # Same monotonic clock in the common parent; no dependent task before first batch is reaped.
    first_raw = c.read(output/'first_control_workers/accepted_native_receipt.json') if first else None
    rest_raw = c.read(output/'remaining_workers/accepted_native_receipt.json') if rest else None
    if first_raw and rest_raw:
        c.require(min(w['started'] for w in rest_raw['workers']) >= max(w.get('last_resource_observation',w['started']) for w in first_raw['workers']),
            'FIRST_BATCH_PRECEDES_REMAINING')
    controls = []; actual = Counter(); strata = {cid:Counter() for cid in eligible}; public_pairs = []
    for row in measured:
        task_id = row['cell_id']+'_'+str(row['replica']); task = by_task[task_id]
        work = output/('first_control_workers' if row['replica'] == 0 else 'remaining_workers')/task_id
        c.require(row == c.read(work/'task_result.json') and row['seed'] == task['seed']
            and row['binding'] == c.BINDING and row['cohort'] == 'EXPLORATION_N8'
            and not row['synthetic'] and row['normalization_unchanged'], 'EXACT_NATIVE_PAIRED_RESULT')
        trajectory = c.read(work/'trajectory.json')
        c.require(trajectory['cell'] == task['cell'] and trajectory['seed'] == task['seed']
            and trajectory['event_digest'] == row['event_digest'] == c.digest(trajectory['events'])
            and trajectory['cohort'] == 'EXPLORATION_N8' and not trajectory['conditional']
            and not trajectory['quantum_weights_changed'], 'FROZEN_EVENT_LAW_AND_SEED')
        c.require(len(trajectory['events']) == task['cell']['q']
            and sum(map(len, trajectory['events'])) == task['cell']['R'], 'OCCURRENCE_AND_EVENT_COUNTS')
        c.require(c.old.event_stratum(trajectory['events']) == row['stratum'], 'RECORDED_STRATUM')
        strata[row['cell_id']][row['stratum']] += 1
        c.old.check_pair(row, c.BINDING['compiler'])
        expected_calls = dict(trajectory=1, occurrence=task['cell']['q'], RTE_event=task['cell']['R'],
            native_evolution=1, wrapper_compile=2)
        if row['replica'] == 0: expected_calls.update(control_probe=4, independent_event_oracle=1)
        c.require(row['calls'] == expected_calls, 'EXACT_PER_TRAJECTORY_BUDGET')
        for kind, count in expected_calls.items():
            ledger = c.read(work/(kind+'_'+str(count)+'_reserved.json'))
            c.require(ledger == dict(kind=kind, count=count if kind != 'wrapper_compile' else 1,
                total=count, nonrefundable=True), 'NONREFUNDABLE_CALL_RESERVATION')
        # Both sequential compile reservations must be retained.
        c.require(c.read(work/'wrapper_compile_1_reserved.json') == dict(kind='wrapper_compile',count=1,total=1,nonrefundable=True),
            'FIRST_AXIS_RESERVED')
        actual.update(row['calls'])
        for axis in c.old.AXES:
            c.require(c.read(work/('measured_'+axis+'.json')) == row['axes'][axis], 'SAVED_AXIS_BYTES')
        if row['replica'] == 0:
            control = c.read(work/'control.json')
            c.require(control['status'] == 'PASS' and control['control_probes'] == 4
                and control['independent_event_oracles'] == 1 and control['tolerance'] == 1e-9
                and math.isfinite(control['max_error']) and control['max_error'] < control['tolerance']
                and control['full_Fock_scalar_once'] and not control['reference_rotated']
                and control['ordinary_ancilla_branch'], 'CONTROL_AND_INDEPENDENT_EVENT_ORACLE')
            controls.append(dict(cell_id=row['cell_id'], **control))
        else: c.require(not (work/'control.json').exists(), 'NO_UNREGISTERED_EXTRA_CONTROL')
        public_pairs.append({k:deepcopy(v) for k,v in row.items() if k not in ('seed','axes')})
        public_pairs[-1]['axes'] = {axis:{k:v for k,v in row['axes'][axis].items() if k != 'seed'} for axis in c.old.AXES}
    limits = c.budgets()['A']
    mapping = dict(trajectory='trajectory_maximum',occurrence='occurrence_maximum',RTE_event='RTE_event_maximum',
        native_evolution='native_evolution_maximum',wrapper_compile='wrapper_compile_maximum',
        control_probe='control_probe_maximum',independent_event_oracle='independent_event_oracle_maximum')
    c.require(all(value <= limits[mapping[key]] for key,value in actual.items()), 'TOTAL_FIXED_CALL_BUDGET')
    c.require(result['actual_new_base_signals'] == 49 and result['actual_new_compiled_trajectories'] == len(tasks)
        and result['actual_new_wrapper_compiles'] == 2*len(tasks), 'ACTUAL_RESULT_CALL_COUNTS')
    for cid, counts in strata.items():
        c.require(dict(counts) == {k:v for k,v in result['new_cost_strata'][cid].items() if v}, 'STRATUM_COUNTS')

    # Inspect the sole saved accounting schema and exact copied FP rows. Never
    # invoke evaluate_pairs, account_stage, statistics, bootstrap, covariance or n design.
    rows = result['rows']
    c.require(len(rows) == 250 and {(r['cell_id'],r['epsilon']) for r in rows}
        == {(x['id'],e) for x in registered for e in c.EPS}, 'ALL_50_X_5_ENDPOINTS')
    for row in rows:
        cid = row['cell_id']; ep = by_id[cid]['precision_rows'][str(row['epsilon'])]['1.0']
        c.require(row['eligible'] == ep['cost_acquisition_eligible'], 'SAVED_ENDPOINT_ELIGIBILITY')
        if cid == c.FP_PR:
            original = deepcopy(next(r for r in fp['saved_rows'] if r['cell_id'] == cid
                and r['epsilon'] == row['epsilon'] and r['bias_mode'] == 'USUAL_SIGNED' and r['allowance_factor'] == 1.))
            original.update(cohort='SAVED_FP_PANEL_N32_ANCHOR',statistics_recomputed=False)
            original.setdefault('G_RZ_exact',None)
            c.require(row == original, 'VERBATIM_FP_SAVED_STATISTICS')
        elif row['eligible']:
            c.require(row['n'] == 8 and row['cohort'] == 'EXPLORATION_N8'
                and Fraction(row['G_RZ_exact']) > 0 and math.isfinite(row['G_RZ_SE']) and row['G_RZ_SE'] >= 0,
                'SAVED_N8_ACCOUNTING_SCHEMA')
            c.require(row['shots'] == {ax:ep['axes'][channel]['shots'] for ax,channel in zip(c.old.AXES,('real','imag'))},
                'SAVED_SHOT_BINDING')
        else: c.require(row['G_RZ_exact'] is None and row['G_RZ_SE'] is None and row['n'] is None, 'INELIGIBLE_NOT_ZERO_FILLED')
    for cov in result['new_cohort_covariance']:
        eps = [r['epsilon'] for r in rows if r['cell_id'] == cov['cell_id'] and r['eligible']]
        matrix = cov['sample_covariance_exact']
        c.require(cov['cell_id'] in eligible and cov['n'] == 8 and cov['epsilons'] == eps
            and len(matrix) == len(eps) and all(len(r) == len(eps) for r in matrix), 'SAVED_COVARIANCE_SCHEMA_ONLY')
    c.require({r['cell_id'] for r in result['new_cohort_covariance']} == eligible, 'COVARIANCE_COVERAGE')
    d_result = c.read(contract['accepted_D']['path'])
    c.require(result['D_result_sha256'] == contract['accepted_D']['sha256'], 'FIXED_D_JOIN')
    c.require(len(result['winners']) == len(result['comparisons']) == 5, 'FIVE_RECORDED_COMPARISONS')
    for winner, comparison in zip(result['winners'], result['comparisons']):
        e = winner['epsilon']; available = [r for r in rows if r['epsilon'] == e and r['G_RZ_exact'] is not None]
        best = winner['best']; pf = next(x['best'] for x in d_result['winners'] if x['epsilon'] == e)
        c.require((best is None and not available) or (best in available and all(
            (Fraction(best['G_RZ_exact']),best['cell_id']) <= (Fraction(r['G_RZ_exact']),r['cell_id']) for r in available)),
            'RECORDED_WINNER_AND_FIXED_TIES')
        c.require(comparison['epsilon'] == e and comparison['PR'] == best and comparison['PF'] == pf, 'VERBATIM_SAVED_COMPARISON_JOIN')
        direction = None if best is None or pf is None else (Fraction(pf['G_RZ_exact']) > Fraction(best['G_RZ_exact']))-(Fraction(pf['G_RZ_exact']) < Fraction(best['G_RZ_exact']))
        c.require(comparison['point_direction'] == direction, 'SAVED_DIRECTION_INTEGRITY')

    c.require(not PUBLIC.exists(), 'NEW_PUBLIC_COMPLETION_DIRECTORY')
    PUBLIC.mkdir(parents=True)
    raw = (output/'result.json').read_bytes()
    c.require(b'/home/' not in raw and b'"seed"' not in raw, 'NO_PRIVATE_PATHS_OR_SEEDS_IN_RESULT')
    (PUBLIC/'result.json').write_bytes(raw)
    (PUBLIC/'signals.json.gz').write_bytes(gzip.compress((output/'signals.json').read_bytes(),mtime=0))
    (PUBLIC/'paired_costs_projection.json.gz').write_bytes(gzip.compress(c.old.encoded(
        dict(rows=public_pairs,raw_seed_event_data_published=False)),mtime=0))
    c.save(PUBLIC/'native_exit.json',native); c.save(PUBLIC/'terminal.json',terminal)
    c.save(PUBLIC/'primitive_validation.json',primitive)
    c.save(PUBLIC/'control_validation.json',dict(configurations=len(controls),probes=4*len(controls),all_PASS=True,rows=controls))
    c.save(PUBLIC/'worker_receipts.json',dict(signal_workers=signal_receipts,first_control_workers=first_receipts,
        remaining_workers=rest_receipts,all_exit0_reaped=True))
    c.save(PUBLIC/'approval_receipt.json',c.read(grant['approval_receipt']))
    c.save(PUBLIC/'seed_refresh_projection.json',refresh)
    resource = c.read(contract['fresh_resources']['receipt'])
    c.save(PUBLIC/'execution_binding.json',dict(SOURCE=a.SOURCE,accepted_A_seal_commit='50479f2f6da76523cdfb3e8bb06bde33213fea67',
        contract_digest=c.digest(contract),grant_sha256=c.sha(grant_path),approval_sha256=grant['approval_receipt_sha256'],
        source_hashes=contract['source_hashes'],input_rows=len(contract['inputs']),provider_files=len(contract['provider_hashes']),
        runtime_files=len(contract['runtime_hashes']),seed_registry_sha256=c.sha(contract['seed_registry']),
        seed_inventory_receipt_sha256=c.sha(contract['seed_inventory_receipt']),worker_cpus=contract['worker_cpus'],
        fresh_resource_receipt_sha256=contract['fresh_resources']['sha256'],effective_memory_available_bytes=resource['effective_memory_available_bytes'],
        agent_imposed_CPU_RAM_worker_wall_caps=None,scientific_conditions_changed=False,one_shot=True,automatic_retry=False))
    records = [dict(path=str(p),sha256=c.sha(p),bytes=p.stat().st_size) for p in files(run)]
    records += [dict(path=str(p),sha256=c.sha(p),bytes=p.stat().st_size) for p in files(ACT)]
    preservation = a.PRIVATE/'completed_A_preservation_manifest_PRIVATE.json'
    c.require(not preservation.exists(), 'NEW_PRIVATE_A_COMPLETION_MANIFEST')
    c.save(preservation, records)
    audit_record = dict(status='COMPLETE_MANDATORY_STOP',SOURCE=a.SOURCE,result_sha256=c.sha(output/'result.json'),
        registered_candidates=50,new_base_signals=49,reused_FP_signals=1,reused_FP_candidates=1,reused_FP_n=32,reused_FP_axes=64,
        new_eligible_candidates=len(eligible),new_trajectories=len(tasks),new_wrapper_compile_axes=2*len(tasks),new_cohort_n=8,
        actual_calls=dict(actual),primitive_probes=120,preparation_reconstructions=3,endpoint_rows=250,endpoint_accounting_pass=1,
        accounting_or_statistics_recomputed_by_auditor=False,saved_FP_raw_reaggregation=False,new_SCF_DF_state_full_reference=0,
        native_exit_code=0,all_workers_reaped=True,worker_pool_maximum=len(contract['worker_cpus']),
        native_worker_processes=len(signal_receipts)+len(first_receipts)+len(rest_receipts),wall_seconds=native['wall_seconds'],
        technical_interruptions=0,retries=0,source_files=closure['SOURCE_files'],input_rows=len(contract['inputs']),
        preserved_prior_records=old_preservation,preserved_A_records=len(records),private_preservation_manifest_sha256=c.sha(preservation),
        seed_refresh_additional_exclusions=refresh['additional_exclusion_values'],seed_collisions=0,
        population_mean_certified=False,strict_bias_certificate=None,S_STATS_P_started=False,independent_scientific_review=False,
        auditor_sha256=c.sha(__file__),mandatory_STOP=True,next_stage_authorized=False)
    c.save(PUBLIC/'completion_audit.json',audit_record)
    c.save(PUBLIC/'artifact_manifest.json',dict(files={p.name:c.sha(p) for p in sorted(PUBLIC.glob('*'))
        if p.is_file() and p.name != 'artifact_manifest.json'},scientific_accounting_recomputed=False))
    return audit_record


if __name__ == '__main__':
    import json
    print(json.dumps(audit()))
