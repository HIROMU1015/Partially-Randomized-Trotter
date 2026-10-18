"""Audit a completed authorized P run without resampling or reaccounting costs.

This author's metadata audit copies the runner's sole accounting result. It does
not claim an independent reviewer, independent rerun, or population guarantee.
Private seeds/events remain in the private run and preservation store.
"""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import shutil
import subprocess

import h4_dp_contract as c

ROOT = Path(__file__).resolve().parents[1]


def audit(run, old_manifests, destination, private_manifest):
    run = Path(run).resolve()
    output = run / 'output'
    spec = c.read(run / 'spec_SEALED_PRIVATE.json')
    grant = c.read(run / 'grant_PRIVATE.json')
    native = c.read(run / 'native_driver_exit.json')
    terminal = c.read(output / 'terminal.json')
    result = c.read(output / 'result.json')
    expected_compiler = c.read(spec['native_config'])['binding']['compiler']
    expected_preparations = c.read(spec['saved_preparation_identities'])['prefixes']
    c.require(grant['spec_digest'] == c.digest(spec) and grant['SOURCE'] == spec['SOURCE'], 'CONTRACT_BINDING')
    c.require(grant['approved_by_user'] and grant['runtime_authorization']
              and grant['one_shot'] and not grant['retry'] and not grant['next_stage_authorized'], 'GRANT_SCOPE')
    c.require(c.sha(grant['approval_receipt']) == grant['approval_receipt_sha256'], 'APPROVAL_RECEIPT')
    c.require(c.read(spec['lock']) == dict(SOURCE=spec['SOURCE'], spec_digest=c.digest(spec)), 'LOCK_BINDING')
    c.require(native['exit_code'] == 0 and native['same_parent_reaped'] is True
              and native['SOURCE'] == spec['SOURCE'] and native['result_sha256'] == c.sha(output / 'result.json'), 'NATIVE_EXIT')
    c.require(terminal == dict(status='COMPLETE_MANDATORY_STOP', SOURCE=spec['SOURCE'],
        native_workers_reaped=True, retry=False, next_stage_authorized=False), 'TERMINAL')
    c.require(not (output / 'STOP.json').exists(), 'UNEXPECTED_STOP')
    for name, expected in spec['source_hashes'].items():
        blob = subprocess.check_output(['git', 'show', spec['SOURCE'] + ':' + name], cwd=ROOT)
        c.require(c.sha(ROOT / name) == expected == hashlib.sha256(blob).hexdigest(), 'SOURCE_CLOSURE')
    for row in spec['inputs']:
        c.require(c.sha(row['path']) == row['sha256'], 'INPUT_CLOSURE:' + row['role'])
    for group in ('runtime_hashes', 'provider_hashes'):
        for name, expected in spec[group].items():
            c.require(c.sha(name) == expected, 'EXECUTION_CLOSURE:' + name)
    for directory, files in spec['frozen_stage_closure'].items():
        for name, expected in files.items():
            c.require(c.sha(Path(directory) / name) == expected, 'FROZEN_STAGE_CLOSURE')
    c.validate_tasks(spec)  # Integer seed/binding metadata only; no scientific provider.
    rows = {(r['cell_id'], r['replica']): r for r in result['paired_costs']}
    c.require(len(rows) == len(result['paired_costs']) == len(spec['tasks']), 'FIXED_N_COVERAGE')
    c.require({p.name for p in (output / 'tasks').iterdir()} == {t['id'] for t in spec['tasks']}, 'EXACT_TASK_DIRECTORIES')
    reserved = Counter()
    controls = []
    completed = Counter()
    strata = {cid: Counter() for cid in spec['n_fixed']}
    new_files = set(p for p in run.iterdir() if p.is_file())
    new_files.update(p for p in output.iterdir() if p.is_file())
    new_files.add(Path(__file__).resolve())
    for name in ('admit_authorized_P_run01.py', 'document_completed_P.py'):
        helper = run.parent / name
        if helper.exists(): new_files.add(helper)
    public_pairs = []
    for task in spec['tasks']:
        d = output / 'tasks' / task['id']
        row = c.read(d / 'result.json')
        c.require(row == rows[(task['cell']['id'], task['replica'])], 'PERSISTED_PAIR')
        c.require(row['seed'] == task['seed'] and row['cohort'] == 'INDEPENDENT_H4_CONFIRMATION', 'COHORT_BINDING')
        tr = c.read(d / 'trajectory.json')
        c.require(tr['cell'] == task['cell'] and tr['master_seed'] == tr['seed'] == task['seed']
                  and tr['step_seeds'] == task['step_seeds'] and not tr['conditional']
                  and not tr['quantum_weights_changed'], 'SEALED_TRAJECTORY')
        c.require(c.digest(tr['events']) == tr['event_digest'] == row['event_digest'], 'EVENT_DIGEST')
        c.require(len(tr['events']) == task['cell']['q']
                  and sum(map(len, tr['events'])) == task['cell']['R'], 'EVENT_COUNTS')
        c.require(c.event_stratum(tr['events']) == row['stratum'], 'SAVED_STRATUM')
        for axis in c.AXES:
            c.require(c.read(d / ('measured_' + axis + '.json')) == row['axes'][axis], 'SAVED_AXIS')
        c.check_pair(row, expected_compiler)
        ledger = c.read(d / 'calls_ledger.json')
        c.require(ledger['nonrefundable'] is True and ledger['reserved'] == row['calls'], 'LEDGER')
        reserved.update(ledger['reserved'])
        if task['replica'] == 0:
            control = c.read(d / 'control.json')
            c.require(control['status'] == 'PASS' and control['control_probes'] == 4
                      and control['independent_event_oracles'] == 1
                      and control['tolerance'] == 1e-9 and control['max_error'] < control['tolerance']
                      and control['full_Fock_scalar_once'] and not control['reference_rotated']
                      and control['ordinary_ancilla_branch'], 'CONTROL')
            controls.append(dict(cell_id=task['cell']['id'], **control))
        else:
            c.require(not (d / 'control.json').exists(), 'UNPLANNED_CONTROL')
        completed[row['cell_id']] += 1
        strata[row['cell_id']][row['stratum']] += 1
        new_files.update(p for p in d.iterdir() if p.is_file())
        public_pairs.append(dict(cell_id=row['cell_id'], replica=row['replica'], cohort=row['cohort'],
            event_digest=row['event_digest'], stratum=row['stratum'], calls=row['calls'],
            axes={a: {k: v for k, v in row['axes'][a].items() if k != 'seed'} for a in c.AXES}))
    reserved['endpoint_accounting_pass'] = 1
    c.require(dict(reserved) == result['actual_calls']
              == {k: v for k, v in spec['calls'].items() if v}, 'EXACT_ACTUAL_CALLS')
    c.require(dict(completed) == spec['n_fixed'], 'EXACT_FIXED_N')
    c.require([r['endpoint'] for r in result['comparisons']] == spec['evaluation_endpoints'], 'EXACT_ENDPOINTS')
    c.require(set(result['within_candidate_covariance']) == set(spec['n_fixed']), 'COVARIANCE_COVERAGE')
    c.require(all(dict(strata[cid]) == {k: v for k, v in result['stratum_counts'][cid].items() if v}
                  for cid in strata), 'STRATUM_COVERAGE')
    worker_receipts = []
    for d in (output / 'worker_contexts').iterdir():
        # The actual parent has joined every worker and checked each exitcode.
        # A later reuse of a PID by another job is not an execution-closure gate.
        prep = c.read(d / 'preparation_receipt.json')
        reconstruction = c.read(d / 'H4_preparation_reconstruction_reserved.json')
        c.require(set(prep['prefixes']) == {'2', '3'} and prep['SCF_DF_state_solver'] == 0
                  and reconstruction['prefixes'] == [2, 3] and reconstruction['nonrefundable'], 'PREPARATION_RECEIPT')
        c.require(all(prep['prefixes'][k]['preparation_hash'] == expected_preparations[k]['preparation_hash']
                      for k in ('2', '3')), 'PREPARATION_IDENTITY')
        worker_receipts.append(prep)
        new_files.update(p for p in d.iterdir() if p.is_file())
    c.require(len(worker_receipts) == spec['workers'], 'WORKER_CONTEXT_COUNT')
    preserved = []
    for manifest in old_manifests:
        prior = c.read(manifest)
        for row in prior:
            c.require(c.sha(row['path']) == row['sha256'] and c.sha(row['copy']) == row['sha256'], 'PRIOR_PRESERVATION')
        preserved.append(dict(manifest_sha256=c.sha(manifest), records=len(prior), unchanged=True))
    private_manifest = Path(private_manifest)
    c.require(not private_manifest.exists(), 'UNUSED_COMPLETION_PRESERVATION')
    store = private_manifest.parent / 'completed_preservation_store'
    store.mkdir(exist_ok=False)
    manifest_rows = []
    for path in sorted(new_files):
        expected = c.sha(path)
        target = store / expected
        if not target.exists(): shutil.copyfile(path, target)
        c.require(c.sha(target) == expected, 'NEW_PRESERVATION_COPY')
        manifest_rows.append(dict(path=str(path), sha256=expected, bytes=path.stat().st_size, copy=str(target)))
    c.save(private_manifest, manifest_rows)
    destination = Path(destination)
    c.require(not destination.exists(), 'UNUSED_PUBLIC_ARTIFACT_DIRECTORY')
    destination.mkdir(parents=True)
    science = {k: v for k, v in result.items() if k != 'paired_costs'}
    science.update(SOURCE=spec['SOURCE'], scientific_result_sha256=native['result_sha256'],
        n_fixed=spec['n_fixed'], seed_plan_digest=spec['seed_plan_digest'],
        candidate_reselection=False, old_exploration_samples_pooled=False,
        statistical_rule='FIXED_N_POINT_DIRECTION_AGAINST_COMPLETED_D', population_mean_certified=False)
    c.save(destination / 'confirmation_result.json', science)
    c.save(destination / 'paired_costs_projection.json', dict(rows=public_pairs,
        private_seeds_and_events_published=False, trajectory_count=len(public_pairs)))
    c.save(destination / 'controls.json', dict(rows=controls, maximum_error=max(x['max_error'] for x in controls)))
    resource = c.read(run / 'fresh_resource_admission_PRIVATE.json')
    approval = c.read(grant['approval_receipt'])
    c.save(destination / 'execution_contract.json', dict(
        status=spec['status'], stage='P', SOURCE=spec['SOURCE'], spec_digest=c.digest(spec),
        approved_by_user=True, runtime_authorization=True, approval_text=approval['user_text'],
        approval_receipt_sha256=grant['approval_receipt_sha256'],
        workflow_technical_recovery_authorized=spec['workflow_technical_recovery_authorized'],
        per_attempt_one_shot=True, per_attempt_retry=False, next_stage_authorized=False,
        source_hashes=spec['source_hashes'], inputs=[dict(role=x['role'], sha256=x['sha256']) for x in spec['inputs']],
        D_result_sha256=spec['D_result_sha256'], S_STATS_result_sha256=spec['stats_result_sha256'],
        seed_plan_digest=spec['seed_plan_digest'], seed_inventory_receipt_sha256=spec['seed_inventory_receipt_sha256'],
        private_seed_manifest_sha256=approval['seed_manifest_sha256'],
        n_fixed=spec['n_fixed'], evaluation_endpoints=spec['evaluation_endpoints'], calls=spec['calls'],
        resource_admission={k: resource[k] for k in (
            'recorded_UTC', 'physical_cores', 'one_second_idle_physical_equivalent', 'effective_cpu_quota',
            'available_memory_bytes', 'planning_bytes_per_worker', 'workers', 'kernel_limits_modified',
            'affinity_modified', 'imposed_resource_caps', 'memory_planning_is_not_enforced_cap',
            'historical_worker_limit_inherited')}))
    acceptance = dict(status='COMPLETE_MANDATORY_STOP', SOURCE=spec['SOURCE'],
        spec_digest=c.digest(spec), grant_sha256=c.sha(run / 'grant_PRIVATE.json'),
        scientific_approval_receipt_sha256=grant['approval_receipt_sha256'],
        scientific_result_sha256=native['result_sha256'], native_driver_exit=native,
        workers=spec['workers'], native_workers_reaped=True, actual_calls=result['actual_calls'],
        n_fixed=spec['n_fixed'], comparisons=dict(Counter(r['status'] for r in result['comparisons'])),
        primary_comparisons=10, diagnostic_comparisons=4, controls_PASS=len(controls),
        preparation_reconstructions_reserved=2*len(worker_receipts),
        new_SCF_DF_state_reference_signal=0, bootstrap=0, new_n_design=0, technical_interruptions=0,
        source_files=len(spec['source_hashes']), input_files=len(spec['inputs']),
        runtime_files=len(spec['runtime_hashes']), provider_files=len(spec['provider_hashes']),
        prior_preservation=preserved, new_records_preserved=len(manifest_rows),
        new_preservation_manifest_sha256=c.sha(private_manifest), population_mean_certified=False,
        author_metadata_audit=True, independent_reviewer=False, automatic_retry=False,
        next_stage_authorized=False, private_seeds_published=False)
    c.save(destination / 'completion_audit.json', acceptance)
    c.save(destination / 'artifact_byte_manifest.json', {p.name: c.sha(p) for p in sorted(destination.iterdir())})
    print(c.encoded(dict(status=acceptance['status'], comparisons=acceptance['comparisons'],
        wall_seconds=native['wall_seconds'], preserved=len(manifest_rows))).decode())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--prior-preservation', type=Path, action='append', required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--private-preservation', type=Path, required=True)
    a = p.parse_args()
    audit(a.run, a.prior_preservation, a.destination, a.private_preservation)


if __name__ == '__main__':
    main()
