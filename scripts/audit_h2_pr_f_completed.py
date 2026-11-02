"""Audit saved H2 F bytes and receipts. Never decode NPZ or rerun science."""
from decimal import Decimal
from pathlib import Path
import shutil

import h2_pr_resource_contract as c


def audit():
    activation = c.PRIVATE / 'activation01'
    preparation = c.PRIVATE / 'phase0_preparation01'
    prepared_public = c.PUBLIC / 'phase0_preparation01'
    public = c.PUBLIC / 'F_completed_run01'
    c.require(not public.exists(), 'NEW_F_COMPLETION_PUBLIC_NAMESPACE')
    spec = c.read(activation / 'F_spec_EXECUTION.json')
    grant = c.read(activation / 'F_grant_APPROVED_PRIVATE.json')
    # The sealed worker and its parent already verified runtime bytes at exit.
    # This metadata audit rechecks SOURCE/provider/manifest; no numerical imports.
    c.verify(spec, grant, runtime=False)
    run = Path(spec['attempt']); output = run / 'output'
    result = c.read(output / 'result.json')
    terminal, native = c.read(output / 'terminal.json'), c.read(run / 'native_exit.json')
    marker, started = c.read(run / 'ONE_SHOT.json'), c.read(run / 'native_started.json')
    native_audit = c.read(run / 'completion_audit.json')
    c.require(result['SOURCE'] == terminal['SOURCE'] == native['SOURCE'] == marker['SOURCE']
              == started['SOURCE'] == native_audit['SOURCE'] == spec['SOURCE'], 'ONE_F_SOURCE')
    c.require(result['status'] == terminal['status'] == native_audit['status']
              == 'COMPLETE_MANDATORY_STOP', 'F_COMPLETION_STATUS')
    c.require(native['exit_code'] == terminal['exit_code'] == 0 and native['wait4_complete']
              and native['same_parent_reaped'] and native['pid'] == started['pid']
              and started['parent_pid'] == marker['parent_pid'], 'NATIVE_EXIT_AND_OWNED_REAP')
    c.require(marker['spec_digest'] == started['spec_digest'] == terminal['spec_digest'] == c.digest(spec)
              and marker['grant_digest'] == c.digest(grant), 'EXECUTION_MARKER_BINDING')
    c.require(not (run / 'driver_STOP.json').exists() and not (output / 'STOP.json').exists(), 'NO_STOP_OR_RETRY')
    c.require(native_audit['result_sha256'] == c.sha(output / 'result.json')
              and c.read(output / 'tol_only_receipt.json') == result, 'RESULT_BYTES_AND_INPUT_RECEIPT')
    c.require(result['scope'] == c.SCOPE and result['stage'] == 'F'
              and result['generation_policy'] == c.plan(), 'H2_ONLY_FIXED_GENERATION_POLICY')
    ledger = c.read(output / 'calls_ledger.json')
    c.require(c.actual_calls(output) == result['actual_calls'] == ledger['actual']
              == native_audit['actual_calls'] == spec['call_budget'] == c.CALLS
              and ledger['nonrefundable'], 'EXACT_RESERVED_AND_COMPLETED_CALLS')
    for name, count in c.CALLS.items():
        path = output / ('reserved_' + name + '.json')
        if count:
            c.require(c.read(path) == dict(category=name, count=count, maximum=count, nonrefundable=True),
                      'NONREFUNDABLE_CALL_RESERVATION:' + name)
        else:
            c.require(not path.exists(), 'NO_UNAPPROVED_CALL:' + name)
    inputs = {}
    for role, name, receipt in [('integrals', 'integrals.npz', result['integrals_receipt']),
                                ('raw_DF', 'raw_df.npz', result['raw_df_receipt']),
                                ('saved_input', 'tol_only_snapshot.npz', result)]:
        path = output / name
        c.require(c.sha(path) == receipt['sha256'] and path.stat().st_size == receipt['bytes'],
                  'GENERATED_INPUT_BYTES:' + role)
        inputs[role] = dict(filename=name, sha256=receipt['sha256'], bytes=receipt['bytes'],
                            arrays=receipt['arrays'], original_preserved_private=True)
    integral = c.read(output / 'integrals_receipt.json')
    c.require(integral['converged'] is True and integral['policy'] == c.plan(), 'SAVED_SCF_CONVERGENCE')
    metadata = result['hamiltonian_metadata']
    c.require(metadata['input_policy'] == 'TOL_ONLY_NO_CONFIG_FALLBACK'
              and metadata['decomposition_kwargs'] == dict(truncation_threshold=1e-8)
              and metadata['final_rank_supplied'] is False and metadata['subsequent_coefficient_cutoff'] == 0.
              and metadata['coefficient_order'] == 'decomposer_generation_order', 'ACTUAL_TOL_ONLY_POLICY')
    c.require(result['sector'] == dict(n_qubits=4, nelec_alpha=1, nelec_beta=1, dimension=4), 'H2_SPIN_SECTOR')
    c.require(type(result['rank']) is int and result['rank'] == metadata['df_rank_actual']
              == result['arrays']['lambdas']['shape'][0]
              == result['arrays']['g_matrices']['shape'][0], 'ACTUAL_RETURNED_RANK')
    c.require(result['independent_sector_error'] < 1e-9 and result['hermiticity_error'] < 1e-10
              and result['state_residual'] < 1e-10 and result['state_gap'] > 1e-12,
              'SAVED_ACCEPTANCE_VALUES')
    c.require(Decimal(result['projection']['eta_hartree_decimal']) <= Decimal('9.9e-11')
              and result['projection']['full_fock_N'] == 4
              and result['projection']['signed_weights_preserved']
              and not result['ground_state_certified']
              and not result['state_renormalized_after_saving'], 'LIMITED_ENGINEERING_ACCEPTANCE')
    prior = c.read(preparation / 'preparation_preservation_manifest_PRIVATE.json')
    for row in prior:
        c.require(c.sha(row['path']) == row['sha256'], 'ORIGINAL_PREPARATION_SEAL_PRESERVED')
    for name, h in c.read(prepared_public / 'artifact_manifest.json')['files'].items():
        c.require(c.sha(prepared_public / name) == h, 'ORIGINAL_PUBLIC_PREPARATION_PRESERVED')
    c.require(all(not v['next_stage_authorized'] for v in [grant, terminal, native, native_audit]),
              'MANDATORY_STAGE_BOUNDARY')
    # Preserve scientific files and execution records; scratch caches are excluded.
    records = [p for p in output.iterdir() if p.is_file()]
    records += [p for p in run.iterdir() if p.is_file()]
    records += [p for p in activation.iterdir() if p.is_file()]
    checkpoint = output / 'scratch/scf.chk'
    if checkpoint.is_file(): records.append(checkpoint)
    preserve = activation / 'F_completed_preservation_manifest_PRIVATE.json'
    c.save(preserve, [dict(path=str(p), sha256=c.sha(p), bytes=p.stat().st_size)
                      for p in sorted(set(records))])
    public.mkdir()
    for origin, name in [(output / 'result.json', 'result.json'),
                         (output / 'terminal.json', 'terminal.json'),
                         (output / 'calls_ledger.json', 'calls_ledger.json'),
                         (output / 'integrals_receipt.json', 'integrals_receipt.json'),
                         (run / 'native_exit.json', 'native_exit.json'),
                         (run / 'native_started.json', 'native_started.json'),
                         (run / 'completion_audit.json', 'native_completion_audit.json'),
                         (run / 'ONE_SHOT.json', 'ONE_SHOT_receipt.json'),
                         (run / 'resource_observation.json', 'resource_observation.json'),
                         (activation / 'approval_receipt.json', 'approval_receipt.json')]:
        shutil.copyfile(origin, public / name)
    c.save(public / 'generated_input_manifest.json', dict(inputs=inputs,
        hamiltonian_hash=result['hamiltonian_hash'], state_hash=result['state_hash'],
        rank=result['rank'], NPZ_decoded_by_auditor=False,
        legacy_H2_or_H4_inputs_reused=False, snapshots_originals_preserved_private=True))
    c.save(public / 'execution_binding.json', dict(SOURCE=spec['SOURCE'],
        preparation_commit='7b8570dc526aa664d918e9346fa964d81e6f5b53',
        spec_digest=c.digest(spec), pending_spec_sha256=spec['pending_spec_sha256'],
        approval_receipt_sha256=grant['approval_receipt_sha256'], grant_sha256=c.sha(activation / 'F_grant_APPROVED_PRIVATE.json'),
        source_hashes=spec['source_hashes'], provider_SOURCE=spec['provider_SOURCE'],
        provider_manifest_digest=c.digest(spec['provider_hashes']), provider_files=len(spec['provider_hashes']),
        runtime_manifest_sha256=spec['runtime_manifest_sha256'], runtime_files=len(c.read(spec['runtime_manifest'])),
        runtime_exit_hash_checks='SEALED_NATIVE_WORKER_AND_PARENT_VERIFY_RUNTIME_TRUE',
        metadata_auditor_runtime_check='MANIFEST_BYTES_ONLY_NO_REDUNDANT_FULL_RUNTIME_PASS',
        source_provider_rechecked_by_metadata_auditor=True, one_shot=True, automatic_retry=False,
        next_stage_authorized=False, publication_authorized_by_F_approval=False))
    report = dict(status='COMPLETE_MANDATORY_STOP', SOURCE=spec['SOURCE'],
        result_sha256=c.sha(output / 'result.json'), rank=result['rank'], sector=result['sector'],
        hamiltonian_hash=result['hamiltonian_hash'], state_hash=result['state_hash'], snapshot_sha256=result['sha256'],
        actual_calls=result['actual_calls'], native_exit_code=0, same_parent_reaped=True,
        native_wall_seconds=native['wall_seconds'], worker_wall_seconds=terminal['wall_seconds'],
        peak_RSS_bytes=native['peak_RSS_bytes'], preparation_private_records_preserved=len(prior),
        completed_private_records_preserved=len(set(records)),
        completed_preservation_manifest_sha256=c.sha(preserve),
        auditor_sha256=c.sha(__file__), scientific_computation_repeated_by_auditor=False,
        NPZ_decoded_by_auditor=False, technical_interruptions=0, retries=0, new_seeds=0,
        FP_D_A_S_STATS_P_executed=False, next_stage_authorized=False,
        independent_scientific_review=False, ground_state_certified=False,
        H4_SOURCE_contracts_or_runtime_modified_by_task=False, new_push=False)
    c.save(public / 'completion_audit.json', report)
    c.save(public / 'artifact_manifest.json', dict(files={p.name:c.sha(p) for p in sorted(public.glob('*.json'))
                                                      if p.name != 'artifact_manifest.json'}))
    return report


if __name__ == '__main__':
    import json
    print(json.dumps(audit(), ensure_ascii=False))
