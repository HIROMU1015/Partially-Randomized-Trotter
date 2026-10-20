"""Frozen C1-3-R1 new one-shot (retry 0). No import of molecular construction/synthesis."""
from fractions import Fraction as F
import datetime
import json
import os
from pathlib import Path
import resource
import time
import traceback
from support import HERE, ROOT, R1, MODEL, digest, environment, git, save, verify_rows

from journal import Journal, classify_failure

def require(condition, reason):
    if not condition: raise RuntimeError(reason)

def main():
    # Exclusive directory and marker: failures, including provenance errors, consume this attempt.
    output = HERE / 'result'
    output.mkdir(exist_ok=False)
    save(output / 'attempt.json', {'scientific_attempt': 1, 'retry': 0, 'source_repair_allowed': False,
                                  'authorization_commit': git('rev-parse', 'HEAD'), 'run_id': 'C1-3-R1-H4-selected-20261021',
                                  'previous_failed_run_resumed': False})
    journal = Journal(output)
    emit = journal.emit
    started = time.monotonic()
    stage, completed = 'PROVENANCE', False
    try:
        contract = json.loads((HERE / 'execution_contract.json').read_text())
        authorization = json.loads((HERE / 'authorization_receipt.json').read_text())
        require(authorization['source_commit'] == git('rev-parse', 'HEAD^'), 'AUTH_NOT_DIRECT_CHILD_OF_SOURCE')
        require(git('diff', '--name-only', 'HEAD^', 'HEAD') == str((HERE / 'authorization_receipt.json').relative_to(ROOT)),
                'AUTH_COMMIT_CHANGED_MORE_THAN_RECEIPT')
        for filename, key in (('execution_contract.json', 'contract_sha256'),
                              ('source_manifest.json', 'source_manifest_sha256'),
                              ('environment_identity.json', 'environment_sha256')):
            require(digest(HERE / filename) == authorization[key], 'AUTH_HASH_MISMATCH:' + filename)
        manifest = json.loads((HERE / 'source_manifest.json').read_text())
        verify_rows(manifest['files'])
        verify_rows(json.loads((HERE / 'input_manifest.json').read_text())['files'])
        verify_rows(json.loads((HERE / 'runtime_manifest.json').read_text())['files'])
        require(environment() == json.loads((HERE / 'environment_identity.json').read_text()), 'RUNTIME_IDENTITY_CHANGED')
        save(output / 'execution_identity.json', {'source_commit': authorization['source_commit'],
             'authorization_commit': git('rev-parse', 'HEAD'), 'authorization_sha256': digest(HERE / 'authorization_receipt.json'),
             'contract_sha256': digest(HERE / 'execution_contract.json'), 'input_model_sha256': MODEL,
             'old_grant_reused': False, 'run': 1, 'retry': 0})
        emit(stage, status='PASS')
        # Scientific code is loaded only after every fixed file is checked.
        import numpy as np
        from qiskit.quantum_info import Operator
        from adapter import DFAdapter, restore_r1
        from state_identity import verify_state_identity
        from reference import saved_basis, q_reference, event_reference, compare
        from trottertracks.algorithm_codesign.g6_f3_generator import prepare
        stage = 'R1_RESTORE'
        core, rows, probabilities, registry, saved = restore_r1()
        require(core['partition_hash'] == contract['partition_hash'], 'PARTITION_MISMATCH')
        adapter = DFAdapter(rows, probabilities, registry, 8, MODEL)
        with np.load(ROOT / contract['snapshot']['path'], allow_pickle=False) as archive:
            state = np.array(archive['state_vector'], copy=True)
        state_identity = verify_state_identity(state, contract['state_sha256'])
        save(output / 'state_identity.json', state_identity)
        require(state.dtype.str == '<c16' and state.shape == (256,), 'STATE_DTYPE_SHAPE_MISMATCH')
        # Existing R1 state-norm criterion; no normalization or physical preparation assumption.
        state_norm_error = abs(float(np.linalg.norm(state)) - 1)
        require(state_norm_error <= contract['state_tolerance'], 'SAVED_STATE_NORM_MISMATCH')
        emit(stage, basis_count=len(saved), label_count=len(rows))
        stage = 'SCHEDULE'
        lam, delta = F(core['lambda_exact']), F(contract['T']) / contract['q_outer'] / contract['r_RTE']
        x = lam * abs(delta)
        require(0 < x <= 1 and contract['sigma'] == (1 if delta > 0 else -1), 'SHORT_STEP_OR_SIGN_MISMATCH')
        prep = json.loads((R1 / 'result/preparation_checkpoint.json').read_text())
        require(prep['partition_hash'] == core['partition_hash'] and prep['tail_hash'] == core['tail_hash']
                and prep['preparation_hash'] == core['preparation_hash'], 'SCHEDULE_INPUT_IDENTITY_MISMATCH')
        ids = [block['block_id'] for block in sorted(prep['deterministic_blocks'], key=lambda b: b['order_index'])]
        schedule = [{'role': 'deterministic_half', 'blocks_in_time_order': ids, 'time': str(F(contract['T']) / 2)},
                    {'role': 'tail_occurrence', 'substeps': [{'index': j, 'time': str(delta),
                      'sigma': contract['sigma'], 'x': str(x)} for j in range(contract['r_RTE'])]},
                    {'role': 'deterministic_half', 'blocks_in_time_order': list(reversed(ids)), 'time': str(F(contract['T']) / 2)}]
        scalar = F(core['constant_exact_from_binary64']) + F(core['tail_extracted_identity_exact_from_binary64'])
        save(output / 'schedule.json', {'scope': 'one outer step; selected event at a tail substep, no full trajectory build',
             'sequence_in_time_order': schedule, 'scalar_total_time': contract['T'],
             'scalar_attribution_for_event_diagnostic': 'commuting constant and tail identity apportioned equally to four substeps',
             'scalar_exact': str(scalar), 'identity_excluded_from_labels': True,
             'time_binary64_rounding_is_diagnostic_not_rigorous': True,
             'physical_state_preparation_certified': False, 'state_norm_error': state_norm_error})
        emit(stage, status='PASS')
        stage = 'FINITE_BIT_PREFLIGHT'
        policy = contract['precision']
        kwargs = {k: (F(v) if k in ('eta', 'rho', 'angle_tau') else v) for k, v in policy.items()}
        session = prepare(probabilities, x, 5, sigma=contract['sigma'], **kwargs)
        save(output / 'preflight.json', session.preflight.to_dict())
        require(session.status()['state'] == 'READY' and session.preflight.read()['ready'], 'P0_P5_NOT_READY')
        emit(stage, status='PASS', preflight_id=session.preflight.digest)
        stage = 'SELECTED_EVENTS'
        selected_labels = sorted({label for item in contract['events'] for label in [*item['word'], item['child']]})
        basis_ids = sorted({rows[label]['basis_id'] for label in selected_labels})
        bases = {key: saved_basis(saved[key], 8) for key in basis_ids}
        q = q_reference(rows, bases, 8)
        results = []
        for index, item in enumerate(contract['events']):
            word, child = tuple(item['word']), item['child']
            record = session.event(word, child, trial_id='selected-formal-' + str(index))
            sidecar = adapter.bind(session, record)
            circuit = adapter.build(sidecar, delta, scalar)
            event = record.read()['event']
            active = event_reference(event, q, 8, contract['sigma'], float(delta * scalar))
            result = compare(circuit, active, state, 8)
            angle = sidecar.read()['angle']
            conversion = abs(F.from_float(float(angle['signed_representative'])) - angle['signed_representative'])
            require(event['proposal'] * event['weight'] == event['coefficient'] and event['proposal'] > 0,
                    'EXACT_PROPOSAL_WEIGHT_FAILED')
            require(angle['precision_check']['passed'], 'ANGLE_GUARD_FAILED')
            save(output / f'event_{index:02d}_sidecar.json', sidecar.to_dict())
            save(output / f'event_{index:02d}_parent.json', session.parent_packet(word).to_dict())
            result.update({'index': index, 'word': list(word), 'child': child, 'labels': selected_labels,
                           'event_id': record.digest, 'sidecar_id': sidecar.digest,
                           'source_id': record.read()['source_id'], 'input_id': record.read()['input_id'],
                           'proposal_weight_exact_identity': True,
                           'angle_math_plus_float_conversion_bound': str(angle['representative_error_upper'] + conversion),
                           'logical_gate_counts_not_native_cost': {str(k): int(v) for k, v in circuit.count_ops().items()},
                           'workspace_beyond_system': 1,
                           'saved_state_XY_is_diagnostic': True})
            save(output / f'event_{index:02d}_comparison.json', result)
            results.append(result)
            emit(stage, index=index, operator_error=result['controlled_Frobenius_error_upper_for_spectral_diagnostic'],
                 numerical_gate=result['operator_numerical_gate_passed'])
            require(result['operator_numerical_gate_passed'], 'CONTROLLED_OPERATOR_NUMERICAL_GATE_FAILED')
            require(result['actual_adjoint_Frobenius_error'] <= contract['operator_tolerance'], 'ACTUAL_ADJOINT_FAILED')
        save(output / 'summary.json', {'status': 'C1_3_CORRECTNESS_PASS', 'scientific_attempts': 1, 'retry': 0,
             'selected_events': len(results), 'selected_labels': selected_labels, 'selected_basis_count': len(bases),
             'restored_basis_count': len(saved), 'restored_labels': len(rows),
             'maximum_operator_Frobenius_error': max(r['controlled_Frobenius_error_upper_for_spectral_diagnostic'] for r in results),
             'maximum_XY_absolute_error_diagnostic': max(v['absolute_error_diagnostic'] for r in results for v in r['XY'].values()),
             'P0_P5_ready': True, 'all_events_or_paths_certified': False,
             'native_provider_or_model_norm_error_certified': False,
             'H_N_or_resource_winner_computed': False, 'C1_4_may_proceed': True})
        completed = True
    except Exception as exc:
        journal.fail(exc, stage, classify_failure(exc, stage))
        emit('FAILURE', failed_at=stage, exception_type=type(exc).__name__, message=str(exc))
        traceback.print_exc()
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        journal.terminal('C1_3_CORRECTNESS_PASS' if completed else 'C1_3_TECHNICAL_INCONCLUSIVE_STOP', stage,
             wall_seconds=time.monotonic() - started,
             CPU_user_seconds=usage.ru_utime, CPU_system_seconds=usage.ru_stime,
             maximum_RSS_KiB=usage.ru_maxrss, resource_limits_added=False,
             shared_configuration_modified=False, no_native_compile_or_sampling=True,
             mandatory_STOP=not completed,
             next='C1-4 documentation only' if completed else 'mandatory STOP; new explicit approval required')
    return 0 if completed else 1

if __name__ == '__main__': raise SystemExit(main())
