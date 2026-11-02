"""One authorized C1-1 input/basis/model audit; no event circuits or sampling."""
from __future__ import annotations

import argparse
import dataclasses
import datetime
from fractions import Fraction as F
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
import types


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULT = HERE / 'result'
RUNTIME = ROOT.parent / 'track-b-c1-1-input-basis-exact-model-20261020.runtime'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def need(value, message):
    if not value:
        raise RuntimeError(message)


def say(stage, **fields):
    print(json.dumps({'stage': stage, **fields}, ensure_ascii=False), flush=True)


def fp(x):
    return str(F(x))


def freeze_environment():
    names = ['numpy', 'scipy', 'qiskit', 'qiskit-nature', 'qiskit-algorithms',
             'openfermion', 'openfermionpyscf', 'pyscf', 'numba', 'h5py']
    rows = {}
    for name in names:
        dist = importlib.metadata.distribution(name)
        files = {}
        for rel in dist.files or ():
            if str(rel).endswith(('/METADATA', '/RECORD')):
                path = Path(dist.locate_file(rel)).resolve()
                files[str(path)] = sha(path)
        rows[name] = {'version': dist.version, 'metadata_file_sha256': files}
    return {'executable': sys.executable, 'executable_realpath': str(Path(sys.executable).resolve()),
            'executable_sha256': sha(Path(sys.executable).resolve()),
            'python_version': sys.version, 'prefix': sys.prefix, 'packages': rows}


def verify_files(rows):
    for row in rows:
        need(sha(ROOT / row['path']) == row['sha256'], 'source/input changed: ' + row['path'])


def audit(contract):
    # Avoid the broad export-only __init__ files; selected fixed modules keep
    # their normal relative imports and unmodified scientific definitions.
    for name, directory in [('trotterlib', ROOT / 'src/trotterlib'),
                            ('trotterlib.df_trotter', ROOT / 'src/trotterlib/df_trotter')]:
        module = types.ModuleType(name)
        module.__path__ = [str(directory)]
        module.__package__ = name
        sys.modules[name] = module
    import numpy as np
    from qiskit.quantum_info import Operator
    from trotterlib import pr2_new_series_validation as original
    from trotterlib import pr2_s0_s1_validation as old
    from trotterlib.df_partial_s2 import prepare_df_partial_s2

    expected = json.loads((ROOT / contract['reference_result']).read_text())
    ref_dev = expected['stages']['V1']['development']
    ref_method = expected['stages']['V2']['ranks'][0]['methods']['B2-G']
    snapshot = ROOT / contract['snapshot']['path']
    say('NPZ_LOAD_BEGIN')
    h, sector, state, sector_state, metadata, layout = original._load_snapshot_once(snapshot)
    digests = original._digest_record(h, sector, state, sector_state)
    identity = {'snapshot_sha256': sha(snapshot), 'layout': layout,
                'expected_digests': ref_dev['digests'], 'actual_digests': digests,
                'digests_equal': digests == ref_dev['digests'],
                'member_allowlist': sorted(original.EXPECTED_KEYS)}
    dump(RESULT / 'input_identity.json', identity)
    need(identity['digests_equal'], 'INPUT_IDENTITY_MISMATCH: stored digest record')
    need(set(layout) == set(original.EXPECTED_LAYOUT), 'INPUT_IDENTITY_MISMATCH: array layout')
    state_checks = {
        'full_norm_error': abs(float(np.linalg.norm(state)) - 1),
        'sector_norm_error': abs(float(np.linalg.norm(sector_state)) - 1),
        'sector_state_max_difference': float(np.max(np.abs(state[sector.basis_indices] - sector_state))),
    }
    outside = np.ones(state.shape[0], dtype=bool)
    outside[sector.basis_indices] = False
    state_checks['outside_sector_max'] = float(np.max(np.abs(state[outside])))
    hermitian = {'one_body': original._relative_hermiticity_residual(h.one_body),
                 'df_fragments': [original._relative_hermiticity_residual(g) for g in h.g_matrices]}
    dump(RESULT / 'state_model_checks.json', {'state_checks': state_checks,
         'state_tolerance_existing': original.STATE_TOLERANCE,
         'hermiticity_residuals': hermitian,
         'hermiticity_tolerance_existing': original.HERMITICITY_TOLERANCE,
         'ground_state_or_physical_preparation_certified': False})
    need(all(v <= original.STATE_TOLERANCE for v in state_checks.values()), 'INPUT_IDENTITY_MISMATCH: state conditions')
    need(hermitian['one_body'] <= original.HERMITICITY_TOLERANCE and
         all(v <= original.HERMITICITY_TOLERANCE for v in hermitian['df_fragments']),
         'INPUT_IDENTITY_MISMATCH: Hermiticity')
    partition = old.generation_partition(h, 3)
    need(list(partition.deterministic_block_indices) == ref_method['ordered_indices'],
         'INPUT_IDENTITY_MISMATCH: deterministic fragment order')
    need(list(partition.randomized_block_indices) == ref_method['randomized_indices'],
         'INPUT_IDENTITY_MISMATCH: randomized fragment order')
    say('INPUT_IDENTITIES_VERIFIED', state_hash=digests['state_hash'])

    # This call only extracts the already-defined Gaussian basis operation
    # sequences and symbolic tail. No event/wrapper builder is called.
    say('BASIS_RECONSTRUCTION_BEGIN', policy='fermionic_gaussian_jw')
    prep = prepare_df_partial_s2(h, partition, identity_policy='extract_identity_phase',
                               coefficient_atol=0.0, partition_policy='explicit_ordered_partition')
    ext = prep.tail_extraction
    need(prep.partition_hash == ref_method['partition_hash'],
         'INPUT_IDENTITY_MISMATCH: partition hash')
    rows = ref_method['sampling_components']
    need(len(rows) == len(ext.components), 'INPUT_IDENTITY_MISMATCH: component count')
    matching = []
    for saved, fresh in zip(rows, ext.components, strict=True):
        identity_equal = (saved['component_id'] == fresh.component_id and
                          saved['df_fragment_id'] == fresh.df_fragment_id and
                          tuple(saved['diagonal_pauli_support']) == fresh.diagonal_pauli_support and
                          saved['coefficient_sign'] == fresh.coefficient_sign and
                          saved['is_identity'] == fresh.is_identity)
        matching.append({'component_id': saved['component_id'], 'identity_equal': identity_equal,
                         'saved_coefficient_hex': float(saved['coefficient']).hex(),
                         'reconstructed_coefficient_hex': fresh.coefficient.hex(),
                         'coefficient_hex_equal': float(saved['coefficient']).hex() == fresh.coefficient.hex()})
    dump(RESULT / 'component_comparison.json', matching)
    need(all(r['identity_equal'] for r in matching), 'INPUT_IDENTITY_MISMATCH: component labels/sign/support')
    basis_sources = {}
    for definition in ext.basis_registry._definitions.values():
        basis_sources[definition.basis_id] = (definition.basis_hash, definition.runtime_operations,
                                             definition.metadata.operations)
    for block in prep.deterministic_blocks:
        basis_sources.setdefault(block.basis_id, (block.basis_hash, block.runtime_basis_operations,
                                                 block.basis_change_operations))
    basis_matrices = {}
    basis_index = []
    dimension = 1 << h.n_qubits
    for index, (basis_id, (basis_hash, operations, descriptors)) in enumerate(sorted(basis_sources.items())):
        matrix = Operator(np.eye(dimension, dtype=np.complex128))
        serialized = []
        for (gate, qubits), descriptor in zip(operations, descriptors, strict=True):
            local = np.asarray(Operator(gate).data, dtype=np.complex128)
            local_sha = hashlib.sha256(np.ascontiguousarray(local).view(np.uint8).tobytes()).hexdigest()
            need(local_sha == descriptor.matrix_sha256, 'BASIS_RECONSTRUCTION_INCONCLUSIVE: local matrix hash')
            serialized.append({'descriptor': dataclasses.asdict(descriptor),
                'local_matrix': {'shape': list(local.shape), 'dtype': 'complex128',
                    'real_hex': [[float(z.real).hex() for z in row] for row in local],
                    'imag_hex': [[float(z.imag).hex() for z in row] for row in local]}})
            matrix = matrix.compose(Operator(local), qargs=list(qubits))
        b = np.asarray(matrix.data)
        basis_matrices[basis_id] = b
        payload = {'basis_id': basis_id, 'basis_hash': basis_hash, 'num_system_qubits': h.n_qubits,
                   'source_operation_order': 'time order; B = last ... first', 'operations': serialized,
                   'unitarity_residual_spectral_numerical': float(np.linalg.norm(b.conj().T @ b - np.eye(dimension), 2)),
                   'unitarity_residual_status': 'diagnostic only; no new tolerance or rigorous certificate',
                   'original_serialized_basis_available': False}
        filename = f'basis_registry/basis_{index:03d}.json'
        dump(RESULT / filename, payload)
        basis_index.append({'basis_id': basis_id, 'basis_hash': basis_hash, 'path': filename,
                            'operation_count': len(operations)})
        say('BASIS_SERIALIZED', index=index, operations=len(operations))
    dump(RESULT / 'basis_registry_index.json', {
        'basis_definitions': basis_index,
        'partition_hash': prep.partition_hash, 'preparation_hash': prep.preparation_hash,
        'tail_hash': ext.tail_hash, 'recorded_preparation_hash': ref_method['preparation_hash'],
        'recorded_tail_hash': ref_method['tail_hash'],
        'preparation_hash_equal_recorded': prep.preparation_hash == ref_method['preparation_hash'],
        'tail_hash_equal_recorded': ext.tail_hash == ref_method['tail_hash'],
        'original_serialized_basis_available': False,
        'original_basis_byte_exact_reproduction_claimed': False,
        'deterministic_blocks': [{k: getattr(v, k) for k in
            ('block_id', 'original_fragment_index', 'basis_id', 'basis_hash', 'order_index')}
            for v in prep.deterministic_blocks]})

    # Adopt saved coefficient binary64 values, not recomputed eigenvalues.
    coefficients = [F.from_float(float(row['coefficient'])) for row in rows]
    need(all(c != 0 for c in coefficients), 'unexpected zero retained coefficient')
    lam = sum(map(abs, coefficients), F())
    probabilities = [abs(c) / lam for c in coefficients]
    exact_rows = []
    for label, (saved, fresh, c, p) in enumerate(zip(rows, ext.components, coefficients, probabilities, strict=True)):
        need(lam * p * saved['coefficient_sign'] == c, 'exact coefficient reconstruction failed')
        exact_rows.append({'label': label, 'component_id': saved['component_id'],
            'df_fragment_id': saved['df_fragment_id'], 'support': saved['diagonal_pauli_support'],
            'sign': saved['coefficient_sign'], 'coefficient_binary64_hex': float(saved['coefficient']).hex(),
            'signed_coefficient_exact': str(c), 'probability_exact': str(p),
            'basis_id': fresh.basis_id, 'basis_hash': fresh.basis_hash,
            'signed_involution_definition': 'sign * B_ideal * Z_support * B_ideal_adjoint',
            'basis_semantics': 'ideal unitary gate semantics; serialized binary64 matrices are numerical evidence, not exact unitary certificates'})
    exact_sum = sum(probabilities, F())
    need(all(p > 0 for p in probabilities) and exact_sum == 1, 'invalid G6 exact distribution')
    candidate_time = F('0.8')
    x = lam * candidate_time / 4
    binary_time = F.from_float(0.8)
    exact = {'model_id': 'C1_1_SAVED_COMPONENT_DYADIC_MODEL_V1',
        'coefficient_source': contract['reference_result'],
        'coefficient_json_pointer': '/stages/V2/ranks/0/methods/B2-G/sampling_components',
        'lambda_exact': str(lam), 'lambda_float_recorded_hex': float(ref_method['exact_rte_lambda_r']).hex(),
        'lambda_minus_recorded_float_exact': fp(lam - F.from_float(float(ref_method['exact_rte_lambda_r']))),
        'probability_sum_exact': str(exact_sum), 'all_positive': True, 'label_bijection': True,
        'coefficient_reconstruction_equalities_exact': True, 'components': exact_rows,
        'identity_policy': prep.identity_policy, 'coefficient_atol': prep.coefficient_atol,
        'extracted_identity_saved_binary64_exact': fp(F.from_float(float(ref_method['tail_identity_coefficient']))),
        'extracted_identity_reconstructed_binary64_exact': fp(F.from_float(prep.extracted_identity_coefficient)),
        'short_step_candidate': {'T_decimal_exact': str(candidate_time), 'q_outer': 1, 'r_RTE': 4, 'm': 5,
             'tail_occurrences_per_partial_s2_step_from_static_source': 1,
             'delta_tail_candidate': str(candidate_time), 'x_exact': str(x), 'x_le_one_exact': x <= 1,
             'binary64_time_image': str(binary_time), 'x_binary64_time_image_exact': fp(lam * binary_time / 4),
             'x_binary64_time_image_le_one_exact': lam * binary_time / 4 <= 1,
             'adopted_C1_schedule': False, 'all_occurrences_status': 'CONDITIONAL until schedule formally fixed'},
        'rigorous_binding_to_saved_DF_Hamiltonian': False, 'model_binding_status': 'MODEL_BINDING_UNCLOSED'}
    dump(RESULT / 'exact_model.json', exact)
    say('EXACT_MODEL_DEFINED', labels=len(exact_rows), normalized_exact=True, candidate_short_step=x <= 1)

    # Numerical operator check, grouped by already common basis to avoid
    # repeated dense products. This changes only audit summation order.
    def grouped_tail(use_saved):
        diagonals = {}
        for saved, fresh in zip(rows, ext.components, strict=True):
            values = np.ones(dimension)
            indices = np.arange(dimension)
            for qubit in fresh.diagonal_pauli_support:
                values *= 1 - 2 * ((indices >> qubit) & 1)
            c = float(saved['coefficient']) if use_saved else fresh.coefficient
            diagonals.setdefault(fresh.basis_id, np.zeros(dimension))
            diagonals[fresh.basis_id] += c * values
        identity = (float(ref_method['tail_identity_coefficient']) if use_saved
                    else prep.extracted_identity_coefficient)
        answer = identity * np.eye(dimension, dtype=np.complex128)
        for key, diagonal in diagonals.items():
            b = basis_matrices[key]
            answer += (b * diagonal[np.newaxis, :]) @ b.conj().T
        return answer
    say('NUMERICAL_MODEL_CHECK_BEGIN')
    full = old._dense_df_operator_qiskit(h)
    deterministic = old._dense_df_operator_qiskit(h.select_blocks(partition.deterministic_block_indices))
    reconstructed = grouped_tail(False)
    exact_as_binary64 = grouped_tail(True)
    normalizer = max(1.0, float(np.linalg.norm(full, 2)))
    values = {'reconstructed_component_model_vs_snapshot_relative_spectral':
              float(np.linalg.norm(deterministic + reconstructed - full, 2) / normalizer),
              'saved_dyadic_component_model_vs_snapshot_relative_spectral':
              float(np.linalg.norm(deterministic + exact_as_binary64 - full, 2) / normalizer),
              'saved_vs_reconstructed_component_models_relative_spectral':
              float(np.linalg.norm(exact_as_binary64 - reconstructed, 2) / normalizer)}
    checks = {k: v <= original.RECONSTRUCTION_TOLERANCE for k, v in values.items()
              if k.endswith('_vs_snapshot_relative_spectral')}
    dump(RESULT / 'model_difference.json', {'residuals': values,
        'existing_relative_spectral_tolerance': original.RECONSTRUCTION_TOLERANCE,
        'tolerance_source': 'pr2_new_series_validation.RECONSTRUCTION_TOLERANCE; same H_D+H_R vs full DF test',
        'numerical_same_comparison_within_existing_tolerance': checks,
        'coefficient_hex_all_equal_saved': all(r['coefficient_hex_equal'] for r in matching),
        'summation': 'common-basis diagonal accumulation; audit only; no sampled events',
        'operator_norm_rigorous_certificate': False, 'model_binding_status': 'MODEL_BINDING_UNCLOSED'})
    need(all(checks.values()), 'BASIS_RECONSTRUCTION_INCONCLUSIVE: existing model tolerance exceeded')
    say('NUMERICAL_MODEL_CHECK_COMPLETE', **values)
    imported = []
    for name, module in sorted(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if filename and Path(filename).is_relative_to(ROOT / 'src'):
            imported.append({'module': name, 'path': str(Path(filename).relative_to(ROOT)), 'sha256': sha(filename)})
    dump(RESULT / 'imported_source_identity.json', imported)
    return {'status': 'C1_1_LIMITED_REANALYSIS_COMPLETE', 'input_identity': 'VERIFIED',
        'basis_registry_reconstructed': 'VERIFIED',
        'original_basis_bytes': 'MISSING_SOURCE',
        'original_basis_reproduction': 'CONDITIONAL',
        'exact_coefficients_and_normalization': 'VERIFIED',
        'model_binding': 'MODEL_BINDING_UNCLOSED', 'numerical_model_check': 'VERIFIED_WITH_EXISTING_TOLERANCE',
        'candidate_short_step': 'VERIFIED_CONDITIONALLY_ON_DRAFT_SCHEDULE',
        'adapter': 'IMPLEMENTATION_REQUIRED', 'scientific_GO': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--authorization', required=True)
    args = parser.parse_args()
    wall, cpu = time.monotonic(), time.process_time()
    contract = json.loads((HERE / 'execution_contract.json').read_text())
    auth_path = Path(args.authorization).resolve()
    auth = json.loads(auth_path.read_text())
    need(auth['contract_sha256'] == sha(HERE / 'execution_contract.json'), 'contract identity mismatch')
    need(auth['source_manifest_sha256'] == sha(HERE / 'source_manifest.json'), 'source manifest mismatch')
    need(auth['runner_sha256'] == sha(__file__), 'runner changed')
    need(auth_path == HERE / 'authorization_receipt.json', 'unexpected authorization file')
    authorization_commit = git('rev-parse', 'HEAD')
    need(git('rev-parse', 'HEAD^') == auth['execution_source_commit'], 'authorization parent mismatch')
    need(git('diff-tree', '--no-commit-id', '--name-only', '-r', 'HEAD') ==
         str(auth_path.relative_to(ROOT)), 'authorization commit has unexpected changes')
    need(not git('status', '--porcelain'), 'worktree not clean at admission')
    need(freeze_environment() == json.loads((HERE / 'environment_identity.json').read_text()), 'runtime changed')
    manifest = json.loads((HERE / 'source_manifest.json').read_text())
    verify_files(manifest['fixed_files'])
    need(sha(ROOT / contract['snapshot']['path']) == contract['snapshot']['sha256'], 'snapshot hash mismatch')
    RESULT.mkdir(exist_ok=False)
    dump(RESULT / 'one_shot_consumed.json', {'run_id': contract['run_id'], 'attempt': 1, 'retry': 0,
        'authorization_file_sha256': sha(auth_path), 'execution_source_commit': auth['execution_source_commit'],
        'authorization_commit': authorization_commit, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
    state = None
    try:
        state = audit(contract)
    except BaseException as exc:
        state = {'status': 'C1_1_REANALYSIS_INCONCLUSIVE', 'stage': 'see raw stdout stage prefix',
                 'error_type': type(exc).__name__, 'error': str(exc), 'retry': 0}
        dump(RESULT / 'failure.json', {**state, 'traceback': traceback.format_exc()})
        traceback.print_exc()
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        state = state or {'status': 'C1_1_REANALYSIS_INCONCLUSIVE'}
        state.update({'run_id': contract['run_id'], 'runs': 1, 'retries': 0,
            'wall_seconds': time.monotonic() - wall, 'own_CPU_seconds': time.process_time() - cpu,
            'peak_RSS_KiB': usage.ru_maxrss, 'mandatory_STOP': True,
            'prohibited_operations': {'SCF_or_new_DF_factorization': 0, 'adapter': 0,
             'event_wrapper_circuit_build': 0, 'compile_synthesis': 0, 'sampling': 0,
             'shot_budget_calculation': 0, 'native_resource_comparison': 0, 'new_nontrivial_proofs': 0},
            'basis_operation_extraction_uses_existing_BogoliubovTransform': True})
        try:
            verify_files(manifest['fixed_files'])
            state['fixed_source_and_input_after_run'] = 'VERIFIED'
        except BaseException as exc:
            dump(RESULT / 'preservation_failure.json', {'error': str(exc), 'traceback': traceback.format_exc()})
            state['status'] = 'C1_1_REANALYSIS_INCONCLUSIVE'
            state['fixed_source_and_input_after_run'] = 'FAILED'
        dump(RESULT / 'terminal.json', state)
        rows = [{'path': str(p.relative_to(HERE)), 'bytes': p.stat().st_size, 'sha256': sha(p)}
                for p in sorted(RESULT.rglob('*')) if p.is_file()]
        dump(RESULT / 'output_inventory.json', rows)
        say('MANDATORY_STOP', **state)
    return 0 if state['status'] == 'C1_1_LIMITED_REANALYSIS_COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
