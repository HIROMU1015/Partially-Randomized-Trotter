"""One authorized R1 input/basis/exact-coefficient audit; no event/provider build."""
from __future__ import annotations

import ast
import dataclasses
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import types

from support import HERE, ROOT, digest, save, emit, require, git, environment, verify_files, run_guarded


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode('utf-8')


def fraction(value):
    return str(F.from_float(float(value)))


def audit(output, contract):
    for name, folder in [('trotterlib', ROOT / 'src/trotterlib'),
                         ('trotterlib.df_trotter', ROOT / 'src/trotterlib/df_trotter')]:
        module = types.ModuleType(name)
        module.__path__ = [str(folder)]
        module.__package__ = name
        sys.modules[name] = module
    import numpy as np
    from qiskit.quantum_info import Operator
    from trotterlib import pr2_s0_s1_validation as old
    from trotterlib import pr2_new_series_validation as prior
    from trotterlib.df_hamiltonian import DFHamiltonian, PhysicalSector
    from trotterlib.df_partial_randomized_pf import df_hamiltonian_hash
    from trotterlib.df_partial_s2 import prepare_df_partial_s2, _partition_hash, _block_hash_payload
    from trotterlib.df_rte_tail import canonical_basis_hash

    tracked = ['generation_partition', 'prepare_df_partial_s2', '_partition_hash', '_dense_df_operator_qiskit']
    forbidden = ['build_df_h_d_from_molecule', 'solve_df_ground_state', 'build_df_trotter_circuit',
                 'build_df_rte_event_sequence_circuit', 'simulate_statevector']
    calls = {name: 0 for name in tracked + forbidden}

    def profile(frame, event, arg):
        if event == 'call' and frame.f_code.co_filename.startswith(str(ROOT / 'src/trotterlib')):
            name = frame.f_code.co_name
            if name in calls:
                calls[name] += 1
            if name in forbidden:
                raise RuntimeError('prohibited operation: ' + name)

    sys.setprofile(profile)
    stage = 'GATE_2_INPUT'
    try:
        emit(stage)
        reference = json.loads((ROOT / contract['reference_result']).read_text())
        method = reference['stages']['V2']['ranks'][0]['methods']['B2-G']
        # Same array materialization as D0 for Hamiltonian members; state
        # members are read from the same single archive without resaving it.
        with np.load(ROOT / contract['snapshot']['path'], allow_pickle=False) as archive:
            require(set(archive.files) == set(contract['npz_members']), 'NPZ members mismatch', 'INPUT_IDENTITY_MISMATCH')
            arrays = {name: np.array(archive[name], copy=True) for name in contract['npz_members']}
        metadata = json.loads(str(arrays['metadata_json'].item()))
        h = DFHamiltonian(constant=float(arrays['constant']), one_body=arrays['one_body'],
             lambdas=arrays['lambdas'], g_matrices=tuple(arrays['g_matrices']),
             metadata=dict(metadata['hamiltonian_metadata']))
        s = metadata['sector']
        sector = PhysicalSector(n_qubits=int(s['n_qubits']), basis_indices=arrays['sector_basis_indices'],
             n_electrons=s['n_electrons'], nelec_alpha=s['nelec_alpha'], nelec_beta=s['nelec_beta'], sz_value=s['sz_value'])
        state, sector_state = arrays['state_vector'], arrays['sector_state_vector']
        digests = prior._digest_record(h, sector, state, sector_state)
        expected = reference['stages']['V1']['development']['digests']
        layouts = {k: {'shape': list(a.shape), 'dtype': str(a.dtype)} for k, a in arrays.items()}
        layout_equal = {k: tuple(layouts[k]['shape']) == shape and layouts[k]['dtype'] == dtype
                        for k, (shape, dtype) in prior.EXPECTED_LAYOUT.items()}
        save(output / 'input_identity.json', {'actual_digests': digests, 'expected_digests': expected,
             'equal': digests == expected, 'snapshot_sha256': digest(ROOT / contract['snapshot']['path']),
             'layouts': layouts, 'layout_equal': layout_equal})
        require(digests == expected and all(layout_equal.values()), 'input digest/layout mismatch', 'INPUT_IDENTITY_MISMATCH')

        partition = old.generation_partition(h, 3)
        hhash = df_hamiltonian_hash(h, weight_rule=partition.weight_rule)
        source_path = ROOT / 'src/trotterlib/df_partial_s2.py'
        fn = next(n for n in ast.parse(source_path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == '_partition_hash')
        require(isinstance(fn.body[0], ast.Assign) and fn.body[0].targets[0].id == 'payload', 'fixed helper structure changed')
        payload = eval(compile(ast.Expression(fn.body[0].value), str(source_path), 'eval'),
             {'asdict': dataclasses.asdict, 'hamiltonian_hash': hhash, 'partition': partition,
              'partition_policy': 'explicit_ordered_partition'})
        encoded = canonical(payload)
        actual_hash = _partition_hash(hhash, partition, partition_policy='explicit_ordered_partition')
        (output / 'partition_payload.canonical.json').write_bytes(encoded)
        save(output / 'partition_payload.json', payload)
        d0_bytes = (ROOT / contract['d0_payload']).read_bytes()
        partition_record = {'old_PR2_expected_hash': contract['historical_partition_hash'],
             'D0_hash': contract['d0_partition_hash'], 'R1_hash': actual_hash,
             'canonical_bytes_sha256': hashlib.sha256(encoded).hexdigest(),
             'D0_bytes_equal': encoded == d0_bytes, 'D0_hash_equal': actual_hash == contract['d0_partition_hash'],
             'old_PR2_hash_equal': actual_hash == contract['historical_partition_hash'],
             'hamiltonian_hash': hhash, 'source_sha256': digest(source_path),
             'partition_policy': 'explicit_ordered_partition',
             'deterministic_indices': list(partition.deterministic_block_indices),
             'randomized_indices': list(partition.randomized_block_indices)}
        save(output / 'partition_identity.json', partition_record)
        require(encoded == d0_bytes and actual_hash == contract['d0_partition_hash'] and
                hashlib.sha256(encoded).hexdigest() == actual_hash,
                'R1 partition differs from fixed D0; no basis work allowed', 'PARTITION_IDENTITY_MISMATCH')
        require(list(partition.deterministic_block_indices) == [0, 1, 2] and
                list(partition.randomized_block_indices) == list(range(3, 12)), 'fragment order mismatch', 'INPUT_IDENTITY_MISMATCH')
        outside = np.ones(state.size, dtype=bool)
        outside[sector.basis_indices] = False
        checks = {'full_norm_error': abs(float(np.linalg.norm(state)) - 1),
             'sector_norm_error': abs(float(np.linalg.norm(sector_state)) - 1),
             'sector_max_difference': float(np.max(np.abs(state[sector.basis_indices] - sector_state))),
             'outside_sector_max': float(np.max(np.abs(state[outside])))}
        hermitian = [prior._relative_hermiticity_residual(h.one_body)] + [prior._relative_hermiticity_residual(g) for g in h.g_matrices]
        save(output / 'state_checks.json', {'checks': checks, 'existing_state_tolerance': prior.STATE_TOLERANCE,
             'Hermiticity_residuals': hermitian, 'existing_Hermiticity_tolerance': prior.HERMITICITY_TOLERANCE,
             'physical_state_preparation_or_exact_ground_state_certified': False})
        require(all(v <= prior.STATE_TOLERANCE for v in checks.values()) and
                all(v <= prior.HERMITICITY_TOLERANCE for v in hermitian), 'state/Hermiticity mismatch', 'INPUT_IDENTITY_MISMATCH')
        emit('D0_PARTITION_REPRODUCED', partition_record)

        stage = 'GATE_3_BASIS'
        emit(stage)
        prep = prepare_df_partial_s2(h, partition, identity_policy='extract_identity_phase',
             coefficient_atol=0.0, partition_policy='explicit_ordered_partition')
        ext = prep.tail_extraction
        preparation_record = {'partition_hash': prep.partition_hash, 'expected_D0_hash': actual_hash,
             'preparation_hash': prep.preparation_hash, 'old_preparation_hash': method['preparation_hash'],
             'tail_hash': ext.tail_hash, 'old_tail_hash': method['tail_hash'],
             'extraction_metadata': dataclasses.asdict(ext.extraction_metadata),
             'deterministic_blocks': [_block_hash_payload(b) for b in prep.deterministic_blocks],
             'identity_policy': prep.identity_policy, 'coefficient_atol': prep.coefficient_atol,
             'constant_coefficient_hex': prep.constant_coefficient.hex(),
             'extracted_identity_coefficient_hex': prep.extracted_identity_coefficient.hex(),
             'old_extracted_identity_coefficient_hex': float(method['tail_identity_coefficient']).hex(),
             'reconstructed_components': [{field.name: getattr(c, field.name) for field in dataclasses.fields(c)
                  if field.name != 'basis_change_operations'} for c in ext.components],
             'basis_operations_storage': 'basis/basis_NNN.json; each component refers to basis_id/hash'}
        # Split component records below; this early full checkpoint also
        # survives a later registry/conversion failure.
        save(output / 'preparation_checkpoint.json', preparation_record)
        require(prep.partition_hash == actual_hash, 'preparation partition differs from D0', 'PARTITION_IDENTITY_MISMATCH')
        definitions = {d.basis_id: (d.basis_hash, d.runtime_operations, d.metadata.operations)
                       for d in ext.basis_registry._definitions.values()}
        for block in prep.deterministic_blocks:
            existing = definitions.get(block.basis_id)
            if existing is not None:
                require(existing[0] == block.basis_hash, 'basis ID conflict', 'BASIS_RECONSTRUCTION_INCONCLUSIVE')
            definitions[block.basis_id] = (block.basis_hash, block.runtime_basis_operations, block.basis_change_operations)
        dimension = 1 << h.n_qubits
        matrices, basis_rows = {}, []
        for i, (basis_id, (basis_hash, operations, descriptors)) in enumerate(sorted(definitions.items())):
            records, local_matrices = [], []
            for (gate, qubits), descriptor in zip(operations, descriptors, strict=True):
                local = np.asarray(Operator(gate).data, dtype=np.complex128)
                local_sha = hashlib.sha256(np.ascontiguousarray(local).view(np.uint8).tobytes()).hexdigest()
                records.append({'descriptor': dataclasses.asdict(descriptor),
                     'gate_class': type(gate).__module__ + '.' + type(gate).__name__, 'actual_matrix_sha256': local_sha,
                     'matrix_hash_equal': local_sha == descriptor.matrix_sha256,
                     'local_matrix': {'shape': list(local.shape), 'real_hex': [[float(z.real).hex() for z in row] for row in local],
                                      'imag_hex': [[float(z.imag).hex() for z in row] for row in local]}})
                local_matrices.append((local, qubits))
            linked = [c.component_id for c in ext.components if c.basis_id == basis_id]
            fragments = sorted({int(c.df_fragment_id.rsplit('-', 1)[1]) for c in ext.components if c.basis_id == basis_id} |
                {b.original_fragment_index for b in prep.deterministic_blocks if b.basis_id == basis_id and b.original_fragment_index is not None})
            path = f'basis/basis_{i:03d}.json'
            basis_doc = {'basis_id': basis_id, 'basis_hash': basis_hash,
                 'canonical_basis_hash_recomputed': canonical_basis_hash(h.n_qubits, descriptors),
                 'system_qubits': h.n_qubits, 'original_fragment_indices': fragments,
                 'component_ids': linked, 'operations_in_time_order': records,
                 'matrix_semantics': 'B = last ... first; operator B Z B_actual_adjoint; actual adjoint is reverse gate order with each local matrix conjugate-transposed',
                 'source': 'existing Gaussian/JW registry; fixed runtime',
                 'original_basis_bytes_available': False, 'status': 'BASIS_RECONSTRUCTED_PROVENANCE_DISTINCT'}
            save(output / path, basis_doc)
            require(all(r['matrix_hash_equal'] for r in records) and basis_doc['canonical_basis_hash_recomputed'] == basis_hash,
                    'basis descriptor/local matrix identity mismatch', 'BASIS_RECONSTRUCTION_INCONCLUSIVE')
            matrix = Operator(np.eye(dimension, dtype=np.complex128))
            for local, qubits in local_matrices:
                matrix = matrix.compose(Operator(local), qargs=list(qubits))
            b = np.asarray(matrix.data)
            matrices[basis_id] = b
            basis_rows.append({'basis_id': basis_id, 'basis_hash': basis_hash, 'path': path,
                 'bytes': (output / path).stat().st_size, 'sha256': digest(output / path), 'operations': len(records),
                 'unitarity_residual_Frobenius_diagnostic_only': float(np.linalg.norm(b.conj().T @ b - np.eye(dimension), 'fro'))})
            emit('BASIS_SAVED', {'index': i, 'operations': len(records)})
        save(output / 'basis_registry_index.json', {'status': 'BASIS_RECONSTRUCTED_PROVENANCE_DISTINCT',
             'basis_definitions': basis_rows, 'original_basis_byte_exact': False,
             'unitarity_diagnostics_have_no_new_PASS_tolerance': True})
        require(all(c.basis_id in matrices and definitions[c.basis_id][0] == c.basis_hash for c in ext.components),
                'component basis reference missing or conflicting', 'BASIS_RECONSTRUCTION_INCONCLUSIVE')

        stage = 'GATE_4_EXACT_MODEL'
        emit(stage)
        saved_rows = method['sampling_components']
        fresh = ext.components
        old_by_id = {row['component_id']: (i, row) for i, row in enumerate(saved_rows)}
        structural = []
        common_old_indices = []
        for i, c in enumerate(fresh):
            old_index, old_row = old_by_id.get(c.component_id, (None, None))
            if old_index is not None:
                common_old_indices.append(old_index)
            good = old_row is None or (
                 old_row['df_fragment_id'] == c.df_fragment_id and
                 tuple(old_row['diagonal_pauli_support']) == c.diagonal_pauli_support and
                 old_row['is_identity'] == c.is_identity)
            structural.append({'label': i, 'old_label': old_index, 'old_component_id': old_row['component_id'] if old_row else None,
                 'new_component_id': c.component_id, 'common_component_structure_equal': good,
                 'old_coefficient_sign': old_row['coefficient_sign'] if old_row else None,
                 'new_coefficient_sign': c.coefficient_sign,
                 'old_coefficient_hex': float(old_row['coefficient']).hex() if old_row else None,
                 'new_coefficient_hex': c.coefficient.hex(),
                 'coefficient_hex_equal': old_row is not None and float(old_row['coefficient']).hex() == c.coefficient.hex()})
        fresh_ids = {c.component_id for c in fresh}
        old_only = [row for row in saved_rows if row['component_id'] not in fresh_ids]
        identical_id_order = [c.component_id for c in fresh] == [r['component_id'] for r in saved_rows]
        legal_components = all(int(c.df_fragment_id.rsplit('-', 1)[1]) in partition.randomized_block_indices and
             1 <= len(c.diagonal_pauli_support) <= 2 and len(set(c.diagonal_pauli_support)) == len(c.diagonal_pauli_support) and
             all(0 <= q < h.n_qubits for q in c.diagonal_pauli_support) and not c.is_identity for c in fresh)
        save(output / 'component_comparison.json', {'rows': structural, 'old_basis_metadata_available': False,
             'old_only_rows': old_only, 'identical_component_ID_order': identical_id_order,
             'common_labels_relative_order_preserved': common_old_indices == sorted(common_old_indices),
             'new_retained_components_legal': legal_components, 'old_list_and_new_input_declared_same': False,
             'note': 'new source retention at threshold zero and reconstructed coefficient signs define new input; old differences are recorded, never masked'})
        require(all(r['common_component_structure_equal'] for r in structural) and legal_components and
                common_old_indices == sorted(common_old_indices), 'component structure/order mapping mismatch', 'COMPONENT_MAPPING_INCONCLUSIVE')
        coefficients = [F.from_float(c.coefficient) for c in fresh]
        require(len({c.component_id for c in fresh}) == len(fresh) and all(c != 0 for c in coefficients),
                'duplicate or zero retained label', 'COMPONENT_MAPPING_INCONCLUSIVE')
        lam = sum(map(abs, coefficients), F())
        probabilities = [abs(c) / lam for c in coefficients]
        exact_rows = []
        for label, (component, c, p) in enumerate(zip(fresh, coefficients, probabilities, strict=True)):
            exact_rows.append({'label': label, 'component_id': component.component_id,
                 'fragment_id': component.df_fragment_id, 'original_fragment_index': int(component.df_fragment_id.rsplit('-', 1)[1]),
                 'coefficient_binary64': component.coefficient, 'coefficient_hex': component.coefficient.hex(),
                 'signed_coefficient_exact': str(c), 'probability_exact': str(p), 'sign': component.coefficient_sign,
                 'support': list(component.diagonal_pauli_support), 'identity': component.is_identity,
                 'basis_id': component.basis_id, 'basis_hash': component.basis_hash,
                 'signed_involution': 'sign * B_ideal * Z_support * B_ideal_adjoint',
                 'exact_coefficient_reconstruction': lam * p * component.coefficient_sign == c})
        chunks = []
        for begin in range(0, len(exact_rows), 36):
            path = f'labels/labels_{begin:04d}.json'
            save(output / path, exact_rows[begin:begin + 36])
            chunks.append({'path': path, 'first_label': begin, 'count': len(exact_rows[begin:begin + 36]), 'sha256': digest(output / path)})
        exact_core = {'schema': 'R1_G6_RECONSTRUCTED_DYADIC_INPUT_V1', 'partition_hash': actual_hash,
             'Hamiltonian_hash': hhash, 'preparation_hash': prep.preparation_hash, 'tail_hash': ext.tail_hash,
             'execution_source_commit': json.loads((HERE / 'authorization_receipt.json').read_text())['execution_source_commit'],
             'snapshot_sha256': contract['snapshot']['sha256'], 'environment_identity_sha256': digest(HERE / 'environment_identity.json'),
             'coefficient_adoption': 'new reconstructed finite binary64 coefficients as exact dyadic rationals; not old float lambda/probabilities',
             'label_count': len(exact_rows), 'label_chunks_in_order': chunks, 'lambda_exact': str(lam),
             'old_PR2_float_lambda_hex_diagnostic_only': float(method['exact_rte_lambda_r']).hex(),
             'new_exact_lambda_minus_old_PR2_float_exact': str(lam - F.from_float(float(method['exact_rte_lambda_r']))),
             'probability_sum_exact': str(sum(probabilities, F())), 'all_positive': all(p > 0 for p in probabilities),
             'label_component_bijection': len({c.component_id for c in fresh}) == len(fresh),
             'coefficient_reconstruction_equalities': all(r['exact_coefficient_reconstruction'] for r in exact_rows),
             'identity_policy': prep.identity_policy, 'coefficient_atol': prep.coefficient_atol,
             'constant_exact_from_binary64': fraction(prep.constant_coefficient),
             'tail_extracted_identity_exact_from_binary64': fraction(prep.extracted_identity_coefficient),
             'scalar_phase_convention': 'exp(-i*t*constant) and exp(-i*t*extracted_tail_identity), outside random labels',
             'operator_semantics': 'ideal unitary gates; local binary64 matrices are numerical evidence, not exact unitary certificates',
             'word_operator_order': 'Q(i1)...Q(il); circuit time order is reversed; each Q applies B-adjoint, Z, B',
             'relative_phase': 'retain sign in Q; G6 parent (-i*sigma)^len(u), child rotation exp(-i*sigma*phi*Q_i); no extra sign multiplier',
             'old_PR2_same_execution_input_claimed': False, 'original_DF_rigorous_binding': 'MODEL_BINDING_UNCLOSED'}
        save(output / 'exact_model_core.json', exact_core)
        model_hash = hashlib.sha256(canonical(exact_core)).hexdigest()
        save(output / 'model_identity.json', {'model_sha256': model_hash,
             'definition': 'SHA256 compact sorted UTF-8 exact_model_core; ordered chunk hashes bind label bytes',
             'core_file_sha256': digest(output / 'exact_model_core.json'), 'new_provenance': True})
        require(exact_core['all_positive'] and exact_core['probability_sum_exact'] == '1' and
                exact_core['coefficient_reconstruction_equalities'], 'exact normalization failed', 'EXACT_MODEL_INCONCLUSIVE')
        emit('EXACT_MODEL_SAVED', {'labels': len(exact_rows), 'model_hash': model_hash})

        stage = 'GATE_5_NUMERICAL_BINDING'
        emit(stage)
        # Cache diagonals within a common basis; labels remain distinct.
        def tail_matrix(use_old_coefficients=False):
            diagonals = {}
            indices = np.arange(dimension)
            for label, component in enumerate(fresh):
                values = np.ones(dimension)
                for qubit in component.diagonal_pauli_support:
                    values *= 1 - 2 * ((indices >> qubit) & 1)
                c = float(saved_rows[label]['coefficient']) if use_old_coefficients else component.coefficient
                diagonals.setdefault(component.basis_id, np.zeros(dimension))
                diagonals[component.basis_id] += c * values
            scalar = float(method['tail_identity_coefficient']) if use_old_coefficients else prep.extracted_identity_coefficient
            answer = scalar * np.eye(dimension, dtype=np.complex128)
            for key, diagonal in diagonals.items():
                b = matrices[key]
                answer += (b * diagonal[np.newaxis, :]) @ b.conj().T
            return answer

        full = old._dense_df_operator_qiskit(h)
        deterministic = old._dense_df_operator_qiskit(h.select_blocks(partition.deterministic_block_indices))
        new_tail = tail_matrix()
        hybrid_old_coefficients_on_new_basis = tail_matrix(True) if identical_id_order else None
        normalizer = max(1.0, float(np.linalg.norm(full, 2)))
        residual = float(np.linalg.norm(new_tail - (full - deterministic), 2) / normalizer)
        hybrid_residual = (float(np.linalg.norm(hybrid_old_coefficients_on_new_basis - (full - deterministic), 2) / normalizer)
                           if hybrid_old_coefficients_on_new_basis is not None else None)
        numerical = {'target': 'saved full DF vs selected DF deterministic (same constant/one-body retained) + newly extracted tail including scalar identity',
             'new_model_relative_spectral_residual': residual, 'existing_tolerance': prior.RECONSTRUCTION_TOLERANCE,
             'existing_tolerance_source': 'pr2_new_series_validation._method_record: residual-(full-deterministic) / max(1,norm(full,2))',
             'within_existing_comparison_tolerance': residual <= prior.RECONSTRUCTION_TOLERANCE,
             'status': 'MODEL_BINDING_NUMERICALLY_SUPPORTED' if residual <= prior.RECONSTRUCTION_TOLERANCE else 'MODEL_MISMATCH',
             'old_coefficients_on_R1_basis_relative_residual_diagnostic_only': hybrid_residual,
             'hybrid_is_not_original_PR2_operator_reproduction': True,
             'original_PR2_full_operator_comparison': 'UNRESOLVED; old basis operation sequence unavailable',
             'rigorous_operator_norm_certificate': False, 'rigorous_binding_status': 'MODEL_BINDING_UNCLOSED',
             'binary64_evaluation_of_exact_coefficients': 'each dyadic coefficient converts exactly back to its adopted binary64; matrix sums/products are floating numerical diagnostics'}
        save(output / 'model_binding.json', numerical)
        require(numerical['within_existing_comparison_tolerance'], 'model reconstruction exceeds existing tolerance', 'MODEL_MISMATCH')
        x = lam * F(4, 5) / 4
        short = {'draft_T_decimal_exact': '4/5', 'q_outer': 1, 'r_RTE': 4, 'm': 5,
             'draft_delta_tail': '4/5', 'x_exact': str(x), 'x_le_one_exact': x <= 1,
             'binary64_time_image_exact': fraction(0.8), 'x_binary64_time_exact': str(lam * F.from_float(0.8) / 4),
             'x_binary64_time_le_one': lam * F.from_float(0.8) / 4 <= 1,
             'static_schedule': 'one tail occurrence at full step_time between two deterministic halves; r_RTE repeats that tail step',
             'C1_schedule_adopted': False, 'all_actual_occurrences_status': 'CONDITIONAL; actual C1 schedule/time/sign not fixed',
             'no_time_or_split_or_degree_modified': True}
        save(output / 'short_step.json', short)
        return {'classification': 'R1_LIMITED_REANALYSIS_COMPLETE', 'diagnosis_complete': True,
             'partition_D0_reproduced': True, 'input_identity': 'VERIFIED',
             'basis_status': 'BASIS_RECONSTRUCTED_PROVENANCE_DISTINCT', 'basis_count': len(basis_rows),
             'label_count': len(exact_rows), 'exact_model_sha256': model_hash, 'exact_normalization': 'VERIFIED',
             'model_numerical_binding': numerical['status'], 'rigorous_model_binding': 'MODEL_BINDING_UNCLOSED',
             'short_step': 'candidate only; all actual occurrences CONDITIONAL', 'adapter': 'IMPLEMENTATION_REQUIRED',
             'scientific_GO_or_submission_GO': False}
    except BaseException:
        save(output / 'failure_stage.json', {'stage': stage, 'message': 'see primary failure traceback and saved Gate checkpoints'})
        raise
    finally:
        sys.setprofile(None)
        save(output / 'call_observations.json', {'listed_Python_function_counts': calls, 'np_load_calls_in_runner': 1,
             'scope': 'selected functions after imports; no claim to profile every C extension/internal basis object'})


def main():
    initially_clean = not git('status', '--porcelain')
    manifest = json.loads((HERE / 'source_manifest.json').read_text())

    def action(output):
        contract = json.loads((HERE / 'execution_contract.json').read_text())
        auth = json.loads((HERE / 'authorization_receipt.json').read_text())
        require(initially_clean, 'worktree dirty at admission')
        for key, filename in [('contract_sha256', 'execution_contract.json'), ('source_manifest_sha256', 'source_manifest.json'),
                              ('runner_sha256', 'run_r1.py'), ('support_sha256', 'support.py')]:
            require(auth[key] == digest(HERE / filename), 'authorization identity mismatch: ' + filename)
        require(git('rev-parse', 'HEAD^') == auth['execution_source_commit'], 'authorization parent mismatch')
        require(git('diff-tree', '--no-commit-id', '--name-only', '-r', 'HEAD') ==
                str((HERE / 'authorization_receipt.json').relative_to(ROOT)), 'unexpected authorization commit changes')
        verify_files(manifest)
        require(environment() == json.loads((HERE / 'environment_identity.json').read_text()), 'runtime changed')
        save(output / 'run_identity.json', {'run_id': contract['run_id'], 'execution_source_commit': auth['execution_source_commit'],
             'authorization_commit': git('rev-parse', 'HEAD'), 'authorization_sha256': digest(HERE / 'authorization_receipt.json'),
             'old_authorization_reused': False, 'environment_identity_sha256': digest(HERE / 'environment_identity.json')})
        return audit(output, contract)

    return run_guarded(action, HERE / 'result', finalizer=lambda: verify_files(manifest))


if __name__ == '__main__':
    raise SystemExit(main())
