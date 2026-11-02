"""Phase B: authorized saved-local-matrix identity only; no quantum operator build."""
import hashlib
import json
import sys
from support import HERE, ROOT, R1, PRIOR, RECOVERY, BASE, MODEL, canonical, digest, environment, save
from bitwise import restore_local, independent_bytes, legacy_local, difference, raw_hash

def detail(raw, expected, actual, context):
    return {**context, 'real_hex': raw['real_hex'], 'imag_hex': raw['imag_hex'],
        'expected_bytes_hex': expected.hex(), 'actual_bytes_hex': actual.hex(),
        'expected_hash_from_hex_bytes': hashlib.sha256(expected).hexdigest(),
        'actual_hash': hashlib.sha256(actual).hexdigest(), **difference(expected, actual)}

def main():
    assert json.loads((HERE / 'phase_a_checks.json').read_text())['status'] == 'PASS'
    output = HERE / 'phase_b'; output.mkdir(exist_ok=False)
    save(output / 'execution_identity.json', {'scope': 'authorized Phase B all saved local matrices only',
         'base_commit': BASE, 'helper_sha256': digest(HERE / 'bitwise.py'), 'runner_sha256': digest(__file__),
         'reference_sha256': digest(HERE / 'reference.py'), 'runtime': environment(),
         'no_H4_NPZ_or_G6_event_or_operator_XY_evaluation': True})
    index_path = R1 / 'result/basis_registry_index.json'
    index = json.loads(index_path.read_text())
    assert len(index['basis_definitions']) == 13 and sys.byteorder == 'little'
    all_rows, failures, legacy_failures, basis_summary = [], [], [], []
    input_rows = [{'path': str(index_path.relative_to(ROOT)), 'sha256': digest(index_path)}]
    for item in index['basis_definitions']:
        path = R1 / 'result' / item['path']; assert digest(path) == item['sha256']
        input_rows.append({'path': str(path.relative_to(ROOT)), 'sha256': digest(path)})
        doc = json.loads(path.read_text()); operations = doc['operations_in_time_order']
        descriptors = [op['descriptor'] for op in operations]
        computed = hashlib.sha256(canonical({'num_system_qubits': doc['system_qubits'], 'operations': descriptors,
                                            'hash_policy': 'ordered_local_basis_operations_v1'})).hexdigest()
        assert computed == doc['basis_hash'] == item['basis_hash'] and doc['basis_id'] == item['basis_id']
        assert len(operations) == item['operations']
        begin = len(all_rows)
        for operation_index, operation in enumerate(operations):
            descriptor, raw = operation['descriptor'], operation['local_matrix']
            expected = independent_bytes(raw); matrix = restore_local(raw); actual = matrix.tobytes(order='C')
            expected_hash = descriptor['matrix_sha256']; restored_hash = raw_hash(matrix)
            context = {'basis_id': doc['basis_id'], 'basis_hash': doc['basis_hash'],
                       'basis_file': item['path'], 'operation_index': operation_index,
                       'gate_type': operation['gate_class'], 'gate_name': descriptor['name'], 'qubits': descriptor['qubits']}
            match = actual == expected and restored_hash == expected_hash == operation['actual_matrix_sha256']
            row = {**context, 'shape': list(matrix.shape), 'dtype': matrix.dtype.str, 'C_contiguous': bool(matrix.flags.c_contiguous),
                   'endianness': sys.byteorder, 'expected_SHA256': expected_hash, 'actual_SHA256': restored_hash,
                   'independent_packed_hex_bytes_SHA256': hashlib.sha256(expected).hexdigest(), 'exact_bytes_and_hash_match': match}
            old_bytes = legacy_local(raw).tobytes(order='C'); old_delta = difference(expected, old_bytes)
            row.update(legacy_SHA256=hashlib.sha256(old_bytes).hexdigest(), legacy_difference=old_delta)
            if not old_delta['equal']: legacy_failures.append(detail(raw, expected, old_bytes, context))
            if not match: failures.append(detail(raw, expected, actual, context) | {'saved_expected_SHA256': expected_hash})
            assert matrix.shape == tuple(raw['shape']) == (1 << len(descriptor['qubits']),) * 2
            all_rows.append(row)
        save(output / ('basis_' + str(len(basis_summary)).zfill(3) + '.json'), {'basis_id': doc['basis_id'],
             'basis_hash': computed, 'operation_order_and_descriptor_hash_preserved': True, 'operations': all_rows[begin:]})
        basis_summary.append({'basis_id': doc['basis_id'], 'operations': len(operations), 'file_SHA256': digest(path)})
    # Reproduce the previous selected-basis traversal using metadata only, no operator composition.
    core = json.loads((R1 / 'result/exact_model_core.json').read_text()); labels = []
    assert hashlib.sha256(canonical(core)).hexdigest() == MODEL
    for chunk in core['label_chunks_in_order']:
        path = R1 / 'result' / chunk['path']; assert digest(path) == chunk['sha256']
        labels.extend(json.loads(path.read_text()))
    previous = json.loads((RECOVERY / 'execution_contract.json').read_text())
    selected = {i for event in previous['events'] for i in [*event['word'], event['child']]}
    selected_bases = sorted({labels[i]['basis_id'] for i in selected})
    first_selected = next((row for bid in selected_bases for row in legacy_failures if row['basis_id'] == bid), None)
    save(output / 'legacy_first_difference.json', {'first_in_all_saved_basis_order': legacy_failures[0] if legacy_failures else None,
         'previous_selected_basis_traversal': selected_bases, 'first_in_previous_selected_basis_order': first_selected,
         'legacy_source_sha256': digest(PRIOR / 'reference.py'),
         'reproduction_scope': 'same old local expression and fixed selected-basis order; no full basis/event/XY build',
         'all_legacy_differences_are_zero_sign_bits': bool(legacy_failures) and all(r['all_differences_are_zero_sign_bits'] for r in legacy_failures)})
    if failures: save(output / 'failure.json', {'classification': 'PHASE_B_LOCAL_MATRIX_MISMATCH_STOP', 'first': failures[0], 'all': failures})
    for row in input_rows: assert digest(ROOT / row['path']) == row['sha256']
    save(output / 'input_manifest.json', {'files': input_rows, 'unchanged_after_verification': True})
    summary = {'status': 'PASS' if not failures else 'FAILED_STOP', 'basis_count': len(basis_summary),
         'local_matrix_count': len(all_rows), 'new_mismatch_count': len(failures), 'legacy_mismatch_count': len(legacy_failures),
         'all_exact_bytes_hash_match': not failures, 'descriptors_operation_order_basis_ID_hash_preserved': True,
         'existing_R1_files_unchanged': True, 'reference_uses_saved_hex_not_adapter_output': True,
         'legacy_root_cause': 'only IEEE754 zero sign bits changed by real + 1j*imag' if legacy_failures and all(r['all_differences_are_zero_sign_bits'] for r in legacy_failures) else 'inspect saved evidence',
         'G6_event_queries_H4_operator_XY_sampling_native': 0, 'phase_C_scientific_attempts': 0}
    save(output / 'summary.json', summary); print(json.dumps(summary), flush=True)
    return 0 if not failures else 1

if __name__ == '__main__':
    import traceback
    try:
        code = main()
    except Exception as exc:
        directory = HERE / 'phase_b'; directory.mkdir(exist_ok=True)
        if not (directory / 'failure.json').exists():
            save(directory / 'failure.json', {'classification': 'PHASE_B_TECHNICAL_STOP',
                 'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
        traceback.print_exc(); code = 1
    save(HERE / 'phase_b/terminal.json', {'status': 'PASS' if code == 0 else 'FAILED_STOP',
         'phase_B_local_verification_executions': 1, 'phase_C_scientific_attempts': 0,
         'mandatory_STOP': code != 0, 'next': 'prepare fixed Phase C' if code == 0 else 'preserve/stop'})
    raise SystemExit(code)
