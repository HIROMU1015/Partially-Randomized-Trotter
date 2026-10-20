"""C1-1-D0: one fixed-input partition-hash diagnosis; no basis reconstruction."""
from __future__ import annotations

import ast
import dataclasses
import datetime
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


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def emit(event_name, details=None):
    # Nested details prevent stage/status keys from colliding with the logger.
    print(json.dumps({'event': event_name, 'details': details or {}}, ensure_ascii=False), flush=True)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def environment():
    packages = {}
    for name in ['numpy', 'scipy', 'qiskit', 'qiskit-nature', 'qiskit-algorithms',
                 'openfermion', 'openfermionpyscf', 'pyscf', 'numba', 'h5py']:
        dist = importlib.metadata.distribution(name)
        files = {}
        for rel in dist.files or ():
            if str(rel).endswith(('/METADATA', '/RECORD')):
                path = Path(dist.locate_file(rel)).resolve()
                files[str(path)] = digest(path)
        packages[name] = {'version': dist.version, 'metadata_file_sha256': files}
    return {'executable': sys.executable, 'executable_realpath': str(Path(sys.executable).resolve()),
            'executable_sha256': digest(Path(sys.executable).resolve()), 'python_version': sys.version,
            'prefix': sys.prefix, 'packages': packages}


def verify_files(manifest):
    for row in manifest['files']:
        path = ROOT / row['path']
        require(path.stat().st_size == row['bytes'] and digest(path) == row['sha256'],
                'fixed file changed: ' + row['path'])


def float_record(value, provenance):
    return {'value': value, 'repr': repr(value), 'hex': float(value).hex(),
            'type': type(value).__module__ + '.' + type(value).__name__, 'provenance': provenance}


def run_guarded(action, output, finalizer=lambda: None, event_writer=emit):
    """Preserve primary failures, secondary failures, terminal, and exit mapping."""
    output = Path(output)
    output.mkdir(exist_ok=False)
    wall, cpu = time.monotonic(), time.process_time()
    secondary = []
    state = {'classification': 'TECHNICAL_INCONCLUSIVE', 'diagnosis_complete': False}
    save(output / 'attempt.json', {'attempt': 1, 'retry': 0,
         'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
    try:
        state = action(output)
        if state['classification'].startswith('HASH_MISMATCH'):
            save(output / 'failure.json', {'kind': 'HASH_COMPARISON_MISMATCH',
                 'exception_traceback': None, 'diagnostic_outcome': state,
                 'meaning': 'preserved mismatch, not an invented Python exception'})
    except BaseException as exc:
        state = {'classification': 'TECHNICAL_INCONCLUSIVE', 'diagnosis_complete': False,
                 'primary_error_type': type(exc).__name__, 'primary_error': str(exc)}
        record = {**state, 'traceback': traceback.format_exc()}
        try:
            save(output / 'failure.json', record)
        except BaseException:
            secondary.append({'where': 'primary_failure_save', 'traceback': traceback.format_exc()})
        traceback.print_exc()
    try:
        finalizer()
        state['preservation_after_run'] = 'VERIFIED'
    except BaseException:
        secondary.append({'where': 'preservation', 'traceback': traceback.format_exc()})
        state['original_classification_before_preservation_failure'] = state['classification']
        state['classification'] = 'TECHNICAL_INCONCLUSIVE'
        state['preservation_after_run'] = 'FAILED'
    state.update({'attempts': 1, 'retries': 0, 'mandatory_STOP': True,
                  'wall_seconds': time.monotonic() - wall, 'own_CPU_seconds': time.process_time() - cpu,
                  'peak_RSS_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    try:
        event_writer('MANDATORY_STOP', {'terminal': state})
    except BaseException:
        secondary.append({'where': 'terminal_logger', 'traceback': traceback.format_exc()})
    if secondary:
        save(output / 'secondary_failures.json', secondary)
        state['secondary_failure_count'] = len(secondary)
        state['diagnostic_classification_before_secondary_failure'] = state['classification']
        state['classification'] = 'TECHNICAL_INCONCLUSIVE'
    exit_code = (1 if state['classification'] == 'TECHNICAL_INCONCLUSIVE' else
                 0 if state['classification'] == 'HASH_MATCH_VERIFIED' else 2)
    state['expected_process_exit'] = exit_code
    save(output / 'terminal.json', state)
    save(output / 'runner_output_inventory.json', [
        {'path': str(p.relative_to(output)), 'bytes': p.stat().st_size, 'sha256': digest(p)}
        for p in sorted(output.rglob('*')) if p.is_file()])
    return exit_code


def scientific_diagnosis(output, contract):
    # Same namespace isolation as C1-1; existing selected module files unchanged.
    for name, folder in [('trotterlib', ROOT / 'src/trotterlib'),
                         ('trotterlib.df_trotter', ROOT / 'src/trotterlib/df_trotter')]:
        module = types.ModuleType(name)
        module.__path__ = [str(folder)]
        module.__package__ = name
        sys.modules[name] = module
    import numpy as np
    from trotterlib.df_hamiltonian import DFHamiltonian
    from trotterlib.df_partial_randomized_pf import df_hamiltonian_hash
    from trotterlib.df_partial_s2 import _partition_hash
    from trotterlib.pr2_s0_s1_validation import generation_partition, _array_hash, WEIGHT_RULE

    tracked = {'generation_partition', 'generation_ranked_fragments', 'split_df_hamiltonian_by_ld',
               'df_fragment_weight', 'df_hamiltonian_hash', '_partition_hash'}
    forbidden = {'prepare_df_partial_s2', 'build_df_blocks', 'build_df_blocks_givens',
                 'build_one_body_gaussian_block', 'U_to_qiskit_ops_jw', 'U_to_qiskit_ops_jw_givens',
                 '_dense_df_operator_qiskit', 'dense_extracted_df_tail', 'solve_df_ground_state',
                 'build_df_h_d_from_molecule', 'build_df_trotter_circuit'}
    calls = {name: 0 for name in sorted(tracked | forbidden)}

    def observe(frame, event, arg):
        if event == 'call' and frame.f_code.co_filename.startswith(str(ROOT / 'src/trotterlib')):
            name = frame.f_code.co_name
            if name in calls:
                calls[name] += 1
            if name in forbidden:
                raise RuntimeError('prohibited scientific call: ' + name)

    sys.setprofile(observe)
    try:
        reference = json.loads((ROOT / contract['reference_result']).read_text())
        method = reference['stages']['V2']['ranks'][0]['methods']['B2-G']
        require(method['partition_hash'] == contract['expected_partition_hash'], 'reference hash changed')
        emit('SNAPSHOT_OPEN')
        with np.load(ROOT / contract['snapshot']['path'], allow_pickle=False) as archive:
            required = set(contract['npz_read_members'])
            require(required <= set(archive.files), 'required NPZ members missing')
            arrays = {key: np.array(archive[key], copy=True) for key in contract['npz_read_members']}
        metadata = json.loads(str(arrays['metadata_json'].item()))
        h = DFHamiltonian(constant=float(arrays['constant']), one_body=arrays['one_body'],
                          lambdas=arrays['lambdas'], g_matrices=tuple(arrays['g_matrices']),
                          metadata=dict(metadata['hamiltonian_metadata']))
        hhash = df_hamiltonian_hash(h, weight_rule=WEIGHT_RULE)
        actual_identity = {'constant_hex': h.constant.hex(), 'one_body_hash': _array_hash(h.one_body),
            'lambdas_hash': _array_hash(h.lambdas), 'g_matrix_hashes': [_array_hash(g) for g in h.g_matrices],
            'hamiltonian_hash': hhash}
        expected = reference['stages']['V1']['development']['digests']
        comparisons = {key: value == expected[key] for key, value in actual_identity.items()}
        save(output / 'input_identity.json', {'snapshot_sha256': digest(ROOT / contract['snapshot']['path']),
             'members_read': {key: {'shape': list(a.shape), 'dtype': str(a.dtype)} for key, a in arrays.items()},
             'actual': actual_identity, 'expected': {key: expected[key] for key in actual_identity},
             'equal': comparisons, 'state_arrays_read': False})
        require(all(comparisons.values()), 'input/Hamiltonian identity mismatch')

        partition = generation_partition(h, 3)
        require(partition.weight_rule == WEIGHT_RULE, 'Hamiltonian hash weight-rule argument mismatch')
        # Evaluate exactly the payload expression in the sealed helper, then
        # cross-check its bytes/hash against a call to the unmodified helper.
        source_path = ROOT / 'src/trotterlib/df_partial_s2.py'
        tree = ast.parse(source_path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_partition_hash')
        assignment = function.body[0]
        require(isinstance(assignment, ast.Assign) and assignment.targets[0].id == 'payload',
                'unexpected fixed payload expression')
        expression = compile(ast.Expression(assignment.value), str(source_path), 'eval')
        payload = eval(expression, {'asdict': dataclasses.asdict, 'hamiltonian_hash': hhash,
            'partition': partition, 'partition_policy': 'explicit_ordered_partition'})
        encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
        actual_hash = _partition_hash(hhash, partition, partition_policy='explicit_ordered_partition')
        with (output / 'partition_payload.canonical.json').open('xb') as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        save(output / 'partition_payload.json', payload)
        float_rows = []
        for group in ['deterministic_fragments', 'randomized_fragments']:
            for i, fragment in enumerate(payload[group]):
                for key in ['lam', 'weight']:
                    provenance = ('stored lambdas[original_index] converted to Python float' if key == 'lam' else
                                  'df_fragment_weight: abs(lam) * norm(g_matrix, fro)**2; no rounding before partition hash')
                    float_rows.append({'path': f'/{group}/{i}/{key}',
                        **float_record(fragment[key], provenance)})
        float_rows.append({'path': '/ranking_proxy_lambda_r', **float_record(partition.ranking_proxy_lambda_r,
            'split_df_hamiltonian_by_ld: float(sum(fragment.weight for fragment in randomized)), source order')})
        save(output / 'float_provenance.json', float_rows)
        identity = {'expected_hash': contract['expected_partition_hash'], 'actual_hash': actual_hash,
            'hamiltonian_hash': hhash, 'algorithm': 'SHA256', 'canonical_bytes': len(encoded),
            'canonical_payload_sha256': hashlib.sha256(encoded).hexdigest(),
            'serialization': 'json.dumps(sort_keys=True, separators=(comma,colon)); default ensure_ascii=True; UTF-8; no trailing newline',
            'source_reference_commit': contract['scientific_reference_commit'],
            'source_path': str(source_path.relative_to(ROOT)), 'source_sha256': digest(source_path),
            'basis_reconstruction_called': False, 'payload_source': 'unmodified helper first assignment AST expression'}
        save(output / 'hash_identity.json', identity)
        # Actual payload, float evidence and identities are durable before any
        # assertion/comparison against the expected partition hash.
        require(actual_hash == identity['canonical_payload_sha256'], 'source helper/payload bytes disagree')
        require(list(partition.deterministic_block_indices) == method['ordered_indices'] and
                list(partition.randomized_block_indices) == method['randomized_indices'], 'partition order mismatch')
        fields = []

        def compare(path, actual, saved, saved_path, kind):
            is_float = isinstance(actual, float) and isinstance(saved, (float, int))
            equal = (float(actual).hex() == float(saved).hex()) if is_float else actual == saved
            row = {'actual_path': path, 'actual': actual, 'saved': saved,
                   'saved_pointer': saved_path, 'comparison_kind': kind, 'equal': equal}
            if is_float:
                row.update({'actual_hex': float(actual).hex(), 'saved_hex': float(saved).hex(),
                            'actual_repr': repr(actual), 'saved_repr': repr(saved)})
            fields.append(row)

        compare('/hamiltonian_hash', hhash, expected['hamiltonian_hash'], '/stages/V1/development/digests/hamiltonian_hash', 'direct saved digest')
        compare('/ld', partition.ld, reference['stages']['V2']['ranks'][0]['rank'], '/stages/V2/ranks/0/rank', 'saved rank')
        compare('/deterministic_indices', list(partition.deterministic_block_indices), method['ordered_indices'],
                '/stages/V2/ranks/0/methods/B2-G/ordered_indices', 'saved order')
        compare('/randomized_indices', list(partition.randomized_block_indices), method['randomized_indices'],
                '/stages/V2/ranks/0/methods/B2-G/randomized_indices', 'saved order')
        compare('/partition_policy', 'explicit_ordered_partition', method['partition_policy'],
                '/stages/V2/ranks/0/methods/B2-G/partition_policy', 'saved policy; policy string differs from hash-policy field by definition')
        by_index = {f['original_index']: (group, i, f) for group in ['deterministic_fragments', 'randomized_fragments']
                    for i, f in enumerate(payload[group])}
        # Existing other rank records are metadata witnesses for the SAME
        # snapshot, not new rank computations or a parameter search.
        for rank_i, rank_row in enumerate(reference['stages']['V2']['ranks']):
            for label in ['B2-G', 'B2-W']:
                for saved_i, fragment in enumerate(rank_row['methods'][label]['deterministic_fragments']):
                    group, i, actual = by_index[fragment['original_index']]
                    for key, saved_key in [('original_index', 'original_index'), ('lam', 'lambda'), ('weight', 'weight')]:
                        compare(f'/{group}/{i}/{key}', actual[key], fragment[saved_key],
                            f'/stages/V2/ranks/{rank_i}/methods/{label}/deterministic_fragments/{saved_i}/{saved_key}',
                            'saved _fragment_record metadata; original hash payload itself not serialized')
        differences = [r for r in fields if not r['equal']]
        save(output / 'field_comparison.json', {'comparisons': fields, 'differing_records': differences,
            'original_complete_hash_payload_available': False,
            'missing_original_payload_fields': ['original exact JSON bytes', 'direct partition rank/weight_rule fields',
                'fragment 9..11 weights', 'ranking_proxy_lambda_r', 'partition_hash_policy literal'],
            'restriction': 'no guessed replacement fields, alternative serialization, hash search, tolerance or runtime/version exploration'})
        matched = actual_hash == contract['expected_partition_hash']
        classification = ('HASH_MATCH_VERIFIED' if matched else
                          'HASH_MISMATCH_CAUSE_PARTIALLY_IDENTIFIED' if differences else 'HASH_MISMATCH_UNRESOLVED')
        outcome = {'classification': classification, 'diagnosis_complete': True, 'expected_hash': contract['expected_partition_hash'],
            'actual_hash': actual_hash, 'hash_match': matched, 'differing_saved_field_record_count': len(differences),
            'cause_scope': ('specific saved field differences are evidence; absent original full payload prevents unique causal attribution'
                            if differences else 'no differing available fields; missing original full payload limits attribution'),
            'original_C1_1_actual_hash_recovered': False,
            'original_C1_1_status_changed': False, 'C1_1_resumed': False, 'basis_or_exact_model_verified': False}
        save(output / 'diagnosis.json', outcome)
        save(output / 'numpy_build_metadata.json', {'numpy_version': np.__version__,
            'build': getattr(np.__config__, 'CONFIG', {}), 'meaning': 'this runtime only; original binary/BLAS identity not certified'})
        emit('HASH_DIAGNOSIS_SAVED', outcome)
        return outcome
    finally:
        sys.setprofile(None)
        save(output / 'scientific_call_observations.json', {'profile_scope': 'listed Python function names in selected project source, after imports',
             'counts': calls, 'np_load_calls_in_runner': 1, 'not_a_global_C_extension_profiler': True})


def main():
    initially_clean = not git('status', '--porcelain')
    manifest = json.loads((HERE / 'source_manifest.json').read_text())

    def action(output):
        contract = json.loads((HERE / 'execution_contract.json').read_text())
        auth = json.loads((HERE / 'authorization_receipt.json').read_text())
        require(initially_clean, 'worktree dirty at admission')
        for key, filename in [('contract_sha256', 'execution_contract.json'),
                              ('source_manifest_sha256', 'source_manifest.json'), ('runner_sha256', 'diagnose_partition.py')]:
            require(auth[key] == digest(HERE / filename), 'authorization identity mismatch: ' + filename)
        require(git('rev-parse', 'HEAD^') == auth['execution_source_commit'], 'authorization parent mismatch')
        require(git('diff-tree', '--no-commit-id', '--name-only', '-r', 'HEAD') ==
                str((HERE / 'authorization_receipt.json').relative_to(ROOT)), 'authorization commit changes unexpected files')
        verify_files(manifest)
        require(environment() == json.loads((HERE / 'environment_identity.json').read_text()), 'runtime changed')
        save(output / 'run_identity.json', {'run_id': contract['run_id'], 'execution_source_commit': auth['execution_source_commit'],
             'authorization_commit': git('rev-parse', 'HEAD'), 'authorization_sha256': digest(HERE / 'authorization_receipt.json'),
             'scientific_reference_commit': contract['scientific_reference_commit'], 'old_authorization_reused': False})
        return scientific_diagnosis(output, contract)

    return run_guarded(action, HERE / 'result', finalizer=lambda: verify_files(manifest))


if __name__ == '__main__':
    raise SystemExit(main())
