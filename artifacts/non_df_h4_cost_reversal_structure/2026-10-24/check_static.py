"""Metadata/SHA audit and artificial checks only. Never imports scientific runners."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = '1caaba2516dd3a30475e2acbd69672810a9fc05a'
OLD = '0c329b30a81accb35c212be8defa66fee0dd0f12'
PILOT = 'artifacts/non_df_h4_pilot/2026-10-23/run01'
C = 'artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01'
REUSED = 'artifacts/non_df_h4_tail_rho_investigation/2026-10-24/preservation_before.json'
SOURCES = ['pauli_signal_adapter.py', 'h4_molecular_inputs.py', 'non_df_h4_readiness.py',
           'fixed_signal_evaluation.py', 'h4_audited_evaluation.py', 'h4_pilot_science.py',
           'rte_event_audit.py', 'rte.py', 'df_partial_randomized_pf.py', 'df_partial_s2.py',
           'df_partial_s2_repeated.py', 'df_rte_qiskit.py', 'df_trotter/ops.py',
           'rte_compiled_cost.py', 'native_df_gate_model.py']

def read(path):
    return json.loads((ROOT / path).read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def git_bytes(ref, path):
    return subprocess.check_output(['git', 'show', ref + ':' + path], cwd=ROOT)

def functions(path):
    tree = ast.parse(path.read_text())
    wanted = {'PauliSignalAdapter', 'append_pauli_evolution', 'append_pauli_event',
              'trajectory', 'resolve_prefixes', 'build_h4_inputs', 'structural_prefix',
              'DFSignalAdapter', 'default_compiler', 'hadamard_wrapper', '_statistics',
              'evaluate_h4_audited', 'event_gates', 'audited_adapter',
              'AuditedPauliSignalAdapter', 'AuditedDFSignalAdapter',
              '_make_event', '_paired_order_weight', 'finite_rte_distribution',
              'df_fragment_weight', 'rank_df_fragments', 'split_df_hamiltonian_by_ld',
              'prepare_df_partial_s2', '_append_block', '_append_step_phases',
              '_build_parts', 'build_step', 'build_additive_circuits',
              'estimate_df_partial_s2_structural_size_upper_bound',
              'make_df_partial_s2_repeated_request', '_build_raw', 'build',
              'estimate_df_rte_structural_size_upper_bound', '_build',
              '_append_central_application', 'U_to_qiskit_ops_jw',
              'one_body_diagonal_primitives', 'df_squared_diagonal_primitives',
              'append_diagonal_primitives', 'transpile_and_measure_cost',
              'get_or_transpile', 'lower_gate', 'assemble', 'count'}
    return [{'symbol': n.name, 'line': n.lineno, 'end_line': n.end_lineno}
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]

def audit():
    # Reuse the previously sealed preservation map instead of regenerating science.
    preserved = read(REUSED)
    for path, expected in preserved.items():
        assert sha(ROOT / path) == expected, path
    receipt = read('artifacts/non_df_h4_cost_reversal_structure/2026-10-24/github_verification.json')
    for row in receipt['files']:
        data = (ROOT / row['path']).read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        assert blob == row['git_blob_sha'], row['path']
        assert data == git_bytes(row['ref'], row['path']), row['path']
    source_map = []
    for name in SOURCES:
        path = 'src/trotterlib/' + name
        # Existing generic model is a reused asset, not original H4 production.
        ref = BASE if name == 'native_df_gate_model.py' else OLD
        assert (ROOT / path).read_bytes() == git_bytes(ref, path), path
        source_map.append(dict(path=path, ref=ref, sha256=sha(ROOT / path),
                               symbols=functions(ROOT / path)))
    runtime_suffixes = ['circuit/gate.py', 'circuit/add_control.py',
                        'circuit/library/pauli_evolution.py',
                        'synthesis/evolution/product_formula.py',
                        'synthesis/evolution/lie_trotter.py',
                        'transpiler/preset_passmanagers/level1.py',
                        'transpiler/preset_passmanagers/builtin_plugins.py']
    seal = read(C + '/seal.json')
    runtime = []
    for suffix in runtime_suffixes:
        matches = [p for p in seal['legacy_runtime_files'] if p.endswith('/qiskit/' + suffix)]
        assert len(matches) == 1, suffix
        p = matches[0]
        assert sha(Path(p)) == seal['legacy_runtime_files'][p], p
        runtime.append(dict(path=p, sha256=sha(Path(p))))
    pr = read(PILOT + '/run/pauli_jw_PR/result.json')
    inp = read(PILOT + '/input_run/input_generation/data/receipt.json')
    labels = pr['model']['retained_labels']
    assert labels == inp['prefixes']['pauli_jw']['retained_labels']
    pauli_input = read(PILOT + '/input_run/input_generation/data/pauli.json')
    assert labels == [term[0] for term in pauli_input['terms'][:27]]
    # Token/descriptor properties, never averages, coefficient sums or H4 arithmetic.
    assert len(labels) == pr['model']['L_D'] == 27
    assert all(set(s) <= set('IZ') and s.count('Z') in (1, 2) for s in labels)
    pf = read(PILOT + '/run/pauli_jw_PF/result.json')
    assert any(set(s) & set('XY') for s in pf['model']['retained_labels'])
    assert pr['state']['hash'] == pf['state']['hash'] == inp['state_hash']
    saved_prices = read('artifacts/non_df_h4_conditional_cng_results/2026-10-24/run01/cost_summary.json')
    for case in ('pauli_jw_PF', 'pauli_jw_PR', 'df_native_PF', 'df_native_PR'):
        result = read(PILOT + '/run/' + case + '/result.json')
        assert result['state']['hash'] == inp['state_hash']
        assert result['compiler'] == pr['compiler']
        # Equality of saved aggregates only: no recomputation from cost samples.
        assert saved_prices[case]['axes'] == result['circuit_cost']['axes']
    df = read(C + '/df_native_model.json')['cases']
    assert df['PR']['metadata']['deterministic_block_order'] == ['df-one-body-correction','df-fragment-0']
    assert df['PR']['blocks'] == df['PF']['blocks'][:2]
    basis_tags = {o['description']['name'] for b in df['PR']['blocks']
                  for o in b['basis']['operations']}
    assert 'xx_plus_yy' in basis_tags
    # No cost aggregation. Search schema keys/program filenames only, within four known cases.
    missing_keys = {'pretranspile_gate_counts', 'posttranspile_gate_counts',
                    'basis_switch_count', 'emitted_basis_change_count',
                    'cancelled_basis_change_pairs', 'pass_trace'}
    found = []
    def visit(obj, path):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in missing_keys:
                    found.append(dict(file=path, key=k))
                visit(v, path)
        elif isinstance(obj, list):
            for v in obj:
                visit(v, path)
    files = []
    programs = []
    for case in ('pauli_jw_PF', 'pauli_jw_PR', 'df_native_PF', 'df_native_PR'):
        directory = ROOT / PILOT / 'run' / case
        selected = [directory/n for n in ('result.json','validation.json','actual_call_accounting.json')]
        selected += sorted((directory/'trajectories').glob('*/price_join.json'))
        for p in selected:
            rel = str(p.relative_to(ROOT)); files.append(rel); visit(json.loads(p.read_text()), rel)
            if p.name == 'price_join.json':
                assert json.loads(p.read_text())['event_costs_additive'] is False
        programs += [str(p.relative_to(ROOT)) for p in directory.rglob('*')
                     if p.is_file() and p.suffix.lower() in ('.qpy','.qasm','.qasm3')]
    assert not found, found
    assert not programs, programs
    return dict(status='PASS_STATIC_ONLY', base_commit=BASE,
        reused_preservation_map=REUSED, preservation_map_sha256=sha(ROOT/REUSED),
        preserved_file_checks=len(preserved), remote_file_checks=len(receipt['files']),
        sources=source_map, runtime_seal=C+'/seal.json', runtime_files=runtime,
        pauli_prefix=dict(labels=labels, all_IZ=True, supports='Z or ZZ',
                          method='descriptor check; no coefficients or prices evaluated'),
        df_retained_basis_tags=sorted(basis_tags),
        bounded_inventory=dict(file_count=len(files),
                               sorted_filenames_sha256=hashlib.sha256('\n'.join(sorted(files)).encode()).hexdigest(),
                               patterns=['each original case: result.json, validation.json, actual_call_accounting.json',
                                         'each original case: trajectories/*/price_join.json',
                                         'each original case recursively: *.qpy, *.qasm, *.qasm3'],
                               searched_keys=sorted(missing_keys),
                               matching_keys=found, saved_programs=programs,
                               scope='selected receipts and price joins in original four case folders only'),
        compiled_component_attribution=None, actual_N_G=None,
        science_calls=0, saved_scientific_numeric_reanalysis=0, sampling=0, build=0, compile=0,
        original_native='UNKNOWN_UNCHANGED', C_L='COMBINATION_BLOCKED',
        compiler_certificate='MISSING_EVIDENCE', independent_review=False)

class PrefixRule(unittest.TestCase):
    def test_artificial_weight_prefix_is_not_item_fraction(self):
        path = ROOT / 'src/trotterlib/non_df_h4_readiness.py'
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'structural_prefix')
        env = {'math': math}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), env)
        self.assertEqual(env['structural_prefix']([9,1,1,1]), 1)
        self.assertEqual(env['structural_prefix']([3,3,2,2]), 2)

def artificial_tests():
    path = ROOT / 'tests/test_native_df_gate_model.py'
    spec = importlib.util.spec_from_file_location('reused_artificial_gate_checks', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.TestSuite([
        PrefixRule('test_artificial_weight_prefix_is_not_item_fraction'),
        module.MatrixChecks('test_basis_cancellation_preserves_noncommuting_order'),
        module.MatrixChecks('test_declared_lowering_all_gates_including_global_phase'),
        module.CountChecks('test_wrapper_axes')])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    return dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                reused_test_file=str(path.relative_to(ROOT)), scope='invented gates only; declared lowering is not Qiskit optimizer')

if __name__ == '__main__':
    result = audit()
    result['artificial'] = artificial_tests()
    result['checker_sha256'] = sha(Path(__file__))
    result['reused_tests_sha256'] = sha(ROOT/'tests/test_native_df_gate_model.py')
    (HERE/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','preserved_file_checks','remote_file_checks','artificial')},ensure_ascii=False))
