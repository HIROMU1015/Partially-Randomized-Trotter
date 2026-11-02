"""Stdlib-only static/saved-data inspection, never imports scientific modules.

ASTs are parsed but not executed. No frame generation, matrix/circuit evaluation,
compile, synthesis, solver, sampling, GPU query, or one-shot launch occurs.
All writes are new audit records outside the protected evidence directories.
"""
from pathlib import Path
import ast
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = 'ce0e286430a2a23869cb9417c82113c98064c72c'
SOURCE = '41349b73157c7781e0b534cf0fe61e06bf439954'
AUTH = 'a1f537b68f0f22419dff0202c553b7511653caa3'
OLD_AUDIT_SOURCE = 'b8c4c446d3b913fa5c85b568dcac2051f4054c1e'
RESULT = 'artifacts/track_b_df_correction_comparison_result/2026-10-11/v2'
ART = 'artifacts/track_b_df_correction_comparison_source_v2/2026-10-11'
OLD = 'scripts/tracks/algorithm_codesign/audit_df_sparse_correction.py'
OLD_E = 'scripts/tracks/algorithm_codesign/audit_df_sparse_correction_native.py'
COMP = 'src/trottertracks/algorithm_codesign/df_correction_comparison/comparison.py'
CERT = 'src/trottertracks/algorithm_codesign/df_correction_comparison/certification.py'
NATIVE = 'src/trottertracks/algorithm_codesign/df_correction_comparison/native_evaluation.py'
COST = 'src/trotterlib/rte_compiled_cost.py'
WORKER = 'src/trottertracks/algorithm_codesign/df_correction_comparison/worker.py'
CHECKS = []


def check(name, condition):
    CHECKS.append({'check': name, 'passed': bool(condition)})
    if not condition:
        raise AssertionError(name)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def identity(p):
    p = Path(p); b = p.read_bytes()
    return {'path': str(p.relative_to(ROOT)), 'bytes': len(b),
            'sha256': hashlib.sha256(b).hexdigest()}


def load(p):
    return json.loads((ROOT / p).read_bytes())


def write(name, obj):
    with (OUT / name).open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False); f.write('\n')
    check('new review-sized file ' + name, (OUT / name).stat().st_size <= 512 * 1024)


def node(path, function, cls=None):
    tree = ast.parse((ROOT / path).read_text())
    scope = tree.body
    if cls:
        scope = next(n for n in scope if isinstance(n, ast.ClassDef) and n.name == cls).body
    return next(n for n in scope if isinstance(n, ast.FunctionDef) and n.name == function)


def excerpt(path, n):
    text = (ROOT / path).read_text().splitlines()
    return {'path': path, 'line_start': n.lineno, 'line_end': n.end_lineno,
            'text': '\n'.join(text[n.lineno - 1:n.end_lineno]),
            'AST': ast.dump(n, include_attributes=False)}


def assigned(path, function, variable):
    return next(n for n in ast.walk(node(path, function)) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == variable for t in n.targets))


def main():
    check('audit base commit', git('rev-parse', 'HEAD').decode().strip() == BASE)
    check('execution source direct authorization parent', git('rev-list', '--parents', '-n', '1', AUTH).decode().split() == [AUTH, SOURCE])
    protected = load('artifacts/track_b_df_correction_comparison_integrity_v3/2026-10-11/source_manifest_v3.json')
    for row in protected['files']:
        check('protected source bytes ' + row['path'], identity(ROOT / row['path']) == row)
    frozen_result_inventory = load('artifacts/track_b_df_correction_comparison_execution_audit/2026-10-12/saved_output_audit_v1.json')['result_file_inventory']
    for row in frozen_result_inventory:
        check('old result identity ' + row['path'], identity(ROOT / row['path']) == row)
    old_record = load('artifacts/track_b_df_sparse_correction_audit/2026-10-11/phase_E_checks.json')
    check('old Phase E script record identity', identity(ROOT / OLD_E)['sha256'] == old_record['source_sha256'])
    for p in (OLD, OLD_E, COST):
        check('old helper/policy unchanged since prior audit ' + p, git('show', OLD_AUDIT_SOURCE + ':' + p) == (ROOT / p).read_bytes())
    for p in (OLD, OLD_E, COST, COMP, CERT, NATIVE, WORKER):
        check('fixed source bytes unchanged ' + p, git('show', SOURCE + ':' + p) == (ROOT / p).read_bytes())
    before = [identity(ROOT / r['path']) for r in frozen_result_inventory]
    before += [identity(ROOT / (ART + '/authorization.json')), identity(ROOT / (ART + '/contract_v2.json'))]
    old_random, new_random = node(OLD, 'random_unitary'), node(COMP, 'random_unitary')
    # The old function has a local NumPy import; normalize only that import.
    old_body = [n for n in old_random.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    new_body = [n for n in new_random.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    old_random_ast = ast.dump(ast.Module(body=old_body, type_ignores=[]), include_attributes=False)
    new_random_ast = ast.dump(ast.Module(body=new_body, type_ignores=[]), include_attributes=False)
    check('QR random-unitary expression AST identical', old_random_ast == new_random_ast)
    # Compare multiplication parenthesization without computing matrices.
    old_primitive = node(OLD, 'primitive')
    old_update = next(n for n in ast.walk(old_primitive) if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'result' for t in n.targets)
        and isinstance(n.value, ast.BinOp) and isinstance(n.value.op, ast.MatMult))
    product = old_update.value.right.right
    check('old scalar multiplication before matmul AST', isinstance(product, ast.BinOp) and
          isinstance(product.op, ast.MatMult) and isinstance(product.left, ast.BinOp) and
          isinstance(product.left.op, ast.Mult) and product.left.left.value == 2)
    new_occupied = assigned(CERT, 'primitive', 'occupied')
    check('new mul then scale AST', isinstance(new_occupied.value, ast.Call) and
          new_occupied.value.func.attr == 'scale' and new_occupied.value.args[0].value == 2 and
          new_occupied.value.func.value.func.attr == 'mul')
    # Compare expm input expression after substituting t.value -> t, no evaluation.
    old_u = assigned(OLD_E, 'main', 'u')
    new_u = assigned(NATIVE, 'native_fixture', 'raw_u')
    class StripBallValue(ast.NodeTransformer):
        def visit_Attribute(self, n):
            if isinstance(n.value, ast.Name) and n.value.id == 't' and n.attr == 'value':
                return ast.Name(id='t', ctx=ast.Load())
            return self.generic_visit(n)
    new_normal = StripBallValue().visit(ast.parse(ast.unparse(new_u.value), mode='eval')).body
    check('expm time/sign/order expression identical modulo t.value',
          ast.dump(old_u.value, include_attributes=False) == ast.dump(new_normal, include_attributes=False))
    old_compiler = assigned(OLD_E, 'main', 'compiler')
    new_compiler = assigned(NATIVE, 'native_fixture', 'compiler')
    check('CompilerSettings constructor AST identical', ast.dump(old_compiler.value, include_attributes=False) == ast.dump(new_compiler.value, include_attributes=False))
    payload = {**old_record['compiler'], 'compiler_hash_policy': 'qiskit_compiler_settings_v1', 'connectivity': 'all_to_all_logical'}
    settings_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    check('source-derived compiler hash equals all prior rows', all(r['compiler_settings_hash'] == settings_hash for r in old_record['central_compilations']))
    # Source identifies INPUT fingerprint calculation before transpile.
    cost_node = node(COST, 'transpile_and_measure_cost')
    fp_call = next(n for n in ast.walk(cost_node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'canonical_qiskit_circuit_fingerprint')
    transpile_call = next(n for n in ast.walk(cost_node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'transpile')
    check('fingerprint is computed on input before transpile', fp_call.lineno < transpile_call.lineno and isinstance(fp_call.args[0], ast.Name) and fp_call.args[0].id == 'circuit')
    check('fingerprint nullable fallback present', any(isinstance(n, ast.Assign) and
        isinstance(n.value, ast.Constant) and n.value.value is None and
        any(isinstance(t, ast.Name) and t.id == 'actual_circuit_fingerprint' for t in n.targets)
        for n in ast.walk(cost_node)))
    numeric_node = node(COST, '_numeric_parameter_payload')
    check('numeric parameters use raw ndarray SHA256', any(isinstance(n, ast.Constant) and n.value == 'array_sha256' for n in ast.walk(numeric_node)))
    guard = next(n for n in ast.walk(node(NATIVE, 'native_fixture')) if isinstance(n, ast.If) and
                 'compiled.actual_circuit_fingerprint' in ast.unparse(n.test))
    source_comparison = [
        {'component': 'seed / draw order', 'status': 'SOURCE_MATCH', 'old': '7101107; random_unitary(4), 3 draws r3, 3 draws r6', 'new': 'same explicit draws; source-level match only', 'old_source': OLD_E + ':76,110', 'new_source': COMP + ':29'},
        {'component': 'random_unitary QR phase convention', 'status': 'AST_IDENTICAL_EXCEPT_LOCAL_IMPORT', 'old_source': OLD + ':168', 'new_source': COMP + ':16'},
        {'component': 'raw frame values', 'status': 'OLD_VALUES_NOT_SAVED', 'new': 'exact float.hex serialized and identical between E-inputs and jobs; old recipe frame values unavailable'},
        {'component': 'frame buffer representation', 'status': 'STATIC_REPRESENTATION_DIFF_RUNTIME_EFFECT_UNPROVEN', 'old': 'random_unitary(r)[:,:2] slices', 'new': 'serialize hex; new np.array in deserialize_frames', 'effect': 'buffer ownership/strides can change; no observed stride records; no matrix constructed in audit'},
        {'component': 'primitive algebra / signs', 'status': 'EXACT_ALGEBRA_MATCH_WITH_FLOAT_DAG_DIFF', 'old': 'result @ (I - (2*a.conj().T) @ a)', 'new': 'result @ (I - float(2)*(a.conj().T @ a))', 'qualification': 'Multiplication by 2 is exact in the ordinary binary64 range. Parenthesization difference alone does not prove unequal bytes or a bias.'},
        {'component': 'triple commutator', 'status': 'EXACT_ALGEBRA_MATCH', 'old': 'comm(Q0,comm(Q1,Q2)); signs (1,-1,1)', 'new': 'Ball comm with same AB-BA and same signs; added enclosures'},
        {'component': 'raw whole unitary', 'status': 'EXPM_EXPRESSION_MATCH', 'old': 'expm(1j*.03125*t/4)', 'new': 'expm(1j*.03125*t.value/4)', 'qualification': 'same expression does not prove same t bytes'},
        {'component': 'uncontrolled/controlled assembly', 'status': 'SOURCE_MATCH', 'old': 'False then True; block_diag(I,raw_u) for True', 'new': 'same; UnitaryGate(target) and QuantumCircuit(r+int(controlled))'},
        {'component': 'compiler settings / versions', 'status': 'CONSTRUCTOR_AST_MATCH_AND_REPORTED_VERSIONS_MATCH', 'source_derived_settings_hash': settings_hash, 'qualification': 'old package/binary trees were not recorded; full historical runtime identity cannot be proven'},
        {'component': 'fingerprint target / policy', 'status': 'CONFIRMED_PRETRANSPILE_INPUT_RECURSIVE_RAW_BYTES', 'policy': 'qiskit_recursive_numeric_circuit_v2', 'qualification': 'includes exact input ndarray bytes and recursive definitions; not a norm equivalence or native count test'},
        {'component': 'selected prior row', 'status': 'MATCHED_BY_SOURCE_AND_PREFIX', 'row': {'active_modes': 3, 'controlled': False}, 'expected': old_record['central_compilations'][0]['circuit_fingerprint']},
    ]
    excerpts = [excerpt(OLD, old_primitive), excerpt(CERT, node(CERT, 'primitive')),
        excerpt(CERT, node(CERT, 'mul', 'Ball')), excerpt(CERT, node(CERT, 'scale', 'Ball')),
        excerpt(OLD, node(OLD, 'comm')), excerpt(CERT, node(CERT, 'comm')),
        excerpt(COMP, node(COMP, 'e_recipe')), excerpt(NATIVE, node(NATIVE, 'serialize_frames')),
        excerpt(NATIVE, node(NATIVE, 'deserialize_frames')), excerpt(OLD_E, old_u),
        excerpt(NATIVE, new_u), excerpt(OLD_E, old_compiler), excerpt(NATIVE, new_compiler),
        excerpt(COST, node(COST, '_numeric_parameter_payload')),
        excerpt(COST, node(COST, '_operation_payload')), excerpt(COST, cost_node), excerpt(NATIVE, guard),
        excerpt('tests/tracks/algorithm_codesign/test_df_correction_comparison_v2.py',
                node('tests/tracks/algorithm_codesign/test_df_correction_comparison_v2.py', 'test_off_domain_float_assembly_is_bit_identical_to_frozen_recipe_helper'))]
    write('source_comparison_v1.json', {'kind': 'STATIC_SOURCE_EQUIVALENCE_AND_DIFFERENCE_AUDIT',
        'base_result_commit': BASE, 'frozen_source': SOURCE, 'authorization': AUTH,
        'comparisons': source_comparison, 'source_excerpts_and_AST': excerpts,
        'scientific_modules_imported': False, 'new_science_calls': 0,
        'old_selected_bit_identity_test_scope': 'one 2-mode rank2 frame and one selected commutator; passes were saved; not rerun; not a full E fingerprint guarantee'})
    frames = load(RESULT + '/E-inputs.json')
    frame_records = []
    for r in ('3', '6'):
        job = load(RESULT + '/E-r' + r + '.job.json')
        check('saved job frames equal E-inputs r' + r, job['frames'] == frames[r])
        digest, complex_entries = hashlib.sha256(), 0
        for frame in frames[r]:
            check('saved frame shape r' + r, len(frame) == int(r) and all(len(row) == 2 for row in frame))
            for row in frame:
                for re, im in row:
                    check('hex round trip r' + r, float.fromhex(re).hex() == re and float.fromhex(im).hex() == im)
                    digest.update((re + '\0' + im + '\0').encode()); complex_entries += 1
        frame_records.append({'r': int(r), 'frames': len(frames[r]), 'shape_per_frame': [int(r), 2],
            'complex_entries': complex_entries, 'ordered_hex_components_sha256': digest.hexdigest(),
            'job_frames_exactly_equal': True, 'hex_to_hex_lossless': True,
            'old_phase_E_frame_bytes_available': False, 'no_array_or_matrix_constructed': True})
    prefix = load(RESULT + '/E-r3.json.prefix.json')
    check('failed prefix before accepted native output', prefix['partial_not_scientific_outcome'] and
          prefix['prefix']['compile_calls'] == 1 and prefix['prefix']['native'] == [])
    status = load(RESULT + '/terminal_status.json')
    check('old classification and STOP retained', status['classification'] == 'TECHNICAL_INCONCLUSIVE' and status['mandatory_STOP'] and status['retries'] == 0)
    write('saved_input_and_failure_metadata_v1.json', {
        'kind': 'SAVED_INPUT_AND_FAILURE_METADATA_ONLY', 'frames': frame_records,
        'classification_unchanged': status['classification'], 'technical_reason': status['technical_reason'],
        'observed_guard': 'ArithmeticError: Frozen E whole fingerprint mismatch; no synthesis',
        'observed_guard_source': NATIVE + ':' + str(guard.lineno),
        'actual_fingerprint': None, 'actual_fingerprint_status': 'NOT_SAVED; cannot distinguish non-null digest mismatch from None fallback',
        'input_unitary_bytes': 'NOT_SAVED', 'failed_native_IR': 'NOT_SAVED',
        'fingerprint_creation_exception': 'NOT_SAVED; TypeError/ValueError may be converted to None in cost helper',
        'old_phase_E_raw_frames': 'NOT_SAVED', 'old_phase_E_raw_generators': 'NOT_SAVED',
        'E_r3_saved_compile_attempts': 1, 'E_r6_compile_attempts': None, 'T_calls': 0,
        'exact_failure_cause_classification': 'INPUT_IDENTITY_GUARD_FAILURE_WITH_CAUSAL_ALLOCATION_UNRESOLVED',
        'additional_matrix_compile_synthesis_calls': 0, 'mandatory_STOP': True})
    inputs = [OLD, OLD_E, COMP, CERT, NATIVE, COST, WORKER,
        'tests/tracks/algorithm_codesign/test_df_correction_comparison_v2.py',
        'artifacts/track_b_df_sparse_correction_audit/2026-10-11/phase_E_checks.json',
        ART + '/contract_v2.json', ART + '/authorization.json', ART + '/runtime_identity_v2.json',
        RESULT + '/E-inputs.json', RESULT + '/E-r3.job.json', RESULT + '/E-r6.job.json',
        RESULT + '/E-r3.json.prefix.json', RESULT + '/E-r3.log', RESULT + '/terminal_status.json',
        RESULT + '/one_shot_marker.json',
        'artifacts/track_b_df_correction_comparison_execution_audit/2026-10-12/prelaunch_gate_receipt.json',
        'artifacts/track_b_df_correction_comparison_execution_audit/2026-10-12/outer_process_receipt.json']
    check('protected result/auth/contract identities unchanged through audit', before ==
          [identity(ROOT / r['path']) for r in frozen_result_inventory] +
          [identity(ROOT / (ART + '/authorization.json')), identity(ROOT / (ART + '/contract_v2.json'))])
    write('input_provenance_and_checks_v1.json', {
        'kind': 'STATIC_READ_ONLY_AUDIT_PROVENANCE', 'base_result_commit': BASE, 'frozen_source': SOURCE,
        'authorization': AUTH, 'old_audit_source_commit': OLD_AUDIT_SOURCE,
        'input_files': [identity(ROOT / p) for p in inputs], 'protected_source_files_verified': len(protected['files']),
        'result_files_verified': len(frozen_result_inventory), 'checks': CHECKS.copy(), 'all_checks_passed': True,
        'source_contract_auth_marker_result_changes': 0, 'science_calls': 0,
        'matrix_circuit_compile_synthesis_sampling_LP_GPU_calls': 0, 'retries': 0, 'mandatory_STOP': True})
    print(json.dumps({'static_checks_passed': len(CHECKS), 'new_science_calls': 0,
        'protected_source_files': len(protected['files']), 'old_result_files': len(frozen_result_inventory),
        'classification_unchanged': status['classification'], 'mandatory_STOP': True}))


if __name__ == '__main__':
    main()
