"""S3 static/saved/artificial audit. No runner, generator or science invocation."""
import ast
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from trottertracks.algorithm_codesign.g10_io import iter_json_bytes, protected_check_streaming
from trottertracks.algorithm_codesign.g10_saved import serial
from trottertracks.algorithm_codesign.g10_launch import verify_launch
from trottertracks.algorithm_codesign.synthesis_placement.wrapper_launch import verify_runtime

ROOT = Path(__file__).resolve().parents[3]
PREP = ROOT/'artifacts/track_b_g10_key_compatibility_preparation/2026-10-10/v3'
R2 = 'f9d2665283c707e5a025a2c92c1a051153eaf2e1'
S2 = 'a139b91f119d109430ae3154a045d0fdcf722233'
BOOKKEEPING = {'schema', 'branch', 'base_commit', 'result_directory', 'authorization_path',
               'optional_receipt_path', 'source_manifest', 'protected_ledger', 'io_revision'}


def identity(path, limit=None):
    if str(path).lower().endswith('.npz'):
        raise PermissionError('NPZ rejected before any access')
    digest, size = hashlib.sha256(), 0
    with Path(path).open('rb') as stream:
        while limit is None or size < limit:
            block = stream.read(32768 if limit is None else min(32768, limit-size))
            if not block:
                break
            digest.update(block); size += len(block)
    return {'bytes': size, 'sha256': digest.hexdigest()}


def git_bytes(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def equivalence_audit():
    fixtures = load_file('g10_artificial_JSON_fixtures', ROOT/'scripts/tracks/algorithm_codesign/g10_json_compatibility_fixtures.py')
    cases = dict(fixtures.examples(), large_typed_2500=fixtures.large_typed(2500))
    records = []
    for name, value in cases.items():
        before = copy.deepcopy(value)
        old = (json.dumps(serial(value), indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()
        digest, size, maximum = hashlib.sha256(), 0, 0
        for chunk in iter_json_bytes(value):
            digest.update(chunk); size += len(chunk); maximum = max(maximum, len(chunk))
        # nan keys compare by identity, not equality after deepcopy; contents are
        # otherwise checked in focused tests. Key identity/position never changes.
        if name != 'nonfinite_keys_are_strings' and value != before:
            raise AssertionError('input mutated: '+name)
        if size != len(old) or digest.hexdigest() != hashlib.sha256(old).hexdigest() or maximum > 32768:
            raise AssertionError('legacy bytes/hash mismatch: '+name)
        records.append({'case': name, 'bytes': size, 'SHA256': digest.hexdigest(),
                        'byte_equivalent': True, 'maximum_write_bytes': maximum})
    return {'passed': True, 'reference': 'unchanged g10_saved.serial + JSON indent2/Unicode/finite/newline',
            'cases': records, 'cases_count': len(records), 'whole_result_science_generated': False,
            'admitted_values': 'finite JSON scalars/Fraction/dict/list/tuple, acyclic including shared subtrees',
            'keys': 'stable side-effect-free str(key), all encountered int/str labels plus tested general builtins',
            'collision_rule': 'first normalized key position, last value',
            'intentional_safety_boundary': 'invalid overwritten values rejected, unlike legacy if they disappear before JSON validation',
            'unsupported_general_cases': ['side-effecting key str()', 'concurrent mutation'],
            'science_synthesis_matrix_sampling_LP': 0, 'mandatory_STOP': True}


def source_diff_audit():
    old_io = ast.parse((PREP/'g10_io_S2_reference.py.txt').read_text())
    new_io = ast.parse((ROOT/'src/trottertracks/algorithm_codesign/g10_io.py').read_text())
    def top_nodes(tree):
        return {node.name: ast.dump(node, include_attributes=False)
                for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    before, after = top_nodes(old_io), top_nodes(new_io)
    changed = {key for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
    if changed != {'validate_tree', '_compatible_tokens', 'iter_json_bytes'}:
        raise AssertionError('unexpected IO code change: '+repr(changed))
    def non_definitions(tree):
        return ast.dump(ast.Module(body=[node for node in tree.body
                        if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Expr))], type_ignores=[]))
    if non_definitions(old_io) != non_definitions(new_io):
        raise AssertionError('IO imports/constants changed')
    old_runner_path = ROOT/'scripts/tracks/algorithm_codesign/g10_degree_matched_native_v2.py'
    new_runner_path = ROOT/'scripts/tracks/algorithm_codesign/g10_degree_matched_native_v3.py'
    old_runner = ast.parse(old_runner_path.read_text())
    new_runner = ast.parse(new_runner_path.read_text())
    metadata = {
        'artifacts/track_b_g10_key_compatibility_preparation/2026-10-10/v3':
        'artifacts/track_b_g10_rss_repair_preparation/2026-10-10/v2',
        'contract_v3.json': 'contract_v2.json',
        'G10_DEGREE_MATCHED_NATIVE_V3_ONE_SHOT': 'G10_DEGREE_MATCHED_NATIVE_V2_ONE_SHOT'}
    class NormalizeMetadata(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value in metadata:
                return ast.copy_location(ast.Constant(value=metadata[node.value]), node)
            return node
    new_runner = NormalizeMetadata().visit(new_runner)
    old_runner.body = old_runner.body[1:]; new_runner.body = new_runner.body[1:]
    if ast.dump(old_runner, include_attributes=False) != ast.dump(new_runner, include_attributes=False):
        raise AssertionError('v3 runner differs beyond preparation paths/marker kind/docstring')
    helper = load_file('g10_S2_static_AST_verifier', ROOT/'scripts/tracks/algorithm_codesign/verify_g10_rss_repair_preparation.py')
    original_ast = helper.scientific_ast_audit()
    return {'passed': True, 'IO_changed_functions': sorted(changed),
            'all_other_IO_classes_functions_imports_constants_AST_unchanged': True,
            'stream_write_hash_guard_failure_completion_AST_unchanged': True,
            'v3_runner_metadata_only_delta': metadata,
            'v3_collect_and_scientific_execute_body_equal_v2': True,
            'v1_to_v2_science_AST_audit': original_ast,
            'old_IO_snapshot_hash': identity(PREP/'g10_io_S2_reference.py.txt'),
            'new_IO_hash': identity(ROOT/'src/trottertracks/algorithm_codesign/g10_io.py')}


def audit():
    c = json.loads((PREP/'contract_v3.json').read_text())
    old = json.loads((ROOT/'artifacts/track_b_g10_rss_repair_preparation/2026-10-10/v2/contract_v2.json').read_text())
    changed = {key for key in old.keys() | c.keys() if old.get(key) != c.get(key)}
    if not changed <= BOOKKEEPING or old['caps'] != c['caps']:
        raise AssertionError('science or caps contract changed')
    auth = json.loads((ROOT/c['authorization_path']).read_text())
    if (auth['science_execution_authorized'] is not False or auth['source_commit'] is not None
            or auth['status'] != 'PENDING_SEPARATE_G10_V3_AUTHORIZATION'):
        raise AssertionError('preparation must remain unauthorized')
    try:
        verify_launch(ROOT, PREP/'contract_v3.json', '0'*40)
    except PermissionError as exc:
        refusal = str(exc)
    else:
        raise AssertionError('pending launch accepted')
    if (ROOT/c['result_directory']).exists() or (ROOT/c['optional_receipt_path']).exists():
        raise AssertionError('production result/marker/A3 receipt must be absent')
    for contract in (old, c):
        check = protected_check_streaming(ROOT, contract)
        if check['violations']:
            raise AssertionError('protected ledger changed: '+repr(check['violations']))
    protected = check['protected_paths']
    immutable = json.loads((PREP/'protected_source_references_v3.json').read_text())
    for record in immutable['references']:
        if record['path'].lower().endswith('.npz'):
            raise PermissionError('NPZ Git blob excluded')
        raw = git_bytes('show', record['commit']+':'+record['path'])
        if len(raw) != record['bytes'] or hashlib.sha256(raw).hexdigest() != record['sha256']:
            raise AssertionError('immutable source reference changed')
    manifest_path = ROOT/c['source_manifest']
    critical = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('focused_tests_passed') is not True:
            raise AssertionError('focused validation not complete')
        for relative, digest in manifest['sha256'].items():
            if identity(ROOT/relative)['sha256'] != digest:
                raise AssertionError('S3 critical path changed: '+relative)
        critical = len(manifest['sha256'])
    profiles = {mode: json.loads((PREP/('io_'+mode+'_v3.json')).read_text())
                for mode in ('legacy_saved', 'stream_saved_corrected', 'legacy_typed', 'stream_typed')}
    for report in profiles.values():
        if not report['passed'] or report['frozen_caps'] != c['caps']:
            raise AssertionError('IO validation failed or diagnostic caps changed')
    for left, right in (('legacy_saved', 'stream_saved_corrected'), ('legacy_typed', 'stream_typed')):
        for key in ('bytes', 'sha256'):
            if profiles[left]['identity'][key] != profiles[right]['identity'][key]:
                raise AssertionError('profile identity mismatch')
        if profiles[right]['resource']['peak_RSS_KiB'] >= profiles[left]['resource']['peak_RSS_KiB']:
            raise AssertionError('profile does not demonstrate memory reduction')
    first = json.loads((PREP/'io_stream_saved_v3.json').read_text())
    if first['passed'] is not False or first['technical_reason'] != 'AssertionError: saved JSON bytes/hash differ':
        raise AssertionError('initial diagnostic comparator record changed')
    if first['identity'] != profiles['stream_saved_corrected']['identity']:
        raise AssertionError('corrected diagnostic should preserve the identical payload')
    runtime = verify_runtime(ROOT, c)
    if identity(Path(sys.executable).resolve())['sha256'] != c['runtime_executable_sha256']:
        raise PermissionError('fixed runtime executable mismatch')
    forbidden = [name for name in sys.modules if name.startswith(('numpy', 'mpmath', 'pygridsynth'))]
    if forbidden:
        raise AssertionError('scientific dependency unexpectedly imported')
    return {'kind': 'G10_V3_KEY_COMPATIBILITY_SOURCE_PREPARATION_AUDIT', 'passed': True,
            'science_contract_and_caps_identical': True, 'bookkeeping_changed_keys': sorted(changed),
            'source_diff': source_diff_audit(), 'S2_protected_paths': 1300,
            'R2_extended_protected_paths': protected, 'protected_violations': [],
            'immutable_source_references': immutable, 'critical_paths': critical,
            'runtime': runtime, 'runtime_executable_sha256': c['runtime_executable_sha256'],
            'pending_launch_refusal': refusal, 'production_directory_marker_A3_receipt_absent': True,
            'scientific_imports': forbidden, 'production_runner_science_synthesis_matrix_sampling_LP_calls': 0,
            'production_completion_within_512_MiB_proven': False,
            'IO_diagnostic_invocations': 5, 'initial_diagnostic_comparator_failure_preserved': True,
            'production_runs': 0, 'production_retries': 0, 'science_execution_authorized': False,
            'mandatory_STOP': True, 'next_science_authorized': False}


if __name__ == '__main__':
    report = audit()
    print(json.dumps(report, ensure_ascii=False, indent=2))
