"""Read-only S3 review-intake provenance. Stdlib only; no runner imports."""
import ast
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
S3 = 'b9ed01455351628c9073748f5ba5751aa794b789'
PREP = 'artifacts/track_b_g10_key_compatibility_preparation/2026-10-10/v3/'
INTAKE = 'artifacts/track_b_g10_s3_final_review_intake/2026-10-10/v1/'
APPENDED = {
    'PROJECT_MAP.md', 'docs/README.md', 'docs/research/研究概要・現状.md',
    'docs/research/研究ノート/README.md', 'docs/tracks/algorithm_codesign/README.md',
}


def safe(relative):
    # Reject before resolve/stat/open or Git blob access.
    path = Path(relative)
    if str(relative).lower().endswith('.npz'):
        raise PermissionError('NPZ access forbidden')
    if path.is_absolute() or '..' in path.parts:
        raise PermissionError('repository-relative path required')
    return ROOT/path


def identity(relative, limit=None):
    path = safe(relative)
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        while limit is None or size < limit:
            chunk = stream.read(32768 if limit is None else min(32768, limit-size))
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return {'bytes': size, 'sha256': digest.hexdigest()}


def git_blob(commit, relative):
    safe(relative)
    return subprocess.check_output(['git', 'show', commit+':'+relative], cwd=ROOT)


def load(relative):
    return json.loads(safe(relative).read_text())


def audit():
    receipt = load(INTAKE+'review_intake_v1.json')
    review = receipt['adopted_review']['path']
    actual_review = identity(review)
    expected_review = {k: receipt['adopted_review'][k] for k in ('bytes', 'sha256')}
    assert actual_review == expected_review
    text = safe(review).read_text()
    assert S3 in text and receipt['review_classification'] in text
    assert '本番実行認可はまだ与えない' in text

    manifest = load(PREP+'source_manifest_v3.json')
    assert manifest['focused_tests_passed'] is True
    critical_full = 0
    prefixes = {}
    for relative, digest in manifest['sha256'].items():
        if relative in APPENDED:
            frozen = git_blob(S3, relative)
            expected = {'bytes': len(frozen), 'sha256': hashlib.sha256(frozen).hexdigest()}
            assert expected['sha256'] == digest
            assert identity(relative, len(frozen)) == expected
            prefixes[relative] = {'frozen_S3_prefix': expected, 'current': identity(relative)}
        else:
            assert identity(relative)['sha256'] == digest, relative
            critical_full += 1
    assert len(manifest['sha256']) == 180 and len(prefixes) == 5

    contract = load(PREP+'contract_v3.json')
    ledger = load(contract['protected_ledger'])
    for relative, record in ledger.items():
        limit = record['bytes'] if relative in contract['append_only_paths'] else None
        assert identity(relative, limit) == record, relative
    assert len(ledger) == 1352

    inventory = load(PREP+'evidence_manifest_v3.json')
    for relative, record in inventory['files'].items():
        limit = record['bytes'] if relative in APPENDED else None
        assert identity(relative, limit) == record, relative

    refs = load(PREP+'protected_source_references_v3.json')['references']
    for record in refs:
        data = git_blob(record['commit'], record['path'])
        assert len(data) == record['bytes']
        assert hashlib.sha256(data).hexdigest() == record['sha256']

    # Parse bytes without importing the science sources or I/O runtime.
    old = ast.parse(safe(PREP+'g10_io_S2_reference.py.txt').read_text())
    new = ast.parse(safe('src/trottertracks/algorithm_codesign/g10_io.py').read_text())
    def definitions(tree):
        return {n.name: ast.dump(n, include_attributes=False) for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    before, after = definitions(old), definitions(new)
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    assert changed == ['_compatible_tokens', 'iter_json_bytes', 'validate_tree']
    old_runner = safe('scripts/tracks/algorithm_codesign/g10_degree_matched_native_v2.py').read_bytes()
    new_runner = safe('scripts/tracks/algorithm_codesign/g10_degree_matched_native_v3.py').read_bytes()
    normalized = new_runner
    for left, right in [
        (b'Future G10 v3.', b'Future G10 v2.'),
        (b'artifacts/track_b_g10_key_compatibility_preparation/2026-10-10/v3',
         b'artifacts/track_b_g10_rss_repair_preparation/2026-10-10/v2'),
        (b'contract_v3.json', b'contract_v2.json'),
        (b'G10_DEGREE_MATCHED_NATIVE_V3_ONE_SHOT', b'G10_DEGREE_MATCHED_NATIVE_V2_ONE_SHOT'),
    ]:
        normalized = normalized.replace(left, right)
    assert normalized == old_runner

    auth = load(contract['authorization_path'])
    assert auth['status'] == 'PENDING_SEPARATE_G10_V3_AUTHORIZATION'
    assert auth['science_execution_authorized'] is False and auth['source_commit'] is None
    assert auth['explicit_execution_instruction'] is None
    assert auth['runs'] == 1 and auth['retries'] == 0 and auth['mandatory_STOP'] is True
    assert not safe(contract['result_directory']).exists()
    assert not safe(contract['optional_receipt_path']).exists()
    frozen_contract = git_blob(S3, PREP+'contract_v3.json')
    assert hashlib.sha256(frozen_contract).hexdigest() == identity(PREP+'contract_v3.json')['sha256']
    assert identity(PREP+'contract_v3.json')['sha256'] == 'f50e9b99a25f859556631c47916dc346295b3d80dd22139ae9cd7af0da0bc7b4'
    return {
        'kind': 'G10_S3_REVIEW_INTAKE_READ_ONLY_PROVENANCE', 'passed': True,
        'frozen_source_S3': S3, 'review_copy_identity': actual_review,
        'verbatim_copy_verified_at_intake': receipt['verbatim_copy_verified'],
        'critical_paths': len(manifest['sha256']), 'full_hash_critical_paths': critical_full,
        'append_only_S3_prefixes': prefixes, 'protected_paths': len(ledger), 'protected_violations': [],
        'S3_preparation_inventory_paths_verified': len(inventory['files']),
        'immutable_Git_source_references_verified': len(refs),
        'IO_changed_definition_names': changed, 'runner_metadata_normalized_bytes_equal_S2': True,
        'contract_identity': identity(PREP+'contract_v3.json'),
        'pending_authorization_identity': identity(contract['authorization_path']),
        'source_manifest_identity': identity(PREP+'source_manifest_v3.json'),
        'science_source_contract_authorization_old_results_markers_STOP_unchanged': True,
        'v3_production_directory_marker_A3_receipt_absent': True,
        'reference_commit_must_not_be_A3_parent': True,
        'focused_or_reviewer_tests_rerun': False,
        'production_runner_invocations': 0, 'new_science_synthesis_matrix_circuit_sampling_LP_GPU': 0,
        'NPZ_access': 0, 'science_execution_authorized': False, 'mandatory_STOP': True,
        'method': 'stdlib streaming identities and static AST/bytes only; no scientific source imports',
    }


if __name__ == '__main__':
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
