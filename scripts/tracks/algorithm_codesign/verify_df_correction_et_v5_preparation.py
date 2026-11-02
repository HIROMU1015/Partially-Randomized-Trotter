"""Read-only source/provenance audit. No registered numerical evaluation."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
BASE = 'a70af62798849ba1add7491375fe4825acdcaaa0'
ART = 'artifacts/track_b_df_correction_et_storage_preparation/2026-10-12/v5'
OLD_ART = 'artifacts/track_b_df_correction_et_resumption_preparation/2026-10-12'
OLD_NS = 'src/trottertracks/algorithm_codesign/df_correction_et_resumption'
NEW_NS = 'src/trottertracks/algorithm_codesign/df_correction_et_storage_v5'
INDEXES = ['PROJECT_MAP.md', 'docs/research/研究概要・現状.md', 'scripts/README.md',
           'src/trotterlib/README.md', 'docs/README.md',
           'docs/tracks/algorithm_codesign/README.md']


def identity(path):
    raw = (ROOT/path).read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def check_rows(rows):
    for row in rows:
        assert identity(row['path']) == row, row['path']
    return len(rows)


def tree(text):
    return ast.dump(ast.parse(text), include_attributes=False)


def main():
    old = json.loads((ROOT/OLD_ART/'contract_v4.json').read_text())
    new = json.loads((ROOT/ART/'contract_v5.json').read_text())
    changed = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
    assert changed == ['base_commit', 'status', 'storage_amendment', 'version'], changed
    identical_contract_sections = sorted(k for k in old if k not in changed)
    preserved_code = []
    for name in ('native_evaluation.py', 'saved_data.py', 'accounting.py'):
        assert (ROOT/OLD_NS/name).read_bytes() == (ROOT/NEW_NS/name).read_bytes(), name
        preserved_code.append(identity(Path(NEW_NS)/name))
    assert (ROOT/'src/trottertracks/algorithm_codesign/df_correction_comparison/guard.py').read_bytes() == (ROOT/NEW_NS/'guard.py').read_bytes()
    code_equivalence = {}
    for name in ('acceptance.py', 'worker.py'):
        a = (ROOT/OLD_NS/name).read_text()
        b = (ROOT/NEW_NS/name).read_text()
        a = a.replace('from ..df_correction_comparison.storage import', 'from .storage import')
        a = a.replace('from ..df_correction_comparison.guard import', 'from .guard import')
        a = a.replace('contract_v4.json', 'contract_v5.json')
        assert tree(a) == tree(b), name
        code_equivalence[name] = 'AST identical after storage/guard import and contract-path substitution'
    a = (ROOT/'scripts/tracks/algorithm_codesign/run_df_correction_et_resumption_v4.py').read_text()
    b = (ROOT/'scripts/tracks/algorithm_codesign/run_df_correction_et_resumption_v5.py').read_text()
    a = a.replace('df_correction_et_resumption', 'df_correction_et_storage_v5')
    a = a.replace('df_correction_comparison.guard', 'df_correction_et_storage_v5.guard')
    a = a.replace('df_correction_comparison.storage', 'df_correction_et_storage_v5.storage')
    a = a.replace('contract_v4.json', 'contract_v5.json')
    assert tree(a) == tree(b), 'coordinator'
    code_equivalence['coordinator'] = 'AST identical after namespace/storage/guard/contract-path substitution'
    source = json.loads((ROOT/OLD_ART/'source_manifest_v4.json').read_text())
    protected = [r for r in source['files'] if r['path'] not in INDEXES]
    source_count = check_rows(protected)
    append_only = []
    for name in INDEXES:
        base = subprocess.check_output(['git', 'show', BASE+':'+name], cwd=ROOT)
        now = (ROOT/name).read_bytes()
        assert now.startswith(base), name
        append_only.append({'path': name, 'base_bytes': len(base), 'appended_bytes': len(now)-len(base),
                            'base_sha256': hashlib.sha256(base).hexdigest()})
    inventory_path = 'artifacts/track_b_df_correction_et_execution_audit/2026-10-12/v1/original_run_inventory.json'
    inventory = json.loads((ROOT/inventory_path).read_text())
    old_result_count = check_rows(inventory['files'])
    old_result_root = ROOT/'artifacts/track_b_df_correction_et_resumption_result/2026-10-12/v1'
    assert sorted(str(p.relative_to(ROOT)) for p in old_result_root.rglob('*') if p.is_file()) == sorted(r['path'] for r in inventory['files'])
    auth = json.loads((ROOT/ART/'authorization.json').read_text())
    assert auth['science_execution_authorized'] is False and auth['runs'] == 0
    assert auth['source_commit'] is None and auth['source_manifest_sha256'] is None
    assert auth['contract_sha256'] == identity(Path(ART)/'contract_v5.json')['sha256']
    assert auth['retries'] == 0 and auth['mandatory_STOP'] is True
    assert not (ROOT/'artifacts/track_b_df_correction_et_resumption_result/2026-10-12/v2').exists()
    sys.path.insert(0, str(ROOT/'src'))
    from trottertracks.algorithm_codesign.df_correction_et_storage_v5.launch import verify_local
    try:
        verify_local(ROOT)
    except PermissionError as exc:
        pending_rejection = str(exc)
        assert pending_rejection == 'Separate explicit one-shot authorization absent'
    else:
        raise AssertionError('Pending authorization was accepted')
    manifest = ROOT/ART/'source_manifest_v5.json'
    manifest_checked = None
    if manifest.exists():
        entries = json.loads(manifest.read_text())['files']
        manifest_checked = check_rows(entries)
    report = {'kind': 'DF_ET_V5_STATIC_SCOPE_AND_PROTECTION_AUDIT', 'status': 'PASS',
              'base_commit': BASE, 'contract_changes': changed,
              'identical_contract_sections': identical_contract_sections,
              'byte_identical_scientific_modules': preserved_code,
              'guard_byte_identical': True, 'code_equivalence': code_equivalence,
              'old_source_non_index_files_verified': source_count, 'indexes_append_only': append_only,
              'old_result_files_verified': old_result_count,
              'old_result_bytes': inventory['bytes'], 'old_result_inventory': identity(inventory_path),
              'old_marker': identity('artifacts/track_b_df_correction_et_resumption_result/2026-10-12/v1/one_shot_marker.json'),
              'old_authorization': identity(Path(OLD_ART)/'authorization.json'),
              'old_terminal': identity('artifacts/track_b_df_correction_et_resumption_result/2026-10-12/v1/terminal_status.json'),
              'pending_authorization_rejected': pending_rejection,
              'fresh_result_namespace_absent': True, 'source_manifest_files_verified': manifest_checked,
              'registered_evaluation_calls': 0, 'native_compile_calls': 0, 'synthesis_calls': 0,
              'matrix_calls': 0, 'GPU_queries': 0, 'mandatory_STOP': True}
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
