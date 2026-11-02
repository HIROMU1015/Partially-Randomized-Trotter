"""Stdlib-only saved-byte and static-source audit; never execute science."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
DOCS = {'PROJECT_MAP.md', 'VALIDATION_STATUS.md', 'docs/README.md',
        'docs/research/README.md', 'docs/research/研究概要・現状.md',
        'scripts/README.md', 'src/trotterlib/README.md'}


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def load(path):
    return json.loads(path.read_text())


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args])


def main():
    before = load(OUT / 'preservation_before_review_v1.json')
    origin = Path(before['worktree'])
    for name, expected in before['hashes'].items():
        assert sha(origin / name) == expected, ('origin', name)
    assert git(origin, 'rev-parse', 'HEAD').decode().strip() == before['head']
    assert git(origin, 'status', '--short', '--untracked-files=all').decode() == before['status']
    assert hashlib.sha256(git(origin, 'diff', '--binary')).hexdigest() == before['diff_sha256']
    changed = [n for n, h in before['hashes'].items() if sha(ROOT / n) != h]
    assert set(changed) == DOCS, changed

    relative = 'artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1'
    old = ROOT / relative
    inventory = load(old / 'pilot_evidence_inventory_v5.json')
    for entry in inventory['entries']:
        path = origin / entry['path']
        assert sha(path) == entry['sha256'] and path.stat().st_size == entry['size_bytes']
        if entry['path'] not in DOCS:
            copied = ROOT / entry['path']
            assert sha(copied) == entry['sha256'] and copied.stat().st_size == entry['size_bytes']
    assert sha(old / 'pilot_evidence_inventory_v5.json') == before['pilot_inventory_sha256']

    freeze = load(ROOT / 'artifacts/resource_applicability/track_a_ax2b_h4_fingerprint_preparation/2026-10-09/source_freeze_v5.json')
    for group in ('science_source_hashes', 'preparation_validation_file_hashes'):
        for name, expected in freeze[group].items():
            assert sha(ROOT / name) == expected, name
    audit = load(old / 'saved_evidence_audit_v5.json')
    terminal = load(old / 'run_v1/terminal_status.json')
    summary = load(old / 'saved_result_analysis_v5.json')
    assert audit['status'] == 'H4_SAVED_PILOT_EVIDENCE_VERIFIED'
    assert audit['complete_pilot_verified']
    assert terminal['status'] == 'H4_TECHNICAL_PILOT_COMPLETE'
    assert not audit['total_numerical_allowance_certified']
    assert audit['accuracy_eligibility'] == 'UNDETERMINED'
    assert summary['science_attempts'] == 1
    for entry in summary['raw_files']:
        path = ROOT / entry['path']
        assert sha(path) == entry['sha256'] and path.stat().st_size == entry['size_bytes']
    assert len(summary['raw_files']) == 447
    junit = ET.parse(old / 'prelaunch_167.junit.xml').getroot()
    assert len(list(junit.iter('testcase'))) == 167
    assert not any(list(junit.iter(tag)) for tag in ('failure', 'error', 'skipped'))

    source_paths = [
        'src/trotterlib/df_hamiltonian.py', 'src/trotterlib/df_gpu_statevector.py',
        'src/trotterlib/pf_c_system_size_validation.py', 'src/trotterlib/rte.py',
        'src/trotterlib/pr2_s0_s1_validation.py',
        'src/trotterlib/pr2_matched_accuracy_m1_execution.py',
        'src/trottertracks/resource_applicability/ax1b_evaluation.py',
        'src/trottertracks/resource_applicability/ax2a_preparation.py',
        'src/trottertracks/resource_applicability/ax2a_state_action.py',
        'src/trottertracks/resource_applicability/ax2b_native_df_v5.py',
        'src/trottertracks/resource_applicability/ax2b_h4_science_v5.py',
        'src/trottertracks/resource_applicability/ax2b_h4_contract_v5.py']
    static = []
    for name in source_paths:
        tree = ast.parse((ROOT / name).read_text())
        definitions = [{'name': node.name, 'line': node.lineno}
                       for node in tree.body
                       if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        static.append({'path': name, 'sha256': sha(ROOT / name), 'definitions': definitions,
                       'method': 'text_and_AST_only', 'module_imported': False})

    gaps = load(OUT / 'validation_gaps_v1.json')
    draft = load(OUT / 'h6_pilot_contract_draft_v1.json')
    assert len(gaps['gaps']) == 11 and len({g['id'] for g in gaps['gaps']}) == 11
    assert not draft['science_authorized'] and not draft['launch_allowed']
    assert draft['assigned_resources'] is None and not draft['H8_tasks']
    assert len(draft['cells']) == 7 and len(draft['wrapper_tasks']) == 36
    random = [c for c in draft['cells'] if c['method'] in ('B2', 'B3')]
    assert sum(c['cost_replicas'] for c in random) == 4
    assert sum(c['q'] * c['cost_replicas'] for c in random) == 8
    assert all(c['R'] % c['q'] == 0 and c['r'] == c['R'] // c['q'] for c in random)
    assert len({task['id'] for task in draft['wrapper_tasks']}) == 36
    assert all(draft['identities'][key] is None for key in draft['identities'])
    result = {
        'schema': 'track_a_h4_post_review_static_input_audit_v1', 'status': 'PASS',
        'origin_worktree': str(origin), 'origin_files_verified': len(before['hashes']),
        'origin_HEAD_status_diff_unchanged': True,
        'changed_copied_existing_files': sorted(changed),
        'original_execution_inventory_entries_verified': len(inventory['entries']),
        'copied_execution_entries_verified_except_current_indexes': len(inventory['entries']) - len(DOCS),
        'raw_files_unchanged': 447, 'science_source_hashes_verified': len(freeze['science_source_hashes']),
        'validation_hashes_verified': len(freeze['preparation_validation_file_hashes']),
        'historical_prelaunch_tests_verified': 167, 'new_tests_run': 0,
        'saved_audit_status': audit['status'], 'static_source_inventory': static,
        'gap_count': 11, 'proposed_H6_cells': 7, 'proposed_H6_wrappers': 36,
        'new_scientific_module_imports': 0, 'new_scientific_calls': 0,
        'new_Hamiltonian_or_state_generation': 0, 'new_source_implementation': 0,
        'total_numerical_allowance_certified': False, 'accuracy_eligibility': 'UNDETERMINED',
        'shot_or_total_cost_evaluation_performed': False, 'winner_claim': False,
        'mandatory_stop': True, 'next_stage_authorized': False,
        'Track_B_changes': 0, 'commit_or_push_performed': False}
    with (OUT / 'static_input_audit_v1.json').open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in
        ('status', 'origin_files_verified', 'original_execution_inventory_entries_verified',
         'science_source_hashes_verified', 'gap_count', 'proposed_H6_cells', 'proposed_H6_wrappers',
         'new_scientific_calls')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
