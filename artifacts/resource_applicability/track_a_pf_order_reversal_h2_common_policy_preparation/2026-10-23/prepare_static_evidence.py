"""Preparation evidence only: stdlib parsing/copying/hashing, no science imports.

Never imports, evaluates, or runs the historical sources. Does not calculate
biases, headroom, allowances, shots, costs, moments, ratios, or ranks.
Run directly with the system Python from this detached preparation worktree.
"""
import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = '366b2fd24f7e86aa82b5b518b6e3774520f65596'
AUDIT_REL = 'artifacts/resource_applicability/track_a_pf_order_reversal_existing_evidence_audit/2026-10-23'
AUDIT = ROOT / AUDIT_REL
ACQ = '4ec26921c6872ed6e8667f167074e1e405cf1a3f'
PROVIDER = '217c971dd537b1cc7fbc13742bc0f6c263ff7f45'
ACCOUNT = '9a77d2b0a0b87e246e85f085b37f5923c18e27bb'
RESULT = '8613b3ecda2030a677b1690fe7aa3da23dec4873'
IDS = ('B1_p3_q4_2nd', 'B1_p3_q2_4th',
       'H2O32_c7ec590a32a859bc', 'H2O32_8a2ecdd0294ffb74')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def save(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    assert git('rev-parse', 'HEAD').decode().strip() == BASE
    assert not git('diff', '--name-only', BASE)
    assert not git('diff', '--cached', '--name-only')
    old_manifest = read(AUDIT / 'acquisition_manifest.json')
    registry = {}
    for row in old_manifest:
        registry[(row['commit'], row['path'])] = row
    used = []

    def source(commit, path):
        key = commit, path
        if key in registry:
            row = registry[key]
            p = AUDIT / row['local_path']
            raw = p.read_bytes()
            assert sha(raw) == row['sha256']
            origin = 'REUSED_FIXED_AUDIT_COPY'
        else:
            raw = git('show', commit + ':' + path)
            p = HERE / 'sources' / commit / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(raw)
            origin = 'ADDITIONAL_FIXED_GIT_BLOB'
        git_blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        assert git_blob == git('rev-parse', commit + ':' + path).decode().strip()
        if not any(r['commit'] == commit and r['path'] == path for r in used):
            used.append(dict(commit=commit, path=path, local_path=str(p.relative_to(ROOT)),
                bytes=len(raw), sha256=sha(raw), git_blob_sha1=git_blob, retrieval=origin,
                url='https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/' + commit + '/' + path))
        if path.endswith('.py'):
            ast.parse(raw, filename=path)
        return raw

    # The prior 75-row audit is not executed; select four existing string rows.
    rows = list(csv.DictReader((AUDIT / 'evidence_rows_75.csv').open()))
    selected = [r for candidate in IDS for r in rows
                if r['candidate_id'] == candidate and r['epsilon'] == '0.0005']
    assert len(selected) == 4 and [r['candidate_id'] for r in selected] == list(IDS)
    with (HERE / 'candidate_records_4.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)

    # Read and hash the nine required entry documents, without reanalysis.
    required = ('REPORT.md', 'comparison_ABC_15.csv', 'evidence_rows_75.csv',
                'identity_hashes_12.csv', 'shot_static_checks.json', 'SOURCE_INDEX.md',
                'acquisition_manifest.json', 'private_metadata_audit.json', 'completion_status.json')
    inherited = []
    for name in required:
        raw = (AUDIT / name).read_bytes()
        assert raw == git('show', BASE + ':' + AUDIT_REL + '/' + name)
        inherited.append(dict(path=AUDIT_REL + '/' + name, sha256=sha(raw), bytes=len(raw)))
    save('inherited_audit_files.json', dict(commit=BASE, files=inherited,
        reused_not_reexecuted=['75-row scientific evidence audit', 'shot-function AST equivalence',
                              'original rankings and saved-cost decomposition']))

    old_result_path = 'artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/D_completed_run02/result.json'
    old = json.loads(source('829d697ae7905b3ea90a61da9d2d4da15d8b757a', old_result_path))
    old_records = []
    for candidate in IDS[:2]:
        r = next(r for r in old['signals'] if r['cell']['id'] == candidate)
        r = dict(r)
        r['precision_rows'] = {'0.0005': r['precision_rows']['0.0005']}
        old_records.append(r)
    saved = json.loads(source(RESULT, 'artifacts/resource_applicability/track_a_h246_saved_cost_results/2026-10-21/evaluation.json'))
    # Copy sufficient statistics and already evaluated epsilon rows; no moments run.
    new_records = []
    for candidate in IDS[2:]:
        g = next(g for g in saved['groups'] if g['candidate']['id'] == candidate)
        new_records.append(dict(candidate=g['candidate'], unit_statistics=g['unit_statistics'],
            cost_provenance=g['cost_provenance'],
            saved_usual_rows=[r for r in g['rows'] if r['epsilon'] == 0.0005 and r['mode'] == 'USUAL_SIGNED']))
    save('saved_input_excerpts.json', dict(scope='EXACT_SAVED_VALUE_COPIES_NO_REACCOUNT',
        old_reference=old['reference'], old_signals=old_records,
        old_measurements=[r for r in old['measurements'] if r['cell_id'] in IDS[:2]],
        old_preparation_identities={k:old['preparation_identities'][k] for k in IDS[:2]},
        new_groups=new_records,
        new_signal_diagnostics='PRIVATE_SIGNALS_JSON_NOT_ACQUIRED; do not impute from eligibility label'))

    requests = [
        (PROVIDER, 'src/trottertracks/resource_applicability/h6_matched_accounting_v1.py'),
        (PROVIDER, 'src/trottertracks/resource_applicability/h6_matched_port_v1.py'),
        (PROVIDER, 'src/trottertracks/resource_applicability/h48_preflight_contract_v1.py'),
        (PROVIDER, 'src/trottertracks/resource_applicability/h6_matched_contract_v1.py'),
        (PROVIDER, 'src/trottertracks/resource_applicability/ax2a_control_plan.py'),
        (PROVIDER, 'src/trottertracks/resource_applicability/ax2b_native_df_v5.py'),
        (PROVIDER, 'src/trottertracks/resource_applicability/ax2b_numerical_accounting.py'),
        ('cf99e4452df96291d7e9d37d76d2e91be5cbc413', 'scripts/h4_native_science.py'),
        ('cf99e4452df96291d7e9d37d76d2e91be5cbc413', 'scripts/h2_panels_science.py'),
        ('cf99e4452df96291d7e9d37d76d2e91be5cbc413', 'scripts/h2_d_recovery_science.py'),
        (ACQ, 'scripts/h46_m5_signal_contract.py'),
        (ACQ, 'scripts/h46_m5_native.py'),
        (ACQ, 'scripts/h46_pr_pilot_native.py'),
        (ACQ, 'scripts/h2_outer32_runtime.py'),
        (ACQ, 'scripts/h2_outer32_contract.py'),
        (ACQ, 'scripts/h2_outer32_pipeline.py'),
        (ACQ, 'scripts/prepare_h2_outer32.py'),
        (ACQ, 'scripts/h2_m5_compiler_phase_guard.py'),
        (ACQ, 'src/trotterlib/h46_m5_plan.py'),
        (ACCOUNT, 'scripts/h246_saved_cost_evaluation.py'),
        (ACCOUNT, 'scripts/prepare_h246_saved_cost_evaluation.py'),
        (ACCOUNT, 'scripts/d0_frozen_shot_rule_v1.py'),
        (ACCOUNT, 'artifacts/resource_applicability/track_a_h246_saved_evaluation_preflight/2026-10-21/policy_adoption_required.json'),
        (RESULT, 'artifacts/resource_applicability/track_a_h246_saved_cost_results/2026-10-21/policy_adoption.json'),
        (RESULT, 'artifacts/resource_applicability/track_a_h246_saved_cost_results/2026-10-21/input_provenance.json'),
        (RESULT, 'artifacts/resource_applicability/track_a_h246_saved_cost_results/2026-10-21/grant_and_scope.json'),
        ('a05f4ed19e87e1eb65f7fc346385b5b4eae69845', 'artifacts/resource_applicability/track_a_h2_outer32_run01/2026-10-21/summary.json'),
        ('a05f4ed19e87e1eb65f7fc346385b5b4eae69845', 'docs/research/track_a_h2_outer32_run01_results_20261021.md'),
        ('829d697ae7905b3ea90a61da9d2d4da15d8b757a', old_result_path.replace('result.json','execution_binding.json')),
    ]
    for commit, path in requests:
        source(commit, path)

    # This comparison concerns Git blobs, never actual imported new-run bytes.
    private = read(AUDIT / 'private_metadata_audit.json')
    comparisons = []
    for r in private['provider_file_comparisons']:
        result = subprocess.run(['git','cat-file','-e',ACQ + ':' + r['path']],cwd=ROOT,capture_output=True)
        item = dict(path=r['path'], old_provider_commit=PROVIDER,
                    old_historical_sealed_sha256=r['sealed_sha256'], new_acquisition_commit=ACQ,
                    actual_new_loaded_provider_verified=False)
        if result.returncode == 0:
            raw = source(ACQ, r['path'])
            item.update(new_git_sha256=sha(raw), git_bytes_match_old_seal=sha(raw)==r['sealed_sha256'],
                        status='GIT_BLOB_ONLY_NOT_RUNTIME_ATTESTATION')
        else:
            item.update(new_git_sha256=None, git_bytes_match_old_seal=None,
                        status='NOT_IN_ACQUISITION_GIT_TREE_EXTERNAL_PROVIDER_REQUIRED')
        comparisons.append(item)
    save('provider_git_comparison.json', dict(scope='STATIC_GIT_BYTES_ONLY', files=comparisons,
        inference='acquisition SOURCE does not identify the complete loaded provider closure'))

    exact = [private['old_spec_locator_from_public_binding'], private['old_runtime_manifest_path'],
        '/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h2-pr-resource-preparation-20261019/.h2_pr_resource_private/2026-10-19/D_interruption_audit01/original_D_preservation_manifest_PRIVATE.json']
    save('exact_locator_checks.json', dict(scope='THREE_REFERENCED_PATHS_ONLY_NO_DIRECTORY_SCAN',
        files=[dict(path=p, exists_here=Path(p).exists()) for p in exact],
        new_private_root=None, remote_host=None, remote_access_attempted=False,
        user_reply='一旦なしで進めて後で渡す',
        disposition='UNACQUIRED_DEFERRED_BY_USER; prior metadata audit remains historical evidence'))

    save('source_manifest.json', used)
    lines = ['# 固定SOURCE・取得ファイル', '',
        '既存監査の保存copyを再利用し、不足分だけ固定Git blobから追加取得した。',
        '以下はファイルbyteの所在であり、新runでの実ロードを保証する一覧ではない。', '',
        '| commit | file | acquisition | SHA-256 |', '|---|---|---|---|']
    for r in used:
        lines.append('| `' + r['commit'] + '` | [' + r['path'] + '](' + r['url'] + ') | ' + r['retrieval'] + ' | `' + r['sha256'] + '` |')
    (HERE / 'SOURCE_INDEX.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status='STATIC_COPY_AND_HASH_PASS', candidates=len(selected),
                         used_source_files=len(used), scientific_calls=0)))


if __name__ == '__main__':
    main()
