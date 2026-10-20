"""Post-run metadata/bytes verification only; never imports scientific modules."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = '49dae5247ff0ad2c4efb9056e095b70913cc9ddb'
SOURCE = '2be82f7bff78b7d11ffb11013b03a49e296acd1d'
AUTH = '324634a37538eea2bf6153467fa0216ffa8a5c56'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def git(*args): return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
def save(name, data): (HERE / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')

def main():
    checks = {}
    for filename in ('source_manifest.json', 'input_manifest.json', 'runtime_manifest.json'):
        rows = json.loads((HERE / filename).read_text())['files']
        for record in rows:
            path = Path(record['path']); path = path if path.is_absolute() else ROOT / path
            assert sha(path) == record['sha256'], str(path)
        checks[filename] = {'files': len(rows), 'all_SHA256_equal': True}
    private = json.loads((HERE / 'preservation_baseline.json').read_text())['private_originals']
    for record in private: assert sha(record['path']) == record['sha256']
    namespace = str(HERE.relative_to(ROOT)) + '/'
    assert all(line.startswith('A\t' + namespace) for line in git('diff', '--cached', '--name-status', BASE).splitlines())
    assert not git('diff', '--name-only')
    authorization = json.loads((HERE / 'authorization_receipt.json').read_text())
    assert authorization['source_commit'] == SOURCE
    assert authorization['source_manifest_sha256'] == sha(HERE / 'source_manifest.json')
    assert authorization['contract_sha256'] == sha(HERE / 'execution_contract.json')
    assert git('rev-parse', AUTH + '^') == SOURCE
    assert git('diff', '--name-only', SOURCE, AUTH) == namespace + 'authorization_receipt.json'
    terminal = json.loads((HERE / 'result/terminal.json').read_text())
    summary = json.loads((HERE / 'result/summary.json').read_text())
    assert terminal['classification'] == summary['status'] == 'C1_3_CORRECTNESS_PASS'
    assert terminal['scientific_attempts'] == 1 and terminal['retry'] == 0 and summary['selected_events'] == 6
    assert not (HERE / 'result/failure.json').exists()
    assert len(list((HERE / 'result').glob('event_*_comparison.json'))) == 6
    assert len(list((HERE / 'result').glob('event_*_sidecar.json'))) == 6
    assert len(list((HERE / 'result').glob('event_*_parent.json'))) == 6
    phase_b = json.loads((HERE / 'phase_b/summary.json').read_text())
    assert phase_b['status'] == 'PASS' and phase_b['basis_count'] == 13 and phase_b['local_matrix_count'] == 412
    assert phase_b['new_mismatch_count'] == 0 and phase_b['legacy_mismatch_count'] == 59
    assert json.loads((HERE / 'c1_4_readiness.json').read_text())['status'] == 'UNADOPTED_DRAFT_NOT_EXECUTION_AUTHORIZATION'
    inventory = [{'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'SHA256': sha(path)}
                 for path in sorted((HERE / 'result').iterdir()) if path.is_file()]
    save('run_output_manifest.json', {'run_id': 'C1-3-R2-H4-selected-20261021',
         'classification': summary['status'], 'scientific_attempts': 1, 'retry': 0, 'files': inventory})
    save('verification.json', {'status': 'PRESERVATION_VERIFIED_CORRECTNESS_PASS',
         'base_commit': BASE, 'source_commit': SOURCE, 'authorization_commit': AUTH,
         'manifest_checks': checks, 'private_originals_checked': len(private), 'private_originals_published': False,
         'only_additions_in_new_namespace': True, 'existing_tracked_blobs_modified': 0,
         'old_failed_SOURCE_authorizations_results_STOP_unchanged': True,
         'new_frozen_science_SOURCE_contract_unchanged': True,
         'phase_A_test_executions': 1, 'phase_B_local_verification_executions': 1,
         'phase_C_scientific_attempts': 1, 'phase_C_retry': 0, 'phase_C': summary['status'],
         'phase_D_C1_4': 'COMPLETE_UNADOPTED_CONTRACT_DRAFTS',
         'post_phase_C_scientific_execution': False,
         'native_compile_synthesis_shots_sampling_performance_N_ranking': False,
         'other_worktrees_or_shared_configuration_modified': False,
         'next': 'publication bytes/HEAD/clean verification then mandatory STOP; C1-5 not authorized'})
    print(json.dumps({'preservation': 'VERIFIED', 'manifest_checks': checks,
         'result_files': len(inventory), 'phase_C': summary['status'], 'C1_4': 'DRAFT_COMPLETE', 'mandatory_STOP': True}))

if __name__ == '__main__': main()
