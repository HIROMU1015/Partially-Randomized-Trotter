"""Terminal bytes/Git check only. Does not load scientific arrays/modules."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = 'e72867b87e12d93d4fc1361f45f105febc6a94c6'
SOURCE = '9a3503375c07037ab3e0b9428a7c0254b66c31f6'
AUTH = '8c892129f0d16211ee2fd2f3eafafeba4f181bd7'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def git(*args): return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
def save(name, data): (HERE / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')

def main():
    checks = {}
    for filename in ('source_manifest.json', 'input_manifest.json', 'runtime_manifest.json'):
        records = json.loads((HERE / filename).read_text())['files']
        for record in records:
            path = Path(record['path'])
            if not path.is_absolute(): path = ROOT / path
            assert sha(path) == record['sha256'], str(path)
        checks[filename] = {'files': len(records), 'all_SHA256_equal': True}
    private = json.loads((HERE / 'preservation_baseline.json').read_text())['private_originals']
    for record in private: assert sha(record['path']) == record['sha256']
    namespace = str(HERE.relative_to(ROOT)) + '/'
    changes = git('diff', '--cached', '--name-status', BASE).splitlines()
    assert all(line.startswith('A\t' + namespace) for line in changes)
    assert not git('diff', '--name-only')
    authorization = json.loads((HERE / 'authorization_receipt.json').read_text())
    assert authorization['source_commit'] == SOURCE
    assert authorization['source_manifest_sha256'] == sha(HERE / 'source_manifest.json')
    assert authorization['contract_sha256'] == sha(HERE / 'execution_contract.json')
    assert git('rev-parse', AUTH + '^') == SOURCE
    assert git('diff', '--name-only', SOURCE, AUTH) == namespace + 'authorization_receipt.json'
    state = json.loads((HERE / 'result/state_identity.json').read_text())
    assert state['matches'] and state['expected'] == state['actual']
    preflight = json.loads((HERE / 'result/preflight.json').read_text())
    assert preflight['ready'] and all(r['passed'] and r['status'] == 'PASS' for r in preflight['predicates'])
    terminal = json.loads((HERE / 'result/terminal.json').read_text())
    assert terminal['classification'] == 'C1_3_TECHNICAL_INCONCLUSIVE_STOP'
    assert terminal['scientific_attempts'] == 1 and terminal['retry'] == 0
    assert not (HERE / 'result/summary.json').exists()
    assert not list((HERE / 'result').glob('event_*'))
    inventory = [{'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'SHA256': sha(path)}
                 for path in sorted((HERE / 'result').iterdir()) if path.is_file()]
    save('run_output_manifest.json', {'run_id': 'C1-3-R1-H4-selected-20261021', 'scientific_attempts': 1, 'retry': 0,
         'classification': terminal['classification'], 'files': inventory})
    save('verification.json', {'status': 'PRESERVATION_VERIFIED_TECHNICAL_STOP_REMAINS',
         'base_commit': BASE, 'source_commit': SOURCE, 'authorization_commit': AUTH,
         'manifest_checks': checks, 'private_originals_checked': len(private), 'private_originals_published': False,
         'only_additions_in_new_namespace': True, 'existing_tracked_blobs_modified': 0,
         'prior_failed_SOURCE_authorization_results_STOP_unchanged': True,
         'new_frozen_source_and_contract_unchanged': True,
         'phase_A': 'PASS', 'phase_A_test_executions': 1, 'phase_A_H4_scientific_attempts': 0,
         'phase_B': terminal['classification'], 'new_scientific_attempts': 1, 'new_retry': 0,
         'state_hash_fixed_expected_matches': True, 'P0_P5_ready': True,
         'event_operator_XY_comparisons_completed': 0, 'phase_C_C1_4': 'NOT_STARTED',
         'postfailure_work': 'static reads, documentation, logs/bytes/Git preservation only',
         'postfailure_H4_NPZ_open_or_scientific_execution': False,
         'native_compile_synthesis_shots_sampling_performance': False,
         'other_worktrees_or_shared_configuration_modified': False,
         'remaining': 'GitHub publication bytes/head check, then mandatory STOP'})
    print(json.dumps({'preservation': 'VERIFIED', 'checks': checks, 'result_files': len(inventory),
         'scientific_classification': terminal['classification'], 'mandatory_STOP': True}))

if __name__ == '__main__': main()
