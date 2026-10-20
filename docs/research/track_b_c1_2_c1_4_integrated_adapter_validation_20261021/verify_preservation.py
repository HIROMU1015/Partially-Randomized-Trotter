"""Bytes/Git-only terminal verification; never loads scientific arrays or modules."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = '2937e6b655ba3cd77f982d6901d766355a8b92c1'
SOURCE = 'f3a6aa1108bbc8359e585c470e23033458c57466'
AUTH = 'c7d444d5c59acfc1d57d6b4ed348a86451522b07'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def git(*args): return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
def save(name, value): (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

def main():
    checks = {}
    for name in ('source_manifest.json', 'input_manifest.json', 'runtime_manifest.json'):
        rows = json.loads((HERE / name).read_text())['files']
        for row in rows:
            path = Path(row['path'])
            if not path.is_absolute(): path = ROOT / path
            assert sha(path) == row['sha256'], str(path)
        checks[name] = {'files': len(rows), 'all_SHA256_equal': True}
    originals = json.loads((HERE / 'preservation_baseline.json').read_text())['private_originals']
    for row in originals: assert sha(row['path']) == row['sha256']
    diff = git('diff', '--cached', '--name-status', BASE).splitlines()
    namespace = str(HERE.relative_to(ROOT)) + '/'
    assert all(line.startswith('A\t' + namespace) for line in diff)
    assert not git('diff', '--name-only'), 'unstaged tracked changes'
    auth = json.loads((HERE / 'authorization_receipt.json').read_text())
    assert auth['source_commit'] == SOURCE
    assert sha(HERE / 'source_manifest.json') == auth['source_manifest_sha256']
    assert sha(HERE / 'execution_contract.json') == auth['contract_sha256']
    assert git('rev-parse', AUTH + '^') == SOURCE
    assert git('diff', '--name-only', SOURCE, AUTH) == namespace + 'authorization_receipt.json'
    terminal = json.loads((HERE / 'result/terminal.json').read_text())
    assert terminal['status'] == 'FAILED_STOP' and terminal['scientific_attempts'] == 1 and terminal['retry'] == 0
    assert not any((HERE / 'result' / f).exists() for f in ('preflight.json', 'schedule.json', 'summary.json'))
    inventory = [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': sha(p)}
                 for p in sorted((HERE / 'result').iterdir()) if p.is_file()]
    save('run_output_manifest.json', {'scientific_attempts': 1, 'retry': 0, 'files': inventory})
    save('verification.json', {'status': 'PRESERVATION_VERIFIED_SCIENTIFIC_FAILURE_REMAINS',
         'base_commit': BASE, 'source_commit': SOURCE, 'authorization_commit': AUTH,
         'manifest_checks': checks, 'private_originals_checked': len(originals),
         'private_originals_published': False, 'existing_tracked_blobs_changed': 0,
         'only_additions_under_new_namespace': True, 'frozen_source_and_contract_changed': False,
         'C1_2_artificial_test_executions': 2, 'C1_2_final_frozen_checks': 'PASS',
         'C1_3_scientific_attempts': 1, 'C1_3_retry': 0, 'C1_3_status': 'FAILED_STOP',
         'C1_4_status': 'NOT_STARTED', 'postfailure_science_execution': False,
         'postfailure_diagnosis': 'static comparison of existing hash function definitions only',
         'new_H4_kernel_or_event_evaluation': False, 'native_compile_synthesis_shots_sampling': False,
         'other_worktrees_or_shared_configuration_modified': False,
         'remaining_action': 'publish bytes/head verification and mandatory STOP'})
    print(json.dumps({'preservation': 'VERIFIED', 'checks': checks, 'result_files': len(inventory),
                      'scientific_status': 'FAILED_STOP', 'mandatory_STOP': True}))

if __name__ == '__main__': main()
