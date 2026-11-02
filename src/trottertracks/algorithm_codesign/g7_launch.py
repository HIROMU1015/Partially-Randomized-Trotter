"""G7 delegated technical scope, fixed source, protected history and fresh marker."""
import hashlib
import json
from pathlib import Path
import re
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def protected_check(root, contract):
    ledger = json.loads((root / contract['protected_ledger']).read_text())
    failures = []
    for path, record in ledger.items():
        if path.lower().endswith('.npz'):
            raise PermissionError('NPZ path forbidden before access')
        data = (root / path).read_bytes()
        if path in contract['append_only_paths']:
            data = data[:record['bytes']]
        if hashlib.sha256(data).hexdigest() != record['sha256']:
            failures.append(path)
    return {'protected_paths': len(ledger), 'violations': failures,
            'old_sources_results_authorizations_markers_STOP_unchanged': not failures}


def verify_source(root, source, contract):
    if not re.fullmatch('[0-9a-f]{40}', source) or git(root, 'rev-parse', 'HEAD') != source:
        raise PermissionError('launch HEAD must be the fixed full source commit')
    if git(root, 'status', '--porcelain', '--untracked-files=all'):
        raise PermissionError('clean source worktree required')
    manifest = json.loads((root / contract['source_manifest']).read_text())
    if manifest['focused_tests_passed'] is not True:
        raise PermissionError('focused verification pending')
    for path, digest in manifest['sha256'].items():
        if sha(root / path) != digest:
            raise PermissionError('source identity mismatch: ' + path)
    check = protected_check(root, contract)
    if check['violations']:
        raise PermissionError('protected history mismatch')
    return check


def consume_marker(directory, receipt):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError('G7 evidence/marker exists; no retry')
    path = directory / 'one_shot_consumed.json'
    with path.open('x') as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    return path
