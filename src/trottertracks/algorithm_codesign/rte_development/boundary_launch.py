"""D2 boundary-specific authorization; rejects before registered inputs."""
import json
from pathlib import Path
import re

from .evidence import digest
from .launch import git, check_protected


def validate_binding(a, contract_sha, source, head, parents, changed, dirty, allowed):
    if (a.get('status') != 'APPROVED_FOR_ONE_D2_BOUNDARY_RUN'
        or a.get('science_execution_authorized') is not True
        or type(a.get('runs')) is not int or a['runs'] != 1
        or type(a.get('retries')) is not int or a['retries'] != 0
        or a.get('mandatory_STOP') is not True
        or not isinstance(a.get('explicit_execution_instruction'),str)
        or len(a['explicit_execution_instruction'].strip()) < 10):
        raise PermissionError('separate D2 one-shot authorization required')
    if (not re.fullmatch('[0-9a-f]{40}', source or '')
        or a.get('source_commit') != source or a.get('contract_sha256') != contract_sha
        or head == source or parents != [source] or dirty or not changed
        or not set(changed) <= set(allowed) or allowed[0] not in changed):
        raise PermissionError('D2 source/contract/direct-child/clean binding')


def verify_launch(root, contract_path, source):
    root = Path(root); c = json.loads(Path(contract_path).read_text())
    a = json.loads((root/c['authorization_path']).read_text())
    if a.get('science_execution_authorized') is not True:
        raise PermissionError('D2 boundary preparation only')
    head = git(root,'rev-parse','HEAD')
    validate_binding(a,digest(contract_path),source,head,
        git(root,'show','-s','--format=%P',head).split(),
        git(root,'-c','core.quotePath=false','diff','--name-only',source,head).splitlines(),
        bool(git(root,'status','--porcelain','--untracked-files=all')),
        [c['authorization_path'],c['optional_receipt_path']])
    if git(root,'remote','get-url','origin') != 'git@github.com:HIROMU1015/Partially-Randomized-Trotter.git':
        raise PermissionError('D2 remote repository mismatch')
    branch = git(root,'branch','--show-current')
    remote = git(root,'ls-remote','--heads','origin','refs/heads/'+branch)
    if not branch or not remote or remote.split()[0] != head:
        raise PermissionError('D2 remote A mismatch')
    manifest = json.loads((root/c['source_manifest']).read_text())
    if manifest.get('focused_tests_passed') is not True:
        raise PermissionError('D2 source tests incomplete')
    for path, sha in manifest['critical_sha256'].items():
        if path.lower().endswith('.npz') or digest(root/path) != sha:
            raise PermissionError('D2 source/inputs identity changed: '+path)
    check_protected(root,c)
    if (root/c['result_directory']).exists():
        raise FileExistsError('D2 result namespace exists; no retry')
    return c,a,head
