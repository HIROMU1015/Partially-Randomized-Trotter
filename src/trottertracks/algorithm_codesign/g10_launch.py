"""Independent G10 launch: source S -> clean authorization-only child A."""
import json
import re
from pathlib import Path
from .g7_launch import protected_check, sha
from .synthesis_placement.wrapper_launch import git


def validate_binding(auth, contract_hash, requested_source, head, parents, changed, dirty, allowed):
    s = auth.get('source_commit')
    if (not isinstance(s, str) or not re.fullmatch('[0-9a-f]{40}', s)
            or auth.get('status') != 'APPROVED_FOR_ONE_G10_RUN'
            or auth.get('science_execution_authorized') is not True
            or type(auth.get('runs')) is not int or auth['runs'] != 1
            or type(auth.get('retries')) is not int or auth['retries'] != 0
            or auth.get('mandatory_STOP') is not True):
        raise PermissionError('separate source-bound G10 one-shot authorization required')
    if (not isinstance(auth.get('explicit_execution_instruction'), str)
            or len(auth['explicit_execution_instruction'].strip()) < 10):
        raise PermissionError('source-bound explicit G10 instruction missing')
    if requested_source != s or auth.get('contract_sha256') != contract_hash:
        raise PermissionError('G10 source/contract identity mismatch')
    if head == s or parents != [s] or dirty or not changed or not set(changed) <= allowed:
        raise PermissionError('clean direct authorization-only child required')


def verify_launch(root, contract_path, requested_source):
    root, contract_path = Path(root), Path(contract_path)
    c = json.loads(contract_path.read_text())
    auth = json.loads((root/c['authorization_path']).read_text())
    if (auth.get('status') != 'APPROVED_FOR_ONE_G10_RUN'
            or auth.get('science_execution_authorized') is not True
            or not isinstance(auth.get('source_commit'), str)
            or not re.fullmatch('[0-9a-f]{40}', auth['source_commit'])):
        raise PermissionError('G10 preparation only; fixed-source one-shot authorization pending')
    source, head = auth['source_commit'], git(root, 'rev-parse', 'HEAD')
    # Disable quoted Unicode names explicitly; the authorization paths are ASCII.
    changed = git(root, '-c', 'core.quotePath=false', 'diff', '--name-only', source, head).splitlines()
    validate_binding(auth, sha(contract_path), requested_source, head,
                     git(root, 'show', '-s', '--format=%P', head).split(), changed,
                     bool(git(root, 'status', '--porcelain', '--untracked-files=all')),
                     {c['authorization_path'], c['optional_receipt_path']})
    if c['authorization_path'] not in changed:
        raise PermissionError('authorization JSON must be committed in A')
    remote = git(root, 'ls-remote', 'origin', 'refs/heads/'+git(root, 'branch', '--show-current'))
    if not remote or remote.split()[0] != head:
        raise PermissionError('remote execution branch does not match A')
    manifest = json.loads((root/c['source_manifest']).read_text())
    if manifest.get('focused_tests_passed') is not True:
        raise PermissionError('focused source verification incomplete')
    for relative, digest in manifest['sha256'].items():
        if relative.lower().endswith('.npz'):
            raise PermissionError('NPZ access forbidden')
        if sha(root/relative) != digest:
            raise PermissionError('critical identity changed: '+relative)
    check = protected_check(root, c)
    if check['violations']:
        raise PermissionError('protected history changed')
    directory = root/c['result_directory']
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError('G10 result/marker already exists; no retry')
    return c, auth, head, check
