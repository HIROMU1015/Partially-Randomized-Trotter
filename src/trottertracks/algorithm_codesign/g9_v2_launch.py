"""G9 v2 needs a NEW explicit authorization; v1 markers stay consumed."""
import json
import re
from pathlib import Path

from .g7_launch import protected_check, sha
from .synthesis_placement.wrapper_launch import git


def validate_binding(auth, contract_hash, requested_source, head, parents,
                     changed, dirty, allowed):
    source = auth.get('source_commit')
    if not isinstance(source, str) or not re.fullmatch('[0-9a-f]{40}', source):
        raise PermissionError('separate G9 v2 source-bound authorization pending')
    if (auth.get('status') != 'APPROVED_FOR_ONE_G9_V2_RUN'
            or auth.get('science_execution_authorized') is not True
            or type(auth.get('runs')) is not int or auth['runs'] != 1
            or type(auth.get('retries')) is not int or auth['retries'] != 0
            or auth.get('mandatory_STOP') is not True):
        raise PermissionError('separate G9 v2 one-shot authorization required')
    instruction = auth.get('explicit_execution_instruction')
    if not isinstance(instruction, str) or len(instruction.strip()) < 10:
        raise PermissionError('new explicit G9 v2 execution instruction required')
    if requested_source != source or auth.get('contract_sha256') != contract_hash:
        raise PermissionError('G9 v2 source/contract binding mismatch')
    if head == source or parents != [source] or dirty:
        raise PermissionError('clean direct authorization-only child required')
    if not changed or not set(changed).issubset(allowed):
        raise PermissionError('authorization child changed a forbidden path')


def verify_launch(root, contract_path, requested_source):
    root = Path(root)
    contract_path = Path(contract_path)
    contract = json.loads(contract_path.read_text())
    auth = json.loads((root / contract['authorization_path']).read_text())
    # Refuse pending preparations BEFORE revision queries or output/marker access.
    source = auth.get('source_commit')
    if (auth.get('status') != 'APPROVED_FOR_ONE_G9_V2_RUN'
            or auth.get('science_execution_authorized') is not True
            or not isinstance(source, str)
            or not re.fullmatch('[0-9a-f]{40}', source)):
        raise PermissionError('G9 v2 preparation only; new one-shot authorization pending')
    head = git(root, 'rev-parse', 'HEAD')
    changed = git(root, 'diff', '--name-only', source, head).splitlines()
    allowed = {contract['authorization_path'], contract['optional_receipt_path']}
    validate_binding(auth, sha(contract_path), requested_source, head,
                     git(root, 'show', '-s', '--format=%P', head).split(),
                     changed, bool(git(root, 'status', '--porcelain', '--untracked-files=all')),
                     allowed)
    if contract['authorization_path'] not in changed:
        raise PermissionError('authorization JSON must be committed in child A')
    manifest = json.loads((root / contract['source_manifest']).read_text())
    if manifest.get('focused_tests_passed') is not True:
        raise PermissionError('focused source verification incomplete')
    for relative, digest in manifest['sha256'].items():
        if sha(root / relative) != digest:
            raise PermissionError('critical source/input changed: ' + relative)
    check = protected_check(root, contract)
    if check['violations']:
        raise PermissionError('protected G9 v1 / prior evidence changed')
    # Existing result directories are never reused, even with a new authorization.
    directory = root / contract['result_directory']
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError('G9 v2 evidence/marker exists; no retry')
    return contract, auth, head, check
