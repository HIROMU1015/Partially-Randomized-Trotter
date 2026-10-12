"""Source-bound gate; no numerical imports, no fixture access before authorization."""
import hashlib
import json
from pathlib import Path
import subprocess

ART = Path('artifacts/track_b_df_correction_et_phase_preparation/2026-10-12/v7')
RESULT = Path('artifacts/track_b_df_correction_et_resumption_result/2026-10-12/v4')
RECEIPT = Path('docs/tracks/algorithm_codesign/df_correction_et_execution_authorization_v7.md')
MANIFEST = Path('artifacts/track_b_df_correction_et_phase_preparation/2026-10-12/v7/source_manifest_v7.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, text=True).strip()


def verify_local(root):
    root = Path(root)
    auth = json.loads((root/ART/'authorization.json').read_text())
    if (auth.get('status') != 'APPROVED_FOR_ONE_DF_CORRECTION_E_T_V7_RUN'
        or auth.get('science_execution_authorized') is not True or auth.get('runs') != 1
        or auth.get('retries') != 0 or auth.get('mandatory_STOP') is not True
        or not isinstance(auth.get('explicit_user_instruction'), str)
        or not auth['explicit_user_instruction'].strip()):
        raise PermissionError('Separate explicit one-shot authorization absent')
    source = auth.get('source_commit')
    if not isinstance(source, str) or len(source) != 40:
        raise PermissionError('Frozen source SHA absent')
    head = git(root, 'rev-parse', 'HEAD')
    parents = git(root, 'rev-list', '--parents', '-n', '1', head).split()[1:]
    if parents != [source]:
        raise PermissionError('HEAD is not the authorization-only direct child')
    allowed = {str(ART/'authorization.json'), str(RECEIPT)}
    diff = set(git(root, 'diff', '--name-only', source, head).splitlines())
    if str(ART/'authorization.json') not in diff or not diff <= allowed:
        raise PermissionError('Authorization-only path guard failed')
    if git(root,'diff','--name-only','HEAD') or git(root,'diff','--cached','--name-only','HEAD'):
        raise PermissionError('Tracked execution source changed after launch')
    manifest = json.loads((root/MANIFEST).read_text())
    if auth.get('source_manifest_sha256') != sha(root/MANIFEST):
        raise PermissionError('Authorization source manifest mismatch')
    if manifest.get('kind') != 'DF_CORRECTION_ET_SOURCE_MANIFEST_V7':
        raise PermissionError('Unexpected manifest namespace')
    for entry in manifest['files']:
        p = root/entry['path']
        if p.stat().st_size != entry['bytes'] or sha(p) != entry['sha256']:
            raise PermissionError('Protected source/input/runtime identity changed: '+entry['path'])
    if auth.get('contract_sha256') != sha(root/ART/'contract_v7.json'):
        raise PermissionError('Authorization contract mismatch')
    return {'source':source, 'authorization':head,
            'authorization_sha256':sha(root/ART/'authorization.json'),
            'contract_sha256':sha(root/ART/'contract_v7.json'),
            'source_manifest_sha256':sha(root/MANIFEST)}


def verify_launch(root):
    info = verify_local(root)
    if git(root, 'status', '--porcelain=v1'):
        raise PermissionError('Execution worktree must be clean')
    branch = git(root, 'branch', '--show-current')
    remote = git(root, 'remote', 'get-url', 'origin')
    if remote != 'git@github.com:HIROMU1015/Partially-Randomized-Trotter.git':
        raise PermissionError('Repository remote not authorized')
    if not branch or branch in ('main', 'master'):
        raise PermissionError('Dedicated execution branch required')
    remote_sha = git(root, 'ls-remote', 'origin', 'refs/heads/'+branch).split()[0]
    if remote_sha != info['authorization']:
        raise PermissionError('Remote execution SHA mismatch')
    if (Path(root)/RESULT).exists() or (Path(root)/RESULT).is_symlink():
        raise PermissionError('Marker already exists; STOP without retry')
    return dict(info, branch=branch, remote_sha=remote_sha)


def verify_child(root, job):
    info = verify_local(root)
    marker = Path(root)/RESULT/'one_shot_marker.json'
    if not marker.exists() or job.get('marker_sha256') != sha(marker):
        raise PermissionError('Authorized marker identity absent')
    saved = json.loads(marker.read_text())
    if any(saved[k] != info[k] for k in ('source','authorization','authorization_sha256','contract_sha256','source_manifest_sha256')):
        raise PermissionError('Marker provenance mismatch')
    if saved.get('runs') != 1 or saved.get('retries') != 0 or saved.get('mandatory_STOP') is not True:
        raise PermissionError('Consumed marker run/STOP conditions mismatch')
    return dict(info, marker_sha256=sha(marker))
