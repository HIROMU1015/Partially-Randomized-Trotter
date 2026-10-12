"""Stdlib-only gate. No registered arrays or scientific imports before approval."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ART=Path('artifacts/track_b_e_r3_identity_diagnostic_preparation/2026-10-12')
RESULT=Path('artifacts/track_b_e_r3_identity_diagnostic_result/2026-10-12/v1')
RECEIPT=Path('docs/tracks/algorithm_codesign/e_r3_identity_diagnostic_execution_authorization_v1.md')
MANIFEST=ART/'source_manifest_v1.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def git(root,*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()


def verify_local(root):
    root=Path(root);auth=json.loads((root/ART/'authorization.json').read_bytes())
    if not (auth.get('status')=='APPROVED_FOR_ONE_E_R3_IDENTITY_ONLY_DIAGNOSTIC'
        and auth.get('diagnostic_execution_authorized') is True
        and auth.get('science_execution_authorized') is False and auth.get('runs')==1
        and auth.get('retries')==0 and auth.get('mandatory_STOP') is True
        and isinstance(auth.get('explicit_user_instruction'),str) and auth['explicit_user_instruction'].strip()):
        raise PermissionError('Separate explicit E-r3 identity-only authorization absent')
    source=auth.get('source_commit')
    if not isinstance(source,str) or not re.fullmatch('[0-9a-f]{40}',source):
        raise PermissionError('Frozen diagnostic source SHA absent')
    head=git(root,'rev-parse','HEAD')
    if git(root,'rev-list','--parents','-n','1',head).split()!=[head,source]:
        raise PermissionError('HEAD is not authorization-only direct child')
    diff=set(git(root,'-c','core.quotePath=false','diff','--name-only',source,head).splitlines())
    if str(ART/'authorization.json') not in diff or not diff<={str(ART/'authorization.json'),str(RECEIPT)}:
        raise PermissionError('Authorization-only path guard failed')
    if git(root,'diff','--name-only','HEAD') or git(root,'diff','--cached','--name-only','HEAD'):
        raise PermissionError('Tracked diagnostic source changed')
    for name,p in [('contract_sha256',ART/'contract_v1.json'),('manifest_sha256',MANIFEST)]:
        if auth.get(name)!=sha(root/p):raise PermissionError('Authorization '+name+' mismatch')
    manifest=json.loads((root/MANIFEST).read_bytes())
    for row in manifest['files']:
        p=root/row['path']
        if p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:
            raise PermissionError('Protected diagnostic identity changed: '+row['path'])
    return {'source':source,'authorization':head,'authorization_sha256':sha(root/ART/'authorization.json'),
            'contract_sha256':sha(root/ART/'contract_v1.json'),'manifest_sha256':sha(root/MANIFEST)}


def verify_launch(root):
    info=verify_local(root)
    if git(root,'status','--porcelain=v1'):raise PermissionError('Execution worktree must be clean')
    branch=git(root,'branch','--show-current')
    if not branch or branch in {'main','master'}:raise PermissionError('Dedicated execution branch required')
    if git(root,'remote','get-url','origin')!='git@github.com:HIROMU1015/Partially-Randomized-Trotter.git':
        raise PermissionError('Repository remote not authorized')
    remote=git(root,'ls-remote','origin','refs/heads/'+branch).split()
    if not remote or remote[0]!=info['authorization']:raise PermissionError('Remote SHA mismatch')
    if (Path(root)/RESULT).exists():
        raise PermissionError('Diagnostic result namespace already exists; no retry')
    return dict(info,branch=branch,remote_SHA=remote[0])


def verify_child(root):
    info=verify_local(root);out=Path(root)/RESULT;marker=out/'one_shot_marker.json'
    saved=json.loads(marker.read_bytes())
    if (any(saved.get(k)!=v for k,v in info.items()) or saved.get('runs')!=1
        or saved.get('retries')!=0 or saved.get('mandatory_STOP') is not True):
        raise PermissionError('Diagnostic marker provenance mismatch')
    if (out/'terminal_status.json').exists() or (out/'terminal_status.json.review.json').exists():
        raise PermissionError('Terminal STOP saved')
    branch=git(root,'branch','--show-current')
    if saved.get('branch')!=branch or saved.get('remote_SHA')!=info['authorization']:
        raise PermissionError('Diagnostic marker branch/remote binding mismatch')
    return dict(info,branch=branch,remote_SHA=saved['remote_SHA'],marker_sha256=sha(marker))
