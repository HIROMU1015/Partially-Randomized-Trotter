"""Zero-compute gate, immutable source binding and job ownership checks."""
import hashlib
import json
import os
from pathlib import Path
import re
from ..rte_development.evidence import digest, safe_relative
from ..rte_development.launch import git

PREFIX='artifacts/track_b_parent_allocation_experiment_source/2026-10-15/v4'
CONTRACT=PREFIX+'/execution_contract_v4.json'
# Set by source-preparation script after the adopted contract is complete.
CONTRACT_SHA='f76a10c6f04972494d5b7607a1a32f425f47c52b5bdb6083f8a983a827056e7d'


def read_contract(root):
    root=Path(root)
    if digest(root/CONTRACT)!=CONTRACT_SHA: raise PermissionError('fixed contract identity changed')
    return json.loads((root/CONTRACT).read_text())


def check_identities(root,c):
    root=Path(root); e=c['execution']
    m=json.loads((root/e['source_manifest']).read_text())
    if m.get('focused_tests_passed') is not True: raise PermissionError('focused validation incomplete')
    protected=json.loads((root/e['protected_ledger']).read_text())
    for section in (m['critical_sha256'],protected['sha256']):
        for path,h in section.items():
            safe_relative(path)
            if digest(root/path)!=h: raise PermissionError('source/input/protected identity changed: '+path)
    historical=c['reuse']['historical_guard_source']
    raw=__import__('subprocess').check_output(['git','show',historical['commit']+':'+historical['path']],cwd=root)
    if len(raw)!=historical['bytes'] or hashlib.sha256(raw).hexdigest()!=historical['sha256']:
        raise PermissionError('historical guard source changed')
    for path,entry in protected.get('historical_source_blobs',{}).items():
        safe_relative(path)
        raw=__import__('subprocess').check_output(['git','show',entry['commit']+':'+path],cwd=root)
        if len(raw)!=entry['bytes'] or hashlib.sha256(raw).hexdigest()!=entry['sha256']:
            raise PermissionError('historical v1 source blob changed: '+path)
    for entry in protected.get('historical_source_versions',[]):
        path=entry['path'];safe_relative(path)
        raw=__import__('subprocess').check_output(['git','show',entry['commit']+':'+path],cwd=root)
        if len(raw)!=entry['bytes'] or hashlib.sha256(raw).hexdigest()!=entry['sha256']:
            raise PermissionError('historical v2 source blob changed: '+path)
    return m


def validate_binding(a, c, source, head, parents, changed, dirty):
    e=c['execution'];allowed={e['authorization_path'],e['optional_receipt_path']}
    if (not re.fullmatch('[0-9a-f]{40}',source or '') or a.get('source_commit')!=source
        or a.get('status')!='APPROVED_FOR_ONE_PARENT_ALLOCATION_RUN'
        or a.get('science_execution_authorized') is not True
        or type(a.get('runs')) is not int or a['runs']!=1
        or type(a.get('retries')) is not int or a['retries']!=0
        or a.get('mandatory_STOP') is not True or a.get('contract_sha256')!=CONTRACT_SHA
        or not isinstance(a.get('explicit_execution_instruction'),str)
        or len(a['explicit_execution_instruction'].strip())<10
        or parents!=[source] or head==source or dirty
        or e['authorization_path'] not in changed or not set(changed)<=allowed):
        raise PermissionError('explicit one-shot clean authorization-only direct child required')


def verify_launch(root,source):
    root=Path(root);c=read_contract(root);e=c['execution']
    a=json.loads((root/e['authorization_path']).read_text())
    # Reject before any runtime import, registered input read or constructor.
    if a.get('science_execution_authorized') is not True:
        raise PermissionError('PREPARATION_ONLY: separate explicit one-shot authorization required')
    head=git(root,'rev-parse','HEAD')
    validate_binding(a,c,source,head,git(root,'show','-s','--format=%P',head).split(),
        git(root,'-c','core.quotePath=false','diff','--name-only',source,head).splitlines(),
        bool(git(root,'status','--porcelain','--untracked-files=all')))
    if git(root,'remote','get-url','origin')!='git@github.com:HIROMU1015/Partially-Randomized-Trotter.git':
        raise PermissionError('repository identity')
    branch=git(root,'branch','--show-current')
    remote=git(root,'ls-remote','--heads','origin','refs/heads/'+branch)
    if not branch or not remote or remote.split()[0]!=head: raise PermissionError('remote A mismatch')
    m=check_identities(root,c)
    if (root/e['result_directory']).exists(): raise FileExistsError('result namespace already exists; no retry')
    return c,a,head,m


def execution_identity(root,c,a,head):
    return dict(source_commit=a['source_commit'],execution_HEAD=head,contract_sha256=CONTRACT_SHA,
        authorization_sha256=digest(Path(root)/c['execution']['authorization_path']),
        source_manifest_sha256=digest(Path(root)/c['execution']['source_manifest']),
        review_sha256=c['review']['sha256'],tool_identity_sha256=c['acquisition']['tool_identity']['sha256'])


def verify_worker(root,job_path,role="key"):
    if role not in ("key","ledger"):raise PermissionError("unknown job role")
    root=Path(root);c=read_contract(root);e=c['execution'];path=Path(job_path).resolve();d=root/e['result_directory']
    if (role=='key' and (path.parent!=(d/'jobs').resolve() or not re.fullmatch(r'key_[0-9]{3}\.json',path.name))) or (role=='ledger' and path!=(d/'ledger_job.json').resolve()):
        raise PermissionError('worker namespace')
    a=json.loads((root/e['authorization_path']).read_text());head=git(root,'rev-parse','HEAD');s=a.get('source_commit')
    if not re.fullmatch('[0-9a-f]{40}',s or ''):raise PermissionError('worker source missing')
    validate_binding(a,c,s,head,git(root,'show','-s','--format=%P',head).split(),
                     git(root,'diff','--name-only',s,head).splitlines(),False)
    check_identities(root,c)
    dirty=git(root,'-c','core.quotePath=false','status','--porcelain','--untracked-files=all').splitlines()
    if any(not line[3:].startswith(e['result_directory']+'/') for line in dirty):
        raise PermissionError('unexpected running worktree modification')
    job=json.loads(path.read_text());identity=execution_identity(root,c,a,head)
    if any(job.get(k)!=v for k,v in identity.items()) or job.get('parent_pid')!=os.getppid():
        raise PermissionError('worker ownership/source identity')
    marker=d/'one_shot_consumed.json';original=json.loads(marker.read_text())
    if digest(marker)!=job.get('marker_sha256') or any(original.get(k)!=v for k,v in identity.items()):
        raise PermissionError('marker identity')
    if any((d/n).exists() for n in ('terminal.json','STOP.json')):raise PermissionError('terminal restart forbidden')
    if role=='ledger':
        if path.with_suffix('.started.json').exists():raise FileExistsError('ledger restart forbidden')
        return c,job,d
    inv=d/'inventory.json'
    if digest(inv)!=job.get('inventory_sha256') or digest(d/'candidate_freeze.json')!=job.get('candidate_freeze_sha256'):
        raise PermissionError('frozen acquisition/candidate identity')
    keys=json.loads(inv.read_text())['keys']
    found=[k for k in keys if k['key']==job.get('key') and k['ratio']==job.get('ratio') and k['acquisition']=='NEW_ONCE']
    if len(found)!=1 or len([k for k in keys if k['acquisition']=='NEW_ONCE'])>c['caps']['new_synthesis_keys']:
        raise PermissionError('worker outside frozen inventory')
    if (path.with_suffix('.started.json')).exists():raise FileExistsError('worker restart forbidden')
    return c,job,d
