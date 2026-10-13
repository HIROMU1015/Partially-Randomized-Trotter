"""Source-bound D3 gate. Pending authority rejects before input/runtime access."""
import json
import os
from pathlib import Path
import re

from ..rte_development.evidence import digest, safe_relative
from ..rte_development.launch import git

DESIGN = 'artifacts/track_b_d3_preparation/2026-10-13/v1/comparison_design_contract_v1.json'
DESIGN_SHA = '76b3e4d95df62a00f6e66bd9dacd3230a09025167d29de838c917de4722d726e'
CONTRACT = 'artifacts/track_b_d3_source/2026-10-13/v1/execution_contract_v1.json'
COMPLETE = 'D3_GUARANTEE_PRICE_MAP_COMPLETE_AWAITING_GPT_REVIEW'
INCONCLUSIVE = 'D3_TECHNICAL_INCONCLUSIVE'


def read_contract(root):
    root = Path(root)
    c = json.loads((root/CONTRACT).read_text())
    e = c['execution']
    # Authority first. No saved scientific input or numeric package is read.
    a = json.loads((root/e['authorization_path']).read_text())
    if a.get('science_execution_authorized') is not True:
        raise PermissionError('D3 source preparation only; separate explicit one-shot authorization required')
    return c, a


def validate_contract(root, c):
    root = Path(root)
    if digest(root/DESIGN) != DESIGN_SHA:
        raise PermissionError('D3 adopted design identity changed')
    design = json.loads((root/DESIGN).read_text())
    except_metadata = {'kind', 'status', 'next_authorized_work_now'}
    if set(c) != set(design) | {'execution'}:
        raise PermissionError('execution contract field set changed')
    for k,v in design.items():
        if k not in except_metadata and c[k] != v:
            raise PermissionError('scientific design changed: '+k)
    if c['kind'] != 'D3_FIXED_EXECUTION_CONTRACT_V1' or c['status'] != 'SOURCE_PREPARED_SEPARATE_AUTHORIZATION_REQUIRED':
        raise PermissionError('D3 fixed execution contract required')
    e = c['execution']
    expected = {'authorization_path':'artifacts/track_b_d3_source/2026-10-13/v1/authorization.json',
                'source_manifest':'artifacts/track_b_d3_source/2026-10-13/v1/source_manifest_v1.json',
                'protected_ledger':'artifacts/track_b_d3_source/2026-10-13/v1/protected_hashes_v1.json',
                'optional_receipt_path':'docs/tracks/algorithm_codesign/d3_execution_authorization_receipt.md',
                'result_directory':'artifacts/track_b_d3_result/2026-10-13/v1',
                'design_contract_sha256':DESIGN_SHA}
    if e != expected:
        raise PermissionError('D3 execution paths/design binding changed')
    return c


def validate_binding(a, contract_sha, source, head, parents, changed, dirty, e):
    if (not re.fullmatch('[0-9a-f]{40}', source or '')
        or a.get('status') != 'APPROVED_FOR_ONE_D3_RUN'
        or a.get('science_execution_authorized') is not True
        or type(a.get('runs')) is not int or a['runs'] != 1
        or type(a.get('retries')) is not int or a['retries'] != 0
        or a.get('mandatory_STOP') is not True
        or not isinstance(a.get('explicit_execution_instruction'),str)
        or len(a['explicit_execution_instruction'].strip()) < 10):
        raise PermissionError('explicit one-shot approval required')
    if (a.get('source_commit') != source or a.get('contract_sha256') != contract_sha
        or a.get('design_contract_sha256') != DESIGN_SHA or head == source
        or parents != [source] or dirty or not changed
        or e['authorization_path'] not in changed
        or not set(changed) <= {e['authorization_path'],e['optional_receipt_path']}):
        raise PermissionError('D3 clean authorization-only direct child binding')


def check_identities(root, c):
    root = Path(root); e = c['execution']
    manifest = json.loads((root/e['source_manifest']).read_text())
    if manifest.get('focused_tests_passed') is not True:
        raise PermissionError('D3 focused validation incomplete')
    for section in (manifest['critical_sha256'],
                    json.loads((root/e['protected_ledger']).read_text())['sha256']):
        for p, h in section.items():
            safe_relative(p)
            if p.lower().endswith('.npz') or digest(root/p) != h:
                raise PermissionError('D3 protected/source/input identity mismatch: '+p)
    return manifest


def verify_launch(root, source):
    root = Path(root); c,a = read_contract(root)
    if not re.fullmatch('[0-9a-f]{40}', source or ''):
        raise PermissionError('full source SHA required')
    validate_contract(root,c); e = c['execution']
    head = git(root,'rev-parse','HEAD')
    changed = git(root,'-c','core.quotePath=false','diff','--name-only',source,head).splitlines()
    validate_binding(a,digest(root/CONTRACT),source,head,
                     git(root,'show','-s','--format=%P',head).split(),changed,
                     bool(git(root,'status','--porcelain','--untracked-files=all')),e)
    if a.get('inventory_sha256') != c['acquisition']['key_inventory']['sha256']:
        raise PermissionError('authorization inventory identity mismatch')
    if git(root,'remote','get-url','origin') != 'git@github.com:HIROMU1015/Partially-Randomized-Trotter.git':
        raise PermissionError('D3 remote repository mismatch')
    branch = git(root,'branch','--show-current')
    remote = git(root,'ls-remote','--heads','origin','refs/heads/'+branch)
    if not branch or not remote or remote.split()[0] != head:
        raise PermissionError('D3 remote authorization SHA mismatch')
    manifest = check_identities(root,c)
    if (root/e['result_directory']).exists():
        raise FileExistsError('D3 result namespace/marker already exists; no retry')
    return c,a,head,manifest


def execution_identity(root,c,a,head,source):
    root = Path(root)
    return dict(source_commit=source,execution_HEAD=head,contract_sha256=digest(root/CONTRACT),
                authorization_sha256=digest(root/c['execution']['authorization_path']),
                design_contract_sha256=DESIGN_SHA,
                source_manifest_sha256=digest(root/c['execution']['source_manifest']),
                inventory_sha256=c['acquisition']['key_inventory']['sha256'],
                tool_identity_sha256=c['acquisition']['tool_identity']['sha256'])


def check_execution_identity(root,c,identity):
    actual=execution_identity(root,c,{},git(root,'rev-parse','HEAD'),identity['source_commit'])
    if any(identity.get(k)!=v for k,v in actual.items()):
        raise PermissionError('running source/contract/authorization/manifest identity changed')
    marker=Path(root)/c['execution']['result_directory']/'one_shot_consumed.json'
    if digest(marker)!=identity.get('marker_sha256'):
        raise PermissionError('consumed marker changed')
    return True


def verify_job(root, job_path, role):
    """No backend call; immutable ownership receipts reject job restarts."""
    root = Path(root); c,a = read_contract(root); validate_contract(root,c)
    directory = root/c['execution']['result_directory']
    job_path = Path(job_path).resolve()
    if role == 'ledger':
        if job_path != (directory/'ledger_job.json').resolve():
            raise PermissionError('ledger job namespace')
    elif role == 'key':
        if job_path.parent != (directory/'jobs').resolve() or not re.fullmatch(r'key_[0-9]{3}\.json',job_path.name):
            raise PermissionError('key job namespace')
    else: raise ValueError('unknown execution role')
    job = json.loads(job_path.read_text())
    head = git(root,'rev-parse','HEAD'); source = a['source_commit']
    if not re.fullmatch('[0-9a-f]{40}',source or ''):
        raise PermissionError('worker source SHA missing')
    validate_binding(a,digest(root/CONTRACT),source,head,
                     git(root,'show','-s','--format=%P',head).split(),
                     git(root,'-c','core.quotePath=false','diff','--name-only',source,head).splitlines(),
                     False,c['execution'])
    if a.get('inventory_sha256') != c['acquisition']['key_inventory']['sha256']:
        raise PermissionError('worker authorization inventory identity mismatch')
    identity = execution_identity(root,c,a,head,source)
    if any(job.get(k) != v for k,v in identity.items()) or job.get('parent_pid') != os.getppid():
        raise PermissionError('job provenance or parent ownership mismatch')
    manifest=json.loads((root/c['execution']['source_manifest']).read_text())
    for p,h in manifest['critical_sha256'].items():
        if p.endswith('.py') or p in (CONTRACT,DESIGN,c['acquisition']['key_inventory']['path'],
                                     c['acquisition']['tool_identity']['path'],c['execution']['protected_ledger']):
            if digest(root/p)!=h:
                raise PermissionError('worker code/contract/inventory identity mismatch: '+p)
    marker = directory/'one_shot_consumed.json'
    if digest(marker) != job.get('marker_sha256'):
        raise PermissionError('job one-shot marker identity mismatch')
    original = json.loads(marker.read_text())
    if any(original.get(k) != v for k,v in identity.items()):
        raise PermissionError('marker source binding mismatch')
    if (directory/'terminal.json').exists() or (directory/'STOP.json').exists():
        raise PermissionError('terminal job may not restart')
    if role == 'key':
        inventory = json.loads((root/c['acquisition']['key_inventory']['path']).read_text())
        match = [v for v in inventory['keys'] if v['new_key'] == job.get('key')]
        if len(match) != 1 or match[0]['ratio'] != job.get('ratio'):
            raise PermissionError('worker key outside fixed inventory')
        if digest(root/c['acquisition']['key_inventory']['path']) != job['inventory_sha256']:
            raise PermissionError('worker inventory changed')
    return c, job, directory
