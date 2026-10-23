"""Standard-library-only fail-closed gate. No science imports or issuance API.

This checks a FUTURE externally approved, fully bound F1 packet. It cannot turn
I/P0 documents into permission. No actual packet/marker is supplied in stage I.
"""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
import re
from pathlib import Path


def file_sha(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def _under(root,path):
    p=(root/path).resolve()
    if not p.is_relative_to(root) or not p.is_file(): raise ValueError('File outside frozen root')
    return p


def verify_packet(packet_dir,source_root,output):
    """Read-only preflight; refuse unbound/default/source-only authorizations."""
    directory=Path(packet_dir).resolve(); root=Path(source_root).resolve()
    out=Path(output).resolve()
    grant_bytes=(directory/'authorization.json').read_bytes()
    grant=json.loads(grant_bytes)
    if (grant.get('stage'),grant.get('authority'),grant.get('status'))!=(
            'F1','USER_EXPLICIT_F1','APPROVED_BOUND_ONE_SHOT'):
        raise PermissionError('Explicit, bound F1 authorization required; I/P0 is insufficient')
    if grant.get('retry_allowed') is not False or grant.get('mandatory_stop') is not True:
        raise PermissionError('One-shot/no-retry/STOP required')
    contract_bytes=(directory/'contract.json').read_bytes()
    inventory_bytes=(directory/'case_inventory.json').read_bytes()
    approval=(directory/'user_approval.txt').read_bytes()
    for name,data in [('contract_sha256',contract_bytes),('inventory_sha256',inventory_bytes),
                      ('approval_sha256',approval)]:
        if grant.get(name)!=file_sha(data): raise PermissionError('Authorization binding mismatch: '+name)
    contract=json.loads(contract_bytes); inventory=json.loads(inventory_bytes)
    if not isinstance(contract.get('input_files'),dict) or not contract['input_files']:
        raise PermissionError('Missing bound inputs')
    if contract.get('stage')!='F1' or contract.get('context')!={'T':'1/5','q':2,'r':2,'K':2}:
        raise PermissionError('F1 context changed')
    if contract.get('contract_status')!='USER_APPROVED_F1_FINAL': raise PermissionError('Draft contract')
    if contract.get('scientific_call_counts_fully_resolved') is not True:
        raise PermissionError('Unresolved scientific call inventory')
    if contract.get('source_closure_sha256')!=file_sha(_canonical(contract.get('source_files'))):
        raise PermissionError('Contract SOURCE digest mismatch')
    if grant.get('source_closure_sha256')!=file_sha(_canonical(contract.get('source_files'))):
        raise PermissionError('Source closure binding mismatch')
    sources=contract.get('source_files')
    if not isinstance(sources,dict) or not sources: raise PermissionError('Missing SOURCE closure')
    required_new={str(p.relative_to(root)) for p in
        (root/'src/trottertracks/pr_dictionary_validation').glob('*.py')}
    required_new.add('scripts/run_pr_dictionary_f1.py')
    if not required_new.issubset(sources): raise PermissionError('New SOURCE omitted from closure')
    for path,expected in sources.items():
        if file_sha(_under(root,path).read_bytes())!=expected: raise PermissionError('SOURCE mismatch')
    qualification=contract.get('interval_provider_qualification',{})
    if qualification:
        provider='src/trottertracks/pr_dictionary_validation/intervals.py'
        if qualification.get('source_sha256')!=sources.get(provider):
            raise PermissionError('Interval provider SOURCE mismatch')
        if qualification.get('environment_lock_sha256')!=contract.get('environment_lock_sha256'):
            raise PermissionError('Interval qualification environment mismatch')
    inputs={}
    for path,expected in contract.get('input_files',{}).items():
        data=_under(directory,path).read_bytes()
        if file_sha(data)!=expected: raise PermissionError('Input hash mismatch')
        inputs[path]=data
    for key in ('fixture_input_sha256','catalog_input_sha256','environment_lock_sha256'):
        value=contract.get(key)
        if not isinstance(value,str) or len(value)!=64 or value not in contract['input_files'].values():
            raise PermissionError('Unbound '+key)
    for distribution,version in contract.get('dependency_versions',{}).items():
        if importlib.metadata.version(distribution)!=version: raise PermissionError('Environment version mismatch')
    if not contract.get('dependency_versions') or not contract.get('environment_file_hashes'):
        raise PermissionError('Environment closure missing')
    for path,expected in contract['environment_file_hashes'].items():
        p=Path(path)
        if not p.is_absolute() or not p.is_file() or file_sha(p.read_bytes())!=expected:
            raise PermissionError('Runtime environment closure mismatch')
    if not isinstance(inventory.get('cases'),list) or not inventory['cases']:
        raise PermissionError('Missing registered case IDs')
    cases=inventory['cases']; ids=[c['id'] for c in cases]
    if any(not isinstance(x,str) or re.fullmatch(r'[A-Za-z0-9_-]+',x) is None for x in ids):
        raise PermissionError('Unsafe registered case ID')
    if len(set(ids))!=len(ids): raise PermissionError('Duplicate registered case ID')
    allowed={'H_ID','REFERENCE','LAW','COMPONENT','MOTIF','WRAPPER','COMPILE','MEAN',
             'CATALOG_EVENT','CATALOG_MEAN','PRICE_DIRECT','PRICE_DP','PRICE_COMPARE',
             'ARM_PRICE','SELECTOR','EXHAUSTIVE'}
    seen=set(); counts={}
    for c in cases:
        if c.get('handler') not in allowed or not set(c.get('requires',[])).issubset(seen):
            raise PermissionError('Unknown handler/invalid dependency order')
        if c.get('call_group') not in contract.get('logical_case_counts',{}):
            raise PermissionError('Unregistered call group')
        counts[c['call_group']]=counts.get(c['call_group'],0)+1; seen.add(c['id'])
    if counts!=contract.get('logical_case_counts'): raise PermissionError('Case count mismatch')
    bundle=json.loads(inputs[contract['fixture_bundle_path']])
    lengths=[len(bundle['arms'][arm]) for arm in ('SS','SP','PS','PP')]
    expected={'F1.H':4,'F1.LAW':sum(n+n**3 for n in lengths),
        'F1.COMPONENT':3*sum(lengths),'F1.MOTIF':96,'F1.WRAPPER':64,'F1.COMPILE':64,
        'F1.MEAN':4,'F1.CATALOG.EVENT':136,'F1.CATALOG.MEAN':2,'F1.PRICE.ENUM':68,
        'F1.PRICE.DP':1,'F1.PRICE.COMPARE':1,'F1.PRICE.ARMS':4,'F1.SELECTOR':1,
        'F1.EXHAUSTIVE':1,'F1.REFERENCE':2*(4+32+68+sum(lengths))}
    if counts!=expected: raise PermissionError('P0 logical counts changed')
    groups={'H_ID':'F1.H','REFERENCE':'F1.REFERENCE','LAW':'F1.LAW',
        'COMPONENT':'F1.COMPONENT','MOTIF':'F1.MOTIF','WRAPPER':'F1.WRAPPER',
        'COMPILE':'F1.COMPILE','MEAN':'F1.MEAN','CATALOG_EVENT':'F1.CATALOG.EVENT',
        'CATALOG_MEAN':'F1.CATALOG.MEAN','PRICE_DIRECT':'F1.PRICE.ENUM',
        'PRICE_DP':'F1.PRICE.DP','PRICE_COMPARE':'F1.PRICE.COMPARE',
        'ARM_PRICE':'F1.PRICE.ARMS','SELECTOR':'F1.SELECTOR','EXHAUSTIVE':'F1.EXHAUSTIVE'}
    for case in cases:
        if groups[case['handler']]!=case['call_group']:
            raise PermissionError('Handler/call group mismatch')
        if case['handler']=='REFERENCE' and case.get('digits') not in (90,128):
            raise PermissionError('Unregistered reference precision')

    if str(out)!=grant.get('fresh_output_namespace') or out.exists():
        raise PermissionError('Fresh authorized output required')
    marker=directory/('consumed-'+file_sha(grant_bytes))
    if marker.exists(): raise PermissionError('Authorization already consumed; no retry')
    return dict(contract=contract,inventory=inventory,inputs=inputs,output=out,
                consumption_path=marker,authorization_digest=file_sha(grant_bytes),source_root=root)


def claim_packet(verified):
    """FUTURE execution only. Atomic consumption; failures never delete marker."""
    marker=verified['consumption_path']
    fd=os.open(marker,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'w') as f:
        f.write(verified['authorization_digest']+'\n'); f.flush(); os.fsync(f.fileno())
    # Atomic mkdir rejects concurrent/symlink output. Failed mkdir still consumes.
    verified['output'].mkdir(parents=False,exist_ok=False)
