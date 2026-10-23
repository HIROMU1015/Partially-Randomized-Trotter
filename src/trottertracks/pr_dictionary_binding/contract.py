"""Stdlib-only contract, schema and one-shot permission checks for FUTURE B.

No grant issuer. Preparation documents have no execution permission.
"""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),
                      allow_nan=False).encode('utf-8')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def schema_validate(spec,value,path='$'):
    """Supported closed subset used by the published B schema; no eval/codegen."""
    if 'const' in spec and (type(value)!=type(spec['const']) or value!=spec['const']):
        raise ValueError(path+': const mismatch')
    if 'enum' in spec and value not in spec['enum']: raise ValueError(path+': enum')
    if 'type' in spec:
        types=spec['type'] if isinstance(spec['type'],list) else [spec['type']]
        matches={'object':type(value) is dict,'array':type(value) is list,
                 'string':type(value) is str,'integer':type(value) is int,
                 'boolean':type(value) is bool,'null':value is None}
        if not any(matches[x] for x in types): raise ValueError(path+': type')
    if isinstance(value,dict):
        if not set(spec.get('required',[])).issubset(value): raise ValueError(path+': missing field')
        properties=spec.get('properties',{})
        if spec.get('additionalProperties') is False and not set(value).issubset(properties):
            raise ValueError(path+': unknown field')
        for key,item in value.items():
            rule=properties.get(key,spec.get('additionalProperties'))
            if isinstance(rule,dict): schema_validate(rule,item,path+'.'+key)
    if isinstance(value,list):
        if len(value)<spec.get('minItems',0): raise ValueError(path+': length')
        if 'items' in spec:
            for i,item in enumerate(value): schema_validate(spec['items'],item,path+'['+str(i)+']')
    if isinstance(value,str):
        if len(value)<spec.get('minLength',0): raise ValueError(path+': empty')
        if 'pattern' in spec and re.fullmatch(spec['pattern'],value) is None:
            raise ValueError(path+': pattern')
    if type(value) is int and value<spec.get('minimum',value): raise ValueError(path+': minimum')


def relative_file(root,name):
    if not isinstance(name,str) or Path(name).is_absolute(): raise ValueError('Relative file required')
    path=(root/name).resolve()
    if not path.is_relative_to(root) or not path.is_file(): raise ValueError('File escapes root')
    return path


def expected_plan():
    # Literal metadata; this function does not construct scientific inputs/events.
    physical='B.INPUT.physical';catalog='B.INPUT.catalog'
    result=[dict(id=physical,family='B.INPUT',requires=[]),
            dict(id=catalog,family='B.INPUT',requires=[physical])]
    result.extend(dict(id='B.ARMS.'+a,family='B.ARMS',requires=[physical]) for a in ('SS','SP','PS','PP'))
    result.extend(dict(id='B.CONTEXTS.'+a,family='B.CONTEXTS',requires=['B.ARMS.'+a]) for a in ('SS','SP','PS','PP'))
    result.append(dict(id='B.CONTEXTS.catalog',family='B.CONTEXTS',requires=[catalog]))
    return result


def verify(packet_dir,source_root,output):
    """FUTURE read-only preflight. No science module is imported here."""
    packet=Path(packet_dir).resolve();root=Path(source_root).resolve();out=Path(output).resolve()
    grant_bytes=(packet/'authorization.json').read_bytes();grant=json.loads(grant_bytes)
    if (grant.get('stage'),grant.get('authority'),grant.get('status'))!=(
            'B','USER_EXPLICIT_B','APPROVED_BOUND_ONE_SHOT'):
        raise PermissionError('B science explicitly approved and bound packet required')
    if grant.get('retry_allowed') is not False or grant.get('mandatory_stop') is not True:
        raise PermissionError('One-shot/STOP required')
    contract_bytes=(packet/'contract.json').read_bytes();contract=json.loads(contract_bytes)
    approval=(packet/'user_approval.txt').read_bytes()
    if grant.get('contract_sha256')!=sha(contract_bytes) or grant.get('approval_sha256')!=sha(approval):
        raise PermissionError('Contract/approval binding mismatch')
    if contract.get('stage')!='B' or contract.get('status')!='USER_APPROVED_B_FINAL':
        raise PermissionError('Preparation is not authorization')
    if contract.get('call_plan')!=expected_plan(): raise PermissionError('B logical cases/DAG changed')
    if contract.get('logical_counts')!={'B.INPUT':2,'B.ARMS':4,'B.CONTEXTS':5}:
        raise PermissionError('B call counts changed')
    if contract.get('scope')!='NO_BUILD_NO_EVENTS_F1_LAWS_ONLY_PRICE_UNBOUND':
        raise PermissionError('B scope changed')
    if not isinstance(contract.get('source_files'),dict) or not contract['source_files']:
        raise PermissionError('Missing SOURCE closure')
    source_sha=sha(canonical(contract['source_files']))
    if grant.get('source_closure_sha256')!=source_sha: raise PermissionError('SOURCE closure mismatch')
    for name,expected in contract['source_files'].items():
        if sha(relative_file(root,name).read_bytes())!=expected: raise PermissionError('SOURCE bytes mismatch')
    mandatory={'src/trotterlib/rte.py','src/trottertracks/pr_dictionary_validation/__init__.py',
        'src/trottertracks/pr_dictionary_validation/fixture.py','src/trottertracks/pr_dictionary_validation/algebra.py',
        'src/trottertracks/pr_dictionary_validation/prototype.py','src/trottertracks/pr_dictionary_validation/cases.py',
        'src/trottertracks/pr_dictionary_binding/__init__.py','src/trottertracks/pr_dictionary_binding/contract.py',
        'src/trottertracks/pr_dictionary_binding/engine.py','scripts/run_pr_dictionary_binding.py'}
    if set(contract['source_files'])!=mandatory: raise PermissionError('B minimal source set mismatch')
    source_commit=contract.get('source_commit')
    if not isinstance(source_commit,str) or re.fullmatch('[0-9a-f]{40}',source_commit) is None:
        raise PermissionError('SOURCE commit unbound')
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    if actual!=source_commit: raise PermissionError('SOURCE HEAD mismatch')
    for name,expected in contract.get('recipe_policy_sources',{}).items():
        if sha(relative_file(root,name).read_bytes())!=expected:raise PermissionError('Recipe SOURCE policy mismatch')
    if not contract.get('recipe_policy_sources'):raise PermissionError('Recipe policy identity unbound')
    for doc in ('B_INPUT_SPECIFICATION.md','B_CALL_INVENTORY.json','B_SCHEMAS.json'):
        name='docs/research/pr_algorithm_improvement_phase2d_binding_preparation_20261023/'+doc
        if contract.get('specification_files',{}).get(name)!=sha(relative_file(root,name).read_bytes()):
            raise PermissionError('Specification/schema/inventory identity mismatch')
    env_bytes=(packet/'environment.json').read_bytes();env=json.loads(env_bytes)
    if grant.get('environment_sha256')!=sha(env_bytes): raise PermissionError('Environment digest mismatch')
    if env.get('status')!='DEDICATED_FROZEN_ENVIRONMENT' or sys.prefix==sys.base_prefix or not sys.flags.isolated:
        raise PermissionError('Dedicated frozen venv required')
    if env.get('python_patch')!='3.11.9' or platform.python_version()!=env.get('python_patch') or env.get('numpy')!='1.26.4':
        raise PermissionError('Python/NumPy environment mismatch')
    if importlib.metadata.version('numpy')!=env['numpy']:
        raise PermissionError('Installed NumPy metadata mismatch')
    if not env.get('wheel_sha256') or not env.get('package_file_hashes') or not env.get('native_runtime_hashes'):
        raise PermissionError('Package/native closure missing')
    prefix=Path(sys.prefix).resolve()
    cfg=(prefix/'pyvenv.cfg').read_text().lower().replace(' ','')
    if 'include-system-site-packages=false' not in cfg:raise PermissionError('System packages must be excluded')
    for name,expected in env['package_file_hashes'].items():
        if sha(relative_file(prefix,name).read_bytes())!=expected:
            raise PermissionError('Environment package-file mismatch')
    for name,expected in env['native_runtime_hashes'].items():
        p=Path(name)
        if not p.is_absolute() or not p.is_file() or sha(p.read_bytes())!=expected:
            raise PermissionError('Native runtime mismatch')
    if env.get('dedicated_prefix_identity')!=sha(str(prefix).encode()):
        raise PermissionError('Wrong private environment prefix')
    run_id=grant.get('approved_run_id')
    if not isinstance(run_id,str) or re.fullmatch('[A-Za-z0-9_-]+',run_id) is None:
        raise PermissionError('No approved run ID')
    expected_out=root/'artifacts/pr_algorithm_improvement_phase2d_binding/2026-10-23'/run_id
    if out!=expected_out.resolve() or str(out)!=grant.get('fresh_output_namespace') or out.exists():
        raise PermissionError('Fresh registered namespace required')
    if not out.parent.is_dir():raise PermissionError('Approved output parent must already exist')
    marker=packet/('B-consumed-'+sha(grant_bytes))
    if marker.exists(): raise PermissionError('Used authorization; retry forbidden')
    schema_name='docs/research/pr_algorithm_improvement_phase2d_binding_preparation_20261023/B_SCHEMAS.json'
    schemas=json.loads(relative_file(root,schema_name).read_bytes())
    schema_validate(schemas['environment'],env)
    return dict(packet=packet,root=root,output=out,contract=contract,environment=env,schemas=schemas,
                marker=marker,grant_digest=sha(grant_bytes),source_closure_sha256=source_sha,
                environment_sha256=sha(env_bytes),run_id=run_id)


def consume(verified):
    """FUTURE only. No actual marker/output is created during B-PRE."""
    fd=os.open(verified['marker'],os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'w') as stream:
        stream.write(verified['grant_digest']+'\n');stream.flush();os.fsync(stream.fileno())
    # Parent directory must be prepared in the approved dedicated area.
    verified['output'].mkdir(parents=False,exist_ok=False)
