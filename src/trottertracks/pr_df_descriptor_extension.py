"""Source-bound supplied DF descriptor around the existing fixed provider/entry.

No descriptor derivation or numerical leaf is invoked by this module's reader.
An explicit new descriptor changes only the prefix lookup/source binding; all
existing canonical, physical, law, precision and numerical gates remain active.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess

INDEX=dict(commit='7a8bc7f132e166390c943e6b32c92cc6abe1baf4',path='artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/SOURCE_INDEX.json')
DERIVATION=dict(commit='7f4c9a171a1d4caeaeaee6c443217edec761f4c9',path='scripts/h46_new2_core.py',function='matrix_only_tail_coefficients')
BINDING_KEYS=('system','basis','geometry','rank','rank_policy','hamiltonian','state','snapshot')


def read(root,locator):
    value=json.loads(subprocess.check_output(['git','show',locator['commit']+':'+locator['path']],cwd=root))
    for k in locator.get('selector',[]):value=value[k]
    return value


def validate(document,locator,configuration):
    spec=configuration['spec'];prefix=spec['L_D'];binding=spec['binding']
    if document.get('schema')!='PR_DF_TAIL_DESCRIPTOR_V1'or document.get('prefix')!=prefix:
        raise ValueError('Explicit DF descriptor prefix/schema mismatch')
    if document.get('derivation_source')!=DERIVATION or document.get('saved_index')!=INDEX:
        raise ValueError('DF descriptor derivation/input SOURCE mismatch')
    if any(k not in document.get('binding',{})or k not in binding or document['binding'][k]!=binding[k]for k in BINDING_KEYS):
        raise ValueError('DF descriptor H/state/snapshot binding mismatch')
    value=document['descriptor']
    if set(value)!= {'lambda_R','scalar'} or any(type(value[k])not in(int,float)or not math.isfinite(value[k])for k in value)or value['lambda_R']<=0:
        raise ValueError('Finite nonempty DF tail descriptor required')
    if not document.get('execution_source')or not document.get('input_source'):
        raise ValueError('DF descriptor numerical SOURCE/input provenance required')
    return deepcopy(value)


def entry(root,locator,base_class):
    """Extend one entry instance; saved prefix descriptors keep original rules."""
    document=read(root,locator)
    base=base_class(root);prefix=document['prefix']
    if str(prefix)in base.info['prefixes']:
        if document['descriptor']!=base.info['prefixes'][str(prefix)]:raise ValueError('New descriptor differs from saved prefix descriptor')
    class Extended(base_class):
        def _support(self,configuration):
            reasons=super()._support(configuration)
            if configuration['spec']['L_D']!=prefix:return reasons
            expected=validate(document,locator,configuration)
            split=configuration['spec']['split']
            if split['lambda_source']!=locator or (Fraction(split['lambda_R']['numerator'],split['lambda_R']['denominator'])if isinstance(split['lambda_R'],dict)else Fraction(split['lambda_R']))!=Fraction(expected['lambda_R']):
                return reasons+['SUPPLIED_DF_DESCRIPTOR_LAMBDA_OR_SOURCE_MISMATCH']
            if 'scalar'in split and (Fraction(split['scalar']['numerator'],split['scalar']['denominator'])if isinstance(split['scalar'],dict)else Fraction(split['scalar']))!=Fraction(expected['scalar']):
                return reasons+['SUPPLIED_DF_DESCRIPTOR_SCALAR_MISMATCH']
            # The sole original check replaced here assumes the saved INDEX's
            # prefix selector. Its replacement above checks the new fixed blob.
            return [r for r in reasons if r!='TAIL_LAMBDA_OR_SOURCE_NOT_SUPPORTED']
    result=Extended.__new__(Extended);result.__dict__.update(base.__dict__)
    result.info=deepcopy(base.info);result.info['prefixes'][str(prefix)]=deepcopy(document['descriptor'])
    return result


def context(root,config,locator,provider):
    """Existing saved-H loader + existing make_context with one supplied tail."""
    document=read(root,locator);descriptor=validate(document,locator,config)
    split=config['spec']['split']
    if split['lambda_source']!=locator:raise ValueError('Configuration does not bind supplied DF descriptor')
    value=split['lambda_R'];lam=Fraction(value['numerator'],value['denominator'])if isinstance(value,dict)else Fraction(value)
    if lam!=Fraction(descriptor['lambda_R']):raise ValueError('Configuration DF lambda differs')
    # Relative module namespace is the fixed provider's own namespace.
    import importlib
    completion=importlib.import_module(provider.__package__+'.h4_pr_rmse_completion')
    index=read(root,INDEX);saved=index['T20_reference_files']['H4'];raw=Path(saved['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=saved['sha256']:raise ValueError('Original saved context bytes differ')
    original=json.loads(raw)
    if any(original['binding'].get(k)!=config['spec']['binding'].get(k)for k in BINDING_KEYS):raise ValueError('Original saved H binding differs')
    loaded,state_metadata=completion.load_context(root,original['binding'])
    tails={config['spec']['L_D']:dict(descriptor,source=deepcopy(locator))}
    result=provider.make_context(loaded.matrices,loaded.state,loaded.ham.constant,tails,
        binding=config['spec']['binding'],source=dict(saved_index=INDEX,descriptor=locator,descriptor_execution_source=document['execution_source']))
    result.state_metadata=state_metadata
    return result
