"""Canonical preparation identities, shared by planning, dry-check and dispatch."""
from copy import deepcopy
from .rpe_q3_coordinator import digest
from .rpe_preparation_artifacts import input_manifest,artifact_path


def normalize_preparation_request(request, *, source_manifest=None, dependencies=()):
    request=deepcopy(request)
    if request.get('route',{}).get('kind')!='preparation':raise ValueError('Preparation route required')
    preparation=request.get('preparation')
    if not isinstance(preparation,dict)or preparation.get('stage')not in ('signal','compile','audit'):
        raise ValueError('Preparation stage must be signal, compile, or audit')
    optional = preparation.get('optional_price_validation', False)
    if type(optional) is not bool or (optional and preparation['stage'] != 'compile'):
        raise ValueError('Optional price validation must be a compile preparation')
    if dependencies:preparation['dependencies']=list(dict.fromkeys(dependencies))
    preparation.setdefault('dependencies',[])
    preparation.setdefault('inputs',{})
    if not preparation.get('source'):
        if preparation.get('artificial')is not True or not source_manifest:
            raise ValueError('Fixed preparation SOURCE manifest required')
        preparation['source']=deepcopy(source_manifest)
    if not isinstance(preparation['source'].get('commit'),str)or not isinstance(preparation['source'].get('files'),dict):
        raise ValueError('Preparation SOURCE must pin commit and files')
    preparation['inputs']=input_manifest(request)
    for name in preparation.get('outputs',[]):artifact_path(name)
    # The derived ID is excluded from its own seed; all execution/scientific
    # fields, including source, inputs, dependencies and command, remain bound.
    provided=request.pop('request_id',None)
    identity='prep-'+digest(request)[:24]
    if provided is not None and provided!=identity:raise ValueError('Preparation request ID differs from its full definition')
    request['request_id']=identity
    return request


def preparation_job_id(request, *, source_manifest=None, dependencies=()):
    return normalize_preparation_request(request,source_manifest=source_manifest,dependencies=dependencies)['request_id']
