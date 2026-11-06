"""Byte-only H6 execution preflight. Scientific permission is supplied in chat.

SOURCE/input/config checks preserve numerical scope; no approval authority,
grant, saved permission text or authentication system is implemented.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

CONTRACT='docs/research/research_c_h6_contract_unapproved_20261106.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configuration(c):
    return {k:v for k,v in c.items() if k!='execution_SOURCE_sha256'}


def preflight(root,path=None,*,fit_python=None,eval_python=None):
    from .c_h6_validation_preparation import specification,workload
    root=Path(root).resolve();path=root/CONTRACT if path is None else Path(path)
    c=json.loads(path.read_text())
    if (c['physical_proposal']!=specification() or c['workload_ceiling']!=workload()
            or c['execution_ready'] is not True or c['scientific_execution_authorized'] is not False
            or c['execution_entry']!='scripts/run_research_c_h6.py'
            or c['retry']!=0 or c['resume']!=0 or c['mandatory_stop'] is not True):
        raise ValueError('H6_FIXED_EXECUTION_SCOPE')
    def inside(name):
        p=(root/name).resolve()
        if not p.is_relative_to(root):raise ValueError('REPOSITORY_RELATIVE_PATH')
        return p
    source=inside(c['execution_SOURCE'])
    if sha(source)!=c['execution_SOURCE_sha256']:raise ValueError('H6_EXECUTION_SOURCE_DIGEST')
    manifest=json.loads(source.read_text())
    if configuration(c)!=manifest['contract']:raise ValueError('H6_FIXED_CONTRACT_CHANGED')
    for name,want in manifest['files'].items():
        if sha(inside(name))!=want:raise ValueError('SOURCE:'+name)
    for value in (c['physical_generation'],c['neutral']):
        name=value.get('input_path',value.get('path'));want=value.get('input_sha256',value.get('sha256'))
        if sha(inside(name))!=want:raise ValueError('FIXED_INPUT:'+name)
    if inside(c['fresh_output']).exists():raise ValueError('FRESH_NO_RETRY_RESUME')
    checked={}
    for role,python in (('fit',fit_python),('evaluation',eval_python)):
        if python is None:continue
        expected=c['environment']['package_pins'][role]
        code='import importlib.metadata as m,json,sys;print(json.dumps(dict(python=sys.version.split()[0],packages={p:m.version(p) for p in '+repr(list(expected))+'})))'
        observed=json.loads(subprocess.check_output([str(python),'-B','-c',code],text=True))
        actual=observed['packages']
        if observed['python']!=c['environment']['python_version']:raise ValueError('PYTHON_VERSION:'+role)
        if actual!=expected:raise ValueError('ENVIRONMENT:'+role)
        checked[role]=actual
    return dict(status='H6_PASSIVE_EXECUTION_PREFLIGHT_PASS',execution_ready=True,
        scientific_execution_authorized=False,scientific_calls=0,molecular_arrays_loaded=0,
        source_sha256=sha(source),source_files=len(manifest['files']),environment_checks=checked,
        fresh_output=c['fresh_output'],maximum_logical_units=218350,
        approval_caveat='technical readiness is not scientific execution permission',mandatory_stop=True)
