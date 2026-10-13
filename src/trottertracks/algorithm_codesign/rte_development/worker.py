"""One frozen synthesis key; authority and job binding precede backend import."""
import json
import os
from pathlib import Path
import resource
import time

from .evidence import atomic_json, digest
from .launch import Guard, git, verify_job, validate_binding


def run_worker(root, job_path):
    root=Path(root); job_path=Path(job_path)
    cpath=root/'artifacts/track_b_rte_development_source/2026-10-13/v1/contract_v1.json'
    c=json.loads(cpath.read_text()); auth=json.loads((root/c['authorization_path']).read_text())
    if auth.get('science_execution_authorized') is not True:
        raise PermissionError('worker refuses pending authorization before backend import')
    directory=root/c['result_directory']
    if job_path.parent.resolve() != (directory/'jobs').resolve():
        raise PermissionError('job outside fresh result namespace')
    job=json.loads(job_path.read_text()); inventory_path=directory/'inventory.json'
    inventory=json.loads(inventory_path.read_text())
    bound={'sha256':digest(inventory_path),'keys':inventory['keys']}
    head=git(root,'rev-parse','HEAD')
    validate_binding(auth,digest(cpath),auth.get('source_commit'),head,
        git(root,'show','-s','--format=%P',head).split(),
        git(root,'-c','core.quotePath=false','diff','--name-only',auth.get('source_commit'),head).splitlines(),
        False,[c['authorization_path'],c['optional_receipt_path']])
    if (git(root,'show','-s','--format=%P',head).split()!=[auth.get('source_commit')]
            or auth.get('contract_sha256')!=digest(cpath)
            or auth.get('status')!='APPROVED_FOR_ONE_RTE_DEVELOPMENT_D1_RUN'
            or auth.get('runs')!=1 or auth.get('retries')!=0 or auth.get('mandatory_STOP') is not True):
        raise PermissionError('worker execution-source mismatch')
    verify_job(job,bound,digest(directory/'one_shot_consumed.json'),head,
               auth.get('source_commit'),digest(root/c['authorization_path']),digest(cpath))
    manifest=json.loads((root/c['source_manifest']).read_text())
    for path,h in manifest['critical_sha256'].items():
        if digest(root/path)!=h:raise PermissionError('worker critical identity mismatch')
    result_path=job_path.with_suffix('.result.json')
    atomic_json(job_path.with_suffix('.started.json'),{'key':job['key'],
        'inventory_sha256':bound['sha256'],'source_commit':auth['source_commit'],
        'execution_HEAD':head,'marker_sha256':job['marker_sha256'],'runs':1,'retries':0})
    limit=int(c['caps']['per_key_CPU_seconds'])
    resource.setrlimit(resource.RLIMIT_CPU,(limit,limit+1))
    start=time.process_time(); wall=time.monotonic()
    try:
        with Guard(c['caps'],directory,worker=True) as guard:
            from ..synthesis_placement.wrapper_launch import verify_runtime
            verify_runtime(root,c)
            import sys
            if digest(Path(sys.executable).resolve())!=c['runtime_executable_sha256']:
                raise PermissionError('worker runtime executable identity changed')
            from fractions import Fraction as F
            from ..rte_reallocation.native import Angle
            from ..rte_reallocation.numeric import configure,synthesize
            configure(c['interval_dps'])
            saved=synthesize(Angle('atan',F(job['ratio'])),c['primitive_error'],
                              c['synthesizer_options'],max_characters=c['caps']['sequence_characters'])
            guard.check()
            if saved['error_pass'] is not True:raise ArithmeticError('strict primitive error failed')
        receipt={'status':'KEY_COMPLETE','key':job['key'],'ratio':job['ratio'],
            'inventory_sha256':bound['sha256'],'source_commit':auth['source_commit'],
            'execution_HEAD':head,'marker_sha256':job['marker_sha256'],
            'contract_sha256':digest(cpath),'authorization_sha256':digest(root/c['authorization_path']),
            'CPU_seconds':time.process_time()-start,'wall_seconds':time.monotonic()-wall,
            'synthesis_calls':1,'retries':0,'saved':saved}
        atomic_json(result_path,receipt,131072)
        return 0
    except BaseException as exc:
        atomic_json(result_path,{'status':'KEY_TECHNICAL_INCONCLUSIVE','key':job['key'],
            'inventory_sha256':bound['sha256'],'reason':type(exc).__name__+': '+str(exc)[:1000],
            'CPU_seconds':time.process_time()-start,'retries':0},16384)
        return 1
