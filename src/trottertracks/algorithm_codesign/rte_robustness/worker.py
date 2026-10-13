"""Called only by an authorized D3 ledger owner. Backend imported after gate."""
import os
from pathlib import Path
import resource
import sys

from .launch import verify_job
from .acquisition import check_saved
from ..rte_development.evidence import atomic_json, digest
from ..rte_development.launch import Guard


def execute(root, job_path):
    root=Path(root);c,job,directory=verify_job(root,job_path,'key')
    path=Path(job_path)
    atomic_json(path.with_suffix('.started.json'),
                dict(**job,pid=os.getpid(),job_path_sha256=digest(path),role='synthesis-key'))
    calls=0;saved=None;guard=Guard(c['caps'],directory,worker=True)
    resource.setrlimit(resource.RLIMIT_CPU,(c['caps']['per_key_CPU_seconds'],
                                           c['caps']['per_key_CPU_seconds']+1))
    def on_call():
        nonlocal calls
        calls+=1
        if calls != 1:raise RuntimeError('second backend call forbidden')
        atomic_json(path.with_suffix('.backend_call_consumed.json'),
                    dict(**job,backend_call_token=1,retries=0,
                         note='exclusive reservation immediately before backend entry; not a completion claim'))
    try:
        with guard:
            if digest(Path(sys.executable).resolve()) != c['acquisition']['runtime_executable_sha256']:
                raise PermissionError('fixed worker runtime required')
            if any(os.environ.get(k) != '1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')):
                raise PermissionError('fixed numeric threads required')
            from ..synthesis_placement.wrapper_launch import verify_runtime
            verify_runtime(root,dict(tool_identity=c['acquisition']['tool_identity']))
            from fractions import Fraction as F
            from ..rte_reallocation.native import Angle
            from ..rte_reallocation.numeric import configure, synthesize
            configure(c['acquisition']['interval_dps'])
            saved=synthesize(Angle('atan',F(job['ratio'])),F(1,10**7),
                             c['acquisition']['options'],on_synthesis=on_call,
                             max_characters=c['caps']['sequence_characters'])
            check_saved(saved,job['ratio'],'1/10000000')
            if calls != 1:raise RuntimeError('backend call coverage')
            guard.check()
        value=dict(**job,status='KEY_COMPLETE',synthesis_calls=calls,saved=saved,
                   retries=0,resource=guard.snapshot())
        code=0
    except BaseException as exc:
        value=dict(**job,status='KEY_TECHNICAL_INCONCLUSIVE',synthesis_calls=calls,saved=saved,
                   reason=type(exc).__name__+': '+str(exc)[:1500],retries=0,
                   prefix_scientific_use=False,resource=guard.snapshot())
        code=1
    atomic_json(path.with_suffix('.result.json'),value,131072)
    return code
