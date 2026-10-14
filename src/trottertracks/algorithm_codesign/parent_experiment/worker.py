"""A single fresh-angle acquisition; backend and guard loaded after job gate."""
import os
from pathlib import Path
import resource
import sys
from .launch import verify_worker
from ..rte_development.launch import Guard
from ..rte_development.evidence import atomic_json,digest


def execute(root,job_path):
    c,job,d=verify_worker(root,job_path);path=Path(job_path)
    atomic_json(path.with_suffix('.started.json'),dict(**job,pid=os.getpid(),job_sha256=digest(path)))
    calls=0;saved=None;guard=Guard(c['caps'],d,worker=True)
    resource.setrlimit(resource.RLIMIT_CPU,(c['caps']['per_key_CPU_seconds'],c['caps']['per_key_CPU_seconds']+1))
    def on_call():
        nonlocal calls
        calls+=1
        if calls!=1:raise RuntimeError('second backend call forbidden')
        atomic_json(path.with_suffix('.backend_call_consumed.json'),dict(**job,retries=0,call_reservation=1))
    try:
        with guard:
            if digest(Path(sys.executable).resolve())!=c['acquisition']['runtime_executable_sha256']:
                raise PermissionError('worker runtime identity')
            if any(os.environ.get(k)!='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')):
                raise PermissionError('numeric threads')
            from ..synthesis_placement.wrapper_launch import verify_runtime
            verify_runtime(root,dict(tool_identity=c['acquisition']['tool_identity']))
            from fractions import Fraction as F
            from ..rte_reallocation.numeric import synthesize,configure
            from ..rte_reallocation.native import Angle
            configure(c['acquisition']['interval_dps'])
            saved=synthesize(Angle('atan',F(job['ratio'])),F(c['primitive_error']),c['acquisition']['options'],
                             on_synthesis=on_call,max_characters=c['caps']['sequence_characters'])
            if calls!=1:raise RuntimeError('backend entry coverage')
            guard.check()
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>c['caps']['worker_RSS_MiB']*1024:
                raise RuntimeError('peak worker RSS cap; no retry')
        value=dict(**job,status='KEY_COMPLETE',saved=saved,synthesis_calls=calls,retries=0,resource=dict(guard.snapshot(),peak_self_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));code=0
    except BaseException as exc:
        value=dict(**job,status='KEY_TECHNICAL_INCONCLUSIVE',saved=saved,synthesis_calls=calls,retries=0,
                   reason=type(exc).__name__+': '+str(exc)[:2000],prefix_scientific_use=False,resource=dict(guard.snapshot(),peak_self_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));code=1
    atomic_json(path.with_suffix('.result.json'),value,262144)
    return code
