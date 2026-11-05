#!/usr/bin/env python3
"""Passive unless --execute; run the fixed integrated scope without retries."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys

sys.dont_write_bytecode=True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--execution-source',type=Path)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--worker',choices=('physical','evaluation','fit'))
    p.add_argument('--worker-config',type=Path)
    a=p.parse_args();root=a.root.resolve()
    sys.path.insert(0,str(root/'src'))
    config=None if a.worker_config is None else json.loads(a.worker_config.read_text())
    if config and config.get('affinity') is not None:os.sched_setaffinity(0,config['affinity'])
    for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'
    os.environ['JAX_ENABLE_X64']='1'
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    label='parent' if config is None else config.get('job_id',config.get('out','worker').replace('/','_'))
    for key,leaf in [('TMPDIR','tmp'),('XDG_CACHE_HOME','cache'),('JAX_COMPILATION_CACHE_DIR','jax'),
                     ('MPLCONFIGDIR','mpl'),('NUMBA_CACHE_DIR','numba')]:
        path=root/'.cache/integrated_execution'/label/leaf
        path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    if a.worker:
        if config is None:raise ValueError('WORKER_CONFIG_REQUIRED')
        def stop(signum,frame):raise RuntimeError('OWN_JOB_DEPENDENCY_STOP')
        signal.signal(signal.SIGTERM,stop)
        if a.worker=='fit':
            from trottertracks.c_integrated_execution import fit_worker
            fit_worker(root,config)
        else:
            from trottertracks.c_integrated_execution_rpc import worker_main
            worker_main(root,a.worker,config)
        return
    from trottertracks.c_integrated_execution import preflight,Run,FAMILY
    source=a.execution_source or root/FAMILY/'execution_preflight/SOURCE.json'
    result=preflight(root,source)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if a.execute:Run(root,source).execute()


if __name__=='__main__':main()
