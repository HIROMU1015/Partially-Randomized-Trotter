#!/usr/bin/env python3
"""H6 fixed experiment: passive by default; --execute only after user permission."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode=True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--contract',type=Path)
    p.add_argument('--fit-python',type=Path)
    p.add_argument('--eval-python',type=Path)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--worker',choices=('fit','physical','evaluation'))
    p.add_argument('--worker-config',type=Path)
    a=p.parse_args();root=a.root.resolve();sys.path.insert(0,str(root/'src'))
    if a.dry_run and a.execute:p.error('--dry-run and --execute are exclusive')
    if a.worker:
        if a.execute or a.worker_config is None:p.error('worker needs config, not --execute')
        config=json.loads(a.worker_config.read_text())
        if config.get('affinity') is not None:os.sched_setaffinity(0,config['affinity'])
        threads=1 if a.worker=='fit' else config.get('threads',max(1,len(os.sched_getaffinity(0))//7))
        for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            os.environ[name]=str(threads)
        os.environ['JAX_ENABLE_X64']='1';os.environ['JAX_PLATFORMS']='cpu'
        for name,leaf in (('XDG_CACHE_HOME','cache'),('TMPDIR','tmp'),('JAX_COMPILATION_CACHE_DIR','jax'),('MPLCONFIGDIR','mpl')):
            folder=root/'.cache/h6_execution'/config.get('job_id',config.get('out','worker').replace('/','_'))/leaf
            folder.mkdir(parents=True,exist_ok=True);os.environ[name]=str(folder)
        if a.worker=='fit':
            if config['operation'] not in ('prepare_codec','specs','tensor_scale','start_identity','cell'):
                raise ValueError('FIXED_H6_FIT_OPERATION')
            from trottertracks.c_integrated_execution import fit_worker
            fit_worker(root,config);return
        from trottertracks.c_integrated_execution_rpc import worker_loop
        from trottertracks.c_h6_runtime import H6PhysicalWorker
        from trottertracks.c_h6_parallel import Actor
        worker=(H6PhysicalWorker(root,root/config['out'],config.get('dependency_stop'))
                if a.worker=='physical' else Actor(root,root/config['out']))
        protocol=os.fdopen(os.dup(sys.stdout.fileno()),'w',buffering=1)
        try:
            os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
            worker_loop(worker,output_stream=protocol)
        finally:protocol.close()
        return
    from trottertracks.c_h6_contract import preflight,CONTRACT
    result=preflight(root,a.contract,fit_python=a.fit_python,eval_python=a.eval_python)
    print(json.dumps(result),flush=True)
    if a.execute:
        if a.fit_python is None or a.eval_python is None:p.error('--execute requires both dedicated Python paths')
        from trottertracks.c_h6_runtime import Run
        c=json.loads((root/CONTRACT if a.contract is None else a.contract).read_text())
        c['source_sha256']=result['source_sha256']
        Run(root,c,fit_python=a.fit_python,eval_python=a.eval_python).execute()


if __name__=='__main__':main()
