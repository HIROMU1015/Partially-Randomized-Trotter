"""Default is a metadata-only draft. Execution needs a new explicit H7 grant."""
import argparse
import json
import os
from pathlib import Path
import sys
import types


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--manifest');p.add_argument('--authorization');a=p.parse_args()
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
    from trottertracks.resource_applicability.h7_open_shell.contract import draft,verify_grant
    if not a.execute:
        if a.manifest or a.authorization:p.error('Execution files need --execute')
        print(json.dumps(draft(),indent=2));return
    if not a.manifest or not a.authorization:p.error('New manifest and H7 authorization required')
    m=json.loads(Path(a.manifest).read_bytes());g=json.loads(Path(a.authorization).read_bytes())
    verify_grant(m,g,m.get('SOURCE',''))  # Reject old/unapproved grants before numeric imports.
    from trottertracks.resource_applicability.h7_open_shell.execution import verify_assets
    verify_assets(root,m)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','QISKIT_NUM_PROCS','RAYON_NUM_THREADS'):
        if os.environ.get(key)!='1':raise ValueError('PROCESS_THREAD_ENV:'+key)
    if os.environ.get('QISKIT_PARALLEL')!='FALSE':raise ValueError('QISKIT_INTERNAL_PARALLELISM')
    for key in ('TMPDIR','MPLCONFIGDIR','NUMBA_CACHE_DIR','XDG_CACHE_HOME'):
        value=Path(os.environ.get(key,'/')).resolve()
        if not value.is_relative_to('/home/AbeHiromu') or not value.is_dir():raise ValueError('PRIVATE_PROCESS_CACHE:'+key)
    # Legacy __init__ imports plots and unrelated research. Expose only the
    # pinned package path; no installed venv/package content is modified.
    package=types.ModuleType('trotterlib');package.__path__=[str(root/'src/trotterlib')];sys.modules['trotterlib']=package
    from trottertracks.resource_applicability.h7_open_shell.execution import execute
    execute(root,m,g)


if __name__=='__main__':main()
