"""Authorized limited compiler-A correctness. Never launches price acquisition."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['QISKIT_PARALLEL']='FALSE'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor,as_completed
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
p=argparse.ArgumentParser();p.add_argument('--common-cache',type=Path,required=True)
p.add_argument('--out',type=Path)
args=p.parse_args()
os.environ['TRACK_C_H4_COMMON_CACHE']=str(args.common_cache.resolve())
from trottertracks import track_c_compiler_a_validation as validation
if args.out is not None:validation.OUT=args.out.resolve()
from trottertracks.track_c_compiler_a_validation import *
if __name__=='__main__':
    if (OUT/'STOP.json').exists():raise SystemExit('Stopped study; preserve this output')
    if OUT.resolve()==args.common_cache.resolve().parent:
        raise SystemExit('Output must be separate from the read-only input cache')
    OUT.mkdir(parents=True,exist_ok=True)
    data=args.common_cache.read_bytes()
    save_json(OUT/'INPUTS.json',dict(base_commit=BASE,common_cache=str(args.common_cache.resolve()),
        common_cache_sha256=hashlib.sha256(data).hexdigest(),read_only_reuse=True,
        profile=PROFILE,new_SCF=0,new_DF_fit=0,new_ground_solves=0,
        new_sampling_batches=0,new_price_cells=0,new_T_synthesis=0))
    print(synthetic_checks(),flush=True)
    tasks=h4_tasks()
    with ProcessPoolExecutor(max_workers=min(len(tasks),len(os.sched_getaffinity(0)))) as pool:
        for f in as_completed([pool.submit(h4_check,t) for t in tasks]):print(f.result(),flush=True)
    print(negative_half_check(),flush=True)
    summary=summarize()
    save_json(OUT/'STOP.json',dict(status='MANDATORY_STOP',limited_correctness=summary['status'],
        all_round_RPE_qualification='UNQUALIFIED',new_price_cells=0,new_T_synthesis=0,
        old_run01_restarted=False,automatic_resume=False))
    print('limited correctness complete; mandatory STOP; all-round RPE remains unqualified',flush=True)
