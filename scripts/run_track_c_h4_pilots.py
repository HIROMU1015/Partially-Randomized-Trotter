"""Run only the adopted finite Track C H4 pilot scope."""
import os
# Small sector matrices are distributed across workers; avoid nested BLAS pools.
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['QISKIT_PARALLEL']='FALSE'
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from trottertracks.track_c_h4 import *

if __name__=='__main__':
    command=sys.argv[1]
    if (OUT/'STOP.json').exists():
        raise SystemExit('This pilot is stopped. Preserve run01; a newly authorized scope needs a new run.')
    if command=='prepare':prepare_common();print('common preparation saved',flush=True)
    elif command=='signals':
        tasks=signal_tasks()
        with ProcessPoolExecutor(max_workers=min(len(tasks),len(os.sched_getaffinity(0)))) as ex:
            for f in as_completed([ex.submit(signal_task,t) for t in tasks]):print(f.result(),flush=True)
    elif command=='cs-signals':
        tasks=cs_tasks()
        with ProcessPoolExecutor(max_workers=min(len(tasks),len(os.sched_getaffinity(0)))) as ex:
            for f in as_completed([ex.submit(cs_signal_task,t) for t in tasks]):print(f.result(),flush=True)
    elif command=='prices':
        from trottertracks.track_c_h4_price import price_task,price_tasks
        tasks=price_tasks()
        with ProcessPoolExecutor(max_workers=min(len(tasks),len(os.sched_getaffinity(0)))) as ex:
            for f in as_completed([ex.submit(price_task,t) for t in tasks]):print(f.result(),flush=True)
    elif command=='prices-T':
        from trottertracks.track_c_h4_cs_price import cs_price_task,cs_price_tasks
        tasks=cs_price_tasks()
        with ProcessPoolExecutor(max_workers=min(len(tasks),len(os.sched_getaffinity(0)))) as ex:
            for f in as_completed([ex.submit(cs_price_task,t) for t in tasks]):print(f.result(),flush=True)
    elif command=='account':
        from trottertracks.track_c_h4_accounting import aggregate
        print(aggregate(),flush=True)
    elif command=='strong-D':
        from trottertracks.track_c_h4_price import strong_correction_task
        tasks=[t for t in signal_tasks() if t['pilot']=='C-N' and t['ld']==3]
        with ProcessPoolExecutor(max_workers=min(len(tasks),len(os.sched_getaffinity(0)))) as ex:
            for f in as_completed([ex.submit(strong_correction_task,t) for t in tasks]):print(f.result(),flush=True)
    elif command=='native-check':
        from trottertracks.track_c_h4_native_check import native_wrapper_check,native_check_tasks
        tasks=native_check_tasks()
        with ProcessPoolExecutor(max_workers=len(tasks)) as ex:
            for f in as_completed([ex.submit(native_wrapper_check,t) for t in tasks]):print(f.result(),flush=True)
    elif command=='minimal-compiler-check':
        from trottertracks.track_c_h4_native_check import compiler_minimal_repro
        print(compiler_minimal_repro(),flush=True)
