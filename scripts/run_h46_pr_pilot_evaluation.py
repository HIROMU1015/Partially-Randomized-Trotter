"""Two independent saved-only evaluations; never draw or compile trajectories."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from pathlib import Path
import h46_pr_pilot_evaluation as e

def run(configs,controller):
    cfgs=[e.read(p) for p in configs]
    e.require(len(cfgs)==2 and {v['stage'] for v in cfgs}==set(e.STAGES),'TWO_INDEPENDENT_APPROVED_EVALUATIONS')
    for cfg in cfgs:e.authorize(cfg)
    root=Path(controller);root.mkdir(exist_ok=False)
    e.save(root/'ENTERED.json',dict(SOURCE=cfgs[0]['SOURCE'],stages=list(e.STAGES),retry=False,new_sampling=0,new_compile=0))
    results=[];errors={}
    context=mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=min(len(cfgs),len(os.sched_getaffinity(0))),mp_context=context) as pool:
        futures=[(cfg,pool.submit(e.execute_stage,cfg)) for cfg in cfgs];processes=list(pool._processes.values())
        for cfg,future in futures:
            try:results.append(future.result())
            except Exception as exc:errors[cfg['stage']]=str(exc)
    workers=[dict(pid=p.pid,exit=p.exitcode,reaped=not p.is_alive()) for p in processes]
    status='STOP' if errors or not all(p['exit']==0 and p['reaped'] for p in workers) else 'PILOT_EVALUATION_COMPLETE_MANDATORY_STOP'
    e.save(root/'completion.json',dict(status=status,SOURCE=cfgs[0]['SOURCE'],results=results,errors=errors,workers=workers,
      H2_evaluation_reused_not_recalculated=True,new_sampling=0,new_compile=0,new_signal=0,new_shot_calculation=0,
      ranking=False,n_E_adopted=False,AD03='DEFERRED',next_phase_started=False,mandatory_STOP=True))
    return 0 if status!='STOP' else 1

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',action='append');p.add_argument('--controller')
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:
        print(__import__('json').dumps(dict(status='SCOPE_ONLY',stages=list(e.STAGES),H2='REUSE_ONLY',new_sampling=0,new_compile=0)));return 0
    try:
        e.require(a.config and a.controller,'EXPLICIT_CONFIG_AND_CONTROLLER');return run(a.config,a.controller)
    except Exception as exc:print(__import__('json').dumps(dict(status='REFUSED_OR_STOP',reason=str(exc),retry=False)));return 2
if __name__=='__main__':raise SystemExit(main())
