#!/usr/bin/env python3
"""Source-fixed T2 only. Independent monitor; no retry/resume or follow-on job."""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trottertracks import ld2_m12_t2 as T


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan-out',type=Path)
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--source')
    ap.add_argument('--plan',type=Path)
    args=ap.parse_args()
    if args.plan_out:
        T.write(args.plan_out,T.plan(ROOT));return
    if not(args.execute and args.source and args.plan):ap.error('--execute --source SHA --plan FILE required')
    started=time.perf_counter()
    fixed=json.loads(args.plan.read_text())
    request,inputs=T.preflight(ROOT,args.source,fixed)
    public=ROOT/fixed['public'];private=Path(fixed['private'])
    public.mkdir(parents=True,exist_ok=False)
    private.mkdir(parents=True,exist_ok=False)
    T.write(public/'PREFLIGHT.json',dict(source=args.source,plan_sha256=T.sha(args.plan),
        inputs=inputs,resources=T.resources(disk_path=private),
        preflight_seconds=time.perf_counter()-started,mandatory_stop=True))
    process=mp.get_context('fork').Process(target=T.run_worker,
        args=(ROOT,args.source,fixed,request,inputs,public,private))
    process.start()
    T.write(public/'PROCESS.json',dict(pid=process.pid,parent_pid=T.os.getpid()))
    previous=None;stop=None
    try:
        with (public/'RESOURCES.jsonl').open('x') as log:
            while process.is_alive():
                current=T.resources(process.pid,private)
                log.write(json.dumps(current)+'\n');log.flush()
                if T.resource_exhaustion(previous,current):
                    # Headroom measured on this run, not a fixed RAM/CPU cap or PSI rule.
                    stop=dict(reason='AVAILABLE_HEADROOM_BELOW_OBSERVED_NEXT_TWO_INTERVAL_GROWTH',
                              previous=previous,current=current)
                    T.write(public/'RESOURCE_STOP.json',stop)
                    T.os.kill(process.pid,signal.SIGTERM)
                    process.join(10)
                    if process.is_alive():process.kill()
                    break
                previous=current
                process.join(fixed['monitor']['interval_seconds'])
    except BaseException:
        if process.is_alive():
            T.os.kill(process.pid,signal.SIGTERM);process.join(10)
            if process.is_alive():process.kill()
        raise
    process.join()
    partial={}
    for name in ('STRUCTURAL.json','cosine.json','sine.json','TECHNICAL_TIMING.json'):
        path=private/'path'/name
        if path.exists():partial[name]=json.loads(path.read_text())
    success=process.exitcode==0 and (public/'T2_PATH_RESULT.json').exists()
    if not (public/'T2_PATH_RESULT.json').exists():
        T.write(public/'T2_PATH_RESULT.json',dict(status='PARTIAL_STOP',request=request,
            partial=partial,private=str(private),source=args.source,
            mean_compiled_price=None,confidence_interval=None,qualified_expectation=False,
            mandatory_stop=True))
    T.write(public/'RESULT.json',dict(status='T2_COMPLETE' if success else 'STOP',exitcode=process.exitcode,
        source=args.source,partial=partial,resource_stop=stop,private=str(private),
        total_pipeline_wall_seconds=time.perf_counter()-started,
        resources_after=T.resources(disk_path=private),mean_compiled_price=None,
        confidence_interval=None,qualified_expectation=False,mandatory_stop=True))
    print('T2_COMPLETE' if success else 'T2_STOP','exitcode',process.exitcode,flush=True)
    if not success:raise SystemExit(1)


if __name__=='__main__':main()
