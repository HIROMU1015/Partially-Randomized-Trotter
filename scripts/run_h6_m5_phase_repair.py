"""One productive worker plus independent parent observation and wait4 witness."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time
from types import FunctionType
import h6_m5_phase_repair_contract as c
import run_h6_m5_saved_phase as observed

supervise=FunctionType(observed.supervise.__code__,dict(observed.__dict__,c=c,__file__=__file__),
                      'supervise',observed.supervise.__defaults__,observed.supervise.__closure__)


def execute(cfg, *, target=None):
    c.authorize(cfg); c.require(sys.executable==cfg['runtime']['python'],'FIXED_REPAIR_PYTHON')
    root=Path(cfg['run_root']); c.require(not root.exists(),'UNUSED_REPAIR_ROOT')
    resource=observed.fresh_resources(cfg); resource['workers']=1; resource['productive_jobs']=1
    root.mkdir(exist_ok=False);Path(cfg['output']).mkdir();Path(cfg['ledger']).mkdir()
    c.save(cfg['marker'],dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant=cfg['grant']))
    c.save(root/'fresh_gate.json',resource)
    ctx=mp.get_context('spawn');stop=ctx.Event();handlers={};signals=[];error=None;process=None
    started=time.monotonic();envkeys=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','QISKIT_PARALLEL');prior={k:os.environ.get(k) for k in envkeys}
    def interrupt(signum,frame):signals.append(signum);stop.set()
    try:
        for key in envkeys:os.environ[key]='FALSE' if key=='QISKIT_PARALLEL' else '1'
        for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,interrupt)
        if target is None:
            from h6_m5_phase_repair_native import worker
            target=worker
        process=ctx.Process(target=target,args=(cfg,stop));process.start()
        c.save(root/'worker_start.json',dict(pid=process.pid,launcher_pid=os.getpid(),SOURCE=cfg['SOURCE']))
        with (root/'resource_observation.jsonl').open('x') as journal:
            while process.is_alive():
                journal.write(json.dumps(dict(elapsed=time.monotonic()-started,worker_pid=process.pid,
                    signals=signals,resources=observed.observe()))+'\n');journal.flush()
                process.join(timeout=1)
        process.join();c.require(process.exitcode==0,'OWN_REPAIR_WORKER_NATIVE_EXIT')
        c.require(not signals,'OWN_REPAIR_SIGNAL')
        c.require(not (Path(cfg['output'])/'STOP.json').exists(),'REPAIR_SCIENTIFIC_OR_TECHNICAL_STOP')
        report=c.read(Path(cfg['output'])/'report.json')
        c.require(report['status']=='REPAIR_COSINE_SINGLE_STATE_PASS_MANDATORY_STOP' and
            report['SOURCE']==cfg['SOURCE'] and report['plan_sha256']==cfg['plan_sha256'], 'REPAIR_REPORT_IDENTITY')
        c.require({p.stem for p in Path(cfg['ledger']).glob('*.json')}==set(c.STAGES),'EXACT_FIVE_STAGE_LEDGER')
        c.require(len(report['checks'])==3 and all(r['PASS'] and r['strictly_below']==1e-9 and
            r['raw_error']<1e-9 for r in report['checks']),'EXACT_THREE_RAW_GATES')
        for row in report['artifacts'].values():
            p=Path(cfg['output'])/row['file'];c.require(c.sha(p)==row['sha256'] and p.stat().st_size==row['bytes'],'NEW_ARTIFACT_SHA')
    except BaseException as exc:
        error=str(exc);stop.set()
        c.save(root/'STOP.json',dict(status='STOP',reason=error,retry=False,old_STOP_remains_STOP=True))
    finally:
        if process is not None:process.join()
        for sig,h in handlers.items():signal.signal(sig,h)
        for key,value in prior.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
    try:c.verify_seals(cfg);c.consumed(cfg)
    except Exception as exc:error=error or str(exc)
    report_path=Path(cfg['output'])/'report.json'
    c.save(root/'completion_witness.json',dict(status='STOP' if error else 'REPAIR_PILOT_PASS_MANDATORY_STOP',reason=error,
        SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],pipeline_identity=cfg['pipeline_identity'],
        worker_pid=None if process is None else process.pid,worker_exit=None if process is None else process.exitcode,
        worker_reaped=process is not None and not process.is_alive(),signals=signals,
        stage_entries=[stage for stage in c.STAGES if (Path(cfg['ledger'])/(stage+'.json')).exists()],
        report_sha256=c.sha(report_path) if report_path.exists() else None,wall_seconds=time.monotonic()-started,
        old_STOP_remains_STOP=True,accepted_cost_rows=0,sine=0,retry=False,next_phase_started=False,mandatory_STOP=True))
    return 1 if error else 0


def main():
    parser=argparse.ArgumentParser();m=parser.add_mutually_exclusive_group();m.add_argument('--execute',action='store_true');m.add_argument('--supervise',action='store_true');parser.add_argument('--config');args=parser.parse_args()
    if not args.execute and not args.supervise:
        print(json.dumps(dict(status='SCOPE_ONLY',scope=c.scope(),science_started=False)));return 0
    try:
        c.require(args.config,'NEW_REPAIR_CONFIG_REQUIRED');cfg=c.read(args.config)
        if args.supervise:return supervise(cfg,args.config)
        c.authorize(cfg)
        c.require(c.read(cfg['supervisor']['entered'])==dict(SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],grant_id=cfg['grant']['grant_id'],retry=False),'REPAIR_SUPERVISOR_REQUIRED')
        return execute(cfg)
    except Exception as exc:
        print(json.dumps(dict(status='REFUSED',reason=str(exc),retry=False)));return 2


if __name__=='__main__':raise SystemExit(main())
