"""Separate one-shot preservation control; existing monitor is reused."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time
from types import FunctionType
import h6_m5_preserve_oneq_contract as c
import run_h6_m5_saved_phase as observed

supervise=FunctionType(observed.supervise.__code__,dict(observed.__dict__,c=c,__file__=__file__),'supervise',observed.supervise.__defaults__,observed.supervise.__closure__)


def execute(cfg,*,target=None):
    c.authorize(cfg);c.require(sys.executable==cfg['runtime']['python'],'FIXED_PRESERVE_ONEQ_PYTHON')
    root=Path(cfg['run_root']);c.require(not root.exists(),'UNUSED_PRESERVE_ONEQ_RUN_ROOT')
    fresh=observed.fresh_resources(cfg);fresh['workers']=fresh['productive_jobs']=1
    root.mkdir(exist_ok=False);Path(cfg['output']).mkdir();Path(cfg['ledger']).mkdir()
    c.save(cfg['marker'],dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant=cfg['grant']));c.save(root/'fresh_gate.json',fresh)
    ctx=mp.get_context('spawn');stop=ctx.Event();process=None;handlers={};signals=[];error=None;started=time.monotonic()
    envkeys=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','QISKIT_PARALLEL');oldenv={k:os.environ.get(k) for k in envkeys}
    def interrupt(signum,frame):signals.append(signum);stop.set()
    try:
        for key in envkeys:os.environ[key]='FALSE' if key=='QISKIT_PARALLEL' else '1'
        for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,interrupt)
        if target is None:
            from h6_m5_preserve_oneq_native import worker
            target=worker
        process=ctx.Process(target=target,args=(cfg,stop));process.start()
        c.save(root/'worker_start.json',dict(pid=process.pid,launcher_pid=os.getpid(),SOURCE=cfg['SOURCE']))
        with (root/'resource_observation.jsonl').open('x') as f:
            while process.is_alive():
                f.write(json.dumps(dict(elapsed=time.monotonic()-started,worker_pid=process.pid,signals=signals,resources=observed.observe()))+'\n');f.flush();process.join(timeout=1)
        process.join();c.require(process.exitcode==0,'PRESERVE_ONEQ_WORKER_NATIVE_EXIT');c.require(not signals,'OWN_PRESERVE_ONEQ_SIGNAL')
        c.require(not (Path(cfg['output'])/'STOP.json').exists(),'PRESERVE_ONEQ_SCIENTIFIC_OR_TECHNICAL_STOP')
        report=c.read(Path(cfg['output'])/'report.json')
        c.require(report['status']=='PRESERVE_ONEQ_SINGLE_STATE_PASS_MANDATORY_STOP' and report['SOURCE']==cfg['SOURCE'] and report['plan_sha256']==cfg['plan_sha256'],'PRESERVE_ONEQ_REPORT_IDENTITY')
        c.require(report['compiler_policy']==c.POLICY and report['new_compile']==1 and report['statevector_replays']==1 and report['reused_initial_raw_gate']==1 and report['accepted_cost_rows']==0 and report['mandatory_STOP'] and report['next_phase_started'] is False,'PRESERVE_ONEQ_REPORT_FIXED_SCOPE')
        c.require(report['oneq_policy_journal_sha256']==c.sha(Path(cfg['output'])/'oneq_policy_passes.jsonl'),'PRESERVE_ONEQ_JOURNAL_SHA')
        c.require({p.stem for p in Path(cfg['ledger']).glob('*.json')}==set(c.STAGES),'EXACT_THREE_PRESERVE_ONEQ_STAGES')
        c.require(len(report['checks'])==3 and report['checks'][0]['reused'] and all(r['PASS'] and r['strictly_below']==1e-9 and r['raw_error']<1e-9 for r in report['checks']),'EXACT_THREE_RAW_CHECKS_ONE_REUSED')
        for row in report['artifacts'].values():
            p=Path(cfg['output'])/row['file'];c.require(c.sha(p)==row['sha256'] and p.stat().st_size==row['bytes'],'PRESERVE_ONEQ_ARTIFACT_SHA')
    except BaseException as exc:
        error=str(exc);stop.set();c.save(root/'STOP.json',dict(status='STOP',reason=error,retry=False,old_STOP_remains_STOP=True))
    finally:
        if process is not None:process.join()
        for sig,h in handlers.items():signal.signal(sig,h)
        for key,value in oldenv.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
    try:c.verify_seals(cfg);c.consumed(cfg)
    except Exception as exc:error=error or str(exc)
    report_path=Path(cfg['output'])/'report.json'
    c.save(root/'completion_witness.json',dict(status='STOP' if error else 'PRESERVE_ONEQ_PILOT_PASS_MANDATORY_STOP',reason=error,SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],pipeline_identity=cfg['pipeline_identity'],
        worker_pid=None if process is None else process.pid,worker_exit=None if process is None else process.exitcode,worker_reaped=process is not None and not process.is_alive(),signals=signals,
        stage_entries=[s for s in c.STAGES if (Path(cfg['ledger'])/(s+'.json')).exists()],report_sha256=c.sha(report_path) if report_path.exists() else None,
        old_STOP_remains_STOP=True,accepted_cost_rows=0,sine=0,retry=False,next_phase_started=False,mandatory_STOP=True,wall_seconds=time.monotonic()-started))
    return 1 if error else 0


def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group();group.add_argument('--execute',action='store_true');group.add_argument('--supervise',action='store_true');parser.add_argument('--config');a=parser.parse_args()
    if not a.execute and not a.supervise:print(json.dumps(dict(status='SCOPE_ONLY',scope=c.scope(),science_started=False)));return 0
    try:
        c.require(a.config,'NEW_PRESERVE_ONEQ_CONFIG_REQUIRED');cfg=c.read(a.config)
        if a.supervise:return supervise(cfg,a.config)
        c.authorize(cfg);c.require(c.read(cfg['supervisor']['entered'])==dict(SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],grant_id=cfg['grant']['grant_id'],retry=False),'PRESERVE_ONEQ_SUPERVISOR_REQUIRED')
        return execute(cfg)
    except Exception as exc:print(json.dumps(dict(status='REFUSED',reason=str(exc),retry=False)));return 2


if __name__=='__main__':raise SystemExit(main())
