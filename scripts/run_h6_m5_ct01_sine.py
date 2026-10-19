"""One-shot sine only, with independent monitoring and external wait4 witness."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time
from types import FunctionType
import h6_m5_ct01_sine_contract as c
import run_h6_m5_saved_phase as observed

supervise=FunctionType(observed.supervise.__code__,dict(observed.__dict__,c=c,__file__=__file__),'supervise',observed.supervise.__defaults__,observed.supervise.__closure__)


def execute(cfg,*,target=None):
    c.authorize(cfg);c.require(sys.executable==cfg['runtime']['python'],'FIXED_CT01_PYTHON')
    root=Path(cfg['run_root']);c.require(not root.exists(),'UNUSED_CT01_RUN_ROOT')
    fresh=observed.fresh_resources(cfg);fresh['workers']=fresh['productive_jobs']=1
    root.mkdir(exist_ok=False);Path(cfg['output']).mkdir();Path(cfg['ledger']).mkdir()
    c.save(cfg['marker'],dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant=cfg['grant']));c.save(root/'fresh_gate.json',fresh)
    ctx=mp.get_context('spawn');stop=ctx.Event();p=None;handlers={};signals=[];error=None;started=time.monotonic()
    keys=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','QISKIT_PARALLEL');env={k:os.environ.get(k) for k in keys}
    def interrupt(sig,frame):signals.append(sig);stop.set()
    try:
        for k in keys:os.environ[k]='FALSE' if k=='QISKIT_PARALLEL' else '1'
        for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,interrupt)
        if target is None:
            from h6_m5_ct01_sine_native import worker
            target=worker
        p=ctx.Process(target=target,args=(cfg,stop));p.start()
        c.save(root/'worker_start.json',dict(pid=p.pid,launcher_pid=os.getpid(),SOURCE=cfg['SOURCE']))
        with (root/'resource_observation.jsonl').open('x') as f:
            while p.is_alive():
                f.write(json.dumps(dict(elapsed=time.monotonic()-started,worker_pid=p.pid,signals=signals,resources=observed.observe()))+'\n');f.flush();p.join(timeout=1)
        p.join();c.require(p.exitcode==0 and not signals,'CT01_NATIVE_EXIT_OR_OWN_SIGNAL')
        c.require(not (Path(cfg['output'])/'STOP.json').exists(),'CT01_SCIENTIFIC_OR_TECHNICAL_STOP')
        r=c.read(Path(cfg['output'])/'report.json')
        c.require(r['status']=='CT01_SINE_PASS_MANDATORY_STOP' and r['SOURCE']==cfg['SOURCE'] and r['plan_sha256']==cfg['plan_sha256'] and r['pipeline_identity']==cfg['pipeline_identity'],'CT01_REPORT_IDENTITY')
        c.require(r['new_compile']==r['compiled_state_actions']==1 and r['new_cosine_compile']==r['new_uncompiled_full_replays']==r['new_sampling']==0 and r['compiler_policy']=='ORIGINAL_PRESET_OPT1_UNMODIFIED','CT01_REPORT_SCOPE')
        c.require(r['reused_cosine']==cfg['cosine_acceptance'] and r['pilot_axis_receipts']==2 and not r['cost_accounting'] and not r['ranking'] and not r['main_table_adoption'],'CT01_REUSE_NOT_ACCOUNTING')
        c.require(len(r['checks'])==2 and [x['strictly_below'] for x in r['checks']]==[1e-9,1e-7] and all(x['PASS'] and 0<=x['raw_error']<x['strictly_below'] for x in r['checks']),'CT01_CHECKS')
        c.require(r['mandatory_STOP'] and not r['retry'] and not r['next_phase_started'] and r['AD03']=='DEFERRED','CT01_MANDATORY_STOP')
        c.require({x.stem for x in Path(cfg['ledger']).glob('*.json')}==set(c.STAGES),'CT01_LEDGER')
        for row in r['artifacts'].values():
            path=Path(cfg['output'])/row['file'];c.require(path.stat().st_size==row['bytes'] and c.sha(path)==row['sha256'],'CT01_RESULT_ARTIFACT_SHA')
        c.require(c.read(Path(cfg['output'])/'cost_sine.json')['status']=='PILOT_AXIS_ACCEPTED_CT01','CT01_SINE_AXIS_RECEIPT')
    except BaseException as exc:
        error=str(exc);stop.set();c.save(root/'STOP.json',dict(status='STOP',reason=error,retry=False))
    finally:
        if p is not None:p.join()
        for sig,h in handlers.items():signal.signal(sig,h)
        for k,v in env.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v
    try:c.verify_seals(cfg);c.consumed(cfg)
    except Exception as exc:error=error or str(exc)
    rp=Path(cfg['output'])/'report.json'
    c.save(root/'completion_witness.json',dict(status='STOP' if error else 'CT01_SINE_COMPLETE_MANDATORY_STOP',reason=error,SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],pipeline_identity=cfg['pipeline_identity'],worker_exit=None if p is None else p.exitcode,worker_pid=None if p is None else p.pid,worker_reaped=p is not None and not p.is_alive(),signals=signals,report_sha256=c.sha(rp) if rp.exists() else None,stage_entries=[s for s in c.STAGES if (Path(cfg['ledger'])/(s+'.json')).exists()],wall_seconds=time.monotonic()-started,retry=False,next_phase_started=False,mandatory_STOP=True))
    return 1 if error else 0


def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group();g.add_argument('--execute',action='store_true');g.add_argument('--supervise',action='store_true');p.add_argument('--config');a=p.parse_args()
    if not a.execute and not a.supervise:print(json.dumps(dict(status='SCOPE_ONLY',scope=c.scope(),science_started=False)));return 0
    try:
        c.require(a.config,'CT01_CONFIG_REQUIRED');cfg=c.read(a.config)
        if a.supervise:return supervise(cfg,a.config)
        c.authorize(cfg);c.require(c.read(cfg['supervisor']['entered'])==dict(SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],grant_id=cfg['grant']['grant_id'],retry=False),'CT01_EXTERNAL_SUPERVISOR_REQUIRED')
        return execute(cfg)
    except Exception as exc:print(json.dumps(dict(status='REFUSED',reason=str(exc),retry=False)));return 2


if __name__=='__main__':raise SystemExit(main())
