"""Scoped migration confirmation controller; same native runner and allocator."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import h246_reconstructed_contract as c


def launch(paths,state):
    from launch_h246_confirmation import allocate
    from run_h46_m5_signal import fresh_resources
    configs=[c.read(p) for p in paths]
    for cfg in configs:c.authorize(cfg)
    state=Path(state).resolve();state.mkdir(parents=True,exist_ok=False)
    pending=list(zip(paths,configs));reports=[];wave=0
    while pending:
        obs=fresh_resources(dict(run_root=str(state/'probe'),systems={'probe':{}},resources=dict(system_RSS_proxy_bytes={'probe':1},disk_proxy_bytes=0)))
        lease=allocate(obs,[cfg for _,cfg in pending]);selected=[x for x in pending if x[1]['stage'][:2] in lease['workers']]
        lease['configs']={cfg['stage']:dict(SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256']) for _,cfg in selected}
        lp=state/f'lease_{wave}.json';c.save(lp,lease);active=[]
        for path,cfg in selected:
            log=(state/(cfg['stage']+'_controller.log')).open('xb')
            env=dict(os.environ,H246_CONFIRMATION_LEASE=str(lp))
            process=subprocess.Popen([cfg['runtime']['python'],'-B',__file__,'--supervise','--config',path],env=env,stdout=log,stderr=subprocess.STDOUT)
            active.append((process,path,cfg,log))
        for process,path,cfg,log in active:
            code=process.wait();log.close()
            au=subprocess.run([cfg['runtime']['python'],'-B',__file__,'--audit','--config',path],capture_output=True,text=True)
            report=dict(stage=cfg['stage'],supervisor_exit=code,audit_exit=au.returncode,audit_stdout=au.stdout,audit_stderr=au.stderr,retry=False)
            c.save(state/(cfg['stage']+'_finish.json'),report)
            if code==0 and au.returncode==0:
                ev=subprocess.run([cfg['runtime']['python'],'-B',__file__,'--evaluate','--config',path],capture_output=True,text=True)
                report.update(evaluation_exit=ev.returncode,evaluation_stdout=ev.stdout,evaluation_stderr=ev.stderr)
                c.save(state/(cfg['stage']+'_evaluation.json'),report)
            reports.append(report)
        pending=[x for x in pending if x not in selected];wave+=1
        if any(r['supervisor_exit'] or r['audit_exit'] or r.get('evaluation_exit',1) for r in reports):break
    c.save(state/'CONTROLLER_FINISHED.json',dict(reports=reports,unstarted=[cfg['stage'] for _,cfg in pending],retry=False,mandatory_STOP=True))
    return int(any(r['supervisor_exit'] or r['audit_exit'] or r.get('evaluation_exit',1) for r in reports))


def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    for mode in ('prepare','launch','supervise','execute','audit','evaluate'):g.add_argument('--'+mode,action='store_true')
    p.add_argument('--private');p.add_argument('--config')
    p.add_argument('--attempt',choices=('confirmation','confirmation_run02'),default='confirmation');a=p.parse_args()
    try:
        if a.prepare:
            c.require(a.private,'PRIVATE_REQUIRED');print(json.dumps(dict(configs=c.make_packs(a.private,a.attempt))));return 0
        if a.launch:
            c.require(a.private,'PRIVATE_REQUIRED');v=c.read(Path(a.private)/a.attempt/'configs_PRIVATE.json')
            return launch(v['configs'],Path(a.private)/a.attempt/'controller_run01')
        c.require(a.config,'CONFIG_REQUIRED');cfg=c.read(a.config);c.authorize(cfg)
        if a.audit:
            result=c.audit(cfg);print(json.dumps({k:v for k,v in result.items() if k not in ('results','external_exit')}));return int(bool(result['errors']))
        if a.evaluate:c.evaluate(cfg);return 0
        rt=c.runtime(cfg)
        if a.supervise:return rt.runner.supervise(cfg,a.config)
        c.require(c.read(cfg['supervisor']['entered'])==dict(SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],grant_id=cfg['grant']['grant_id'],retry=False),'OWN_EXTERNAL_SUPERVISOR_REQUIRED')
        return rt.runner.execute(cfg)
    except Exception as exc:
        print(json.dumps(dict(status='REFUSED',reason=str(exc),retry=False)));return 2


if __name__=='__main__':raise SystemExit(main())
