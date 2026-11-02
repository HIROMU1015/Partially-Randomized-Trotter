"""Future explicit launch only. Coordinate owned processes using fresh resources."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import h246_confirmation_contract as c

def allocate(observation,configs):
    """Greedy marginal workload reduction; no historical constant worker cap.

    One worker per admitted stage first. Disk includes all pending final output,
    two raw/partial buffers per active worker, plus one restoration pair.
    """
    cpu=math.floor(observation['idle_CPU_equivalents']);ram=observation['effective_available_RAM_bytes']
    disk=observation['disk_free_bytes'];final=sum(x['resources']['final_growth_proxy_bytes'] for x in configs)
    restore=max(x['resources']['raw_pair_proxy_bytes'] for x in configs)
    spare=disk-final-restore
    workers={};by={x['stage'][:2]:x for x in configs}
    for system,x in by.items():
        r=x['resources'];scratch=2*r['raw_pair_proxy_bytes']
        if cpu>=1 and ram>=r['worker_RSS_proxy_bytes'] and spare>=scratch:
            workers[system]=1;cpu-=1;ram-=r['worker_RSS_proxy_bytes'];spare-=scratch
    c.require(workers,'FRESH_RESOURCES_CANNOT_ADMIT_ONE_STAGE')
    while cpu>=1:
        feasible=[s for s in workers if workers[s]<len(by[s]['tasks'])
          and ram>=by[s]['resources']['worker_RSS_proxy_bytes']
          and spare>=2*by[s]['resources']['raw_pair_proxy_bytes']]
        if not feasible:break
        system=max(feasible,key=lambda s:by[s]['resources']['workload_proxy_seconds']/(workers[s]*(workers[s]+1)))
        workers[system]+=1;cpu-=1;ram-=by[system]['resources']['worker_RSS_proxy_bytes']
        spare-=2*by[system]['resources']['raw_pair_proxy_bytes']
    return dict(workers=workers,peak_growth_proxy_bytes=disk-spare,
      proxy_is_upper_bound=False,runtime_caps=False,host_PSI_kill=False,
      sequential_fallback=len(workers)<len(configs),observation=observation)

def launch(paths,state):
    configs=[c.read(p) for p in paths]
    c.require(len({x['stage'] for x in configs})==len(configs),'ONE_CONFIG_PER_STAGE')
    for cfg in configs:c.authorize(cfg)
    state=Path(state).resolve();state.mkdir(parents=True,exist_ok=False)
    pending=list(zip(paths,configs));reports=[];wave=0
    while pending:
        from run_h46_m5_signal import fresh_resources
        obs=fresh_resources(dict(run_root=str(state/'unused'),systems={'probe':{}},
          resources=dict(system_RSS_proxy_bytes={'probe':1},disk_proxy_bytes=0)))
        lease=allocate(obs,[v for _,v in pending]);selected=[x for x in pending if x[1]['stage'][:2] in lease['workers']]
        lease['configs']={v['stage']:dict(SOURCE=v['SOURCE'],plan_sha256=v['plan_sha256']) for _,v in selected}
        lp=state/('lease_'+str(wave)+'.json');c.save(lp,lease);active=[]
        for path,cfg in selected:
            env=dict(os.environ,H246_CONFIRMATION_LEASE=str(lp))
            log=(state/(cfg['stage']+'_controller.log')).open('xb')
            process=subprocess.Popen([cfg['runtime']['python'],'-B',str(c.ROOT/'scripts/run_h246_confirmation.py'),
              '--supervise','--config',str(path)],env=env,stdout=log,stderr=subprocess.STDOUT)
            active.append((process,path,cfg,log))
        # A failure in an owned stage does not send signals to unrelated stages.
        for process,path,cfg,log in active:
            code=process.wait();log.close()
            audit=subprocess.run([cfg['runtime']['python'],'-B',str(c.ROOT/'scripts/run_h246_confirmation.py'),
              '--audit','--config',str(path)],capture_output=True,text=True)
            reports.append(dict(stage=cfg['stage'],supervisor_exit=code,audit_exit=audit.returncode,
              audit_stdout=audit.stdout,audit_stderr=audit.stderr,automatic_retry=False))
        pending=[x for x in pending if x not in selected];wave+=1
        # Any unresolved failure prevents starting further stages. Active jobs
        # were allowed to finish; scientific conditions are never revised.
        if any(r['supervisor_exit'] or r['audit_exit'] for r in reports):break
    c.save(state/'CONTROLLER_FINISHED.json',dict(reports=reports,unstarted=[v['stage'] for _,v in pending],
      fixed_comparison_evaluation_started=False,retry=False,mandatory_STOP=True))
    return int(any(r['supervisor_exit'] or r['audit_exit'] for r in reports))

def main():
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--config',action='append',default=[])
    p.add_argument('--state-dir');a=p.parse_args()
    if not a.launch:print(json.dumps(dict(status='SCOPE_ONLY',science_authorized=False,mandatory_STOP=True)));return 0
    try:
        c.require(a.config and a.state_dir,'EXPLICIT_LAUNCH_ARGUMENTS');return launch(a.config,a.state_dir)
    except Exception as exc:print(json.dumps(dict(status='REFUSED',reason=str(exc))));return 2

if __name__=='__main__':raise SystemExit(main())
