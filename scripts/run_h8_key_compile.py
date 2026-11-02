"""Default: print an unapproved plan. Real compile requires a new sealed grant."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import h8_key_compile_contract as c
import h8_key_compile_parallel as pool


def observe():
    memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
            if line.startswith(('MemAvailable:','MemFree:'))}
    root=Path('/sys/fs/cgroup');name=next(x.split(':',2)[2] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
    group=root/name.lstrip('/');available=memory['MemAvailable'];groups=[]
    while group.is_relative_to(root):
        row=dict(group=str(group))
        for key in ('memory.events','memory.pressure','memory.max','memory.current'):
            path=group/key
            if path.exists():row[key]=path.read_text()
        if row.get('memory.max','max').strip()!='max':
            available=min(available,max(0,int(row['memory.max'])-int(row['memory.current'])))
        groups.append(row)
        if group==root:break
        group=group.parent
    return dict(monotonic_seconds=time.monotonic(),memory=memory,effective_available_bytes=available,
        host_memory_PSI=Path('/proc/pressure/memory').read_text(),cgroups=groups,
        PSI_STOP_gate=False,own_affinity=sorted(os.sched_getaffinity(0)))


def cpu_observation(cpus):
    """Read-only role placement check, unrelated Git state is irrelevant."""
    selected=set(cpus);physical=[];pinned=[]
    for cpu in cpus:
        path=Path('/sys/devices/system/cpu')/('cpu'+str(cpu))/'topology'
        physical.append(((path/'physical_package_id').read_text().strip(),(path/'core_id').read_text().strip()))
    c.require(len(set(physical))==len(cpus),'ROLES_NEED_DISTINCT_PHYSICAL_CORES')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            if proc.stat().st_uid!=os.getuid():continue
            argv=(proc/'cmdline').read_bytes().split(b'\0')
            if not argv or b'python' not in Path(os.fsdecode(argv[0])).name.lower().encode():continue
            if not any(b'/home/AbeHiromu/projects/' in x and x.endswith(b'.py') for x in argv):continue
            affinity=os.sched_getaffinity(int(proc.name))
            if len(affinity)<=16:pinned.append(dict(identity=pool.identity(int(proc.name)),cpus=sorted(affinity)))
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    c.require(not any(selected&set(row['cpus']) for row in pinned),'ROLE_CPU_OVERLAP_OWN_PINNED_JOB')
    return dict(distinct_physical_cores=physical,own_pinned_science_jobs=pinned,
                policy='READ_ONLY_NO_IDLE_THRESHOLD_NO_UNRELATED_WORKTREE_FREEZE')


def fresh(config, grant):
    cpus=c.authorize(config,grant);c.verify(config,runtime=True)
    out,control,lock=c.paths(config);c.require(out.parent.is_dir(),'EXCLUSIVE_STAGE_PARENT_REQUIRED')
    CPU=cpu_observation(cpus)
    state=observe();estimate=c.plan()['resource_policy']['estimates'];fs=os.statvfs(out.parent)
    c.require(state['effective_available_bytes']>=estimate['aggregate_peak_RSS_bytes']+estimate['headroom_bytes'],
              'INSUFFICIENT_EFFECTIVE_HEADROOM_FOR_ESTIMATE')
    c.require(fs.f_bavail*fs.f_frsize>=estimate['stage_bytes'] and fs.f_favail>=estimate['inodes'],
              'INSUFFICIENT_BLOCKS_OR_INODES')
    mount=max((line.split() for line in Path('/proc/mounts').read_text().splitlines()
               if out.parent.is_relative_to(line.split()[1])),key=lambda x:len(x[1]))
    c.require(mount[2] in ('ext4','xfs') and not any(('quota' in opt and not opt.startswith('no')) or
              opt in ('uquota','gquota','pquota') for opt in mount[3].split(',')),'QUOTA_NEEDS_EFFECTIVE_LIMIT_PROOF')
    # Process / CPU activity is observed only; no freeze of unrelated worktrees.
    return dict(SOURCE=config['SOURCE'],config_sha256=c.digest(config),grant_sha256=c.digest(grant),
        allowed_cpus=cpus,CPU_observation=CPU,observation=state,monotonic_seconds=time.monotonic(),
        source_input_runtime_verified=True,quota_verified=True,filesystem=mount,
        free_bytes=fs.f_bavail*fs.f_frsize,free_inodes=fs.f_favail)


def driver(task, directory, config, grant):
    c.require(Path(sys.executable).resolve()==Path(config['python']).resolve() and
              all(os.environ.get(k)=='1' for k in c.THREADS),'PINNED_PYTHON_THREAD_ENV')
    c.verify(config,runtime=False,git=False)
    output=Path(config['output'])
    try:
        from h8_key_compile_handover import wait_for_release
        wait_for_release(config,grant)
        from h8_key_compile_adapter import evaluate
        result=evaluate(config,grant,output)
        c.verify(config,runtime=False,git=False)
        c.save(output/'result.json',result)
        terminal=dict(status='COMPLETE_MANDATORY_STOP',SOURCE=config['SOURCE'],actual_calls=result['actual_calls'],
                      next_stage_authorized=False,cost_reaccount_authorized=False,automatic_retry=False)
        c.save(output/'terminal.json',terminal)
        return terminal
    except BaseException as error:
        calls={key:len(list(output.glob('reservation_'+key+'_*.json'))) for key in c.CAPS}
        c.save(output/'terminal.json',dict(status='STOP',first_stop=str(error) or type(error).__name__,
            SOURCE=config['SOURCE'],reserved_calls_nonrefundable=calls,automatic_retry=False,next_stage_authorized=False))
        raise


def launch(config_path, grant_path):
    config,grant=map(c.read,(config_path,grant_path))
    c.authorize(config,grant) # Invalid draft rejected before filesystem writes, forks or scientific imports.
    c.require(Path(sys.executable).resolve()==Path(config['python']).resolve(),'PINNED_LAUNCH_PYTHON')
    gate=fresh(config,grant);out,control,lock=c.paths(config)
    c.require(time.monotonic()-gate['monotonic_seconds']<=30,'STALE_FRESH_GATE')
    with lock.open('x') as f:f.write(c.digest(grant));f.flush();os.fsync(f.fileno())
    out.mkdir(mode=0o700);control.mkdir(mode=0o700);c.save(control/'fresh_gate.json',gate)
    first=[];handlers={};samples=0;last=0.;observation_errors=[];old_affinity=os.sched_getaffinity(0)
    def stop(signum,frame):
        if not first:first.append('OWN_LAUNCHER_SIGNAL:'+str(signum))
    def guard():
        nonlocal samples,last
        c.require(not first,first[0] if first else 'OWN_SIGNAL')
        now=time.monotonic()
        if now-last>=5:
            last=now
            try:
                state=observe();samples+=1
                state['prior_observation_errors']=list(observation_errors)
                with (control/'observations.jsonl').open('a') as stream:
                    stream.write(json.dumps(state,allow_nan=False)+'\n');stream.flush()
                observation_errors.clear()
            except Exception as error:
                # Restore observation while the owned scientific process continues.
                # A final native PASS still requires a fresh successful observation.
                observation_errors.append(dict(monotonic_seconds=now,error=type(error).__name__+':'+str(error)))
    # Changes are confined to this launcher and inherited children.
    for key in c.THREADS:os.environ[key]='1'
    os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['TMPDIR']=str(control)
    try:
        for sig in (signal.SIGTERM,signal.SIGINT):handlers[sig]=signal.signal(sig,stop)
        os.sched_setaffinity(0,{grant['driver_cpu']})  # Thin launcher shares the driver core.
        receipt=pool.map_owned([dict(id='driver')],lambda task,d:driver(task,d,config,grant),
            cpus=[grant['driver_cpu']],limits=dict(max_workers=1,cleanup_TERM_seconds=30.),directory=control/'driver',
            deadline=math.inf,font_seed=config['font_seed'],guard=guard)
        guard();c.require(receipt['status']=='PASS' and receipt['all_reaped'] and not receipt['cleanup_errors'],
                          'NATIVE_DRIVER_CLEANUP')
        c.require(c.read(out/'terminal.json')['status']=='COMPLETE_MANDATORY_STOP','DRIVER_NOT_COMPLETE')
        c.verify(config,runtime=True);final_observation=observe()
        c.save(control/'final_gate.json',dict(status='PASS_MANDATORY_STOP',SOURCE=config['SOURCE'],
            native=receipt,first_stop=None,monitor_samples=samples,next_stage_authorized=False,
            final_observation=final_observation,prior_observation_errors=observation_errors,
            cost_reaccount_authorized=False))
    except BaseException as error:
        reason=first[0] if first else str(error)
        if not first and (out/'terminal.json').is_file():
            reason=c.read(out/'terminal.json').get('first_stop',reason)
        c.save(control/'STOP.json',dict(status='STOP',first_stop=reason,
            SOURCE=config['SOURCE'],automatic_retry=False,next_stage_authorized=False))
        raise
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
        os.sched_setaffinity(0,old_affinity)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--launch-approved',action='store_true')
    p.add_argument('--config',type=Path);p.add_argument('--grant',type=Path);a=p.parse_args()
    if not a.launch_approved:
        if a.config or a.grant:p.error('real paths need separately approved execution mode')
        print(json.dumps(c.plan(),indent=2));return
    if not a.config or not a.grant:p.error('new sealed config/grant required')
    launch(a.config,a.grant)


if __name__=='__main__':main()
