"""Detached, bounded native-wait witness. Drafts reject before any process launch."""
import argparse,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import h8_phase_b2_recovery_contract as c
import h8_phase_b2_recovery_runtime as r
import track_a_phase_a_parallel as owned


def durable_json(path,value):
 owned.atomic_json(Path(path),value)
 fd=os.open(Path(path).parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)


def verify_witness(config,g):
 path=Path(config['control']).parent/'witness_started.json';v=r.read(path)
 identity=owned.identity(os.getppid())
 if (identity is None or list(identity)!=v['witness']['identity'] or v['witness']['pid']!=os.getppid()
  or os.sched_getaffinity(os.getppid())!={g['witness_cpu']}):raise ValueError('LIVE_DETACHED_WITNESS_PARENT_REQUIRED')
 return v


def wait_once(command,env,directory,*,limits,witness_cpu,config_sha,grant_sha,allowed_cpus=None):
 """Internal OS primitive; production caller validates config/grant before entry."""
 directory=Path(directory);parent=os.getpid();old=os.sched_getaffinity(0);os.sched_setaffinity(0,{witness_cpu})
 output=directory/'native_exit.json'
 if output.exists():raise ValueError('NATIVE_WAIT_ALREADY_RECORDED')
 with (directory/'witness_attempt').open('x') as f:f.write('ONE_NATIVE_WAIT_ONLY\n');f.flush();os.fsync(f.fileno())
 started=time.monotonic();process=None
 durable_json(directory/'witness_started.json',dict(witness=dict(pid=parent,identity=list(owned.identity(parent))),config_sha256=config_sha,grant_sha256=grant_sha,witness_cpu=witness_cpu,session_id=os.getsid(0),automatic_retry=False))
 try:
  with (directory/'observer_stdout.log').open('xb') as out,(directory/'observer_stderr.log').open('xb') as err:
   if len(list(Path('/proc/self/task').iterdir()))!=1:raise ValueError('SINGLE_THREAD_WITNESS_FORK_REQUIRED')
   selected=set(allowed_cpus or old)
   process=subprocess.Popen(command,env=env,stdout=out,stderr=err,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,selected))
   identity=owned.identity(process.pid)
   durable_json(directory/'observer_started.json',dict(pid=process.pid,identity=list(identity) if identity else None,command=command))
   # Apply after fork: the observer/driver must not inherit witness AS/CPU caps.
   if limits.get('apply_OS_limits',True):
    resource.setrlimit(resource.RLIMIT_AS,(limits['witness_AS_bytes'],limits['witness_AS_bytes']))
    resource.setrlimit(resource.RLIMIT_CPU,(limits['witness_CPU_seconds'],limits['witness_CPU_seconds']))
   deadline=started+limits['wall_seconds']+120
   stop_requested=False
   def stop(*args):
    nonlocal stop_requested
    stop_requested=True
   previous={s:signal.signal(s,stop) for s in (signal.SIGTERM,signal.SIGINT)}
   try:
    while True:
     try:code=process.wait(timeout=1);break
     except subprocess.TimeoutExpired:
      if stop_requested or time.monotonic()>deadline:
       process.terminate()
       try:code=process.wait(timeout=12)
       except subprocess.TimeoutExpired:process.kill();code=process.wait()
       break
   finally:
    for sig,handler in previous.items():signal.signal(sig,handler)
   durable_json(output,dict(status='NATIVE_LAUNCHER_WAIT_RECORDED',pid=process.pid,identity=list(identity) if identity else None,native_exit_code=code,wait_completed=True,wall_seconds=time.monotonic()-started,witness_stop_requested=stop_requested,automatic_retry=False,next_stage_authorized=False))
   return code
 finally:os.sched_setaffinity(0,old)


def dispatch(config_path,grant_path):
 cfg,g=map(r.read,(config_path,grant_path));r.verify_authorization(cfg,g,config_sha=r.sha(config_path))
 for path in (config_path,grant_path):r.private_path(path)
 if config_path.parent!=grant_path.parent:raise ValueError('SIBLING_PRIVATE_CONFIG_GRANT')
 directory=config_path.parent
 if any((directory/name).exists() for name in ('dispatch_attempt','witness_started.json','native_exit.json','observer_started.json','launcher_admission01')):raise ValueError('RECOVERY_DISPATCH_ALREADY_SPENT')
 for key in ('output','control','lock'):
  if Path(cfg[key]).parent!=directory or Path(cfg[key]).exists():raise ValueError('NEW_EXCLUSIVE_RECOVERY_PATHS')
 with (directory/'dispatch_attempt').open('x') as f:f.write(cfg['SOURCE']+'\n');f.flush();os.fsync(f.fileno())
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
 for key in r.THREAD_KEYS:env[key]='1'
 command=[cfg['python'],'-B',str(Path(__file__).resolve()),'--witness-approved','--config',str(config_path),'--grant',str(grant_path)]
 with (directory/'witness_stdout.log').open('xb') as out,(directory/'witness_stderr.log').open('xb') as err:
  process=subprocess.Popen(command,env=env,stdout=out,stderr=err,start_new_session=True,close_fds=True)
 durable_json(directory/'dispatch_receipt.json',dict(pid=process.pid,identity=list(owned.identity(process.pid)),detached_session=True,scientific_completion_claimed=False,automatic_retry=False))
 return dict(status='DETACHED_WITNESS_DISPATCHED_FRESH_GATE_PENDING',pid=process.pid)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dispatch-approved',action='store_true');p.add_argument('--witness-approved',action='store_true')
 p.add_argument('--config',type=Path);p.add_argument('--grant',type=Path);a=p.parse_args()
 if not a.dispatch_approved and not a.witness_approved:
  if a.config or a.grant:p.error('private paths require explicit approved mode')
  print(json.dumps(dict(status='PLAN_ONLY_NOT_AUTHORIZED',scope=c.SCOPE)));return
 if a.dispatch_approved==a.witness_approved or not a.config or not a.grant:p.error('one approved mode and both private paths required')
 if a.dispatch_approved:print(json.dumps(dispatch(a.config,a.grant)));return
 cfg,g=map(r.read,(a.config,a.grant));r.verify_authorization(cfg,g,config_sha=r.sha(a.config))
 if not (a.config.parent/'dispatch_attempt').exists():raise ValueError('DETACHED_DISPATCH_RECEIPT_REQUIRED')
 command=[cfg['python'],'-B',str(c.ROOT/'scripts/launch_h8_phase_b2_recovery.py'),'--launch-approved','--config',str(a.config),'--grant',str(a.grant),'--fresh-gate',str(a.config.parent/'fresh_gate.json')]
 code=wait_once(command,dict(os.environ),a.config.parent,limits=g['resources'],witness_cpu=g['witness_cpu'],config_sha=r.sha(a.config),grant_sha=r.sha(a.grant),allowed_cpus=g['allowed_cpus'])
 raise SystemExit(0 if code==0 else 1)


if __name__=='__main__':main()
