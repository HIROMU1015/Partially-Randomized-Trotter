"""One child, independent stdlib resource observation, exclusive one-shot lock.

No affinity or shared settings are changed. This launcher only signals its own
direct Popen child, using a pidfd; all exits are waited/reaped.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,sort_keys=True,allow_nan=False)


def cgroup_paths():
    relative=next(x.split(':',2)[2] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
    root=Path('/sys/fs/cgroup');current=root/relative.lstrip('/');result=[]
    while current.is_relative_to(root):
        if (current/'memory.events').exists():result.append(current)
        if current==root:break
        current=current.parent
    return result


def snapshot(paths):
    mem={k:int(v.split()[0])*1024 for k,v in (x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines()) if k=='MemAvailable'}
    pressures={};events={}
    for label,path in [('host',Path('/proc/pressure/memory'))]+[(str(x),x/'memory.pressure') for x in paths]:
        pressures[label]={line.split()[0]:{k:float(v) for k,v in (p.split('=') for p in line.split()[1:])}
                          for line in path.read_text().splitlines()}
    for path in paths:
        events[str(path)]={k:int(v) for k,v in (line.split() for line in (path/'memory.events').read_text().splitlines())}
        for name in ('memory.max','memory.high'):
            value=(path/name).read_text().strip()
            if value!='max':mem['cgroup_available:'+str(path)+':'+name]=max(0,int(value)-int((path/'memory.current').read_text()))
    return dict(memory=mem,pressure=pressures,events=events)


def reason(current,baseline,admission):
    if min(current['memory'].values())<admission:return 'MEMORY_HEADROOM'
    for label,rows in current['pressure'].items():
        if any(v['avg10']>=1 if label=='host' else v['avg10']>0 for v in rows.values()):return 'MEMORY_PRESSURE'
    for group,row in current['events'].items():
        for key in ('oom','oom_kill','oom_group_kill'):
            if row.get(key,0)>baseline['events'][group].get(key,0):return 'OOM_INCREMENT'
    return None


def child_setup():
    # Parent death cannot leave an unobserved numerical child running.
    parent=os.getppid();libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(1,signal.SIGTERM,0,0,0)!=0:raise OSError(ctypes.get_errno(),'prctl')
    if os.getppid()!=parent:os._exit(125)
    os.nice(10)


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--control',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();m=json.loads(args.manifest.read_text());limits=m['limits']
    if m.get('approved') is not True or m.get('runtime_authorization') is not True:raise RuntimeError('UNAUTHORIZED')
    for path in (args.manifest,args.control,args.output):
        if not path.resolve().is_relative_to('/home/AbeHiromu'):raise RuntimeError('HOME_ONLY')
    if args.control.exists() or args.output.exists():raise RuntimeError('EXCLUSIVE_PATH_REQUIRED')
    args.control.mkdir();control=args.control
    resource.setrlimit(resource.RLIMIT_AS,(512*2**20,limits['AS_bytes']))
    resource.setrlimit(resource.RLIMIT_FSIZE,(limits['log_bytes'],limits['output_bytes']))
    with (control/'one_shot.lock').open('x') as f:f.write(hashlib.sha256(args.manifest.read_bytes()).hexdigest())
    if subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()!=m['SOURCE']:raise RuntimeError('SOURCE_HEAD_MISMATCH')
    for relative,h in m['source_hashes'].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=h:raise RuntimeError('SOURCE_BYTES_MISMATCH')
    for path,h in m['environment_file_hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=h:raise RuntimeError('ENVIRONMENT_FILE_MISMATCH')
    paths=cgroup_paths();base=snapshot(paths);stop=reason(base,base,limits['admission_bytes'])
    fs=os.statvfs(control)
    if fs.f_bavail*fs.f_frsize<limits['disk_reserve_bytes'] or fs.f_favail<limits['inode_reserve']:stop=stop or 'FILESYSTEM_HEADROOM'
    write(control/'fresh_gate_v1.json',dict(snapshot=base,affinity=sorted(os.sched_getaffinity(0)),affinity_changed=False,
          filesystem=dict(available_bytes=fs.f_bavail*fs.f_frsize,available_inodes=fs.f_favail),
          quota=m['quota_status'],first_stop_reason=stop))
    if stop:raise RuntimeError('FRESH_GATE:'+stop)
    env=os.environ.copy();env.update(m['process_environment'])
    command=[m['python'],'-P','-B',str(ROOT/'scripts/run_native_df_theory_evaluation.py'),'--manifest',str(args.manifest),'--output',str(args.output)]
    start=time.monotonic();last=start;maximum_gap=0.;peak=0;fd=None;child=None;exitcode=None
    def signal_own(sig):
        if child.poll() is None:
            try:signal.pidfd_send_signal(fd,sig)
            except ProcessLookupError:pass
    try:
        with (control/'stdout.log').open('xb') as out,(control/'stderr.log').open('xb') as err,(control/'observations.jsonl').open('x') as log:
            child=subprocess.Popen(command,env=env,stdout=out,stderr=err,preexec_fn=child_setup)
            fd=os.pidfd_open(child.pid)
            write(control/'launch_receipt_v1.json',dict(pid=child.pid,parent_pid=os.getpid(),SOURCE=m['SOURCE'],command=command))
            while child.poll() is None:
                now=time.monotonic();gap=now-last;maximum_gap=max(maximum_gap,gap);last=now
                if gap>5:stop=stop or 'MONITOR_STALENESS'
                before=time.monotonic();current=snapshot(paths);stop=stop or reason(current,base,limits['admission_bytes'])
                try:
                    status=dict(line.split(':',1) for line in Path('/proc',str(child.pid),'status').read_text().splitlines() if ':' in line)
                    rss=int(status['VmRSS'].split()[0])*1024;as_bytes=int(status['VmSize'].split()[0])*1024;peak=max(peak,rss)
                    if rss>limits['RSS_bytes'] or as_bytes>limits['AS_bytes']:stop=stop or 'CHILD_MEMORY_LIMIT'
                    if int(status['Threads'])!=1:stop=stop or 'CHILD_THREADS'
                except (FileNotFoundError,KeyError,ProcessLookupError):
                    # Exit may race proc reading. wait/poll decides actual exit.
                    if child.poll() is None:
                        time.sleep(.01)
                        if child.poll() is None:stop=stop or 'LIVE_PROC_MISSING'
                    rss=as_bytes=None
                duration=time.monotonic()-before
                if duration>5:stop=stop or 'OBSERVATION_DURATION'
                fs=os.statvfs(control)
                if fs.f_bavail*fs.f_frsize<limits['disk_reserve_bytes'] or fs.f_favail<limits['inode_reserve']:stop=stop or 'FILESYSTEM_HEADROOM'
                log.write(json.dumps(dict(elapsed=time.monotonic()-start,interval_seconds=gap,observation_seconds=duration,RSS=rss,AS=as_bytes,resources=current,first_stop_reason=stop))+'\n');log.flush()
                if log.tell()>limits['log_bytes'] or any((control/name).stat().st_size>limits['log_bytes'] for name in ('stdout.log','stderr.log')):stop=stop or 'LOG_CAP'
                if stop:break
                time.sleep(1)
            if stop:signal_own(signal.SIGTERM)
            try:exitcode=child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                signal_own(signal.SIGKILL);exitcode=child.wait(timeout=5)
    except BaseException as exc:
        stop=stop or type(exc).__name__+':'+str(exc)
        if child is not None:
            if fd is None:child.terminate()
            else:signal_own(signal.SIGTERM)
            try:exitcode=child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill();exitcode=child.wait(timeout=5)
        raise
    finally:
        if fd is not None:os.close(fd)
        write(control/'launcher_terminal_v1.json',dict(native_exit_code=exitcode,first_stop_reason=stop,
              wall_seconds=time.monotonic()-start,peak_sampled_RSS=peak,max_monitor_gap_seconds=maximum_gap,
              child_reaped=child is not None and child.returncode is not None,pidfd_closed=True,
              automatic_retry=False,next_stage=False,subagents=0))
    if stop or exitcode!=0:raise SystemExit(2)


if __name__=='__main__':main()
