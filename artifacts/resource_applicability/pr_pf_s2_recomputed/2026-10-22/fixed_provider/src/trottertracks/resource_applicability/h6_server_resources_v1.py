"""Own-process H6 Phase A limits and read-only shared-host observation."""
from __future__ import annotations
import errno
import json
import os
from pathlib import Path
import resource
import signal
import time
from .h6_server_quota_v1 import quota_readonly

GiB = 2**30
THREAD_KEYS = ('NUMBA_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
    'NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS','QISKIT_NUM_PROCS')


def resources(cpus):
    if (not isinstance(cpus,list) or len(cpus)!=4 or cpus!=sorted(set(cpus))
        or any(type(x) is not int or x<0 for x in cpus)):
        raise ValueError('SERVER_FOUR_CPU_PROPOSAL')
    return dict(assigned_cpus=cpus,coordinator_cpu_set=cpus[2:],cost_cpu_sets=[cpus[:2]],
        cost_workers=1,cost_threads=1,signal_threads=1,
        cost_address_space_bytes=32*GiB,signal_address_space_bytes=32*GiB,
        coordinator_address_space_bytes=2*GiB,coordinator_hard_address_space_bytes=32*GiB,
        aggregate_rss_bytes=40*GiB,job_ram_budget_bytes=64*GiB,
        parent_exit_guard='PR_SET_PDEATHSIG_SIGKILL_AND_IDENTITY_RECHECK',
        memory_is_reservation=False,gpu=False)


def observation_policy():
    return dict(interval_seconds=1.,maximum_observation_seconds=5.,
        journal_emit_seconds=10.,journal_bytes=32*2**20,
        fresh_host_available_bytes=int(216.25*GiB),fresh_effective_available_bytes=64*GiB,
        running_host_available_bytes=int(168.25*GiB),running_effective_available_bytes=16*GiB,
        h4_admission_margin_bytes=int(152.25*GiB),headroom_bytes=16*GiB,
        host_full_psi_stop_percent=5.,cgroup_full_psi_stop_percent=0.,
        stop_on_new_oom=True,missing_observation='STOP',first_reason_preserved=True)


def process_identity(pid):
    raw=Path(f'/proc/{pid}/stat').read_text()
    fields=raw[raw.rfind(')')+2:].split()
    return dict(pid=pid,start_ticks=int(fields[19]),pgrp=int(fields[2]),session=int(fields[3]),state=fields[0])


def process_memory(pid):
    row={}
    for line in Path(f'/proc/{pid}/status').read_text().splitlines():
        k,_,v=line.partition(':')
        if k in ('VmRSS','VmSize','VmHWM','VmPeak'):row[k+'_bytes']=int(v.split()[0])*1024
    if 'VmRSS_bytes' not in row or 'VmSize_bytes' not in row:raise ValueError('PROCESS_MEMORY_MISSING')
    return row


def install_parent_guard(expected):
    """A coordinator death kills only its own worker, including native/GIL work."""
    import ctypes
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(1,signal.SIGKILL,0,0,0)!=0:raise OSError(ctypes.get_errno(),'PR_SET_PDEATHSIG')
    # Covers death between fork/exec, guard installation and identity lookup.
    if os.getppid()!=expected['pid'] or process_identity(expected['pid'])['start_ticks']!=expected['start_ticks']:
        raise ValueError('PARENT_EXIT_OR_IDENTITY_CHANGE')


def host_memory():
    """Read memory/PSI/OOM on all visible ancestors; create no cgroups."""
    mem={k:int(v.split()[0])*1024 for k,v in
        (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
        if v.split() and v.split()[0].isdigit()}
    available=mem['MemAvailable'];headrooms=[available];oom={};psi={}
    def pressure(path):
        lines=path.read_text().splitlines()
        full=next(x for x in lines if x.startswith('full '))
        return float(dict(x.split('=',1) for x in full.split()[1:])['avg10'])
    psi['host']=pressure(Path('/proc/pressure/memory'))
    cg=Path('/sys/fs/cgroup')
    rel=next((x.split(':',2)[2] for x in Path('/proc/self/cgroup').read_text().splitlines()
              if x.startswith('0::')),None)
    if rel is None:raise ValueError('CGROUP_V2_REQUIRED')
    node=cg/rel.lstrip('/');ancestors=[]
    while node!=cg:
        ancestors.append(node);node=node.parent
    for node in ancestors:
        events=dict(x.split() for x in (node/'memory.events').read_text().splitlines())
        oom[str(node)]={k:int(events[k]) for k in ('oom','oom_kill')}
        psi[str(node)]=pressure(node/'memory.pressure')
        limit=(node/'memory.max').read_text().strip()
        if limit!='max':headrooms.append(max(0,int(limit)-int((node/'memory.current').read_text())))
    return dict(host_available_bytes=available,effective_available_bytes=min(headrooms),
                oom=oom,full_psi_avg10=psi)


def memory_reason(row, baseline, policy, *, fresh=False):
    prefix='fresh' if fresh else 'running'
    if row['host_available_bytes']<policy[prefix+'_host_available_bytes']:return 'HOST_MEMORY_HEADROOM'
    if row['effective_available_bytes']<policy[prefix+'_effective_available_bytes']:return 'CGROUP_MEMORY_HEADROOM'
    if baseline is not None and row['oom']!=baseline['oom']:return 'OOM_COUNTER_CHANGE'
    for scope,v in row['full_psi_avg10'].items():
        if v<0 or v!=v or v==float('inf'):return 'NONFINITE_PRESSURE'
        threshold=policy['host_full_psi_stop_percent'] if scope=='host' else policy['cgroup_full_psi_stop_percent']
        if (v>=threshold if scope=='host' else v>threshold):return 'MEMORY_PRESSURE:'+scope
    return None


def fresh_resources(assigned, policy, output):
    if not set(assigned['assigned_cpus'])<=os.sched_getaffinity(0):raise ValueError('CPU_UNAVAILABLE')
    hard=resource.getrlimit(resource.RLIMIT_AS)[1]
    if hard!=resource.RLIM_INFINITY and hard<assigned['cost_address_space_bytes']:raise ValueError('INHERITED_HARD_AS')
    fs_hard=resource.getrlimit(resource.RLIMIT_FSIZE)[1]
    if fs_hard!=resource.RLIM_INFINITY and fs_hard<2*GiB:raise ValueError('INHERITED_FSIZE')
    output=Path(output)
    if output.exists() or output.is_symlink():raise ValueError('OUTPUT_USED')
    if not output.parent.is_dir() or not os.access(output.parent,os.W_OK):raise ValueError('OUTPUT_PARENT')
    fs=os.statvfs(output.parent)
    if fs.f_bavail*fs.f_frsize<10*GiB or fs.f_favail<100000:raise ValueError('FILESYSTEM_HEADROOM')
    quota=quota_readonly(output.parent)
    if quota['status']!='KNOWN':raise ValueError('QUOTA_UNKNOWN')
    for item in quota['items']:
        if item['status']=='ACTIVE':
            if item['available_bytes'] is not None and item['available_bytes']<10*GiB:raise ValueError('QUOTA_BYTES')
            if item['available_inodes'] is not None and item['available_inodes']<100000:raise ValueError('QUOTA_INODES')
    start=time.monotonic();row=host_memory();duration=time.monotonic()-start
    if duration>policy['maximum_observation_seconds']:raise ValueError('OBSERVATION_STALE')
    reason=memory_reason(row,None,policy,fresh=True)
    if reason:raise ValueError(reason)
    return dict(memory=row,observation_seconds=duration,available_bytes=fs.f_bavail*fs.f_frsize,
        available_inodes=fs.f_favail,block_bytes=fs.f_frsize,
        quota=quota,memory_is_reservation=False,actual_cpus=sorted(os.sched_getaffinity(0)))


def stop_owned_group(process, identity):
    """Keep the unreaped leader identity pinned until killpg, then wait/reap.

    Supervisors use waitid(WNOWAIT), never poll/reap before group retirement.
    ESRCH alone means the owned group already exited; other errors propagate.
    """
    if process.returncode is not None:raise ValueError('LEADER_REAPED_BEFORE_CLEANUP')
    current=process_identity(process.pid)
    if ((identity is not None and current['start_ticks']!=identity['start_ticks']) or current['pgrp']!=process.pid
        or current['session']!=process.pid):raise ValueError('OWNERSHIP_MISMATCH')
    try:os.killpg(process.pid,signal.SIGKILL)
    except OSError as e:
        if e.errno!=errno.ESRCH:raise
    return process.wait(timeout=10)
