"""Read-only own-affinity CPU admission, before scientific input or RNG.

No fixed worker cap and no process/cgroup/affinity changes. Finite waiting is
not a scientific retry. Only idle ticks count; iowait is recorded separately.
"""
import math
import os
from pathlib import Path
import time
from fractions import Fraction as F
from . import a1_n3_lifecycle_observer as obs


def cpu_ticks(text):
    rows={}
    for line in text.splitlines():
        fields=line.split()
        if fields and fields[0].startswith('cpu') and fields[0][3:].isdigit():
            values=tuple(map(int,fields[1:9]))
            if len(values)!=8:raise ValueError('missing CPU tick fields')
            rows[int(fields[0][3:])]=values
    return rows


def available_from_ticks(before,after,cpus):
    """Exact sum of idle fractions over allowed CPUs; guest ticks excluded."""
    if not cpus:raise ValueError('empty CPU affinity')
    rows=[];capacity=F(0)
    for cpu in sorted(cpus):
        if cpu not in before or cpu not in after:raise ValueError('allowed CPU missing')
        delta=tuple(y-x for x,y in zip(before[cpu],after[cpu]))
        if len(delta)!=8 or min(delta)<0 or sum(delta)<=0:raise ValueError('invalid CPU tick delta')
        capacity+=F(delta[3],sum(delta))
        rows.append(dict(cpu=cpu,before=before[cpu],after=after[cpu],delta=delta))
    return min(len(cpus),math.floor(capacity)),dict(cpus=rows,idle_core_equivalents_exact=str(capacity))


def wait_for_resources(out,stop,*,wait_seconds=1800,interval_seconds=1,recheck_seconds=15,
                       read=None,affinity=None,sleep=time.sleep,clock=time.monotonic):
    """Two consecutive observed windows; timeout or STOP before any input."""
    read=read or (lambda:cpu_ticks(Path('/proc/stat').read_text()))
    affinity=affinity or (lambda:set(os.sched_getaffinity(0)))
    out=Path(out);start=clock();observations=0
    def check():
        if stop():raise RuntimeError('STOP during pre-input resource wait')
    while True:
        check();allowed=set(affinity());counts=[]
        before=read()
        for _ in range(2):
            check();sleep(interval_seconds);check();after=read()
            if set(affinity())!=allowed:raise ValueError('CPU affinity changed during admission')
            count,detail=available_from_ticks(before,after,allowed);counts.append(count);observations+=1
            obs.event(out/'RESOURCE_ADMISSION.jsonl','CPU_OBSERVATION',observation=observations,
                sampled_seconds=interval_seconds,available_workers=count,affinity_CPUs=len(allowed),**detail)
            before=after
        workers=min(counts)
        if workers>=1:
            obs.event(out/'RESOURCE_ADMISSION.jsonl','ADMITTED',workers=workers,
                elapsed_seconds=clock()-start,resources=obs.memory_snapshot(),policy='minimum of two observed idle capacities; no fixed worker cap')
            return workers
        elapsed=clock()-start
        if elapsed>=wait_seconds:raise RuntimeError('finite pre-input resource wait expired')
        obs.event(out/'RESOURCE_ADMISSION.jsonl','WAITING',elapsed_seconds=elapsed,science_calls=0,words_drawn=0)
        sleep(min(recheck_seconds,wait_seconds-elapsed))
