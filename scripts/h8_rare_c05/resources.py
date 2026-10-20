"""Read-only admission observations; no guessed H8 memory footprint or W limit."""
from pathlib import Path
import math
import os
import time
from .sampling import require


class Admission:
    def __init__(self,private_peak_bytes,threads_per_axis,io_slots):
        require(type(private_peak_bytes) is int and private_peak_bytes>0,'APPROVED_PRIVATE_MEMORY_ESTIMATE_REQUIRED')
        require(type(threads_per_axis) is int and threads_per_axis>0 and io_slots>0,'OBSERVED_THREAD_IO_DEMAND_REQUIRED')
        self.peak=private_peak_bytes;self.threads=threads_per_axis;self.io=io_slots;self.active=[];self.previous=None
    def observe(self,rows):self.active=rows
    def capacity(self):
        affinity=os.sched_getaffinity(0);cpu={}
        for line in Path('/proc/stat').read_text().splitlines():
            fields=line.split()
            if fields[0].startswith('cpu') and fields[0][3:].isdigit() and int(fields[0][3:]) in affinity:
                v=list(map(int,fields[1:]));cpu[int(fields[0][3:])]=(sum(v[:8]),v[3]+v[4])
        if self.previous is None:
            self.previous=cpu;time.sleep(.03);return self.capacity()
        idle={}
        for n,(total,free) in cpu.items():
            old=self.previous.get(n,(total,free));delta=total-old[0]
            fraction=max(0.,min(1.,(free-old[1])/delta)) if delta else 0.
            topology=Path('/sys/devices/system/cpu')/('cpu'+str(n))/'topology'
            key=((topology/'physical_package_id').read_text().strip(),(topology/'core_id').read_text().strip())
            idle[key]=max(idle.get(key,0.),fraction)
        self.previous=cpu
        mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
        available=int(mem['MemAvailable'].split()[0])*1024
        quota=len(idle)
        cg=next((x.split('::',1)[1] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::')),None)
        if cg is not None:
            base=Path('/sys/fs/cgroup');p=base/cg.lstrip('/')
            while p.is_relative_to(base):
                f=p/'cpu.max'
                if f.exists():
                    v=f.read_text().split()
                    if v[0]!='max':quota=min(quota,float(v[0])/float(v[1]))
                m,c=p/'memory.max',p/'memory.current'
                if m.exists() and c.exists() and m.read_text().strip()!='max':available=min(available,max(0,int(m.read_text())-int(c.read_text())))
                if p==base:break
                p=p.parent
        # Existing admitted demand plus observed idle capacity; quota remains binding.
        own=sum(h['cpu_demand'] for h in self.active)
        io_pressure=Path('/proc/pressure/io').read_text()
        line=next(x for x in io_pressure.splitlines() if x.startswith('some '))
        io_percent=float(dict(x.split('=') for x in line.split()[1:])['avg10'])
        available_io=max(0,math.ceil(self.io*(1-min(100.,io_percent)/100)))
        return dict(cpu=min(math.floor(quota),own+math.floor(sum(idle.values()))),memory=available,io=available_io,
                    physical_cores_visible=len(idle),CPU_idle_is_observation_not_reservation=True,
                    io_pressure=Path('/proc/pressure/io').read_text(),memory_pressure=Path('/proc/pressure/memory').read_text())
    def demand(self,task,capacity):
        width=2 if capacity['cpu']>=2*self.threads else 1
        return dict(cpu=width*self.threads,memory=self.peak,io=1,axis_width=width)
