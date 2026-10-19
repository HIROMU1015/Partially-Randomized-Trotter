"""Exclusive nested CPU/wall accounting, with explicitly unobserved residual."""
from contextlib import contextmanager
from collections import Counter
import time
from .guard import require


def clock():return time.process_time_ns(),time.monotonic_ns()


class Meter:
    def __init__(self):
        self.begin=clock();self.stack=[];self.cpu=Counter();self.wall=Counter();self.calls=Counter()

    @contextmanager
    def span(self,name):
        start=clock();frame=[start,0,0];self.stack.append(frame);self.calls[name]+=1
        try:yield
        finally:
            end=clock();self.stack.pop();cpu=end[0]-start[0];wall=end[1]-start[1]
            self.cpu[name]+=cpu-frame[1];self.wall[name]+=wall-frame[2]
            if self.stack:self.stack[-1][1]+=cpu;self.stack[-1][2]+=wall

    def snapshot(self):
        require(not self.stack,'METER_ACTIVE')
        end=clock();outer=[end[i]-self.begin[i] for i in (0,1)]
        residual=[outer[0]-sum(self.cpu.values()),outer[1]-sum(self.wall.values())]
        require(min(residual)>=0,'EXCLUSIVE_PARTITION_OVERFLOW')
        return dict(outer_cpu_ns=outer[0],outer_wall_ns=outer[1],exclusive_cpu_ns=dict(self.cpu),
            exclusive_wall_ns=dict(self.wall),calls=dict(self.calls),residual_cpu_ns=residual[0],
            residual_wall_ns=residual[1],coarse_fine_added=False,scientific_cost='NOT_ESTABLISHED')
