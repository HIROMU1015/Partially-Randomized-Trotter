"""Versioned cached observer; same selection/exclusive semantics as measured P1."""
from pathlib import Path
from . import g6_w2c_b3_parallel_driver as D
BASE_OBSERVER=D.FunctionObserver

class ScopedObserver(BASE_OBSERVER):
    def __init__(self, paths, read_clock=D.clock):
        super().__init__([*paths, __file__], read_clock)


class CachedObserver(ScopedObserver):
    """Same call/return selection and exclusive partition; cached code labels.

    Each callback uses one clock read like the original Meter observer. No
    function is sampled, suppressed or shifted out of the measured bundle.
    The outer CPU still includes all callback work and residual overhead.
    """
    def __init__(self, paths, read_clock=D.clock):
        super().__init__(paths, read_clock)
        self.cache, self.read_clock = {}, read_clock

    def hook(self,frame,event,arg):
        if event not in ('call','return'):return
        code=frame.f_code
        try:descriptor=self.cache[id(code)]
        except KeyError:
            name=getattr(code,'co_qualname',code.co_name)
            key=(Path(code.co_filename).name+':'+str(code.co_firstlineno)+':'+name) \
                if code.co_filename in self.paths and name not in ('clock','delta','need') else None
            descriptor=[code,key,'self' in code.co_varnames or 'self' in code.co_freevars or 'self' in code.co_cellvars,None]
            self.cache[id(code)]=descriptor
        if descriptor[1] is None:return
        if descriptor[2] and isinstance(frame.f_locals.get('self'),(D.Meter,BASE_OBSERVER)):return
        now=self.read_clock()
        if event=='call':self.frames.append([frame,descriptor,now[0],now[1],0,0])
        else:
            if not self.frames or self.frames[-1][0] is not frame:raise D.Stop('PROFILE_STACK_MISMATCH')
            item=self.frames.pop();cpu,wall=now[0]-item[2],now[1]-item[3]
            if cpu<item[4] or wall<item[5]:raise D.Stop('NEGATIVE_EXCLUSIVE_COST')
            if self.frames:
                self.frames[-1][4]+=cpu;self.frames[-1][5]+=wall
            row=descriptor[3]
            if row is None:
                row=self.meter.rows.get(descriptor[1])
                if row is None:
                    row=dict(calls=0,cpu_ns=0,wall_ns=0);self.meter.rows[descriptor[1]]=row
                descriptor[3]=row
            row['calls']+=1;row['cpu_ns']+=cpu-item[4];row['wall_ns']+=wall-item[5]
