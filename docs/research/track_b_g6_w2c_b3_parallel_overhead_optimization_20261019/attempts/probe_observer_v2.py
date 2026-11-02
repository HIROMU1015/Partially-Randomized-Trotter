"""Additional artificial candidate; run after the sealed v1 suite completes."""
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[4]
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests/tracks/algorithm_codesign'),str(ROOT/'scripts/tracks/algorithm_codesign')]
import run_b3_overhead_artificial as B
O=B.O


class RowCachedObserver(O.CachedObserver):
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
        if descriptor[2] and isinstance(frame.f_locals.get('self'),(O.D.Meter,O.BASE_OBSERVER)):return
        now=self.read_clock()
        if event=='call':self.frames.append([frame,descriptor,now[0],now[1],0,0])
        else:
            if not self.frames or self.frames[-1][0] is not frame:raise O.D.Stop('PROFILE_STACK_MISMATCH')
            item=self.frames.pop();cpu,wall=now[0]-item[2],now[1]-item[3]
            if cpu<item[4] or wall<item[5]:raise O.D.Stop('NEGATIVE_EXCLUSIVE_COST')
            if self.frames:
                self.frames[-1][4]+=cpu;self.frames[-1][5]+=wall
            row=descriptor[3]
            if row is None:
                row=self.meter.rows.get(descriptor[1])
                if row is None:
                    row=dict(calls=0,cpu_ns=0,wall_ns=0);self.meter.rows[descriptor[1]]=row
                descriptor[3]=row
            row['calls']+=1;row['cpu_ns']+=cpu-item[4];row['wall_ns']+=wall-item[5]


def main():
    old=O.CachedObserver;out=Path(sys.argv[1]);out.mkdir(exist_ok=False,parents=True)
    rows=[];semantic=signature=None
    for repeat in range(3):
        for name,cls in ([('v1',old),('v2',RowCachedObserver)] if repeat%2==0 else [('v2',RowCachedObserver),('v1',old)]):
            with patch.object(O,'CachedObserver',cls):
                metrics,report=B.run_one(B.CONFIGS[-1],'light',128,out,'probe')
            if semantic is None:semantic=metrics['semantic'];signature=metrics['core_function_calls']
            assert metrics['semantic']==semantic and metrics['core_function_calls']==signature
            metrics.update(probe=name,repeat=repeat);rows.append(metrics)
            print(name,metrics['total_cpu_s'],metrics['wall_s'],flush=True)
    (out/'results.json').write_bytes(B.P.safe_json(dict(source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),runtime_source_sha256=sha256(Path(O.__file__).read_bytes()).hexdigest(),rows=rows,scientific_calls=0,real_claims=0))+b'\n')


if __name__=='__main__':main()
