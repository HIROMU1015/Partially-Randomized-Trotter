"""Two tiny owned exit fixtures, at most three total processes, no science."""
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
def load(name,file):
    s=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
r=load('minimal_exit_runner','h6_b0i_pilot_runner_v1.py');g=r.g
p=load('minimal_exit_backend','h6_b0i_parallel_compile_v1.py')

def main(base):
    base.mkdir(mode=0o700)
    plan=dict(max_child_processes=2,max_total_processes=3,threads=1,AS_bytes=256*2**20,
        RSS_bytes=128*2**20,CPU_seconds=30,wall_seconds=20,output_bytes=2**20,
        own_affinity_only=True,scientific_operations=0,compile=0,subagents=0)
    (base/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    resource.setrlimit(resource.RLIMIT_AS,(256*2**20,256*2**20));resource.setrlimit(resource.RLIMIT_CPU,(30,32))
    resource.setrlimit(resource.RLIMIT_FSIZE,(2**20,2**20))
    backend=p.ForkBackend.__new__(p.ForkBackend);backend.g=g;backend.parent=os.getpid()
    backend.clock_ticks=os.sysconf('SC_CLK_TCK');backend.handles=[]
    backend.grant=SimpleNamespace(contract=dict(budget=dict(address_space_bytes=256*2**20,
        rss_bytes=128*2**20,cpu_seconds=30,monitor_max_seconds=5.)))
    cpu=min(os.sched_getaffinity(0));started=time.monotonic()
    try:
        for i in range(2):
            rd,wr=os.pipe();os.set_blocking(rd,False);pid=os.fork()
            if pid==0:
                try:
                    os.close(rd);p.parent_death_guard(backend.parent);os.sched_setaffinity(0,{cpu})
                    time.sleep(.15+.05*i);os.write(wr,b'{}');os.close(wr)
                finally:os._exit(0)
            os.close(wr);identity=g.identity(pid);assert identity is not None
            backend.handles.append(dict(pid=pid,identity=identity,fd=rd,
                task=dict(name='minimal_'+str(i),cpu=cpu),payload=bytearray(),exit_code=None,
                reaped=False,cpu_seconds=0.,peak_rss_bytes=0,started=time.monotonic(),affinity_confirmed=False))
        active=list(backend.handles)
        while active:
            assert time.monotonic()-started<20,'MINIMAL_EXIT_WALL_CAP'
            for handle in backend.poll(active):active.remove(handle)
            time.sleep(.005)
    finally:receipt=backend.cleanup()
    assert receipt['all_reaped'] and not receipt['errors'] and all(x['native_exit_code']==0 for x in receipt['workers'])
    assert all(g.identity(x['pid']) is None for x in receipt['workers'])
    assert all(x['fd'] is None for x in backend.handles)
    forbidden=[x for x in sys.modules if x.split('.')[0] in ('numpy','qiskit','pyscf','cupy')];assert not forbidden
    result=dict(status='PASS',cleanup=receipt,original_identities_absent=True,parent_pipe_FDs_closed=True,
        wall_seconds=time.monotonic()-started,maximum_total_processes=3,scientific_imports=forbidden,compile=0)
    (base/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main(Path(sys.argv[1]))
