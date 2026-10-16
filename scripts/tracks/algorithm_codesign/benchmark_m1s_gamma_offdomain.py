"""Off-domain scalar workload only. Never opens saved science inputs."""
from fractions import Fraction as F
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.m1s_saved_signal_gamma_v1.bounds import Bounds
from trottertracks.algorithm_codesign.m1s_saved_signal_gamma_v1.boundary import compare
from trottertracks.algorithm_codesign.m1s_saved_signal_gamma_v1.spec import ART,RESOURCES


def deny_science(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        p=Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).absolute()
        if ROOT/'artifacts' in p.parents and not str(p).startswith(str(ROOT/ART)+'/'):
            raise RuntimeError('science input open forbidden in off-domain benchmark')
sys.addaudithook(deny_science)


def main():
    resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2))
    resource.setrlimit(resource.RLIMIT_CPU,(60,61))
    signal.signal(signal.SIGALRM,lambda a,b: (_ for _ in ()).throw(RuntimeError('technical benchmark wall cap')))
    signal.alarm(60)
    rows=[]
    for bits in [32,64,128,256]:
        for repeat in range(3):
            B=Bounds(bits,RESOURCES)
            cpu,wall=time.process_time(),time.monotonic()
            pi=B.pi()
            B.log_above_one(F(80))  # artificial scalar; no adopted alpha implied
            # Full-size scalar workload cardinality, no artificial H4 result.
            for i in range(72):
                x=F(2**50+i*982451653,2**53)
                B.sin_positive(x);B.cos_positive(x)
            for i in range(48):
                a=F(900+i,1000)
                r=compare(a,F(2,5),F(4,5),F(1,5),B,pi)
                if r['status']!='UNIQUE_TRADEOFF_BOUNDARY':
                    raise RuntimeError('off-domain expected strict tradeoff branch')
            rows.append({'bits':bits,'repeat':repeat,'cpu_seconds':time.process_time()-cpu,
                         'wall_seconds':time.monotonic()-wall,'bound_steps':B.steps})
    signal.alarm(0)
    # Empirical operational cutoff candidates, never worst-case completion bounds.
    cpu_rate=max(x['cpu_seconds']/x['bound_steps'] for x in rows)
    wall_rate=max(x['wall_seconds']/x['bound_steps'] for x in rows)
    cpu_cap=10*math.ceil(4*cpu_rate*RESOURCES['max_bound_steps']/10)
    wall_cap=10*math.ceil(max(2*cpu_cap,4*wall_rate*RESOURCES['max_bound_steps'])/10)
    result={'schema':'m1s_gamma_offdomain_resource_probe','scope':'ARTIFICIAL_SCALAR_KERNELS_ONLY',
            'rows':rows,'cpu_candidates_seconds':cpu_cap,'wall_candidates_seconds':wall_cap,
            'candidate_formula':'CPU=round_up_10(4*max(observed_CPU/bound_steps)*1000000); wall=round_up_10(max(2*CPU,4*max(observed_wall/bound_steps)*1000000))',
            'factor_4':'explicit engineering headroom; not a probabilistic or worst-case guarantee',
            'peak_RSS_KiB_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'logical_cpus':os.cpu_count(),'cpu_affinity_count':len(os.sched_getaffinity(0)),
            'workers':1,'GPU_used':False,'science_inputs_opened':0,
            'H4_gamma_boundaries_evaluated':0,'saved_signal_reanalysis':0,'shot_forecasts':0,
            'resource_caps_adopted':False,'completion_time_guaranteed':False,'mandatory_STOP':True}
    (ROOT/ART/'offdomain_resource_probe.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['scope','cpu_candidates_seconds','wall_candidates_seconds','peak_RSS_KiB_linux']}))


if __name__=='__main__':main()
