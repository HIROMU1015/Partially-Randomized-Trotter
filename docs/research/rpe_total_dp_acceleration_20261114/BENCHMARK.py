"""Reproduce artificial DP timing; no saved signals, Monte Carlo or real H4."""
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from check_rpe_total_decoder_risk import make
from check_rpe_total_dp_parallel import tables
from trottertracks import rpe_total_dp_parallel as P

for name,n,frontier,limit,workers in (
    ('small',12,512,2000000,(1,2,4,8)),
    ('medium',24,2048,8000000,(1,2,4,8,16))):
    args=make(vector=[n]*5,raws=[(.213847162847,.47193847582)]*5,partial=None,total=None)
    tt=tables(args);scope=dict(source={'fixture':'SYNTHETIC_ONLY'},
        maximum_frontier_states=frontier,maximum_transition_pairs=limit)
    start=time.perf_counter();reference=P.R.total_state_dp(tt,h=10.,target=0.,scope=scope)
    elapsed=time.perf_counter()-start
    print(json.dumps(dict(case=name,backend='original',wall_seconds=elapsed,
                         transition_pairs=reference['transition_pairs'])),flush=True)
    for count in workers:
        engine=P.Accelerator(workers=count);start=time.perf_counter()
        result=engine(tt,h=10.,target=0.,scope=scope);wall=time.perf_counter()-start
        if result!=reference:raise ValueError('Full exact DP output mismatch')
        print(json.dumps(dict(case=name,backend='accelerated',workers=count,wall_seconds=wall,
            speedup=elapsed/wall,exact_output_match=True,observations=engine.observations)),flush=True)
