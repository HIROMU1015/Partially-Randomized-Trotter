"""Fixed artificial cost probes only; no saved means/targets/results are read."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import ExitStack
from fractions import Fraction
import json
import math
from pathlib import Path
import resource
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import rpe_deterministic_risk as D

SOURCE=dict(scope='ARTIFICIAL_BENCHMARK_ONLY_NO_REAL_SIGNAL',
    path='docs/research/rpe_dp_finite_scope_preparation_20261114/BENCHMARK_SYNTHETIC.py')


def domains(groups,limit):
    return groups[0]+sum(min(limit,(1<<(m-1))*groups[m-1])*groups[m]
                          for m in range(1,len(groups)))


def artificial_case(spec):
    wall=time.perf_counter(); cpu=time.process_time()
    with ExitStack() as guards:
        for module,name in ((D.S,'simulate'),(D.S,'simulate_prepared'),(D.K,'diagnose'),
                            (D.p,'run_missing'),(D.p.SavedArtifacts,'read')):
            guards.enter_context(patch.object(module,name,
                side_effect=AssertionError('MC/physical provider/saved data forbidden')))
        tables=[]
        for m,n in enumerate(spec['N_XY_each_axis']):
            phase=.7*(1<<m)
            raw=dict(real=.85*math.cos(phase),imag=.85*math.sin(phase))
            tables.append(D.round_distribution(dict(m=m,raw=raw,N_X=n,N_Y=n),
                h=10.,target_energy=-.07,radius=1.,probability_rule=D.EXACT_AFFINE,
                maximum_outcomes=(n+1)**2))
        groups=[len({(v['zero'],float(v['principal']).hex()) for v in t['outcomes']}) for t in tables]
        limit=spec['maximum_frontier_states']; maximum_pairs=domains(groups,limit)
        cells=[t['cells'] for t in tables]
        certificate=D.verify_good_transitions(tables,
            maximum_pairs=max(1,sum(a*b for a,b in zip(cells,cells[1:]))))
        assert certificate['status']=='VERIFIED_FINITE_BINARY64_SUPPORT'
        dp=D._refine_states(tables,h=10.,target=-.07,lo=Fraction(0),hi=Fraction(20),
            resolved_limit=Fraction(10),scope=dict(source=SOURCE,
                maximum_frontier_states=limit,maximum_transition_pairs=maximum_pairs))
        assert dp['MSE_upper'] is not None
    bits=[];denominator=1
    for table in tables:
        denominator*=table['denominator'];bits.append(denominator.bit_length())
    return dict(label=spec['label'],scope='ARTIFICIAL_ONLY_NO_MOLECULAR_MODEL',source=SOURCE,
        N_XY_each_axis=spec['N_XY_each_axis'],h=10.,target=-.07,
        raw_recipe='0.85 * exp(i * 0.7 * 2**m)',
        maximum_frontier_states=limit,maximum_transition_pairs=maximum_pairs,
        grouped_count_outcomes=groups,prefix_denominator_bits=bits,
        certificate=certificate,status=dp['status'],transition_pairs=dp['transition_pairs'],
        rounds=dp['rounds'],residual_probability=dp['residual_probability'],
        wall_seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-cpu,
        peak_worker_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        real_H4_evaluations=0,MC_calls=0)


def geometry():
    # Dimensions of the unchanged control vector. Zero synthetic raw/target
    # makes this a count-grid computation, independent of any saved signal.
    shots=[64,60,56,52,47,43,38,34,30,25,21,17,12]
    rows=[]
    for m,n in enumerate(shots):
        table=D.round_distribution(dict(m=m,raw=dict(real=0.,imag=0.),N_X=n,N_Y=n),
            h=10.,target_energy=0.,radius=1.,probability_rule=D.EXACT_AFFINE,
            maximum_outcomes=(n+1)**2)
        g=len({(v['zero'],float(v['principal']).hex()) for v in table['outcomes']})
        rows.append(dict(m=m,N_X=n,N_Y=n,count_cells=(n+1)**2,grouped_count_outcomes=g))
    groups=[row['grouped_count_outcomes'] for row in rows]
    proposals=[]
    for limit in (4096,8192,16384):
        proposals.append(dict(maximum_frontier_states=limit,maximum_transition_pairs=domains(groups,limit),
            transient_state_count_bound=max((1<<m)*g for m,g in enumerate(groups))))
    return dict(scope='COUNT_GEOMETRY_ONLY_ARTIFICIAL_ZERO_RAW_AND_ZERO_TARGET',
        probability_or_risk_of_saved_signal_computed=False,rounds=rows,proposals=proposals)


def arithmetic_probes():
    rows=[];repetitions=16384
    for left_bits,right_bits in ((16384,8192),(65536,2048),(65536,8192)):
        left=(1<<left_bits)-(1<<(left_bits//2))+12345
        right=(1<<right_bits)-(1<<(right_bits//2))+54321
        start=time.process_time();check=0
        for _ in range(repetitions):
            check^=(left*right+left)&1
        rows.append(dict(left_bits=left_bits,right_bits=right_bits,repetitions=repetitions,
            cpu_seconds=time.process_time()-start,checksum=check))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path);args=parser.parse_args()
    args.output.mkdir()
    def write(name,data):
        with (args.output/name).open('x') as stream:
            json.dump(data,stream,indent=2,allow_nan=False);stream.write('\n')
    write('COUNT_GEOMETRY.json',geometry())
    write('ARITHMETIC_PROBES.json',arithmetic_probes())
    cases=[dict(label='PREFIX_N64_N60_K8192',N_XY_each_axis=[64,60],maximum_frontier_states=8192),
           dict(label='ARTIFICIAL_13ROUND_N8_K4096',N_XY_each_axis=[8]*13,maximum_frontier_states=4096),
           dict(label='ARTIFICIAL_13ROUND_N8_K8192',N_XY_each_axis=[8]*13,maximum_frontier_states=8192)]
    with ProcessPoolExecutor(max_workers=len(cases)) as pool:
        futures={pool.submit(artificial_case,spec):spec['label'] for spec in cases}
        for future in as_completed(futures):
            result=future.result();write(result['label']+'.json',result)
            print(result['label'],'DONE',result['wall_seconds'],'seconds',flush=True)
    print('COMPLETE: artificial-only cost probes; no saved signal or result read',flush=True)


if __name__=='__main__':main()
