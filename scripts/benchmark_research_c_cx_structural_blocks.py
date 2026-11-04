#!/usr/bin/env python3
"""Artificial-only fresh-process benchmark, using the existing column baseline."""
import argparse
import json
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def records(scenario):
    from trottertracks.c_pr_aware_df_fixed_certificate import identity
    from benchmark_research_c_compiled_raw_local import gate
    a,b=identity(9),identity(9)
    if scenario=='unmatched_full512':
        for q in (0,2,4,6):
            a['gates'] += [gate('cx',[q,q+1]),gate('rz',[q],.125),gate('cx',[q,q+1])]
            b['gates'] += [gate('rz',[q],.125)]
        a['phase_hex']=(2**-40).hex()
    elif scenario=='coarse_telescope':
        a['gates']=[gate('rz',[0],.25),gate('cx',[0,1]),gate('rz',[0],-.25)]
        b['gates']=[gate('cx',[0,1])]
    elif scenario=='raw_primitive':
        from benchmark_research_c_compiled_raw_local import records as old_records
        return old_records('raw_primitives')
    else:raise ValueError('ARTIFICIAL_SCENARIO')
    return a,b


def worker(mode,scenario):
    from fractions import Fraction as F
    from trottertracks import c_cx_structural_blocks as new,c_compiled_raw_local as strict,c_arb_column_batch as old
    from trottertracks.c_pr_aware_df_zsxx_enclosure import runtime_model
    runtime_model();a,b=records(scenario)
    before=[resource.getrusage(k) for k in (resource.RUSAGE_SELF,resource.RUSAGE_CHILDREN)]
    tick=time.perf_counter()
    if mode=='columns':result=old.compare(a,b,parallel=True,batch_size=16)
    elif mode=='strict':result=strict.compare(a,b,parallel=True)
    else:result=new.compare(a,b,parallel=True)
    wall=time.perf_counter()-tick
    after=[resource.getrusage(k) for k in (resource.RUSAGE_SELF,resource.RUSAGE_CHILDREN)]
    cpu=sum(v.ru_utime+v.ru_stime-u.ru_utime-u.ru_stime for u,v in zip(before,after))
    return dict(mode=mode,scenario=scenario,status=result['status'],wall_seconds=wall,CPU_seconds=cpu,
        parent_RSS_KiB=after[0].ru_maxrss,child_RSS_KiB=after[1].ru_maxrss,
        RSS_scope='per-process lifetime high-water,not aggregate peak',
        upper_exact=result['upper_exact'],upper_display=None if result['upper_exact'] is None else float(F(result['upper_exact'])),
        input_gates=[len(x['gates']) for x in (a,b)],input_CX=[sum(g['name']=='cx' for g in x['gates']) for x in (a,b)],
        dimension=1<<a['qubits'],internal_blocks=result.get('internal_block_occurrences'),
        max_support=result.get('maximum_support'),unique_blocks=result.get('unique_blocks'),
        resource_plan={k:v for k,v in result.items() if k.endswith('_resources') or k=='resource_plan'})


def benchmark():
    rows=[]
    for scenario in ('unmatched_full512','raw_primitive','coarse_telescope'):
        for rep in range(3):
            modes=('columns','strict','blocks') if rep%2==0 else ('blocks','strict','columns')
            for mode in modes:
                result=subprocess.run([sys.executable,'-B',__file__,'--worker',mode,'--scenario',scenario],
                    cwd=ROOT,check=True,capture_output=True,text=True)
                rows.append(dict(json.loads(result.stdout),repeat=rep))
    summary=[]
    for scenario in dict.fromkeys(x['scenario'] for x in rows):
        groups={}
        for mode in ('columns','strict','blocks'):
            values=[x for x in rows if x['scenario']==scenario and x['mode']==mode]
            groups[mode]=dict(median_wall_seconds=statistics.median(x['wall_seconds'] for x in values),
                median_CPU_seconds=statistics.median(x['CPU_seconds'] for x in values),
                max_parent_RSS_KiB=max(x['parent_RSS_KiB'] for x in values),max_child_RSS_KiB=max(x['child_RSS_KiB'] for x in values),
                upper_display=values[0]['upper_display'],status=values[0]['status'])
        summary.append(dict(scenario=scenario,methods=groups,
            columns_over_blocks_wall=groups['columns']['median_wall_seconds']/groups['blocks']['median_wall_seconds'],
            tightness_qualified_for_downstream=False if scenario=='coarse_telescope' else None))
    return dict(kind='ARTIFICIAL_ONLY',repetitions=3,rows=rows,summary=summary,
                real_H4_operator_evaluations=0,real_H4_speedup_measured=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=['columns','strict','blocks']);p.add_argument('--scenario')
    a=p.parse_args();print(json.dumps(worker(a.worker,a.scenario) if a.worker else benchmark(),ensure_ascii=False))
