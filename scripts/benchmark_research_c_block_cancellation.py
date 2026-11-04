#!/usr/bin/env python3
"""Artificial fresh-process comparison of tightness and cost. No H4 records."""
import argparse,json,resource,statistics,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def records(scenario):
    from trottertracks.c_pr_aware_df_fixed_certificate import identity
    from benchmark_research_c_compiled_raw_local import gate
    if scenario=='artificial_P0':
        from benchmark_research_c_compiled_raw_local import records as old_records
        return old_records('raw_primitives')
    a,b=identity(9),identity(9)
    if scenario=='Pauli_cancellation_full512':
        for c,t in ((0,1),(2,3),(4,5),(6,7)):
            a['gates'] += [gate('z',[c]),gate('cx',[c,t]),gate('z',[c])]
            b['gates'] += [gate('cx',[c,t])]
        a['phase_hex']=(2**-40).hex()
    elif scenario=='continuous_cancellation_loss':
        a['gates']=[gate('rz',[0],.25),gate('cx',[0,1]),gate('rz',[0],-.25)]
        b['gates']=[gate('cx',[0,1])]
    else:raise ValueError('ARTIFICIAL_SCENARIO')
    return a,b


def worker(mode,scenario):
    from fractions import Fraction as F
    from trottertracks import c_arb_column_batch as columns,c_cx_structural_blocks as blocks
    from trottertracks import c_pauli_frame_enclosure as frame,c_relative_pauli_enclosure as relative
    from trottertracks.c_pr_aware_df_zsxx_enclosure import runtime_model
    runtime_model();a,b=records(scenario)
    before=[resource.getrusage(k) for k in (resource.RUSAGE_SELF,resource.RUSAGE_CHILDREN)];tick=time.perf_counter()
    method=dict(columns=columns,blocks=blocks,frame=frame,relative=relative)[mode]
    value=method.compare(a,b,parallel=True)
    wall=time.perf_counter()-tick;after=[resource.getrusage(k) for k in (resource.RUSAGE_SELF,resource.RUSAGE_CHILDREN)]
    cpu=sum(v.ru_utime+v.ru_stime-u.ru_utime-u.ru_stime for u,v in zip(before,after))
    return dict(mode=mode,scenario=scenario,status=value['status'],upper_exact=value['upper_exact'],
        upper_display=None if value['upper_exact'] is None else float(F(value['upper_exact'])),
        wall_seconds=wall,CPU_seconds=cpu,parent_RSS_KiB=after[0].ru_maxrss,child_RSS_KiB=after[1].ru_maxrss,
        RSS_scope='per-process lifetime high-water,not aggregate peak',dimension=1<<a['qubits'],
        gate_counts=[len(r['gates']) for r in (a,b)],CX_counts=[sum(g['name']=='cx' for g in r['gates']) for r in (a,b)],
        internal_blocks=value.get('internal_blocks',value.get('internal_block_occurrences')),
        maximum_support=value.get('maximum_support'),used_unique_frame_jobs=value.get('used_unique_frame_jobs'),
        resources={k:v for k,v in value.items() if k.endswith('resources') or k=='resource_plan'})


def benchmark():
    rows=[]
    for scenario in ('Pauli_cancellation_full512','artificial_P0','continuous_cancellation_loss'):
        for rep in range(3):
            modes=('columns','blocks','frame','relative') if rep%2==0 else ('relative','frame','blocks','columns')
            for mode in modes:
                result=subprocess.run([sys.executable,'-B',__file__,'--worker',mode,'--scenario',scenario],
                    cwd=ROOT,check=True,capture_output=True,text=True)
                rows.append(dict(json.loads(result.stdout),repeat=rep))
    summary=[]
    for scenario in dict.fromkeys(x['scenario'] for x in rows):
        methods={}
        for mode in ('columns','blocks','frame','relative'):
            group=[x for x in rows if x['scenario']==scenario and x['mode']==mode]
            methods[mode]=dict(median_wall_seconds=statistics.median(x['wall_seconds'] for x in group),
                median_CPU_seconds=statistics.median(x['CPU_seconds'] for x in group),
                parent_RSS_KiB=max(x['parent_RSS_KiB'] for x in group),child_RSS_KiB=max(x['child_RSS_KiB'] for x in group),
                upper_display=group[0]['upper_display'],status=group[0]['status'])
        summary.append(dict(scenario=scenario,methods=methods,columns_over_frame_wall=methods['columns']['median_wall_seconds']/methods['frame']['median_wall_seconds']))
    return dict(kind='ARTIFICIAL_ONLY_NO_SAVED_MOLECULAR_OPERATOR_EVALUATION',repetitions=3,rows=rows,summary=summary,
        new_H4_Arb_evaluations=0,real_H4_speedup_measured=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=['columns','blocks','frame','relative']);p.add_argument('--scenario')
    args=p.parse_args();print(json.dumps(worker(args.worker,args.scenario) if args.worker else benchmark(),ensure_ascii=False))
