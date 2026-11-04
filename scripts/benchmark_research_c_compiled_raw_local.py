#!/usr/bin/env python3
"""Invented circuits only; isolated repetitions, evaluator CPU/RSS and profiles."""
import argparse
import cProfile
import io
import json
from pathlib import Path
import pstats
import resource
import subprocess
import sys
import time
from statistics import median

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def gate(name,qs,*params):
    return dict(name=name,qubits=qs,params_hex=[float(x).hex() for x in params])


def records(scenario):
    from trottertracks import c_pr_aware_df_fixed_certificate as f
    if scenario in ('no_common_full512','profile'):
        n=9 if scenario=='no_common_full512' else 4
        a,b=f.identity(n),f.identity(n)
        for j in range(6 if n==9 else 12):
            q=j%(n-1);alpha=(j+1)*.03125;beta=.1
            b['gates'] += [gate('rz',[q],alpha),gate('rz',[q],beta),gate('cx',[q,q+1])]
            a['gates'] += [gate('rz',[q],alpha+beta),gate('cx',[q,q+1])]
        b['gates'] += [gate('rz',[0],.1),gate('rz',[0],.3)]
        a['gates'] += [gate('rz',[0],.1+.3)]
        return a,b
    if scenario=='coarse_commuting_error':
        a,b=f.identity(3),f.identity(3)
        a['gates']=[gate('rz',[0],.25),gate('cx',[0,1]),gate('rz',[0],-.25)]
        b['gates']=[gate('cx',[0,1])];return a,b
    if scenario=='common_full512':
        a,b=records('no_common_full512');a['gates']=list(b['gates']);a['phase_hex']=(2**-40).hex()
        return a,b
    if scenario=='raw_primitives':
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import XXPlusYYGate,RZZGate
        from trottertracks.c1_h4_native_fair import compile_checked,CompilerProfile
        q=QuantumCircuit(3);q.global_phase=.125
        for j in range(2):
            q.append(XXPlusYYGate(.375+j*.0625,.25),[0,1]);q.crz(.125,1,0)
            q.append(RZZGate(.25).control(1,annotated=False),[2,0,1])
        return f.record(compile_checked(q,CompilerProfile())),f.record(q)
    raise ValueError('ARTIFICIAL_SCENARIO')


def evaluate(mode,a,b,parallel=True):
    from trottertracks import c_arb_column_batch as old
    from trottertracks import c_compiled_raw_local as new
    return (old.compare(a,b,parallel=parallel,batch_size=16) if mode=='old' else
            new.compare(a,b,parallel=parallel,commute_CX=(mode=='local_commute')))


def worker(mode,scenario):
    from fractions import Fraction as F
    from trottertracks.c_pr_aware_df_zsxx_enclosure import runtime_model
    runtime_model();a,b=records(scenario)
    before=[resource.getrusage(k) for k in (resource.RUSAGE_SELF,resource.RUSAGE_CHILDREN)]
    tick=time.perf_counter();r=evaluate(mode,a,b);wall=time.perf_counter()-tick
    after=[resource.getrusage(k) for k in (resource.RUSAGE_SELF,resource.RUSAGE_CHILDREN)]
    cpu=sum((v.ru_utime+v.ru_stime-u.ru_utime-u.ru_stime) for u,v in zip(before,after))
    return dict(mode=mode,scenario=scenario,wall_seconds=wall,CPU_seconds_including_children=cpu,
        max_parent_RSS_KiB=after[0].ru_maxrss,max_child_RSS_KiB=after[1].ru_maxrss,
        RSS_scope='per-process lifetime high-water marks;not aggregate peak or a before/after subtraction',
        upper_exact=r['upper_exact'],upper_display=float(F(r['upper_exact'])),status=r['status'],
        resources={k:v for k,v in r.items() if k.endswith('resources') or k=='resource_plan'},
        input_gate_counts=[len(a['gates']),len(b['gates'])],dimension=1<<a['qubits'])


def benchmark():
    rows=[]
    for scenario in ('no_common_full512','raw_primitives','coarse_commuting_error','common_full512'):
        for repeat in range(3):
            for mode in (('old','local') if repeat%2==0 else ('local','old')):
                result=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),
                    '--worker',mode,'--scenario',scenario],cwd=ROOT,check=True,capture_output=True,text=True)
                rows.append(dict(json.loads(result.stdout),repeat=repeat))
    # Explicit counterexample to treating every fast upper bound as useful.
    a,b=records('coarse_commuting_error');extra=worker('local_commute','coarse_commuting_error')
    summary=[]
    for scenario in dict.fromkeys(x['scenario'] for x in rows):
        selected=[x for x in rows if x['scenario']==scenario];v={}
        for mode in ('old','local'):
            group=[x for x in selected if x['mode']==mode]
            v[mode]=dict(median_wall_seconds=median(x['wall_seconds'] for x in group),
                median_CPU_seconds=median(x['CPU_seconds_including_children'] for x in group),
                max_parent_RSS_KiB=max(x['max_parent_RSS_KiB'] for x in group),
                max_child_RSS_KiB=max(x['max_child_RSS_KiB'] for x in group),
                upper_exact=group[0]['upper_exact'],upper_display=group[0]['upper_display'])
        summary.append(dict(scenario=scenario,methods=v,
            ratio_old_over_local=v['old']['median_wall_seconds']/v['local']['median_wall_seconds'],
            useful_bound_demonstration=scenario!='coarse_commuting_error'))
    profiles={}
    for mode in ('old','local'):
        a,b=records('profile');prof=cProfile.Profile();prof.enable();evaluate(mode,a,b,parallel=False);prof.disable()
        stream=io.StringIO();pstats.Stats(prof,stream=stream).sort_stats('cumulative').print_stats(20)
        profiles[mode]=stream.getvalue()
    return dict(kind='ARTIFICIAL_ONLY_NO_MOLECULAR_RECORDS',repetitions=3,rows=rows,summary=summary,
        coarse_counterexample_known_commutation_variant=extra,profiles=profiles,
        actual_H4_speedup_measured=False,new_H4_scientific_calls=0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=['old','local','local_commute']);p.add_argument('--scenario')
    a=p.parse_args();print(json.dumps(worker(a.worker,a.scenario) if a.worker else benchmark(),ensure_ascii=False))
