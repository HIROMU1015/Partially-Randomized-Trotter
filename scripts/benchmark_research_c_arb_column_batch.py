#!/usr/bin/env python3
"""Invented records only. No molecular file, compiler or RNG is accessed."""
import json,os,platform,statistics,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_arb_column_batch as fast
from trottertracks import c_pr_aware_df_fixed_certificate as old


def pair(n,common):
    a=old.identity(n);b=old.identity(n)
    gates=[dict(name='rz',qubits=[i%n],params_hex=[(.125+i/64).hex()]) if i%2==0 else
           dict(name='cx',qubits=[i%n,(i+1)%n],params_hex=[]) for i in range(common)]
    a['gates']=gates+[dict(name='ry',qubits=[0],params_hex=[(.2).hex()])]+list(reversed(gates))
    b['gates']=gates+[dict(name='ry',qubits=[0],params_hex=[(.2+2**-32).hex()])]+list(reversed(gates))
    a['phase_hex']=(.125).hex();b['phase_hex']=(.125+2**-35).hex()
    return a,b


if __name__=='__main__':
    results=[]
    for name,n,common,cancel,parallel in [('batch_only',4,8,False,False),('common_unitaries_full512',9,12,True,True)]:
        a,b=pair(n,common);times={'old':[],'batch':[]};bounds={}
        for rep in range(3):
            for method in (('old','batch') if rep%2==0 else ('batch','old')):
                tick=time.monotonic();v=(old.compare(a,b,parallel=parallel) if method=='old' else fast.compare(a,b,batch_size=16,parallel=parallel,cancel_common=cancel));times[method].append(time.monotonic()-tick)
                bounds[method]=dict(upper_exact=v['upper_exact'],lower_exact=v['lower_exact'],columns_evaluated=v['columns_evaluated'],precision_bits=v['precision_bits'],resource_plan=v['resource_plan'])
        results.append(dict(name=name,qubits=n,common_gates_per_side=common,cancellation=cancel,parallel=parallel,
            times_seconds=times,median_seconds={k:statistics.median(v) for k,v in times.items()},
            measured_median_ratio_old_over_batch=statistics.median(times['old'])/statistics.median(times['batch']),bounds=bounds))
    p=ROOT/'artifacts/research_c_stage_p_totality_calibration/2026-11-04/technical/artificial_benchmark.json';p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(dict(kind='ARTIFICIAL_ONLY',python=platform.python_version(),CPU_affinity=len(os.sched_getaffinity(0)),results=results,H4_speedup_measured=False,new_H4_scientific_calls=0),indent=2)+'\n')
    print(json.dumps(results))
