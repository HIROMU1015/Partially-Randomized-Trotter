#!/usr/bin/env python3
"""Invented benchmark only; no scientific input paths or execution switch."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from fractions import Fraction as F
import json
import multiprocessing
import os
from pathlib import Path
import resource
import statistics
import time


def job(spec):
    from flint import ctx,acb_mat
    from trottertracks import c_direct_round_signal as direct
    from trottertracks import c_per_word_population_range as population
    from trottertracks import c_pr_aware_df_fixed_certificate as fixed
    from trottertracks import c_pr_aware_df_zsxx_enclosure as z
    from trottertracks import c_fresh_iid_calibration as fresh
    from trottertracks.c_pr_aware_df_fixed_car import pack,unpack
    from trottertracks.c_direct_round_artificial import fixture
    n,m,method,repeat=spec
    args=fixture(m_round=m,n=n)
    # Four SX gates and paired CX are exact identities, with nontrivial
    # intermediate state actions. The scalar literal target remains the same.
    for i,part in enumerate(args['occurrences']):
        record=deepcopy(part['compiled']['record'])
        for q in range(n-1):record['gates'] += [dict(name='sx',qubits=[q],params_hex=[])]*4
        for q in range(n-2):record['gates'] += [dict(name='cx',qubits=[q,q+1],params_hex=[])]*2
        from trottertracks import c_iid_calibration_integration as integration
        part['raw']=integration.bind_operator(record,args['sample'],args['context'],i,role='raw')
        part['compiled']=integration.bind_operator(record,args['sample'],args['context'],i,
            role='compiled',raw_id=part['raw']['record_id'])
        part['actual_inverse']=direct.bind_inverse(fixed.adjoint(record),part['compiled'],args['sample'],args['context'],i)
    # Warm the same pinned runtime in both workers. Otherwise only direct's
    # first call pays for Qiskit import inside runtime_model().
    z.runtime_model()
    # Import and fixture preparation are excluded equally; literal ROUND
    # assembly is charged to the operator method. Both evaluate the same word.
    wall=time.perf_counter();cpu=time.process_time()
    if method=='direct':
        result=direct.evaluate_round(**args)
        upper=max(abs(F(x)) for a in ('X','Y') for x in result[a])
        qualification=direct.QUALIFICATION;columns=0
    else:
        record=fixed.identity(n)
        for part in args['occurrences']:record['gates']+=part['compiled']['record']['gates']
        reference=deepcopy(args['occurrences'][0]['reference'])
        with z._BALL_LOCK,ctx.workprec(192):
            value=acb_mat([[1]])
            for part in args['occurrences']:
                value=unpack(part['reference']['blocks'][0]['entries'],1)*value
            for block in reference['blocks']:block['entries']=pack(value)
        result=population.compare_literal(record,reference,parallel=False)
        upper=F(result['upper_exact']);columns=result['columns_evaluated']
        qualification='full_operator_2_via_all_columns'
    elapsed=time.perf_counter()-wall;used=time.process_time()-cpu
    return dict(qubits=n,system_dimension=1 << (n-1),controlled_dimension=1 << n,m=m,
        occurrences=1 << m,round_length=1 << m,method=method,repeat=repeat,
        forward_gates_per_occurrence=len(args['occurrences'][0]['compiled']['record']['gates']),
        forward_round_gates=sum(len(p['compiled']['record']['gates']) for p in args['occurrences']),
        wall_seconds=elapsed,CPU_seconds=used,
        process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        RSS_scope='isolated worker process high-water mark;includes imports and fixture',
        columns_evaluated=columns,qualification=qualification,upper_exact=str(upper),
        precision_bits=192,artificial_only=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    # 24 small independent CPU tasks. Each has a fresh process so peak RAM
    # and CPU measurements are attributable, with no reused worker high-water.
    specs=[(n,m,method,r) for n,m in ((3,0),(5,0),(5,2),(6,2))
           for method in ('direct','operator') for r in range(3)]
    from trottertracks.c_pr_aware_df_fixed_certificate import resource_plan
    plan=resource_plan(1 << 6);workers=min(plan['workers'],len(specs))
    rows=[];tick=time.perf_counter();failure=None
    try:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                                 max_tasks_per_child=1) as pool:
            futures=[pool.submit(job,s) for s in specs]
            with (a.output/'measurements.jsonl').open('x') as stream:
                for f in as_completed(futures):
                    row=f.result();rows.append(row)
                    stream.write(json.dumps(row,sort_keys=True)+'\n');stream.flush()
        summary=[]
        for n,m in ((3,0),(5,0),(5,2),(6,2)):
            group=[x for x in rows if x['qubits']==n and x['m']==m]
            timing={k:statistics.median(x['wall_seconds'] for x in group if x['method']==k)
                    for k in ('direct','operator')}
            summary.append(dict(qubits=n,m=m,median_wall_seconds=timing,
                observed_operator_over_direct_wall_ratio=timing['operator']/timing['direct']))
        result=dict(kind='ARTIFICIAL_ONLY',measurements=len(rows),repetitions=3,results=summary,
            resource_plan=plan,workers_used=workers,whole_batch_wall_seconds=time.perf_counter()-tick,
            comparison='same invented native/literal pair;direct includes shells and state inverse witnesses;operator certifies all columns without shell/inverse costs',
            parallelism='independent isolated cases;each tiny operator uses serial columns to avoid nested oversubscription',
            runtime_imports_warmed_for_both_methods=True,
            scientific_calls=0,real_H4_speedup_claimed=False)
        (a.output/'benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result))
    except BaseException as exc:
        failure=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        (a.output/'STOP.json').write_text(json.dumps(dict(mandatory_stop=True,artificial_only=True,
            completed=len(rows),failure=failure,retry=0,resume=0,scientific_calls=0),indent=2)+'\n')


if __name__=='__main__':main()
