#!/usr/bin/env python3
"""Artificial CDF8 benchmark only; never loads saved molecular arrays/results."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests'/'management')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=8)
    parser.add_argument('--r',type=int,default=32)
    args=parser.parse_args()
    if not 1<=args.samples<=32 or args.r<1:parser.error('samples1..32,r>=1')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    from threadpoolctl import threadpool_limits
    from test_research_c_native_process import toy,caller,folder,policy,old_samples
    from trottertracks import c_native_process as process
    from trottertracks import c_pr_integrated_performance as perf
    def forbidden(*a,**kw):raise RuntimeError('NO_SAVED_SCIENTIFIC_ARRAYS')
    perf.loader.load=forbidden
    setup=time.monotonic()
    with threadpool_limits(limits=1):a,prepared=toy(n=4,rank=8,r=args.r)
    # Both schedulers start with exactly the same warmed Gaussian cache. The
    # thread baseline must not warm the process copy before measurement.
    process_prepared=dict(prepared,builder=deepcopy(prepared['builder']))
    setup=time.monotonic()-setup
    c=dict(source_manifest_sha256='ARTIFICIAL_PROCESS_BENCHMARK_V1')
    p=policy(args.r);indices=range(args.samples);results={}
    def measure(name,fn):
        tick=time.monotonic();cpu=time.process_time()
        child=resource.getrusage(resource.RUSAGE_CHILDREN)
        rows=fn()
        end=resource.getrusage(resource.RUSAGE_CHILDREN)
        results[name]=dict(wall_seconds=time.monotonic()-tick,
            CPU_seconds=time.process_time()-cpu+end.ru_utime+end.ru_stime-child.ru_utime-child.ru_stime)
        results[name]['average_CPU_cores']=results[name]['CPU_seconds']/results[name]['wall_seconds']
        print(json.dumps(dict(stage=name,**results[name])),flush=True)
        return rows
    old=folder(out/'threads');new=folder(out/'process')
    baseline=measure('threads',lambda:old_samples(c,old,caller(),a,p,prepared,indices))
    call=caller();extra={}
    def parallel():
        rows,plan,reports=process.run_samples(c,new,call,a,p,process_prepared,indices=indices)
        extra.update(plan=plan,worker_pids=sorted({x['pid'] for x in reports}),
            worker_CPU_seconds=sum(x['CPU_seconds'] for x in reports),
            max_worker_peak_RSS_KiB=max(x['peak_RSS_KiB'] for x in reports),
            all_reference_mmaps_readonly=all(x['reference_readonly'] for x in reports))
        return rows
    rows=measure('process',parallel)
    if rows!=baseline:raise AssertionError('METRIC_MISMATCH')
    for i in indices:
        for kind in ('events','raw','compiled','adjoint','XY','metrics'):
            path=Path('samples')/f'sample{i:02d}_{kind}.json'
            if json.loads((old/path).read_text())!=json.loads((new/path).read_text()):
                raise AssertionError(str(path))
    result=dict(status='ARTIFICIAL_EQUIVALENCE_AND_TIMING_COMPLETE',molecular_inputs_loaded=0,
        invented_fixture='seed724 symmetric random J, QR real U; n4/rank8/LD2 linear-Z relocated',
        samples=args.samples,events_per_sample=args.r,full_input_output_dimension=512,
        checks='raw/compiled/actual raw+compiled adjoint/full512/phase/XY; same events and all six metrics',
        setup_seconds=setup,measurements=results,speedup=results['threads']['wall_seconds']/results['process']['wall_seconds'],
        process=extra,scientific_performance_claim=False,
        limitations='single paired local artificial timing; current fixed scientific run is unchanged; startup and per-sample output included')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
