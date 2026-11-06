#!/usr/bin/env python3
"""Invented CDF8/r16,32,64 circuits only; no saved scientific input is read."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def task(r):
    for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    import numpy as np
    from scipy.linalg import expm
    from threadpoolctl import threadpool_limits
    from trottertracks import c_pr_aware_df_stage_b2 as b2
    from trottertracks.c1_h4_strong_baseline import StrongBuilder
    from trottertracks.c1_h4_native_fair import decode, compile_checked
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    from trottertracks.pr_dictionary_validation.f2_execution import circuit_metrics

    with threadpool_limits(limits=1):
        rng = np.random.default_rng(2026110601)
        j = rng.normal(size=(8,4,4))*.003
        j += j.transpose(0,2,1)
        skews = rng.normal(size=(8,4,4))*.15
        u = np.asarray([expm(x-x.T) for x in skews])
        h = np.diag(np.linspace(-.2,.3,4)); h[0,1] = h[1,0] = .07
        g = np.zeros((4,)*4)
        arm = b2.b.prepare_arm(h,g,.31,j,u,role='proposed',
            source_identity='INVENTED_CDF8',input_identity='SYNTHETIC',ld=2)
        tail, config, distribution = b2.b.configure(arm,Fraction(1,10),r,0)
        events = b2.b.independent_occurrence(arm,tail,distribution,config,
            master=2026110601,round_index=-1,axis='ARTIFICIAL_BENCHMARK',shot=0,occurrence=0)
        tick = time.perf_counter()
        raw = b2.b.build_kernel(arm,Fraction(1,10),events,config,
            builder=StrongBuilder(relative_fusion=True))
        build_seconds = time.perf_counter()-tick
        tick = time.perf_counter(); compiled = compile_checked(raw,CompilerProfile())
        compile_seconds = time.perf_counter()-tick
        circuits = dict(raw=raw,compiled=compiled,
            raw_adjoint=b2.b.native.actual_adjoint(raw),
            compiled_adjoint=b2.b.native.actual_adjoint(compiled))
        timings, differences = {}, {}
        for name,qc in circuits.items():
            timings[name] = {}
            matrices = {}
            for backend in ('aer_full','operator'):
                tick = time.perf_counter();matrices[backend] = decode(qc,backend=backend)
                timings[name][backend] = time.perf_counter()-tick
                print(json.dumps(dict(r=r,circuit=name,backend=backend,
                    seconds=timings[name][backend])),flush=True)
            delta = matrices['aer_full']-matrices['operator']
            differences[name] = dict(max_entry=float(np.max(abs(delta))),
                frobenius=float(np.linalg.norm(delta,'fro')))
            if differences[name]['max_entry'] > 1e-10:
                raise ValueError('ARTIFICIAL_FULL_MATRIX_PARITY_FAILURE')
        return dict(r=r,qubits=9,dimension=512,raw_operations=len(raw.data),
            compiled_metrics=circuit_metrics(compiled),build_seconds=build_seconds,
            compile_seconds=compile_seconds,timings=timings,differences=differences,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            scientific_performance_claim=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('FRESH_BENCHMARK_OUTPUT_REQUIRED')
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows=list(pool.map(task,(16,32,64)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(dict(scope='ARTIFICIAL_ONLY',results=rows),stream,indent=2)
        stream.write('\n')


if __name__=='__main__':main()
