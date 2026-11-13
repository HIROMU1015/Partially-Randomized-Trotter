"""Integrated B scalar/batch comparison on artificial 13-round inputs only.

Reuse A's benchmark method: fresh sequential subprocess per path, repeated wall
time and Linux process peak RSS. No physical artifact is read or decoded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src')]


def worker(method, trials, repeats):
    import math
    import numpy as np
    from trottertracks import rpe_decoder_statistics as S
    from trottertracks import rpe_decoder_fast as F
    # Equal import baseline in both fresh workers, outside timing. This is the
    # reference decoder, not a call on any saved scientific input.
    from trotterlib.rpe_four_round_phase_validation import reconstruct_rpe_phase
    rows=[dict(m=m,raw=dict(real=.43*math.cos(.237*(1 << m)),imag=.43*math.sin(.237*(1 << m))),
        N_X=31,N_Y=37) for m in range(13)]
    policy=dict(kind=S.IID,source=dict(fixture='ARTIFICIAL_R3_BENCHMARK'),target_law='ideal_finite_RTE_law')
    extra=dict(batch_decoder=F.decode_sampled_counts) if method=='batch' else {}
    baseline=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    times=[]; digests=[]
    for _ in range(repeats):
        start=time.perf_counter()
        result=S.simulate(rows,M=12,h=.1,prior=[-4.,4.],repetitions=trials,seed=441,policy=policy,**extra)
        times.append(time.perf_counter()-start)
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        for trial in result['trials']: trial.pop('decoder_acceleration_source',None)
        digests.append(hashlib.sha256(json.dumps(result,sort_keys=True,allow_nan=False).encode()).hexdigest())
        del result
    if len(set(digests))!=1: raise AssertionError('Repeated seeded output differs')
    return dict(method=method,trials=trials,rounds=13,repeats=repeats,seconds=times,
        median_seconds=statistics.median(times),baseline_peak_RSS_MiB=baseline/1024.,
        peak_RSS_MiB=peak/1024.,output_digest=digests[0],numpy_version=np.__version__)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--trials',type=int,default=512)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--worker',choices=('scalar','batch'))
    a=parser.parse_args()
    if min(a.trials,a.repeats)<1: parser.error('Positive artificial trial/repeat counts required')
    if a.worker:
        print(json.dumps(worker(a.worker,a.trials,a.repeats),allow_nan=False));return
    if a.out is None: parser.error('--out required')
    if a.out.exists(): raise ValueError('Fresh output required')
    records=[]
    for method in ('scalar','batch'):
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker',method,
            '--trials',str(a.trials),'--repeats',str(a.repeats)]
        record=json.loads(subprocess.check_output(command,cwd=ROOT,text=True))
        records.append(record);print(json.dumps(record),flush=True)
    if records[0]['output_digest']!=records[1]['output_digest']: raise AssertionError('Scalar/batch outputs differ')
    result=dict(scope='ARTIFICIAL_ONLY_NO_H4_OR_SAVED_SCIENCE',python=sys.version,platform=platform.platform(),
        B_base='bf515a976f1987ad6a19c61d14689b27b0894542',A_acceleration='7f1f9053dfbbb81646ac9e483a0354d01cbec535',
        rows_definition='m=0..12; raw=(.43*cos(.237*2**m), .43*sin(.237*2**m)); N_X=31, N_Y=37',
        PCG64_seed=441,records=records,all_trial_outputs_identical_except_acceleration_source=True,
        median_speedup=records[0]['median_seconds']/records[1]['median_seconds'],
        timing='Full integrated simulate includes binomial/count conversion/math.atan2/branches/trial dicts; excludes imports, startup, digest',
        memory='Fresh Linux subprocess ru_maxrss includes interpreter/imports/input/output and previous-repeat digest high-water; not isolated allocation',
        resource_caps_added=False,real_S3_executed=False)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__': main()
