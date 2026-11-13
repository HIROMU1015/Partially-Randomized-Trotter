"""Artificial R3 decoder benchmark, fresh process RSS and output digests.

No real molecular signals, N, physical reanalysis or scientific S3 simulation.
Run with --out <fresh.json>; each finite artificial case uses a fresh worker.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]


def worker(method, M, n, repeats):
    import numpy as np
    from trottertracks import rpe_decoder_fast as F
    from trotterlib.rpe_four_round_phase_validation import reconstruct_rpe_phase, _vectorized_reconstruction
    from rpe_decoder_r3_fixture import b_statistics
    B = b_statistics(patched=True)
    q = [1 << m for m in range(M+1)]
    rng = np.random.default_rng(314159)
    phases = [np.asarray(F._wrap(qq*.237 + rng.normal(0., .1, n))) for qq in q]
    rows = [dict(m=m,raw=dict(real=.43*math.cos(.237*qq),imag=.43*math.sin(.237*qq)),N_X=31,N_Y=37)
            for m,qq in enumerate(q)]
    policy = dict(kind=B.IID,source={'fixture':'ARTIFICIAL_R3_BENCHMARK'},target_law='ideal_finite_RTE_law')
    times = []; digest = None
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for _ in range(repeats):
        start = time.perf_counter()
        if method == 'scalar':
            result = [reconstruct_rpe_phase([p[i] for p in phases], q) for i in range(n)]
        elif method == 'vector': result = _vectorized_reconstruction(phases, q)
        elif method == 'fast_vector': result = F.vectorized_reconstruction_fast(phases, q)
        elif method == 'fast_scalar_batch': result = F.reconstruct_batch(phases, q)
        elif method in ('B_scalar', 'B_batch'):
            extra = dict(batch_decoder=F.decode_sampled_counts) if method == 'B_batch' else {}
            result = B.simulate(rows,M=M,h=.1,prior=[-4.,4.],repetitions=n,seed=441,policy=policy,**extra)
        else: raise ValueError(method)
        times.append(time.perf_counter()-start)
        # Hash comparisons happen outside the timing and after the measured RSS
        # is sampled; no in-process legacy decoder reference run pollutes RSS.
        rss_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if method == 'scalar':
            values = [r['final_phase_estimate'] for r in result]
            branches = [[r['rounds'][m]['selected_branch'] for r in result] for m in range(M+1)]
            ambiguity = [any(math.isclose(r['distance_from_previous'],r['next_smallest_branch_distance'],abs_tol=1e-12,rel_tol=0.)
                             for r in record['rounds'][1:]) for record in result]
        elif method in ('vector','fast_vector'):
            values,branches,ambiguity = result
        elif method == 'fast_scalar_batch':
            values = result['final_phase_estimate']; branches = [r['selected_branch'] for r in result['rounds']]; ambiguity = result['ambiguous']
        else:
            for trial in result['trials']: trial.pop('decoder_acceleration_source',None)
            digest = hashlib.sha256(json.dumps(result,sort_keys=True,allow_nan=False).encode()).hexdigest()
        if not method.startswith('B_'):
            h = hashlib.sha256(np.asarray(values,dtype=np.float64).tobytes())
            for b in branches: h.update(np.asarray(b,dtype=np.int64).tobytes())
            h.update(np.asarray(ambiguity,dtype=bool).tobytes()); digest = h.hexdigest()
        del result
    return dict(method=method,M=M,rounds=M+1,trials=n,repeats=repeats,seconds=times,median_seconds=statistics.median(times),
        peak_RSS_MiB=rss_peak/1024.,baseline_peak_RSS_MiB=rss_before/1024.,digest=digest,
        memory_measurement='Linux ru_maxrss in fresh worker; includes interpreter/imports/input/output and previous-repeat digest high-water; no isolated allocation claim')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--worker',choices=('scalar','vector','fast_vector','fast_scalar_batch','B_scalar','B_batch'))
    parser.add_argument('--M',type=int,default=12)
    parser.add_argument('--trials',type=int,default=128)
    parser.add_argument('--repeats',type=int,default=3)
    a = parser.parse_args()
    if a.worker:
        print(json.dumps(worker(a.worker,a.M,a.trials,a.repeats),allow_nan=False)); return
    if a.out is None: parser.error('--out required')
    if a.out.exists(): raise ValueError('Fresh benchmark output required')
    cases = [(method,12,n) for n in (32,128,512)
             for method in ('scalar','vector','fast_vector','fast_scalar_batch','B_scalar','B_batch')]
    cases += [(method,12,n) for n in (4096,16384) for method in ('fast_scalar_batch','B_batch')]
    records = []
    for method,M,n in cases:
        command = [sys.executable,'-B',str(Path(__file__).resolve()),'--worker',method,'--M',str(M),'--trials',str(n),'--repeats',str(a.repeats)]
        record = json.loads(subprocess.check_output(command,cwd=ROOT,text=True))
        records.append(record)
        print(json.dumps({k:record[k] for k in ('method','trials','median_seconds','peak_RSS_MiB')}),flush=True)
    for n in (32,128,512):
        groups = [[r['digest'] for r in records if r['trials']==n and r['method'] in methods]
                  for methods in (('scalar','vector','fast_vector','fast_scalar_batch'),('B_scalar','B_batch'))]
        if any(len(set(g))!=1 for g in groups): raise AssertionError('Artificial scientific output mismatch')
    result = dict(scope='ARTIFICIAL_ONLY_NO_HAMILTONIAN_OR_SAVED_SCIENCE',seed_phases=314159,seed_counts=441,
        B_source='170047ef6e7a72aef5f4c6ff9c73f942097a694e',python=sys.version,platform=platform.platform(),
        numpy_version=__import__('numpy').__version__,
        records=records,all_shared_case_output_digests_identical=True,
        no_thread_resource_caps_set=True,measurements='wall time per repeated call; Linux peak RSS fresh subprocess',
        timing_excludes='process creation, imports, fixture SOURCE load, output digest; B timings include RNG/count conversion/trial dictionaries',
        estimates=dict(legacy_final_complex_temporary_bytes='trials * 2**M * 16 (single array only, not peak RSS)',
                       fast_lattice_width_at_m_ge_2=4),mandatory_stop=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as stream: stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__': main()
