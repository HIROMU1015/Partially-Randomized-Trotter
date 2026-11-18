"""Artificial DP boundary/worker benchmarks; never load physical H4 probabilities.

Uses the existing fixed-view fixture, original guarded accelerator, and the D2
prefix store. Raw runs stay in a new output directory; RESULT.json has medians.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_boundary_fixture', ROOT / 'tests/test_rpe_dp_prefix_reuse.py')
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def cpu_seconds():
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def fixture():
    _, tables, scope = F.example(shots=4, states=256, pairs=10000000)
    tail = F.changed_tail(tables, shots=3)
    middle = deepcopy(tables)
    middle[6] = F.changed_tail(tables[:7], shots=3)[-1]
    return [tables, tail, middle], scope


def digest(value):
    return hashlib.sha256(F.C._bytes(F.C._encode(value))).hexdigest()


def evaluate(trial, scope, workers, cache, boundaries, namespace, barrier=None):
    cls = F.C.accelerator_class(F.P) if cache is not None else F.P.Accelerator
    options = dict(workers=workers, outcome_batch=4)
    if cache is not None:
        store = F.C.PrefixStore(cache)
        if barrier is not None:
            ordinary_get = store.get
            def get(key, mass_type):
                snapshot = ordinary_get(key, mass_type)
                # Both cold readers finish looking before either can compute/save.
                barrier.wait(timeout=60)
                return snapshot
            store.get = get
        options.update(store=store, checkpoint_rounds=boundaries, cache_namespace=namespace)
    start = time.perf_counter(); cpu = cpu_seconds()
    engine = cls(**options)
    value = engine(trial, h=10., target=0., scope=scope)
    return dict(wall_seconds=time.perf_counter()-start, CPU_seconds=cpu_seconds()-cpu,
                output_hash=digest(value), events=getattr(engine, 'cache_events', []),
                MSE={key:value[key] for key in ('status', 'MSE_lower', 'MSE_upper')},
                peak_self_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                worker_observations=engine.observations)


def boundary_worker(output, boundaries):
    trials, scope = fixture(); start = time.perf_counter(); cpu = cpu_seconds()
    with tempfile.TemporaryDirectory(prefix='artificial-rpe-boundary-') as tmp:
        cache = None if boundaries is None else Path(tmp) / 'cache'
        rows = [evaluate(trial, scope, 1, cache, boundaries, {'fixture':'ARTIFICIAL_BOUNDARY'}) for trial in trials]
        stored = [] if cache is None else list(cache.glob('*.json.gz'))
        size = sum(path.stat().st_size for path in stored)
        result = dict(kind='BOUNDARY_SEQUENCE', boundaries=boundaries, jobs=rows,
                      wall_seconds=time.perf_counter()-start, CPU_seconds=cpu_seconds()-cpu,
                      cache_bytes=size, cache_files=len(stored),
                      IO_seconds={key:sum(event.get(key,0) for row in rows for event in row['events'])
                                  for key in ('hash_seconds','read_seconds','write_seconds')})
    write(output, result)


def profile_job(path, workers, cache=None, barrier=None):
    trials, scope = fixture()
    try:
        result=evaluate(trials[0],scope,workers,cache,[11],{'fixture':'ARTIFICIAL_PROFILE'},barrier)
        write(path,dict(result,pid=os.getpid(),completed=True))
    except BaseException as exc:
        write(path,dict(completed=False,error=repr(exc),pid=os.getpid()))
        raise


def profile_worker(output, slots, workers, cold):
    ctx = mp.get_context('fork');start = time.perf_counter();cpu = cpu_seconds()
    with tempfile.TemporaryDirectory(prefix='artificial-rpe-profile-') as tmp:
        tmp=Path(tmp);jobs=[];exits=[]
        cache=tmp/'cache' if cold else None
        barrier=ctx.Barrier(2) if cold else None
        count=2 if cold else 4
        for begin in range(0,count,slots):
            processes=[]
            for index in range(begin,min(count,begin+slots)):
                path=tmp/('job_%d.json'%index)
                proc=ctx.Process(target=profile_job,args=(path,workers,cache,barrier))
                proc.start();processes.append((path,proc))
            for path,proc in processes:
                proc.join();exits.append(dict(exitcode=proc.exitcode,signal=-proc.exitcode if proc.exitcode<0 else None))
                jobs.append(json.loads(path.read_text()))
                if proc.exitcode!=0 or not jobs[-1]['completed']:raise RuntimeError('Artificial child failed')
        stored=[] if cache is None else list(cache.glob('*.json.gz'))
        result=dict(kind='CONCURRENT_COLD' if cold else 'FOUR_INDEPENDENT_JOBS',slots=slots,dp_workers=workers,
                    jobs=jobs,wall_seconds=time.perf_counter()-start,CPU_seconds=cpu_seconds()-cpu,
                    summed_job_CPU_seconds=sum(j['CPU_seconds'] for j in jobs),process_exits=exits,
                    cache_bytes=sum(p.stat().st_size for p in stored),cache_files=len(stored))
        if cold:
            trials,scope=fixture()
            result['warm_after_both_complete']=evaluate(trials[1],scope,workers,cache,[11],{'fixture':'ARTIFICIAL_PROFILE'})
            result['duplicate_save_events']=sum(e['event']=='SAVE' for j in jobs for e in j['events'])-len(stored)
    write(output,result)


def narrow_worker(output, workers, boundaries):
    trials, scope = fixture();start=time.perf_counter();cpu=cpu_seconds()
    with tempfile.TemporaryDirectory(prefix='artificial-rpe-narrow-') as tmp:
        cache=None if boundaries is None else Path(tmp)/'cache'
        rows=[evaluate(trial,scope,workers,cache,boundaries,{'fixture':'ARTIFICIAL_NARROW'}) for trial in trials[:2]]
        files=[] if cache is None else list(cache.glob('*.json.gz'))
        result=dict(kind='NARROW_SEQUENTIAL_PAIR',slots=1,dp_workers=workers,boundaries=boundaries,
            jobs=rows,wall_seconds=time.perf_counter()-start,CPU_seconds=cpu_seconds()-cpu,
            cache_files=len(files),cache_bytes=sum(p.stat().st_size for p in files))
    write(output,result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repetitions',type=int,default=3);p.add_argument('--profile-repetitions',type=int,default=1)
    p.add_argument('--worker',choices=('boundary','profile','cold','narrow'));p.add_argument('--boundaries')
    p.add_argument('--slots',type=int,default=2);p.add_argument('--dp-workers',type=int,default=1)
    p.add_argument('--narrow-repetitions',type=int,default=0)
    p.add_argument('--boundary-sets',default='[null,[11],[5,11],[2,5,8,11]]')
    a=p.parse_args()
    if a.worker:
        if a.worker=='boundary':return boundary_worker(a.output,json.loads(a.boundaries))
        if a.worker=='narrow':return narrow_worker(a.output,a.dp_workers,json.loads(a.boundaries))
        return profile_worker(a.output,a.slots,a.dp_workers,a.worker=='cold')
    a.output.mkdir(parents=True,exist_ok=False);rows=[];reference=None
    def run(name,args):
        dest=a.output/(name+'.json');start=time.perf_counter()
        print('START',name,flush=True)
        process=subprocess.run([sys.executable,'-B',__file__,'--output',str(dest),*args],cwd=ROOT)
        if process.returncode!=0:raise RuntimeError('%s failed with %s'%(name,process.returncode))
        value=json.loads(dest.read_text());value.update(case=name,driver_wall_seconds=time.perf_counter()-start,
                                                       process_exitcode=process.returncode)
        rows.append(value);print('END',name,round(value['wall_seconds'],4),flush=True)
        return value
    boundary_sets=json.loads(a.boundary_sets)
    if not boundary_sets or boundary_sets[0] is not None:raise ValueError('First boundary mode must be ordinary/null')
    for repeat in range(a.repetitions):
        baseline=None
        for boundaries in boundary_sets:
            name='boundary_%d_%s'%(repeat,'ordinary' if boundaries is None else '_'.join(map(str,boundaries)))
            value=run(name,['--worker','boundary','--boundaries',json.dumps(boundaries)])
            hashes=[j['output_hash'] for j in value['jobs']]
            if baseline is None:baseline=value;reference=hashes
            value['same_entire_DP_outputs']=hashes==reference
            if not value['same_entire_DP_outputs']:raise ValueError('Artificial boundary outputs differ')
            value['speedup_vs_paired_ordinary']=baseline['wall_seconds']/value['wall_seconds']
    for repeat in range(a.profile_repetitions):
        for slots,workers in ((1,1),(2,2),(4,1),(1,4)):
            value=run('profile_%d_%d_%d'%(repeat,slots,workers),['--worker','profile','--slots',str(slots),'--dp-workers',str(workers)])
            value['same_entire_DP_outputs']=all(j['output_hash']==reference[0] for j in value['jobs'])
            if not value['same_entire_DP_outputs']:raise ValueError('Artificial worker profile outputs differ')
        value=run('cold_%d'%repeat,['--worker','cold','--slots','2','--dp-workers','1'])
        value['same_entire_DP_outputs']=(all(j['output_hash']==reference[0] for j in value['jobs'])
                                        and value['warm_after_both_complete']['output_hash']==reference[1])
        if not value['same_entire_DP_outputs']:raise ValueError('Artificial cold/warm outputs differ')
        if any(e['event']=='HIT' for j in value['jobs'] for e in j['events']):raise ValueError('Cold barrier did not force misses')
    for repeat in range(a.narrow_repetitions):
        for workers in (1,2,4):
            for boundaries in (None,[5,11]):
                mode='ordinary' if boundaries is None else '5_11'
                value=run('narrow_%d_%d_%s'%(repeat,workers,mode),['--worker','narrow','--dp-workers',str(workers),'--boundaries',json.dumps(boundaries)])
                value['same_entire_DP_outputs']=[j['output_hash'] for j in value['jobs']]==reference[:2]
                if not value['same_entire_DP_outputs']:raise ValueError('Artificial narrow profile outputs differ')
    summary=[]
    for boundaries in boundary_sets:
        group=[r for r in rows if r['kind']=='BOUNDARY_SEQUENCE' and r['boundaries']==boundaries]
        summary.append(dict(boundaries=boundaries,repetitions=len(group),
            median_wall_seconds=statistics.median(r['wall_seconds'] for r in group),
            median_CPU_seconds=statistics.median(r['CPU_seconds'] for r in group),
            median_speedup_vs_paired_ordinary=statistics.median(r['speedup_vs_paired_ordinary'] for r in group),
            median_cache_bytes=statistics.median(r['cache_bytes'] for r in group),
            median_IO_seconds={k:statistics.median(r['IO_seconds'][k] for r in group) for k in ('hash_seconds','read_seconds','write_seconds')},
            hits_per_job=[[e.get('reused_rounds') for e in j['events'] if e['event']=='HIT'] for j in group[0]['jobs']]))
    result=dict(schema='ARTIFICIAL_RPE_PREFIX_BOUNDARIES_V1',fixture=dict(rounds=13,shots=4,frontier=256,
        transition_pair_limit=10000000,changes=[None,'m12 X/Y=3','m6 X/Y=3'],raws=[.25,-.5],h=10.,target=0.),
        scope='ARTIFICIAL_ONLY_NO_REAL_H4_Q3',real_Q3_calls=0,python=sys.version,
        source_files={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                      (Path(__file__), ROOT/'src/trottertracks/rpe_dp_prefix_reuse.py',ROOT/'tests/test_rpe_dp_prefix_reuse.py')},
        boundary_summary=summary,rows=rows,
        caveats=['Worker profiles have one repetition by default; they do not establish a robust optimum.',
                 'CPU includes benchmark parent and waited descendants; job CPU separately excludes controller.',
                 'RSS is individual process peak, not simultaneous tree memory; shared pages are not summed.',
                 'The cold-reader barrier forces two misses before either computes, demonstrating duplicate work.',
                 'Synthetic DP timing excludes real Q3 metadata, tables, partial/coarse DP and closure.',
                 'No historical prefix states exist; measured synthetic capacity is not actual H4 capacity.'])
    write(a.output/'RESULT.json',result)


if __name__=='__main__':main()
