"""Final-source artificial overhead comparison; no scientific run/claim API."""
import argparse
from collections import Counter
import csv
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import random
import statistics as stat
import subprocess
import sys
import tarfile
import tempfile
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests/tracks/algorithm_codesign'),str(Path(__file__).parent)]
import run_b3_parallel_artificial as R
from trottertracks.algorithm_codesign import g6_w2c_b3_parallel_overhead as O
F,P=R.F,R.P
CONFIGS=[('S0','legacy','legacy',1,True),('S0_cached','legacy','cached',1,True),
         ('before1','legacy','legacy',1,False),('before8','legacy','legacy',8,False),
         ('before_exact8','legacy','legacy',8,False),
         ('observer8','legacy','cached',8,False),('ipc8','coalesced','legacy',8,False),
         ('after1','coalesced','cached',1,False),('after2','coalesced','cached',2,False),
         ('after4','coalesced','cached',4,False),('after8','coalesced','cached',8,False)]
DIAG=[('S0_off','legacy','off',1,True),('before_off','legacy','off',8,False),
      ('after_off','coalesced','off',8,False)]


def write(path,value):Path(path).write_bytes(P.safe_json(value)+b'\n')


def run_one(config,workload,n,out,run_id,keep=False,diagnostic=False):
    name,transport,observer,workers,original=config
    s=P.Settings(mode='S' if original else 'P1',workers=workers,profile_n=n,workload=workload)
    with tempfile.TemporaryDirectory(prefix='b3-r1-artificial-') as temp:
        root=Path(temp);c,l,a=O.prepare(F,root,s,transport,observer)
        if name=='before_exact8':
            # Frozen parent, without our common message Counter/ACK validation.
            from unittest.mock import patch
            with patch.object(O,'AccountingCoordinator',O.BASE_COORDINATOR):
                report=O.invoke(F,root,c,l,a,s,transport=transport,observer=observer,original=original,diagnostic_state=diagnostic)
        else:
            report=O.invoke(F,root,c,l,a,s,transport=transport,observer=observer,original=original,diagnostic_state=diagnostic)
        semantic,files=R.fingerprint(root/'artificial-output')
        rows=report['pairs']
        for pair in rows:
            if original:
                task=pair;key=f"{task['phase']}_{task['pass_name']}_{task['arm']}_{task['layer']}.jsonl"
                expected=pair['file_sha256'];chain=pair['output_chain_sha256'];result=pair
            else:
                key=pair['task']['key']+'.jsonl';expected=pair['file_sha'];chain=pair['chain'];result=pair['outcome']
            record=next(f for f in files if f['path']==key)
            assert record['sha256']==expected and record['chain']==chain
            assert all(a['H_bits_supplied']==160*a['N_executed'] for a in result['axes'])
        if not original:
            states=(root/'artificial-output/trial_states.bin').read_bytes()
            assert states==bytes([P.CODE['COMPLETED']])*report['planned']
            assert report['reserved']==report['started_observed']==report['planned']
        metrics=R.metrics(report,original)
        metrics.update(config=name,run_id=run_id,workload=workload,N=n,transport=transport,observer=observer,
            workers=workers,original=original,files=files,semantic=semantic,
            parent_messages=report.get('parent_messages'),state_cost=report.get('diagnostic_state_cost'),
            core_function_calls=[{k:v['calls'] for k,v in (pair.get('fine',pair.get('functions')) or {}).items()
                if k.startswith(('g6_w2c_b3_driver.py:', 'b3_parallel_artificial_fixture.py:'))
                and ':delta.<locals>.' not in k}  # Clock-validation work falls with fewer RPCs; still profiled.
                for pair in rows if pair.get('fine',pair.get('functions')) is not None])
        if keep:
            with tarfile.open(out/(run_id+'-raw.tar.gz'),'w:gz') as tar:
                tar.add(root/'artificial-output',arcname='artificial-output')
        return metrics,report


def benchmark(out,repeats,diagnostic=False):
    rng=random.Random(2026101902);rows=[];reference={};signatures={}
    configs=DIAG if diagnostic else CONFIGS
    with gzip.open(out/'reports.jsonl.gz','wb') as reports:
        for workload,n in R.NS.items():
            for cfg in configs:
                m,r=run_one(cfg,workload,16,out,'warmup',diagnostic=diagnostic)
                print(json.dumps(dict(warmup=workload,config=cfg[0],wall_s=m['wall_s'])),flush=True)
            for repeat in range(repeats):
                order=list(configs);rng.shuffle(order)
                for cfg in order:
                    run_id=f'{workload}-{cfg[0]}-{repeat}'
                    m,r=run_one(cfg,workload,n,out,run_id,keep=repeat==0 and cfg[0] in ('S0','before8','after8'),diagnostic=diagnostic)
                    assert m['semantic']==reference.setdefault(workload,m['semantic'])
                    if m['observer']!='off' and not m['original']:
                        # Core engine/endpoint calls remain identical across both changes.
                        assert m['core_function_calls']==signatures.setdefault(workload,m['core_function_calls'])
                    m['repeat']=repeat;rows.append(m)
                    reports.write(P.safe_json(dict(run_id=run_id,report=r))+b'\n');reports.flush()
                    write(out/'runs.json',rows)
                    print(json.dumps({k:m[k] for k in ('run_id','wall_s','total_cpu_s','ipc_cpu_s')}),flush=True)
    summary=[]
    for w in R.NS:
        for cfg in configs:
            group=[x for x in rows if x['workload']==w and x['config']==cfg[0]]
            v=dict(workload=w,config=cfg[0],repeats=len(group))
            for key in ('wall_s','total_cpu_s','parent_cpu_s','worker_cpu_s','ipc_cpu_s','result_bytes','peak_sampled_rss_bytes'):
                values=[x[key] for x in group if x[key] is not None]
                for suffix,fn in [('',stat.median),('_min',min),('_max',max)]:v[key+suffix]=fn(values) if values else None
            summary.append(v)
    write(out/'summary.json',summary)
    with (out/'summary.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(summary[0]),lineterminator='\n');writer.writeheader();writer.writerows(summary)


def faults(out):
    # Reuse the frozen 17-case generator with optimized private process wiring.
    invoke=F.invoke
    from unittest.mock import patch
    def optimized(*args,**kwargs):
        with O.install('coalesced','cached'):return invoke(*args,**kwargs)
    with patch.object(F,'invoke',optimized):R.faults(out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite',choices=('benchmark','diagnostic','faults'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    source_paths=[Path(O.__file__),Path(__file__),Path(F.__file__),Path(P.__file__),Path(R.__file__)]
    hashes={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in source_paths}
    write(args.output/'environment.json',dict(artificial_only=True,git_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256=hashes,python=sys.version,platform=platform.platform(),affinity=sorted(os.sched_getaffinity(0)),
        meminfo=Path('/proc/meminfo').read_text(),loadavg=Path('/proc/loadavg').read_text(),cpu_stat=Path('/proc/stat').read_text().splitlines()[0],
        start_wall_time_ns=time.time_ns(),repeats=args.repeats,off_observer_is_diagnostic_only=args.suite=='diagnostic'))
    if args.suite=='faults':faults(args.output)
    else:benchmark(args.output,args.repeats,args.suite=='diagnostic')
    assert hashes=={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in source_paths}
    assert not [m for m in sys.modules if m.rsplit('.',1)[-1] in P.FORBIDDEN]
    write(args.output/'verification.json',dict(all_assertions_passed=True,source_unchanged=True,science_imports=[],scientific_calls=0,real_claims=0,mandatory_STOP=True))


if __name__=='__main__':main()
