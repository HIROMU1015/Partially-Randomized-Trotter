"""ARTIFICIAL ONLY: reproducible ordered/parallel host benchmarks and faults.

System Python, no scientific numerical modules, no CLI production admission.
Each invocation owns only fresh /tmp/b3-r1-artificial-* fake scopes.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import gzip
from hashlib import sha256
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests/tracks/algorithm_codesign')]
import b3_parallel_artificial_fixture as F
P = F.P
CONFIGS = [('S0', 'S', 1, 1, True), ('S', 'S', 1, 1, False),
           ('P1_1','P1',1,1,False), ('P1_2','P1',2,2,False),
           ('P1_4','P1',4,4,False), ('P1_8','P1',8,8,False),
           ('P2_2','P2',8,2,False), ('P2_4','P2',8,4,False)]
NS = {'light':128, 'mixed':64, 'heavy':32}


def write_json(path, value):
    Path(path).write_bytes(P.safe_json(value)+b'\n')


def fingerprint(target):
    inventory, semantic = [], {}
    for path in sorted(target.glob('*.jsonl')):
        if path.name in ('control.jsonl','summary.jsonl'):
            continue
        parsed = F.read_stream(path)
        assert parsed['partial_bytes'] == 0
        normalized = json.dumps(parsed['rows'],sort_keys=True,separators=(',',':')).encode()
        semantic[path.name] = sha256(normalized).hexdigest()
        inventory.append(dict(path=path.name,bytes=path.stat().st_size,sha256=parsed['file_sha'],
                              chain=parsed['chain'],normalized_sha256=semantic[path.name],rows=len(parsed['rows'])))
    assert len(inventory)==24
    return semantic, inventory


def metrics(report, original):
    outer = report['host_outer']
    rows = report['pairs']
    if original:
        # Meter reconciliation's total is explicit; don't infer from stage sums.
        cpus = [r['reconciliation']['outer_total_ns'][0] for r in rows]
        walls = [r['reconciliation']['outer_total_ns'][1] for r in rows]
    else:
        cpus = [r['cpu_ns'] for r in rows]
        walls = [r['wall_ns'] for r in rows]
    return dict(wall_s=outer['wall_ns']/1e9, parent_cpu_s=outer['parent_cpu_ns']/1e9,
                worker_cpu_s=outer['worker_cpu_ns']/1e9,
                total_cpu_s=(outer['parent_cpu_ns']+outer['worker_cpu_ns'])/1e9,
                pair_cpu_s=[x/1e9 for x in cpus],pair_wall_s=[x/1e9 for x in walls],
                phase_cpu_max_to_mean=[max(cpus[i:i+8])/statistics.mean(cpus[i:i+8]) for i in (0,8,16)],
                peak_sampled_rss_bytes=report.get('peak_sampled_rss_bytes'),
                max_worker_memory=max((r['memory']['VmHWM_bytes'] for r in rows),default=0),
                parent_io=outer['io'], worker_io=[r.get('io',r.get('artificial_io')) for r in rows],
                ipc_cpu_s=sum(r.get('control_rpc',{}).get('cpu_ns',0) for r in rows)/1e9,
                ipc_wall_sum_s=sum(r.get('control_rpc',{}).get('wall_ns',0) for r in rows)/1e9,
                worker_launch_cost=report.get('worker_launch_cost'),
                result_bytes=sum(r['output_bytes'] for r in rows),
                global_output=report.get('output'),state_update_io_bytes=report.get('state_update_io_bytes'))


def benchmark(out, repeats):
    rows, reference, warmups = [], {}, []
    rng = random.Random(20261019)
    with gzip.open(out/'run_reports.jsonl.gz','wb') as reports:
        for workload in NS:
            # Fresh pair workers cannot retain JIT state; warm-up primes OS/import/file caches.
            for config in CONFIGS:
                name,mode,workers,cap,original=config
                s=P.Settings(mode=mode,workers=workers,inflight_cap=cap,profile_n=16,workload=workload)
                r=F.run(s,original=original)
                warmups.append(dict(workload=workload,config=name,wall_ns=r['host_outer']['wall_ns']))
                print(json.dumps(dict(warmup=workload,config=name)),flush=True)
            for repeat in range(repeats):
                order=list(CONFIGS);rng.shuffle(order)
                for name,mode,workers,cap,original in order:
                    run_id=f'{workload}-{name}-{repeat}'
                    s=P.Settings(mode=mode,workers=workers,inflight_cap=cap,profile_n=NS[workload],workload=workload)
                    with tempfile.TemporaryDirectory(prefix='b3-r1-artificial-') as tmp:
                        root=Path(tmp)
                        c,ledger,auth=F.fixture(root,s)
                        r=F.invoke(root,c,ledger,auth,s,original=original)
                        semantic,files=fingerprint(root/'artificial-output')
                        if workload not in reference:reference[workload]=semantic
                        assert semantic==reference[workload], 'ORIGINAL_SERIAL_PARALLEL_SEMANTIC_MISMATCH'
                        if not original:
                            states=(root/'artificial-output/trial_states.bin').read_bytes()
                            assert states==bytes([P.CODE['COMPLETED']])*r['planned']
                            assert len(r['pairs'])==24 and r['complete']
                        row=dict(run_id=run_id,workload=workload,config=name,repeat=repeat,
                                 settings=asdict(s),original=original,semantic_equal=True,
                                 claim_status=r['fake_claim_status'],files=files,**metrics(r,original))
                        rows.append(row)
                        reports.write(P.safe_json(dict(run_id=run_id,report=r))+b'\n')
                        reports.flush()
                        if repeat==0 and name in ('S0','P1_8','P2_2'):
                            with tarfile.open(out/(run_id+'-raw.tar.gz'),'w:gz') as tar:
                                tar.add(root/'artificial-output',arcname='artificial-output')
                        print(json.dumps({k:row[k] for k in ('run_id','wall_s','total_cpu_s','ipc_cpu_s')}),flush=True)
                        write_json(out/'runs.json',rows)
    write_json(out/'warmups.json',warmups)
    summaries=[]
    for workload in NS:
        baseline=statistics.median(r['wall_s'] for r in rows if r['workload']==workload and r['config']=='S0')
        serial=statistics.median(r['wall_s'] for r in rows if r['workload']==workload and r['config']=='S')
        for name,*_ in CONFIGS:
            group=[r for r in rows if r['workload']==workload and r['config']==name]
            value=dict(workload=workload,config=name,repeats=len(group))
            for key in ('wall_s','parent_cpu_s','worker_cpu_s','total_cpu_s','ipc_cpu_s','result_bytes'):
                a=[r[key] for r in group]
                value[key]=statistics.median(a);value[key+'_min']=min(a);value[key+'_max']=max(a)
            value.update(speedup_vs_S0=baseline/value['wall_s'],speedup_vs_S=serial/value['wall_s'])
            summaries.append(value)
    write_json(out/'summary.json',summaries)
    import csv
    with (out/'summary.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(summaries[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(summaries)


def faults(out):
    cases=[('sample',dict(faults=((0,0,'sample',3),)),False),
           ('event',dict(faults=((0,0,'event',0),)),False),
           ('after_event',dict(faults=((0,0,'after_event',0),)),False),
           ('parent_dispatch',dict(parent_fault='dispatch'),False),
           ('angle',dict(faults=((1,0,'angle',0),)),False),
           ('write',dict(faults=((0,0,'write',0),)),False),
           ('flush',dict(faults=((0,0,'flush',0),)),False),
           ('worker_exit',dict(faults=((0,0,'exit',0),)),False),
           ('reserved_exit',dict(faults=((0,0,'reserved_exit',0),)),False),
           ('result_cap',dict(result_bytes=1500),False),
           ('log_cap',dict(log_bytes=12000),False),
           ('correctness_zero',dict(faults=((0,0,'zero',0),)),False),
           ('stop_save',dict(parent_fault='stop_save',faults=((0,0,'sample',0),)),False),
           ('P1_inflight',dict(mode='P1'),True),
           ('P2_inflight',dict(mode='P2',inflight_cap=2),True),
           ('simultaneous',dict(mode='P1'),'all')]
    values=[]
    for name,kw,sync in cases:
        s=P.Settings(profile_n=16,workload='light',**kw)
        with tempfile.TemporaryDirectory(prefix='b3-r1-artificial-') as tmp:
            root=Path(tmp);c,ledger,auth=F.fixture(root,s)
            try:F.invoke(root,c,ledger,auth,s,synchronized_failure=sync)
            except P.BundleFailure as exc:
                r=exc.report
                assert not r['complete']
                state=(root/'artificial-output/trial_states.bin').read_bytes()
                counts=Counter(P.STATES[x] for x in state)
                assert counts==Counter(r['states'])
                assert r['reserved']<=r['planned'] and r['started_observed']<=r['reserved']
                r['state_file_sha256']=sha256(state).hexdigest()
                write_json(out/(name+'.json'),r)
                with tarfile.open(out/(name+'-raw.tar.gz'),'w:gz') as tar:
                    tar.add(root/'artificial-output',arcname='artificial-output')
                values.append(dict(case=name,started=r['started_observed'],reserved=r['reserved'],
                    states=r['states'],inflight_at_stop=len(r['representative_failure'].get('inflight_snapshot',{})),
                    post_stop_parent_cpu_ns=r['post_stop_parent_cpu_ns'],
                    post_stop_worker_cpu_estimate_ns=r['post_stop_worker_cpu_estimate_ns'],
                    resolution_ns=r['post_stop_cpu_resolution_ns'],stop_to_join_ns=r['stop_to_join_ns'],
                    fake_claim_status=ledger.row(c['scope'])[4]))
            else:raise AssertionError('FAULT_NOT_OBSERVED:'+name)
        print(json.dumps(values[-1]),flush=True)
    # Abrupt death has no parent report: retain raw states and explicit unknowns.
    with tempfile.TemporaryDirectory(prefix='b3-r1-artificial-') as tmp:
        root=Path(tmp);s=P.Settings(profile_n=16,workload='light',parent_fault='after_reserved')
        c,ledger,auth=F.fixture(root,s)
        ctx=mp.get_context('fork');watch=ctx.RawArray('b',24)
        process=ctx.Process(target=F.invoke,args=(root,c,ledger,auth,s),kwargs=dict(watch=watch))
        process.start();process.join(20)
        assert not process.is_alive() and process.exitcode==86
        deadline=time.monotonic()+10
        while sum(watch[:8])<8 and time.monotonic()<deadline:time.sleep(.01)
        assert sum(watch[:8])==8
        state=(root/'artificial-output/trial_states.bin').read_bytes()
        counts=Counter(P.STATES[x] for x in state)
        assert counts['START_RESERVED']>=1
        try:F.invoke(root,c,ledger,auth,s)
        except Exception:pass
        else:raise AssertionError('CRASH_SCOPE_REUSED')
        row=dict(case='parent_crash',started=None,reserved=len(state)-counts['NOT_STARTED'],states=dict(counts),
                 inflight_at_stop=None,post_stop_parent_cpu_ns=None,post_stop_worker_cpu_estimate_ns=None,
                 resolution_ns=None,stop_to_join_ns=None,fake_claim_status=ledger.row(c['scope'])[4],
                 all_started_children_exited_on_EOF=True,retry_rejected=True,
                 conservative_unknown=counts['START_RESERVED']+counts['STARTED']+counts['IN_FLIGHT'],
                 state_file_sha256=sha256(state).hexdigest(),complete=False)
        write_json(out/'parent_crash.json',row)
        with tarfile.open(out/'parent_crash-raw.tar.gz','w:gz') as tar:
            tar.add(root/'artificial-output',arcname='artificial-output')
        values.append(row)
        print(json.dumps(row),flush=True)
    write_json(out/'summary.json',values)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite',choices=('benchmark','faults'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    paths=[Path(P.__file__),Path(F.__file__),Path(__file__)]
    source={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in paths}
    environment=dict(artificial_only=True,source_sha256=source,python=sys.version,platform=platform.platform(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        affinity=sorted(os.sched_getaffinity(0)),meminfo=Path('/proc/meminfo').read_text(),
        loadavg=Path('/proc/loadavg').read_text(),cpu_stat=Path('/proc/stat').read_text().splitlines()[0],
        monotonic_ns=time.monotonic_ns(),wall_time_ns=time.time_ns(),repeats=args.repeats,
        scientific_calls=0,real_claims=0,mandatory_STOP=True)
    write_json(args.output/'environment.json',environment)
    if args.suite=='benchmark':benchmark(args.output,args.repeats)
    else:faults(args.output)
    assert source=={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in paths}
    assert not [x for x in sys.modules if x.rsplit('.',1)[-1] in P.FORBIDDEN]
    write_json(args.output/'verification.json',dict(source_unchanged=True,science_imports=[],
        scientific_calls=0,real_claims=0,all_assertions_passed=True,mandatory_STOP=True))


if __name__=='__main__':main()
