"""Versioned v3 host benchmarks/fixtures. Artificial endpoints only, no science."""
import argparse
from dataclasses import asdict
import gzip
from hashlib import sha256
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import resource
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests/tracks/algorithm_codesign')]
import b3_parallel_integrated_fixture as F
from test_g6_w2c_b3_parallel_integration import FAULTS, crash_parent
P,D=F.P,F.D
CONFIGS=[('S','reference','S',1),('SF','fused','S',1),('P','reference','P1',8),('PF','fused','P1',8)]
NS={'light':128,'mixed':64,'heavy':32}
FORBIDDEN=P.FORBIDDEN


def write(path,value):Path(path).write_bytes(P.safe_json(value)+b'\n')


def observation():
    return dict(time_ns=time.time_ns(),affinity=sorted(os.sched_getaffinity(0)),
                meminfo=Path('/proc/meminfo').read_text(),loadavg=Path('/proc/loadavg').read_text(),
                cpu_stat=Path('/proc/stat').read_text().splitlines()[0],
                cpu_pressure=Path('/proc/pressure/cpu').read_text(),memory_pressure=Path('/proc/pressure/memory').read_text(),
                io_pressure=Path('/proc/pressure/io').read_text())


def source_hashes():
    names=[*F.V.ROLES.values(),'tests/tracks/algorithm_codesign/b3_parallel_integrated_fixture.py',
           'tests/tracks/algorithm_codesign/test_g6_w2c_b3_parallel_integration.py',
           str(Path(__file__).relative_to(ROOT))]
    return {p:sha256((ROOT/p).read_bytes()).hexdigest()for p in names}


def fingerprint(target):
    files=[];semantics={}
    for path in sorted(target.glob('*.jsonl')):
        if path.name in ('control.jsonl','summary.jsonl'):continue
        parsed=F.read_stream(path);assert parsed['partial_bytes']==0
        normalized=json.dumps(parsed['rows'],sort_keys=True,separators=(',',':')).encode()
        semantics[path.name]=sha256(normalized).hexdigest()
        files.append(dict(path=path.name,bytes=path.stat().st_size,sha256=parsed['file_sha'],
                          chain=parsed['chain'],rows=len(parsed['rows']),normalized_sha256=semantics[path.name]))
    return semantics,files


def run_one(config,workload,n,out,run_id,keep=False):
    name,codec,mode,workers=config
    settings=P.Settings(codec=codec,mode=mode,workers=workers,profile_n=n,workload=workload)
    with tempfile.TemporaryDirectory(prefix='b3-r1-artificial-')as temp:
        root=Path(temp);c,l,a=F.fixture(root,settings);report=F.invoke(root,c,l,a,settings)
        semantics,files=fingerprint(root/'artificial-output')
        assert len(files)==24 and report['complete']
        assert report['reserved']==report['started_observed']==report['planned']
        assert (root/'artificial-output/trial_states.bin').read_bytes()==bytes([P.CODE['COMPLETED']])*report['planned']
        for pair in report['pairs']:
            file=next(f for f in files if f['path']==pair['task']['key']+'.jsonl')
            assert pair['file_sha']==file['sha256'] and pair['chain']==file['chain']
            for axis in pair['outcome']['axes']:assert axis['H_bits_supplied']==160*axis['N_executed']
            for part in ('reconciliation','fine_reconciliation'):
                rec=pair.get(part)
                if rec:assert all(x>=0 for x in rec['unattributed_and_observer_residual_ns'])
        outer=report['host_outer'];pairs=report['pairs']
        metrics=dict(run_id=run_id,config=name,workload=workload,N=n,settings=asdict(settings),semantics=semantics,files=files,
            wall_s=outer['wall_ns']/1e9,parent_cpu_s=outer['parent_cpu_ns']/1e9,worker_cpu_s=outer['worker_cpu_ns']/1e9,
            total_cpu_s=(outer['parent_cpu_ns']+outer['worker_cpu_ns'])/1e9,
            parent_io=outer['io'],worker_io=[x['io']for x in pairs],
            pair_cpu_ns=[x['cpu_ns']for x in pairs],pair_wall_ns=[x['wall_ns']for x in pairs],
            worker_startup_and_unattributed_cpu_ns=outer['worker_cpu_ns']-sum(x['cpu_ns']for x in pairs),
            peak_sampled_rss_bytes=report['peak_sampled_rss_bytes'],max_worker_hwm_bytes=max(x['memory']['VmHWM_bytes']for x in pairs),
            result_bytes=sum(x['output_bytes']for x in pairs),parent_messages=report['parent_messages'],
            source_pins_sha256=D.identity(c['source_pins']),runtime_manifest_sha256=D.identity(c['runtime_manifest']),
            claim_status=report['fake_claim_status'],measurement_condition=report['measurement_condition'],
            unknown_bit_operations=None,observer_effect_is_in_outer=True)
        if keep:
            with tarfile.open(out/(run_id+'-raw.tar.gz'),'w:gz')as tar:tar.add(root/'artificial-output',arcname='artificial-output')
            write(out/(run_id+'-contract.json'),c)
        return metrics,report


def benchmark(out,repeats):
    rng=random.Random(2026101903);rows=[];reference={};observations=[]
    with gzip.open(out/'reports.jsonl.gz','wb')as reports:
        for workload,n in NS.items():
            for cfg in CONFIGS:run_one(cfg,workload,16,out,'warmup')
            for repeat in range(repeats):
                order=list(CONFIGS);rng.shuffle(order)
                for cfg in order:
                    observations.append(dict(run=f'{workload}-{cfg[0]}-{repeat}',before=observation()))
                    metrics,report=run_one(cfg,workload,n,out,f'{workload}-{cfg[0]}-{repeat}',keep=repeat==0)
                    assert metrics['semantics']==reference.setdefault(workload,metrics['semantics'])
                    metrics['repeat']=repeat;rows.append(metrics)
                    observations[-1]['after']=observation()
                    reports.write(P.safe_json(dict(run_id=metrics['run_id'],report=report))+b'\n');reports.flush()
                    write(out/'runs.json',rows);write(out/'resource_observations.json',observations)
                    print(json.dumps({k:metrics[k]for k in ('run_id','wall_s','parent_cpu_s','worker_cpu_s','total_cpu_s')}),flush=True)
    summary=[]
    for workload in NS:
        for cfg in CONFIGS:
            group=[r for r in rows if r['workload']==workload and r['config']==cfg[0]]
            value=dict(workload=workload,config=cfg[0],repeats=len(group))
            for key in ('wall_s','parent_cpu_s','worker_cpu_s','total_cpu_s','peak_sampled_rss_bytes','max_worker_hwm_bytes','result_bytes'):
                values=[r[key]for r in group]
                value[key]=statistics.median(values);value[key+'_min']=min(values);value[key+'_max']=max(values)
            summary.append(value)
    write(out/'summary.json',summary)


def faults(out):
    results=[]
    with gzip.open(out/'reports.jsonl.gz','wb')as reports:
        for codec in ('reference','fused'):
            for name,fields in FAULTS.items():
                s=P.Settings(profile_n=16,workload='light',codec=codec,**fields)
                with tempfile.TemporaryDirectory(prefix='b3-r1-artificial-')as temp:
                    root=Path(temp);c,l,a=F.fixture(root,s)
                    try:F.invoke(root,c,l,a,s)
                    except P.BundleFailure as exc:r=exc.report
                    else:raise AssertionError('FAULT_NOT_REACHED:'+name)
                    assert not r['complete'] and l.row(c['scope'])[4]=='STOPPED'
                    states=(root/'artificial-output/trial_states.bin').read_bytes()
                    assert all(states.count(i)==r['states'][state]for i,state in enumerate(P.STATES))
                    run_id=codec+'-'+name
                    reports.write(P.safe_json(dict(run_id=run_id,report=r))+b'\n');reports.flush()
                    with tarfile.open(out/(run_id+'-raw.tar.gz'),'w:gz')as tar:tar.add(root/'artificial-output',arcname='artificial-output')
                    if codec=='fused' and name in ('sample','flush_in_emit','unacked_exit','P1_inflight'):
                        write(out/(run_id+'-contract.json'),c)
                    row=dict(run_id=run_id,states=r['states'],reserved=r['reserved'],started_observed=r['started_observed'],
                             output=r['output'],stop_to_join_ns=r['stop_to_join_ns'],
                             post_stop_parent_cpu_ns=r['post_stop_parent_cpu_ns'],
                             post_stop_worker_cpu_estimate_ns=r['post_stop_worker_cpu_estimate_ns'],
                             post_stop_cpu_resolution_ns=r['post_stop_cpu_resolution_ns'],
                             failure_count=len(r['failures']),log_errors=r['log_errors'],secondary_stop_error=r.get('secondary_stop_error'))
                    results.append(row);write(out/'summary.json',results)
                    print(json.dumps(dict(run_id=run_id,states=r['states'])),flush=True)
        # The dead parent cannot produce a trustworthy final report. Preserve raw
        # START_RESERVED and explicit unknowns rather than manufacture totals.
        with tempfile.TemporaryDirectory(prefix='b3-r1-artificial-')as temp:
            root=Path(temp);s=P.Settings(profile_n=16,workload='light',codec='fused',parent_fault='after_reserved')
            c,l,a=F.fixture(root,s);ctx=mp.get_context('spawn');watch=ctx.RawArray('b',24)
            process=ctx.Process(target=crash_parent,args=(root,c,l,a,s,watch));process.start();process.join(60)
            assert process.exitcode==86
            until=time.monotonic()+30
            while sum(watch)<8 and time.monotonic()<until:time.sleep(.05)
            assert sum(watch)==8 and l.row(c['scope'])[4]=='ADMITTED'
            state=(root/'artificial-output/trial_states.bin').read_bytes()
            assert state.count(P.CODE['START_RESERVED'])==1
            records=[json.loads(x)for x in (root/'artificial-output/control.jsonl').read_text().splitlines()]
            pids=[r['pid']for r in records if r['event']=='WORKER_START']
            status={}
            for pid in pids:
                try:status[str(pid)]=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]
                except FileNotFoundError:status[str(pid)]='EXITED'
            assert all(x in ('EXITED','Z')for x in status.values())
            value=dict(schema='b3_parallel_crash_observation_v3',artificial_only=True,parent_exitcode=86,
                       raw_state_counts={n:state.count(i)for i,n in enumerate(P.STATES)},
                       reserved_unknown_lower_bound=1,started_observed=None,completed=None,
                       parent_cpu_ns=None,worker_cpu_ns=None,total_cpu_ns=None,complete=False,
                       claim_status='ADMITTED_SPENT',worker_status=status,worker_cleanup_markers=sum(watch),
                       reason='PARENT_DIED_BEFORE_FINAL_REPORT',mandatory_STOP=True,retry=0,resume=False)
            write(out/'fused-parent_crash-observation.json',value)
            with tarfile.open(out/'fused-parent_crash-raw.tar.gz','w:gz')as tar:tar.add(root/'artificial-output',arcname='artificial-output')
            write(out/'fused-parent_crash-contract.json',c)
            reports.write(P.safe_json(dict(run_id='fused-parent_crash',report=value))+b'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('suite',choices=('benchmark','faults'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--repeats',type=int,default=5)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);hashes=source_hashes()
    write(args.output/'environment.json',dict(observation=observation(),source_sha256=hashes,
         git_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),python=sys.version,
         platform=platform.platform(),artificial_only=True,observer='cached_all_call_return',start_method='spawn',repeats=args.repeats))
    if args.suite=='benchmark':benchmark(args.output,args.repeats)
    else:faults(args.output)
    assert source_hashes()==hashes
    assert not [m for m in sys.modules if m.rsplit('.',1)[-1]in FORBIDDEN]
    write(args.output/'verification.json',dict(source_unchanged=True,all_assertions_passed=True,science_imports=[],
          scientific_calls=0,real_authorizations_grants_claims=0,mandatory_STOP=True))


if __name__=='__main__':main()
