"""Fixed distributed SOURCE -> detached workers -> guarded DP, artificial tables.

Two loopback HTTP worker processes use distinct spools/caches on ONE OS. This
acceptance does not claim actual remote hosts or measure real H4 acceleration.
The default production source is 71667549e; RPE_DISTRIBUTED_SOURCE_ROOT/SHA
select another explicitly pinned checkout without changing that source. The
only numerical substitution is its existing
test_pr_width_prefix_execution.fixed_artificial_dp_leaf.
"""
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

SOURCE_ROOT=Path(os.environ.get('RPE_DISTRIBUTED_SOURCE_ROOT',
    '/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-distributed-q3-source-71667549e')).resolve()
SOURCE_SHA=os.environ.get('RPE_DISTRIBUTED_SOURCE_SHA','71667549e16c77c53c8b5c02cdfcf98b468aacf1')
PYTHON=Path('/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python')
DEFAULT_OUTPUT=Path('/tmp/rpe-distributed-71667549e-distributed-prefix')
sys.path[:0]=[str(SOURCE_ROOT/'src'),str(SOURCE_ROOT/'scripts'),str(SOURCE_ROOT/'tests')]
import test_pr_width_prefix_execution as P
import check_rpe_distributed_q3 as F
from trottertracks.rpe_q3_store import JobStore


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_equal(actual,expected,context):
    if actual!=expected:raise AssertionError(context+': '+repr(actual)+' != '+repr(expected))


def shim(path):
    """Delegate supervisors unchanged; reuse only the existing artificial DP leaf."""
    path=Path(path)
    path.write_text('#!/bin/sh\ncase "$2" in\n'+
        '  */run_rpe_q3_worker.py) exec '+shlex.quote(str(PYTHON))+' "$@" ;;\n'+
        '  */run_rpe_optimizer_q3_job.py) exec '+shlex.quote(str(PYTHON))+' -B '+
        shlex.quote(str(SOURCE_ROOT/'tests/test_pr_width_prefix_execution.py'))+' --artificial-dp-worker "$@" ;;\n'+
        '  *) printf "%s\\n" "Unsupported artificial shim invocation" >&2; exit 2 ;;\n'+
        'esac\n')
    path.chmod(0o755)
    return str(path)


def call(args,driver):
    with redirect_stdout(io.StringIO()):return P.call(args,driver)


def acceptance(output=DEFAULT_OUTPUT):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    before=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE_ROOT,text=True).strip()
    assert_equal(before,SOURCE_SHA,'Fixed production SOURCE')
    assert_equal(P.ROOT,SOURCE_ROOT,'Existing artificial fixture SOURCE')
    args,driver=P.setup_run(output/'distributed',True)
    run=output/'distributed/run'
    interpreter=shim(run/'ARTIFICIAL_DP_INTERPRETER')
    port=F.free_port();hosts=['simulated-server2','simulated-server1'];profiles=[]
    for host in hosts:
        profile=dict(host_id=host,root=str(SOURCE_ROOT),work_dir=str(output/'workers'/host),
            interpreter=interpreter,route_views={},prefix_cache_directory=str(output/'caches'/host),slots=1)
        path=output/('PROFILE_'+host+'.json');write(path,profile);profiles.append((path,profile))
    spec=json.loads((output/'distributed/ROUND.json').read_text())
    spec['executor'].update(wait_for_running_jobs=True)
    spec['executor']['distributed']=dict(bind='127.0.0.1',port=port,required_hosts=hosts,
        affinity_wait_seconds=.5)
    write(output/'distributed/ROUND.json',spec)
    processes=[];logs=[];commands=[]
    started=time.monotonic()
    try:
        for path,profile in profiles:
            command=[str(PYTHON),'-B',str(SOURCE_ROOT/'scripts/run_rpe_q3_worker.py'),
                '--profile',str(path),'--controller','http://127.0.0.1:'+str(port),'--poll-seconds','.05']
            log=(output/('WORKER_'+profile['host_id']+'.log')).open('w');logs.append(log)
            process=subprocess.Popen(command,cwd=SOURCE_ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
            processes.append((profile['host_id'],process));commands.append(command)
        first=call(args,driver)
        distributed_seconds=time.monotonic()-started
    finally:
        exits=[]
        for host,process in processes:
            if process.poll() is None:process.terminate()
            code=process.wait(timeout=20)
            exits.append(dict(host_id=host,returncode=code if code>=0 else None,signal=-code if code<0 else None))
        for log in logs:log.close()
        write(output/'WORKER_PROCESS_EXITS.json',exits)
    got=P.outcomes(first.run);hits=P.events(first.run)
    assert_equal(len(got),8,'Artificial distributed completed Q3 count')
    rows=JobStore(run/'distributed/QUEUE.sqlite3').rows()
    owner={row['job_id']:row['host_id']for row in rows}
    if set(owner.values())!=set(hosts):raise AssertionError('Both simulated worker profiles must complete jobs')
    hit_rows=[e for e in hits if e['event']=='HIT']
    if not hit_rows:raise AssertionError('Distributed host-local DP prefix reuse was never exercised')
    donor_checks=[]
    for hit in hit_rows:
        job=Path(hit['job']);request=json.loads((job/'REQUEST.json').read_text())
        assert_equal(hit['reused_rounds'],12,'Guarded DP reused rounds')
        donors=[]
        for path in run.glob('jobs/*/*/REQUEST.json'):
            donor=json.loads(path.read_text())
            if donor['method']==request['method']and donor['configuration_id']==request['configuration_id']and donor['N_X']==[1]*12+[2]:
                donors.append(path.parent.name)
        assert_equal(len(donors),1,'Unique donor for the identical configuration')
        donor=donors[0]
        assert_equal(owner[job.name],owner[donor],'Donor and successor use the same host-local cache')
        donor_checks.append(dict(host_id=owner[job.name],donor_job=donor,successor_job=job.name,reused_rounds=12))
    cache_directories=[profile['prefix_cache_directory']for _,profile in profiles]
    assert_equal(len(set(cache_directories)),2,'Separate host cache directories')
    if not all(list(Path(path).glob('*.json'))for path in cache_directories):
        raise AssertionError('Each simulated host must persist its own actual DP checkpoints')
    reference_args,reference_driver=P.setup_run(output/'reference',False)
    # Same fixture shim, with no cache requested by the legacy reference.
    shim(output/'reference/run/ARTIFICIAL_DP_INTERPRETER')
    reference_start=time.monotonic();reference=call(reference_args,reference_driver)
    reference_seconds=time.monotonic()-reference_start
    assert_equal(got,P.outcomes(reference.run),'All lower/upper MSE bounds and decisions match the serial reference')
    assert_equal(P.events(reference.run),[],'Reference prefix-cache disabled')
    markers={str(path):path.read_bytes()for path in run.glob('jobs/*/*/LAUNCH.json')}
    states=(run/'completion/ARTIFICIAL_SERVICE_STATE.json').read_bytes()
    before_launches=sum(e['event']=='ASSIGN'for e in JobStore(run/'distributed/QUEUE.sqlite3').events())
    resumed=call(args,driver)
    assert_equal(markers,{str(path):path.read_bytes()for path in run.glob('jobs/*/*/LAUNCH.json')},'Controller resume preserves launch markers')
    assert_equal(states,(run/'completion/ARTIFICIAL_SERVICE_STATE.json').read_bytes(),'Resume has no new preparation stage')
    assert_equal(got,P.outcomes(resumed.run),'Resume preserves all numerical outcomes')
    after_launches=sum(e['event']=='ASSIGN'for e in JobStore(run/'distributed/QUEUE.sqlite3').events())
    assert_equal(after_launches,before_launches,'Resume launches zero completed scientific job duplicates')
    summary=F.check(output/'distributed');reference_summary=F.check(output/'reference')
    assert_equal(F.canonical_minima(summary),F.canonical_minima(reference_summary),'Both method minima match the serial reference')
    assert_equal(subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE_ROOT,text=True).strip(),SOURCE_SHA,'SOURCE remained fixed')
    result=dict(status='PASS',production_source=SOURCE_SHA,source_root=str(SOURCE_ROOT),normal_schedule_entry=True,
        numerical_scope='EXISTING_GUARDED_DP_ON_ARTIFICIAL_TABLES_ONLY',actual_OS_hosts=1,simulated_worker_host_ids=hosts,
        shared_filesystem_required=False,distinct_worker_spools=[p['work_dir']for _,p in profiles],
        distinct_host_cache_directories=cache_directories,artificial_DP_Q3_calls_distributed=8,
        artificial_DP_Q3_calls_reference=8,all_MSE_lower_upper_status_exact_equal=True,method_minima_exact_equal=True,
        cache_HITs=len(hit_rows),same_host_donor_successor_checks=donor_checks,resume_new_launches=after_launches-before_launches,
        resume_completed_stage_restarts=0,scientific_H4_calls=0,distributed_wall_seconds=distributed_seconds,
        reference_wall_seconds=reference_seconds,artificial_wall_ratio_reference_over_distributed=reference_seconds/distributed_seconds,
        source_fixture=str(SOURCE_ROOT/'tests/test_pr_width_prefix_execution.py'),
        source_fixture_sha256=sha(SOURCE_ROOT/'tests/test_pr_width_prefix_execution.py'),
        generated_shim_sha256=sha(run/'ARTIFICIAL_DP_INTERPRETER'),kernel_view=str(P.VIEW),runtime=str(PYTHON),
        worker_process_exits=exits,by_host=summary['by_host'],queue_wait_seconds=summary['common_queue_wait_seconds'],
        transfer_seconds=summary['worker_transfer_roundtrip_seconds'],worker_commands=commands)
    write(output/'SUMMARY.json',result)
    write(output/'OUTCOMES_EXACT.json',[dict(method=k[0],configuration_id=k[1],N_X=list(k[2]),N_Y=list(k[3]),
        status=v[0],MSE_lower=v[1],MSE_upper=v[2])for k,v in sorted(got.items())])
    (output/'DP_PREFIX_EVENTS.jsonl').write_text(''.join(json.dumps(e)+'\n'for e in hits))
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    options=parser.parse_args()
    print(json.dumps(acceptance(options.output),indent=2))
