"""Plan-only by default; saved numerical analysis needs a new approved grant."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

import track_a_v1_runner as base

ROOT=Path(__file__).resolve().parents[1]
model=base.load('_v2_saved_residual',ROOT/'src/trotterlib/track_a_v2_residual.py')
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
         'NUMBA_NUM_THREADS','QISKIT_NUM_PROCS','RAYON_NUM_THREADS')
INPUTS={'theory','connections','index','binding','registry'}


def verify_grant(config,grant,fresh,*,source,config_sha,now=None):
    now=time.monotonic() if now is None else now
    if (grant.get('schema')!='track_a_v2_residual_grant_v1' or grant.get('approved_by_user') is not True
            or grant.get('runtime_authorization') is not True or grant.get('metadata_conditional_interpretation_approved') is not True
            or grant.get('scope')!=model.SCOPE or grant.get('SOURCE')!=source or not source
            or grant.get('config_sha256')!=config_sha or grant.get('policy_sha256')!=model.digest(model.policy())
            or config.get('policy_sha256')!=model.digest(model.policy())
            or grant.get('limits')!=model.LIMITS or grant.get('caps')!=model.CAPS
            or grant.get('one_shot') is not True or grant.get('retry') is not False or grant.get('resume') is not False
            or grant.get('next_stage_authorized') is not False):raise ValueError('NEW_APPROVED_RESIDUAL_GRANT_REQUIRED')
    cpus=[grant.get('driver_cpu'),grant.get('monitor_cpu')]
    if (grant.get('allowed_cpus')!=cpus or any(type(x) is not int for x in cpus)
            or len(set(cpus))!=2 or not set(cpus)<=os.sched_getaffinity(0)):raise ValueError('DISJOINT_CPUS_REQUIRED')
    if (fresh.get('source')!=source or fresh.get('config_sha256')!=config_sha or fresh.get('allowed_cpus')!=cpus
            or type(fresh.get('monotonic_seconds')) not in (float,int)
            or not 0<=now-fresh['monotonic_seconds']<=30 or any(fresh.get(k) is not True for k in
            ('profile_verified','resources_idle_verified','protected_jobs_unchanged','external_RSS_monitor_ready'))):
        raise ValueError('FRESH_RESOURCE_PROFILE_MONITOR_GATE_REQUIRED')


def verify_bytes(config,source,*,git_blobs=False):
    if set(config['inputs'])!=INPUTS:raise ValueError('EXACT_INPUT_SET_REQUIRED')
    for name,expected in config['source_hashes'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('SOURCE_PATH')
        if base.sha(ROOT/name)!=expected:raise ValueError('SOURCE_CHANGED:'+name)
        if git_blobs and hashlib.sha256(subprocess.check_output(['git','show',source+':'+name],cwd=ROOT)).hexdigest()!=expected:
            raise ValueError('SOURCE_GIT_BLOB_CHANGED')
    for name,expected in config['runtime_file_hashes'].items():
        if base.sha(name)!=expected:raise ValueError('RUNTIME_PROFILE_CHANGED')
    for name,item in config['inputs'].items():
        path=Path(item['path'])
        if path.is_symlink() or path.stat().st_size!=item['bytes'] or base.sha(path)!=item['sha256']:raise ValueError('INPUT_CHANGED:'+name)


def bounded_json(path,*,compressed=False,expected=None):
    if compressed:
        with gzip.open(path,'rb') as stream:raw=stream.read(16*2**20+1)
        if len(raw)>16*2**20:raise ValueError('GZIP_DECOMPRESSED_CAP')
        if expected is None or len(raw)!=expected['bytes'] or hashlib.sha256(raw).hexdigest()!=expected['sha256']:
            raise ValueError('GZIP_EXPORT_IDENTITY')
    else:
        if Path(path).stat().st_size>16*2**20:raise ValueError('JSON_INPUT_CAP')
        raw=Path(path).read_bytes()
    return base.accounting.strict_json(raw)


def guard(output,started):
    if time.monotonic()-started>model.LIMITS['wall_seconds']:raise ValueError('WALL_CEILING')
    status=dict(line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    if int(status['VmHWM'].split()[0])*1024>model.LIMITS['RSS_bytes']:raise ValueError('RSS_CEILING')
    if int(status['Threads'])!=1:raise ValueError('THREAD_ONE_REQUIRED')
    if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>model.LIMITS['output_bytes']:raise ValueError('OUTPUT_CEILING')


def write(path,value):
    with path.open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def execute(config,grant,source):
    verify_bytes(config,source)
    if Path(sys.executable).resolve()!=Path(config['python']).resolve():raise ValueError('PINNED_PYTHON_REQUIRED')
    if any(os.environ.get(name)!='1' for name in THREADS):raise ValueError('THREAD_ENV_REQUIRED')
    from launch_track_a_v1_phase_a import owned
    parent=base.read_json(Path(config['control'])/'started.json')['driver']
    identity=owned.identity(os.getpid())
    if identity is None or list(identity)!=parent['identity']:raise ValueError('EXTERNAL_OBSERVER_OWNERSHIP_REQUIRED')
    os.sched_setaffinity(0,{grant['driver_cpu']})
    output=Path(config['output']);lock=Path(config['lock'])
    for p in (output,lock):
        if not p.is_absolute() or p.resolve()!=p or p.exists() or not p.is_relative_to('/home/AbeHiromu/projects'):
            raise ValueError('NEW_EXCLUSIVE_PATH_REQUIRED')
    write(lock,dict(SOURCE=source,scope=model.SCOPE));output.mkdir(mode=0o700)
    resource.setrlimit(resource.RLIMIT_AS,(model.LIMITS['AS_bytes'],model.LIMITS['AS_bytes']))
    resource.setrlimit(resource.RLIMIT_CPU,(model.LIMITS['CPU_seconds'],model.LIMITS['CPU_seconds']))
    def stop(*args):raise ValueError('STOP_SIGNAL_OR_WALL')
    for s in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(s,stop)
    signal.alarm(model.LIMITS['wall_seconds']);started=time.monotonic();calls=0
    try:
        data={name:bounded_json(item['path'],compressed=name=='connections',
              expected=config['decompressed_connection']) for name,item in config['inputs'].items()}
        if data['binding']['SOURCE']!='7e59df8ac3c356007d85bfd9218ce1a224a40d52':raise ValueError('THEORY_EXECUTION_SOURCE')
        rows,audit=model.prepare_observations(data['theory'],data['connections'],data['index'],data['registry'])
        write(output/'input_and_provenance_audit.json',audit)
        def reserve(key,endpoint):
            nonlocal calls
            guard(output,started);calls+=1
            if calls>model.CAPS['fits']:raise ValueError('FIT_CAP')
            with (output/'fit_reservations.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(fit=calls,stratum=list(key),endpoint=endpoint,refund=False))+'\n')
                stream.flush();os.fsync(stream.fileno())
        def emit(item):
            guard(output,started);write(output/('fit_%03d.json'%calls),item)
        result=model.evaluate(rows,reserve=reserve,emit=emit)
        write(output/'analysis_status.json',result)
        if result['status']=='STOP':raise ValueError(result['first_stop'])
        verify_bytes(config,source);guard(output,started)
        write(output/'terminal.json',dict(status='COMPLETE_MANDATORY_STOP',exit_code=0,SOURCE=source,
            fits=calls,scientific_status=result['scientific_status'],all_strata_passed=result['all_strata_passed'],
            full_numeric_envelope_certified=False,new_compile=0,new_ranking=0,next_stage_authorized=False))
    except BaseException as exc:
        write(output/'terminal.json',dict(status='STOP',exit_code=1,SOURCE=source,error=str(exc),fits=calls,
            automatic_retry=False,next_stage_authorized=False));raise
    finally:signal.alarm(0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--launch-approved',action='store_true');p.add_argument('--execute-approved',action='store_true')
    for name in ('config','grant','fresh-gate'):p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    if not a.launch_approved and not a.execute_approved:
        if a.config or a.grant or a.fresh_gate:p.error('explicit approved mode required')
        print(json.dumps(model.policy(),indent=2));return
    if a.launch_approved==a.execute_approved or not all((a.config,a.grant,a.fresh_gate)):p.error('one mode and sealed files required')
    config,grant,fresh=map(base.read_json,(a.config,a.grant,a.fresh_gate))
    source=grant.get('SOURCE') if a.execute_approved else subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    verify_grant(config,grant,fresh,source=source,config_sha=base.sha(a.config))
    if a.execute_approved:return execute(config,grant,source)
    if subprocess.check_output(['git','status','--porcelain=v1'],cwd=ROOT):raise ValueError('DIRTY_SOURCE')
    verify_bytes(config,source,git_blobs=True)
    verify_grant(config,grant,fresh,source=source,config_sha=base.sha(a.config))
    for path in (a.config,a.grant,a.fresh_gate,*[Path(config[k]) for k in ('control','output','lock')]):
        if not path.is_absolute() or path.resolve()!=path or not path.is_relative_to('/home/AbeHiromu/projects'):raise ValueError('BOUND_PRIVATE_PATH')
    if any(Path(config[k]).exists() for k in ('control','output','lock')):raise ValueError('ONE_SHOT_ALREADY_SPENT')
    fs=os.statvfs(Path(config['control']).parent)
    if fs.f_bavail*fs.f_frsize<model.LIMITS['stage_storage_bytes']+2*2**30 or fs.f_favail<4096:raise ValueError('STORAGE_GATE')
    from launch_track_a_v1_phase_a import supervise
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',**{name:'1' for name in THREADS})
    command=[config['python'],'-B',str(Path(__file__).resolve()),'--execute-approved','--config',str(a.config),
             '--grant',str(a.grant),'--fresh-gate',str(a.fresh_gate)]
    old=os.sched_getaffinity(0)
    try:
        os.sched_setaffinity(0,{grant['monitor_cpu']})
        supervise(command,env=env,control=config['control'],output=config['output'],limits=model.LIMITS,allowed_cpus=grant['allowed_cpus'])
    finally:os.sched_setaffinity(0,old)


if __name__=='__main__':main()
