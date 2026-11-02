"""Default is plan-only. P7 science requires a new user-approved sealed grant."""
import argparse
import hashlib
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

import track_a_v1_runner as base

ROOT=base.ROOT
contract=base.load('_v1_p7_contract',ROOT/'src/trotterlib/track_a_v1_p7_contract.py')


def verify_grant(config,grant,fresh,*,source,config_sha,now=None):
    fixed=contract.plan();now=time.monotonic() if now is None else now
    if (grant.get('schema')!='track_a_v1_p7_compile_grant_v1' or grant.get('approved_by_user') is not True
            or grant.get('runtime_authorization') is not True or grant.get('one_shot') is not True
            or grant.get('retry') is not False or grant.get('resume') is not False
            or grant.get('scope')!=fixed['scope'] or grant.get('SOURCE')!=source or not source
            or grant.get('config_sha256')!=config_sha or grant.get('plan_sha256')!=contract.digest(fixed)
            or config.get('plan_sha256')!=contract.digest(fixed)
            or grant.get('call_caps')!=fixed['call_caps'] or grant.get('resources')!=fixed['resources']
            or grant.get('frozen_model_sha256')!=contract.MODEL_SHA
            or config.get('frozen_model_sha256')!=contract.MODEL_SHA
            or grant.get('p7_signal_digest')!=config.get('p7_signal_digest')
            or not config.get('p7_signal_digest') or grant.get('next_stage_authorized') is not False):
        raise ValueError('NEW_USER_APPROVED_P7_GRANT_REQUIRED')
    cpus=[grant.get('driver_cpu'),grant.get('monitor_cpu')]
    if (grant.get('allowed_cpus')!=cpus or any(type(x) is not int for x in cpus)
            or len(set(cpus))!=2 or not set(cpus)<=os.sched_getaffinity(0)):
        raise ValueError('DISJOINT_DRIVER_OBSERVER_CPUS_REQUIRED')
    if (fresh.get('source')!=source or fresh.get('config_sha256')!=config_sha
            or fresh.get('allowed_cpus')!=cpus or type(fresh.get('monotonic_seconds')) not in (int,float)
            or not 0<=now-fresh['monotonic_seconds']<=30 or any(fresh.get(k) is not True for k in
                ('profile_verified','resources_idle_verified','protected_jobs_unchanged','external_RSS_monitor_ready'))):
        raise ValueError('FRESH_P7_GATE_REQUIRED')


def verify_bytes(config,source,*,git_blobs=False):
    if config['provider_source']!=contract.PROVIDER:raise ValueError('PROVIDER_SOURCE_CHANGED')
    for root,hashes,fixed in [(ROOT,config['source_hashes'],source),
            (Path(config['provider_root']),config['provider_hashes'],contract.PROVIDER)]:
        for name,expected in hashes.items():
            if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('SOURCE_PATH')
            if base.sha(root/name)!=expected:raise ValueError('SOURCE_BYTES_CHANGED:'+name)
            if git_blobs and hashlib.sha256(subprocess.check_output(['git','show',fixed+':'+name],cwd=root)).hexdigest()!=expected:
                raise ValueError('GIT_SOURCE_BYTES_CHANGED:'+name)
    for name,expected in config['runtime_file_hashes'].items():
        if base.sha(name)!=expected:raise ValueError('RUNTIME_PROFILE_CHANGED')
    for item in config['files']:
        p=Path(item['path'])
        if p.is_symlink() or p.stat().st_size!=item['bytes'] or base.sha(p)!=item['sha256']:
            raise ValueError('INPUT_BYTES_CHANGED:'+item['logical_name'])


def execute(config,grant,source):
    verify_bytes(config,source)
    if Path(sys.executable).resolve()!=Path(config['python']).resolve():raise ValueError('PINNED_PYTHON_REQUIRED')
    os.sched_setaffinity(0,{grant['driver_cpu']})
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS','QISKIT_NUM_PROCS','RAYON_NUM_THREADS'):
        if os.environ.get(name)!='1':raise ValueError('THREAD_ONE_ENV_REQUIRED')
    output=Path(config['output']);lock=Path(config['lock'])
    for path in (output,lock):
        if not path.is_absolute() or path.resolve()!=path or not path.is_relative_to('/home/AbeHiromu/projects') or path.exists():
            raise ValueError('NEW_EXCLUSIVE_BOUND_PATH_REQUIRED')
    with lock.open('x') as f:f.write(source+'\n');f.flush();os.fsync(f.fileno())
    output.mkdir();limits=contract.plan()['resources']
    resource.setrlimit(resource.RLIMIT_AS,(limits['AS_bytes'],limits['AS_bytes']))
    resource.setrlimit(resource.RLIMIT_CPU,(limits['CPU_seconds'],limits['CPU_seconds']))
    permit=base.Permit(config,grant,output);permit.base_contract=base.contract
    writer=base.Writer(output,permit.guard)
    def stop(*args):raise ValueError('STOP_SIGNAL_OR_WALL')
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,stop)
    signal.alarm(limits['wall_seconds'])
    try:
        scratch=output/'scratch';scratch.mkdir();mpl=scratch/'matplotlib';mpl.mkdir();font=config['font_seed']
        if Path(font['name']).name!=font['name'] or base.sha(font['path'])!=font['sha256']:raise ValueError('FONT_SEED_CHANGED')
        (mpl/font['name']).write_bytes(Path(font['path']).read_bytes())
        os.environ['MPLCONFIGDIR']=str(mpl);os.environ['NUMBA_CACHE_DIR']=str(scratch/'numba')
        os.environ['PYTHONDONTWRITEBYTECODE']='1'
        sys.path.insert(0,str(Path(config['provider_root'])/'src'))
        adapter=base.load('_v1_p7_adapter',ROOT/'scripts/track_a_v1_p7_compile.py')
        adapter.evaluate(config,permit,contract,writer)
        verify_bytes(config,source)
        writer.write('terminal.json',dict(status='COMPLETE_MANDATORY_STOP',exit_code=0,SOURCE=source,
            scope=contract.plan()['scope'],call_counts=permit.used,cost_reaccounting_performed=False,
            refit=False,next_stage_authorized=False))
    except BaseException as exc:
        base.Writer(output).write('terminal.json',dict(status='STOP',exit_code=1,SOURCE=source,error=str(exc),
            scope=contract.plan()['scope'],call_counts=permit.used,retry=False,next_stage_authorized=False))
        raise
    finally:signal.alarm(0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',action='store_true');p.add_argument('--launch-approved',action='store_true');p.add_argument('--execute-approved',action='store_true')
    for name in ('config','grant','fresh-gate'):p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    if not a.launch_approved and not a.execute_approved:
        if any((a.config,a.grant,a.fresh_gate)):p.error('sealed inputs require explicit approved mode')
        import json
        print(json.dumps(contract.plan(),indent=2));return
    if a.launch_approved==a.execute_approved or not all((a.config,a.grant,a.fresh_gate)):p.error('one mode and all sealed inputs required')
    config,grant,fresh=map(base.read_json,(a.config,a.grant,a.fresh_gate))
    source=grant.get('SOURCE') if a.execute_approved else subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    verify_grant(config,grant,fresh,source=source,config_sha=base.sha(a.config))
    if a.execute_approved:return execute(config,grant,source)
    verify_bytes(config,source,git_blobs=True)
    verify_grant(config,grant,fresh,source=source,config_sha=base.sha(a.config))
    for path in (a.config,a.grant,a.fresh_gate,Path(config['control'])):
        if not path.is_absolute() or path.resolve()!=path or not path.is_relative_to('/home/AbeHiromu/projects'):raise ValueError('PRIVATE_ABSOLUTE_PATH_REQUIRED')
    if any(Path(config[k]).exists() for k in ('output','lock','control')):raise ValueError('ONE_SHOT_ALREADY_SPENT')
    fs=os.statvfs(Path(config['control']).parent)
    if fs.f_bavail*fs.f_frsize<contract.plan()['resources']['stage_storage_bytes']+2*2**30 or fs.f_favail<4096:
        raise ValueError('FRESH_STORAGE_GATE')
    from launch_track_a_v1_phase_a import supervise
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS','QISKIT_NUM_PROCS','RAYON_NUM_THREADS'):env[name]='1'
    command=[config['python'],'-B',str(Path(__file__).resolve()),'--execute-approved','--config',str(a.config),
        '--grant',str(a.grant),'--fresh-gate',str(a.fresh_gate)]
    old=os.sched_getaffinity(0)
    try:
        os.sched_setaffinity(0,{grant['monitor_cpu']})
        supervise(command,env=env,control=config['control'],output=config['output'],limits=contract.plan()['resources'],allowed_cpus=grant['allowed_cpus'])
    finally:os.sched_setaffinity(0,old)


if __name__=='__main__':main()
