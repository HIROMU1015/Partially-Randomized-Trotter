"""Five-core recovery gate: two preparation workers, one serial compile, detached witness."""
import argparse
import json
import math
import os
from pathlib import Path
import time
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))

from h8_phase_b2_recovery_witness import verify_witness,wait_witness_idle
import h8_phase_b2_recovery_contract as c
import h8_phase_b2_recovery_runtime as r
import h8_phase_b2_recovery_observer as observer


def snapshot():
    value=observer.snapshot()
    if not value['events'] or 'host' not in value['pressure']:raise ValueError('OOM_OR_PSI_UNKNOWN')
    for fields in value['events'].values():
        if not {'oom','oom_kill'}<=set(fields):raise ValueError('OOM_COUNTERS_UNKNOWN')
        if any(int(v)<0 for v in fields.values()):raise ValueError('INVALID_OOM_COUNTER')
    for raw in value['pressure'].values():
        full=[s for s in raw.splitlines() if s.startswith('full ')]
        if len(full)!=1:raise ValueError('FULL_PSI_UNKNOWN')
        x=float(dict(p.split('=') for p in full[0].split()[1:])['avg10'])
        if not math.isfinite(x) or x<0:raise ValueError('INVALID_PSI')
    return value


def fresh(config,g,config_path,*,idle_seconds=1.,bindings_verified=False,launch_affinity=None,cpu_record=lambda value:None,resource_record=lambda value:None):
    cpus=r.verify_authorization(config,g,config_sha=r.sha(config_path),affinity=launch_affinity);r.protected(config)
    siblings=r.topology(cpus)
    pids=[os.getpid(),os.getppid()];started=Path(config['control'])/'started.json'
    if started.exists():pids.append(r.read(started)['driver']['pid'])
    CPU=r.measure_idle(siblings,seconds=idle_seconds,process_pids=pids)
    cpu_record(CPU)  # Preserve failing counters BEFORE raising or forking.
    if CPU['status']!='PASS':raise r.CPUIdleError(CPU)
    value=snapshot();limits=g['resources']
    reason=observer.resource_reason(value,value,limits['aggregate_RSS_bytes']+limits['admission_reserve_bytes'])
    resource_record(observer.resource_failure(value,value,limits['aggregate_RSS_bytes']+limits['admission_reserve_bytes'],reason))
    if reason:raise ValueError('FRESH_RESOURCE:'+reason)
    fs=r.filesystem(Path(config['control']).parent)
    if fs['bytes']<limits['fresh_free_disk_bytes'] or fs['inodes']<limits['fresh_free_inodes']:
        raise ValueError('FRESH_DISK_INODES')
    return dict(source=config['SOURCE'],config_sha256=r.sha(config_path),grant_sha256=c.digest(g),
        allowed_cpus=cpus,monotonic_seconds=time.monotonic(),profile_verified=True,
        observer_identity=list(observer.owned.identity(os.getpid())),
        source_input_bytes_verified=True,physical_cores_verified=True,resources_idle_verified=True,
        protected_jobs_unchanged=True,quota_verified=True,external_RSS_monitor_ready=True,
        effective_available_bytes=value['effective_available_bytes'],filesystem=fs,
        driver_bindings_verified=bindings_verified,CPU_observation=CPU)


def replace(path,value):
    tmp=path.with_suffix('.refresh')
    with tmp.open('x') as f:json.dump(value,f,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def launch(config_path,grant_path,fresh_path):
    config,g=map(r.read,(config_path,grant_path))
    launch_affinity=set(g.get('allowed_cpus',[]))
    r.verify_authorization(config,g,config_sha=r.sha(config_path),affinity=launch_affinity)  # Draft rejects before side effects.
    verify_witness(config,g,config_sha=r.sha(config_path),grant_sha=r.sha(grant_path))
    for path in (config_path,grant_path,fresh_path):r.private_path(path)
    control,output,lock=[r.private_path(config[k]) for k in ('control','output','lock')]
    if (fresh_path.name!='fresh_gate.json' or fresh_path.exists() or
            fresh_path in (config_path,grant_path) or
            any(p.parent!=config_path.parent for p in (control,output,lock))):
        raise ValueError('NEW_DISTINCT_FRESH_GATE_AND_SIBLING_STAGE_PATHS_REQUIRED')
    if fresh_path.parent!=config_path.parent or any(p.exists() for p in (control,output,lock)):
        raise ValueError('UNUSED_ONE_SHOT_PRIVATE_PATHS_REQUIRED')
    admission=config_path.parent/'launcher_admission01'
    if admission.exists():raise ValueError('LAUNCH_ADMISSION_ATTEMPT_ALREADY_SPENT')
    admission.mkdir(mode=0o700)  # Irrevocable even if the fresh gate rejects.
    log=r.Writer(admission);stage='SOURCE_INPUT_RUNTIME';old=os.sched_getaffinity(0)
    try:
        log.write('attempt_started.json',dict(SOURCE=config['SOURCE'],config_sha256=r.sha(config_path),
            grant_sha256=r.sha(grant_path),scope=c.SCOPE,retry=False,next_stage_authorized=False))
        log.write('witness_idle_handshake.json',wait_witness_idle(config,g))
        if os.sched_getaffinity(0)!={g['monitor_cpu']}:raise ValueError('EARLY_OBSERVER_AFFINITY_CHANGED')
        log.write('early_observer_affinity.json',dict(pid=os.getpid(),affinity=sorted(os.sched_getaffinity(0)),before_source_runtime_verification=True))
        r.verify_files(config)
        # Only this process is pinned. Its tiny sample overhead still counts
        # against the unchanged raw 5% busy threshold; nothing is subtracted.
        os.sched_setaffinity(0,{g['monitor_cpu']})
        stage='INITIAL_FRESH_CPU_RESOURCE_GATE'
        initial=fresh(config,g,config_path,launch_affinity=launch_affinity,
            cpu_record=lambda value:log.write('CPU_initial.json',value),resource_record=lambda value:log.write('resource_initial.json',value))
        r.verify_grant(config,g,initial,config_sha=r.sha(config_path),affinity=launch_affinity)
        replace(fresh_path,initial)
        stage='SUPERVISED_DRIVER'
        supervise_validation(config,g,config_path,grant_path,fresh_path,control,output,log,launch_affinity)
        log.write('terminal.json',dict(status='COMPLETE_MANDATORY_STOP',SOURCE=config['SOURCE'],
            reported_exit_code=0,driver_started=True,next_stage_authorized=False,automatic_retry=False))
    except BaseException as error:
        state={k:Path(config[k]).exists() for k in ('control','output','lock')}
        value=dict(status='STOP',SOURCE=config['SOURCE'],stage=stage,
            error_type=type(error).__name__,reason=str(error),reported_exit_code=1,
            native_wait_status_recorded_here=False,path_state=state,
            driver_started=(control/'started.json').exists(),
            science_reservation_journal_exists=(output/'reservations.jsonl').exists(),
            automatic_retry=False,CPUs_reassigned=False,next_stage_authorized=False,mandatory_stop=True)
        if isinstance(error,r.CPUIdleError):value['CPU_failure']=error.receipt
        log.write('terminal.json',value)
        raise
    finally:os.sched_setaffinity(0,old)


def supervise_validation(config,g,config_path,grant_path,fresh_path,control,output,log,launch_affinity):
    def refresh():
        fs=r.filesystem(control.parent)
        if fs['bytes']<2*2**30:raise ValueError('RUN_DISK_HEADROOM')
        logs=[control/n for n in ('stdout.log','stderr.log','observations.jsonl')]
        if sum(p.stat().st_size for p in logs if p.exists())>g['resources']['logs_bytes']:raise ValueError('COMBINED_LOG_CEILING')
        if (control/'science_affinity_bound.json').exists():
            started=r.read(control/'started.json')['driver']
            try:
                identity=observer.owned.identity(started['pid'])
                if identity is not None and list(identity)!=started['identity']:raise ValueError('DRIVER_IDENTITY_CHANGED')
                if identity is not None and os.sched_getaffinity(started['pid'])!={g['driver_cpu']}:
                    raise ValueError('DRIVER_CPU_CHANGED')
            except ProcessLookupError:pass  # Native exit is checked by the supervisor.
        ready=control/'driver_bindings_verified.json'
        if ready.exists():
            if r.read(ready)!=dict(source=config['SOURCE'],config_sha256=r.sha(config_path)):
                raise ValueError('DRIVER_BINDING_HANDSHAKE')
            # Driver has not pinned itself yet; selected cores must still be idle.
            replace(fresh_path,fresh(config,g,config_path,bindings_verified=True,
                launch_affinity=launch_affinity,cpu_record=lambda value:log.write('CPU_driver_refresh.json',value),resource_record=lambda value:log.write('resource_driver_refresh.json',value)));ready.unlink()
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    for k in r.THREAD_KEYS:env[k]='1'
    command=[config['python'],'-B',str(c.ROOT/'scripts/run_h8_phase_b2_recovery.py'),'--execute-approved',
        '--config',str(config_path),'--grant',str(grant_path),'--fresh-gate',str(fresh_path)]
    pressure_baseline=snapshot()
    observer.supervise(command,env=env,control=control,output=output,limits=g['resources'],
        allowed_cpus=g['allowed_cpus'],probe=snapshot,refresh_gate=refresh,initial_driver_cpu=g['monitor_cpu'])
    try:
        final_snapshot=snapshot()
        reason=observer.resource_reason(final_snapshot,pressure_baseline,g['resources']['admission_reserve_bytes'])
        r.Writer(control).write('resource_final.json',observer.resource_failure(final_snapshot,pressure_baseline,g['resources']['admission_reserve_bytes'],reason))
        if reason:raise ValueError('FINAL_RESOURCE:'+reason)
        r.protected(config)
    except BaseException as error:
        r.Writer(control).write('B2_final_gate.json',dict(status='STOP',reason=str(error),next_stage_authorized=False))
        raise
    r.Writer(control).write('B2_final_gate.json',dict(status='PASS_MANDATORY_STOP',
        SOURCE=config['SOURCE'],profile_sha256=c.PROFILE,next_stage_authorized=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--launch-approved',action='store_true')
    for name in ('config','grant','fresh-gate'):p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    if not a.launch_approved:
        if any((a.config,a.grant,a.fresh_gate)):p.error('real paths require approved launch mode')
        print(json.dumps(dict(status='PLAN_ONLY_NOT_AUTHORIZED',scope=c.SCOPE,axes=len(c.AXES))));return
    if not all((a.config,a.grant,a.fresh_gate)):p.error('all private inputs required')
    launch(a.config,a.grant,a.fresh_gate)


if __name__=='__main__':main()
