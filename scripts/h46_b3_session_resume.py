"""Continue the fixed B3 pass after the user manager's external shutdown.

Reuse accepted trials and the validated fast worker without changing either.
Run in the existing login-session scope, detached from the terminal, because
this host has KillUserProcesses=false but does not keep the user manager alive.
No service, login, resource-control, or shared environment setting is changed.
"""
from pathlib import Path
import fcntl
import os
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import h46_b3 as b
import h46_b3_accelerated_run as accelerated
import h46_b3_trace_recovery as traced
from h46_b3_durable import save

WORK=b.PRIVATE/'interruptions/user_manager_exit_20261027'
SCRIPT=Path(__file__).resolve()
RECORD='interruption_user_manager_20261027.json'
OLD_UNITS=('prt-h46-b3-20261026-switch-fast.service',
           'prt-h46-b3-20261026-complete-switch-fast.service')
SUFFIX='_unfinished_user_manager_exit_20261027'


def inspect_trials(rows,root):
    """Validate the whole proposed continuation before moving any directory."""
    complete=0;unfinished=[];unstarted=0
    for row in rows:
        for index in range(9):
            directory=root/row['id']/str(index)
            if (directory/'STOP.json').exists() or (directory/'technical_exit.json').exists():
                raise ValueError('PRESERVED_TRIAL_STOP:'+str(directory))
            if (directory/'result.json').exists():
                traced.check_trial(b.read(directory/'result.json'),row,index)
                events=[directory/'events_PRIVATE.json',directory/'events_PRIVATE.json.gz']
                if not any(p.exists() and p.stat().st_size for p in events):
                    evidence=directory/'event_reconstruction.json'
                    restored=directory/'events_PRIVATE.reconstructed.json.gz'
                    if not (evidence.exists() and restored.exists() and restored.stat().st_size):
                        raise ValueError('COMPLETED_EVENTS_MISSING:'+str(directory))
                    check=b.read(evidence)
                    if check['status']!='PASS' or check['result_sha256_unchanged']!=b.signal.sha(directory/'result.json'):
                        raise ValueError('PREVIOUS_RECONSTRUCTION_CHANGED')
                complete+=1
            elif directory.exists():
                if directory.with_name(directory.name+SUFFIX).exists():
                    raise ValueError('INTERRUPTED_TRIAL_ALREADY_PRESERVED')
                unfinished.append(directory)
            else:unstarted+=1
    return dict(complete=complete,unfinished=unfinished,unstarted=unstarted)


def preflight():
    traced.check_sources()
    provenance=b.read(b.PUBLIC/'scientific_provenance.json')
    for name,expected in provenance['saved_baseline_files'].items():
        if b.signal.sha(b.BASE_PUBLIC/name)!=expected:raise ValueError('SAVED_BASELINE_CHANGED:'+name)
    for system,config in b.signal.origin_config()['systems'].items():
        if system not in provenance['inputs']:continue
        for name,item in config['inputs'].items():
            expected=provenance['inputs'][system]['files'][name]['sha256']
            if item['sha256']!=expected or b.signal.sha(item['path'])!=expected:
                raise ValueError('SAVED_SCIENTIFIC_INPUT_CHANGED:'+name)
    validation=b.read(b.PUBLIC/'acceleration_validation.json')
    for name,expected in validation['sources'].items():
        if b.signal.sha(b.ROOT/name)!=expected:raise ValueError('VALIDATED_FAST_SOURCE_CHANGED:'+name)
    for pair in validation['rows']:
        for mode in ('original','fast'):
            check=pair[mode]
            result=b.PRIVATE/'acceleration/validation'/mode/'cost/B3_EXPLORATION'/pair['candidate']/'0/result.json'
            if check['status']!='PASS' or b.signal.sha(result)!=check['result_sha256']:
                raise ValueError('MATCHED_INPUT_VALIDATION_CHANGED')
    old=b.read(b.PUBLIC/'interruption_run03.json')
    if b.signal.sha(b.ROOT/old['handoff_source'])!=old['handoff_source_sha256']:
        raise ValueError('USED_HANDOFF_SOURCE_CHANGED')
    for unit in OLD_UNITS:
        if traced.is_active(unit):raise ValueError('OLD_STUDY_SERVICE_STILL_ACTIVE')
    progress=b.read(b.PRIVATE/'B3_EXPLORATION_progress.json')
    if progress['errors']:raise ValueError('NO_RETRY_OF_RECORDED_SCIENTIFIC_OR_WORKER_FAILURE')
    rows=b.read(b.PUBLIC/'native_selection.json')['candidates']
    return progress,inspect_trials(rows,b.PRIVATE/'cost/B3_EXPLORATION')


def preserve(progress,inventory):
    previous=WORK/'previous_state';previous.mkdir(parents=True,exist_ok=False)
    for path in (b.PRIVATE/'B3_EXPLORATION_progress.json',b.PRIVATE/'final_status.json',
                 b.PRIVATE/'acceleration/handoff_status.json',
                 b.PRIVATE/'acceleration/switch_fast_stderr.log',
                 b.PUBLIC/'b3_exploration_cost_summary.json',b.PUBLIC/'b3_exploration_costs.json.gz'):
        if path.exists():shutil.copy2(path,previous/path.name)
    journal=subprocess.run(['journalctl','--user','--since','2026-10-27 09:34:00',
        '--until','2026-10-27 09:40:00','--no-pager'],capture_output=True,text=True,check=True).stdout
    if not all(token in journal for token in ('Activating special unit exit.target',
            'Stopping '+OLD_UNITS[0],'Reached target shutdown.target')):
        raise ValueError('EXTERNAL_USER_MANAGER_SHUTDOWN_NOT_ESTABLISHED')
    (previous/'user_manager_journal.log').write_text(journal)
    for directory in inventory['unfinished']:
        directory.rename(directory.with_name(directory.name+SUFFIX))
    record=dict(status='EXTERNAL_USER_MANAGER_SHUTDOWN_FIXED_PASS_CONTINUED',
        stopped_at='2026-10-27T09:36:46+09:00',
        cause='User systemd manager entered exit.target and stopped the B3 services',
        shutdown_requester='Not identified from the accessible user journal',
        user_manager_linger=False,existing_KillUserProcesses=False,
        continuation_backend='Detached process in the existing login-session scope',
        shared_settings_changed=False,completed_trials_reused=inventory['complete'],
        interrupted_trials_preserved=len(inventory['unfinished']),unstarted_trials=inventory['unstarted'],
        previous_progress=progress,preserved_private_subdirectory=str(previous.relative_to(b.PRIVATE)),
        source=str(SCRIPT.relative_to(b.ROOT)),source_sha256=b.signal.sha(SCRIPT),
        fixed_candidates_n_cohorts_seeds_inputs_compiler_unchanged=True,
        successful_trials_repeated=False,scientific_stops_reopened=False,
        recovery_policy='Continuation after an external shutdown; no automatic retry of new failures',
        resumed_epoch=time.time())
    save(b.PUBLIC/RECORD,record)


def science():
    # The supervisor lock excludes a second continuation. The existing cohort
    # lock also excludes any old/new study controller during preservation.
    with (b.PRIVATE/'B3_EXPLORATION_runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        progress,inventory=preflight();preserve(progress,inventory)
    traced.WORK=WORK
    accelerated.configure()  # Same immutable fast worker, new trace directory.
    traced.recovery.mp=SimpleNamespace(get_context=lambda _:SimpleNamespace(Process=traced.TracedProcess))
    traced.recovery.cached_context=b.native_context
    b.save=save;b.run_cost=traced.recovery.run_cost
    with (WORK/'archive.log').open('a') as output:
        archive=subprocess.Popen([sys.executable,'-B',str(b.ROOT/'scripts/h46_b3_archive.py')],stdout=output,stderr=output)
        try:
            traced.recovery.run_cost('B3_EXPLORATION','native_selection.json')
            import h46_b3_finish as finish
            finish.pipeline()
        finally:
            if (b.PUBLIC/'stages_complete.json').exists():
                if archive.wait()!=0:raise ValueError('EVENT_ARCHIVE_EXIT')
            else:archive.terminate();archive.wait()


def process_state(process):
    code=process.poll()
    return dict(ActiveState='active' if code is None else 'inactive',
        SubState='running' if code is None else 'exited',
        Result='success' if code==0 else 'running' if code is None else 'exit-code',
        ExecMainStatus=str(code) if code is not None else '',
        backend='detached_login_session_process',MainPID=process.pid)


def supervise():
    # Inherited SIG_IGN survives exec for all science/trace/archive children.
    # No automatic restart is configured, and SIGTERM is not ignored.
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    WORK.mkdir(parents=True,exist_ok=True)
    with (WORK/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (WORK/'previous_state').exists():raise ValueError('CONTINUATION_ALREADY_STARTED')
        observed=subprocess.run(['busctl','--system','get-property','org.freedesktop.login1',
            '/org/freedesktop/login1','org.freedesktop.login1.Manager','KillUserProcesses'],
            capture_output=True,text=True,check=True).stdout.strip()
        if observed!='b false':raise ValueError('LOGIN_SESSION_PERSISTENCE_NOT_ESTABLISHED')
        save(WORK/'supervisor.json',dict(pid=os.getpid(),session=os.getsid(0),
            cgroup=Path('/proc/self/cgroup').read_text().strip(),started_epoch=time.time(),
            source_sha256=b.signal.sha(SCRIPT),SIGHUP_ignored=True,automatic_retries=0))
        with (WORK/'science_stdout.log').open('ab') as output,(WORK/'science_stderr.log').open('ab') as error:
            child=subprocess.Popen([sys.executable,'-B',str(SCRIPT),'science'],
                cwd=b.ROOT,stdin=subprocess.DEVNULL,stdout=output,stderr=error)
            import h46_b3_complete as complete
            complete.state=lambda:process_state(child)
            # Preserve the old final_status before the existing completion
            # monitor writes this continuation's first running observation.
            while not (b.PUBLIC/RECORD).exists() and child.poll() is None:time.sleep(.1)
            try:complete.main()
            except BaseException as exc:
                status=b.read(complete.STATUS) if complete.STATUS.exists() else {}
                if status.get('stage')=='SCIENTIFIC_STAGE_FAILED_STOP_PRESERVED':
                    status['stderr']=str(WORK/'science_stderr.log');save(complete.STATUS,status)
                else:
                    save(complete.STATUS,dict(stage='TECHNICAL_COMPLETION_FAILED',reason=str(exc),
                        scientific_results_retained=True,new_scientific_work_started=False,
                        science_process=process_state(child)))
                raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['inspect','science','supervise'])
    args=parser.parse_args()
    if args.command=='inspect':
        progress,inventory=preflight()
        print(dict(complete=inventory['complete'],unfinished=len(inventory['unfinished']),unstarted=inventory['unstarted']))
    elif args.command=='science':science()
    else:supervise()
