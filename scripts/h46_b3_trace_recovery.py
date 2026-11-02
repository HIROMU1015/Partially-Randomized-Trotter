"""One fixed-seed recovery pass, after healthy run03 workers finish naturally.

Use the existing science kernels, durable writes, cohort registry, scheduler and
completion checks. strace records the actual SIGBUS siginfo without replacing
the compiler or installing a Python signal handler. Diagnostics use separate
trial directories and are adopted only after the original controller exits.
"""
from __future__ import annotations
import faulthandler
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import h46_b3 as b
import h46_b3_recover as recovery
from h46_b3_durable import save

WORK=b.PRIVATE/'diagnostic_run03'
OLD_UNIT='prt-h46-b3-20261026-run03.service'
UNIT='prt-h46-b3-20261026-run04.service'
SCRIPT=Path(__file__).resolve()


def state(unit):
    result=subprocess.run(['systemctl','--user','show',unit,'-p','ActiveState','-p','Result','-p','ExecMainStatus'],
        capture_output=True,text=True,check=True)
    return dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)


def is_active(unit):
    return state(unit)['ActiveState'] in ('active','activating','deactivating')


def check_sources():
    for name,expected in b.read(b.PUBLIC/'scientific_provenance.json')['new_source_files'].items():
        if b.signal.sha(b.ROOT/name)!=expected:
            raise ValueError('PINNED_SOURCE_CHANGED:'+name)


def fixed_row(identity,index,cohort='B3_EXPLORATION'):
    if cohort not in ('B3_EXPLORATION','B3_CONFIRMATION'): raise ValueError('UNKNOWN_COHORT')
    plan=b.read(b.PUBLIC/('native_selection.json' if cohort=='B3_EXPLORATION' else 'confirmation_selection.json'))
    row=next(r for r in plan['candidates'] if r['id']==identity)
    if not 0<=index<plan.get('n_by_candidate',{}).get(identity,9):
        raise ValueError('TRIAL_INDEX_OUTSIDE_FIXED_PLAN')
    return row


def technical_failures(progress):
    if not progress['errors']: raise ValueError('NO_TECHNICAL_HOLES')
    if any(e['exitcode']!=-7 or e['native_STOP'] for e in progress['errors']):
        raise ValueError('ONLY_TECHNICAL_SIGBUS_HOLES; preserve scientific STOPs')
    return progress['errors']


def check_trial(result,row,index,cohort='B3_EXPLORATION'):
    if result['status']!='PASS' or result['candidate']!=row or result['index']!=index or result['cohort']!=cohort:
        raise ValueError('RECOVERED_TRIAL_BINDING')
    if result['conditional'] or sum(result['orders'].values())!=row['s']:
        raise ValueError('RECOVERED_EVENT_COUNT')


def diagnostic(identity,index):
    """Replay only a recorded technical hole, without touching its original."""
    check_sources();faulthandler.enable(all_threads=True)
    failures=technical_failures(b.read(b.PRIVATE/'B3_EXPLORATION_progress.json'))
    if not any(e['candidate']['id']==identity and e['index']==index for e in failures):
        raise ValueError('NOT_A_RECORDED_TECHNICAL_HOLE')
    row=fixed_row(identity,index);native=b.setup();native.save=save
    native.CONTEXT[row['system']]=b.native_context(row['system'],[0])
    root=WORK/'replays';root.mkdir(exist_ok=True)
    native.PRIVATE=root
    original_compile=native.compile_cached
    def shared_compile(*args):
        native.PRIVATE=b.PRIVATE
        try: return original_compile(*args)
        finally: native.PRIVATE=root
    recovery.NATIVE=native;recovery.ORIGINAL_COMPILE=shared_compile
    native.compile_cached=recovery.compiled_with_representative_action
    result=native.cost_trial((row,'B3_EXPLORATION',index))
    check_trial(result,row,index)
    directory=root/'cost/B3_EXPLORATION'/identity/str(index)
    old=b.PRIVATE/'cost/B3_EXPLORATION'/identity/str(index)/'events_PRIVATE.json'
    matched=[]
    if old.exists() and old.stat().st_size:
        if recovery.event_digest(old)!=recovery.event_digest(directory/'events_PRIVATE.json'):
            save(directory/'STOP.json',dict(reason='FAILED_RUN03_EVENT_REPLAY_MISMATCH'))
            raise ValueError('FAILED_RUN03_EVENT_REPLAY_MISMATCH')
        matched.append(str(old))
    save(directory/'diagnostic_replay_check.json',dict(status='PASS',matched_sources=matched,
        cohort='B3_EXPLORATION',fixed_index=index,same_fixed_seed=True,
        result_sha256=b.signal.sha(directory/'result.json'),additional_independent_samples=0,
        helper_source_sha256=b.signal.sha(SCRIPT)))
    print(json.dumps(dict(status='PASS',candidate=identity,index=index,wall_seconds=result['wall_seconds'])),flush=True)


def worker(jobfile):
    check_sources()
    row,cohort,index=b.read(jobfile)
    if fixed_row(row['id'],index,cohort)!=row: raise ValueError('FIXED_ROW_CHANGED')
    from h46_b3_spawn_continue import fresh_trial
    fresh_trial((row,cohort,index))


class TracedProcess:
    """The existing scheduler's Process interface, with a fresh traced child."""
    def __init__(self,target,args):
        self.job=args[0]

    def start(self):
        row,cohort,index=self.job
        directory=WORK/'run04_traces'/cohort/row['id']/str(index)
        directory.mkdir(parents=True,exist_ok=False)
        job=directory/'job.json';save(job,self.job)
        self.output=(directory/'stdout.log').open('wb')
        self.error=(directory/'stderr.log').open('wb')
        self.process=subprocess.Popen(['strace','-qq','-ff','-tt','-e','trace=none',
            '-e','signal=SIGBUS,SIGSEGV','-o',str(directory/'signals'),
            sys.executable,'-B',str(SCRIPT),'worker','--job',str(job)],
            cwd=b.ROOT,stdout=self.output,stderr=self.error)
        # Report the science child's RSS to the unchanged memory scheduler,
        # rather than the small strace parent's RSS.
        children=Path('/proc',str(self.process.pid),'task',str(self.process.pid),'children')
        self.pid=self.process.pid
        deadline=time.monotonic()+5
        while time.monotonic()<deadline and self.process.poll() is None:
            try: values=children.read_text().split()
            except FileNotFoundError: values=[]
            if values:
                self.pid=int(values[0]);break
            time.sleep(.01)

    def is_alive(self): return self.process.poll() is None

    def join(self):
        self.process.wait();self.output.close();self.error.close()

    @property
    def exitcode(self): return self.process.returncode


def preserve_and_adopt(progress):
    technical_failures(progress)
    rows=b.read(b.PUBLIC/'native_selection.json')['candidates']
    # Validate all proposed moves first. A STOP or incomplete diagnostic must
    # leave the original trial directories untouched.
    for row in rows:
        for index in range(9):
            directory=b.PRIVATE/'cost/B3_EXPLORATION'/row['id']/str(index)
            if (directory/'STOP.json').exists(): raise ValueError('PRESERVED_SCIENTIFIC_STOP')
            if (directory/'result.json').exists(): check_trial(b.read(directory/'result.json'),row,index)
            diagnostic=WORK/'replays/cost/B3_EXPLORATION'/row['id']/str(index)
            if not diagnostic.exists(): continue
            if (diagnostic/'STOP.json').exists() or not (diagnostic/'diagnostic_replay_check.json').exists():
                raise ValueError('DIAGNOSTIC_NOT_ACCEPTED')
            check_trial(b.read(diagnostic/'result.json'),row,index)
            evidence=b.read(diagnostic/'diagnostic_replay_check.json')
            if evidence['status']!='PASS' or b.signal.sha(diagnostic/'result.json')!=evidence['result_sha256']:
                raise ValueError('DIAGNOSTIC_RESULT_CHANGED')
    preserved=b.PRIVATE/'interruptions/run03_SIGBUS'
    preserved.mkdir(exist_ok=False)
    for path in (b.PRIVATE/'B3_EXPLORATION_progress.json',b.PUBLIC/'b3_exploration_cost_summary.json',
                 b.PUBLIC/'b3_exploration_costs.json.gz',b.PRIVATE/'run03_stderr.log'):
        if path.exists(): shutil.copy2(path,preserved/path.name)
    renamed=0;adopted=[]
    for row in rows:
        for index in range(9):
            directory=b.PRIVATE/'cost/B3_EXPLORATION'/row['id']/str(index)
            if (directory/'STOP.json').exists(): raise ValueError('PRESERVED_SCIENTIFIC_STOP')
            if (directory/'result.json').exists():
                check_trial(b.read(directory/'result.json'),row,index);continue
            if directory.exists():
                directory.rename(directory.with_name(str(index)+'_unfinished_run03'));renamed+=1
            diagnostic=WORK/'replays/cost/B3_EXPLORATION'/row['id']/str(index)
            if not diagnostic.exists(): continue
            if (diagnostic/'STOP.json').exists() or not (diagnostic/'diagnostic_replay_check.json').exists():
                raise ValueError('DIAGNOSTIC_NOT_ACCEPTED')
            result=b.read(diagnostic/'result.json');check_trial(result,row,index)
            evidence=b.read(diagnostic/'diagnostic_replay_check.json')
            if evidence['status']!='PASS' or b.signal.sha(diagnostic/'result.json')!=evidence['result_sha256']:
                raise ValueError('DIAGNOSTIC_RESULT_CHANGED')
            directory.parent.mkdir(exist_ok=True)
            diagnostic.rename(directory)
            adopted.append(dict(id=row['id'],index=index,result_sha256=evidence['result_sha256']))
    record=dict(status='RUN04_FIXED_SEED_RECOVERY_STARTED',previous_errors=progress['errors'],
        completed_before_recovery=progress['complete'],preserved=str(preserved),
        unfinished_directories_preserved=renamed,adopted_diagnostic_trials=adopted,
        helper_source='scripts/h46_b3_trace_recovery.py',helper_source_sha256=b.signal.sha(SCRIPT),
        scientific_kernel_inputs_compiler_seed_plan_n_unchanged=True,
        original_workers_interrupted=False,successful_trials_repeated=False,
        recovery_policy='One recovery pass; later failures stay stopped; no recursive retries',
        SIGBUS_root_cause='UNRESOLVED_UNLESS_NEW_SIGINFO_IDENTIFIES_CAUSE',
        diagnostic_artificial_SIGBUS='Artificial signal tracing test, not scientific data')
    save(b.PUBLIC/'interruption_run03.json',record)


def continuation():
    WORK.mkdir(exist_ok=True)
    while is_active(OLD_UNIT):
        save(WORK/'continuation_status.json',dict(stage='WAITING_FOR_HEALTHY_RUN03_WORKERS',updated_epoch=time.time()))
        time.sleep(20)
    progress=b.read(b.PRIVATE/'B3_EXPLORATION_progress.json')
    if progress['active'] or progress['pending']: raise ValueError('ORIGINAL_WORK_NOT_DRAINED')
    check_sources()
    with (b.PRIVATE/'B3_EXPLORATION_runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        preserve_and_adopt(progress)
    recovery.mp=SimpleNamespace(get_context=lambda _:SimpleNamespace(Process=TracedProcess))
    recovery.cached_context=b.native_context
    b.save=save;b.run_cost=recovery.run_cost
    with (WORK/'archive_run04.log').open('a') as output:
        archive=subprocess.Popen([sys.executable,'-B',str(b.ROOT/'scripts/h46_b3_archive.py')],stdout=output,stderr=output)
        try:
            recovery.run_cost('B3_EXPLORATION','native_selection.json')
            import h46_b3_finish as finish
            finish.pipeline()
        finally:
            if (b.PUBLIC/'stages_complete.json').exists():
                if archive.wait()!=0: raise ValueError('EVENT_ARCHIVE_EXIT')
            else:
                archive.terminate();archive.wait()


def completion():
    while is_active('prt-h46-b3-20261026-complete.service'): time.sleep(10)
    import h46_b3_complete as complete
    complete.UNIT=UNIT
    complete.main()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['diagnostic','worker','continue','complete'])
    parser.add_argument('--candidate');parser.add_argument('--index',type=int);parser.add_argument('--job')
    args=parser.parse_args()
    if args.command=='diagnostic': diagnostic(args.candidate,args.index)
    elif args.command=='worker': worker(args.job)
    elif args.command=='continue': continuation()
    else: completion()
