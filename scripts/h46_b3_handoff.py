"""Drain existing B3 workers before resuming queued work with validated fast SHA.

Pause only this study's verified admission controller. No science worker or
process group receives a signal. Observe all worker exits before interrupting
the controller; its existing multiprocessing exit handler joins finished tasks.
"""
from pathlib import Path
import fcntl
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import h46_b3 as b
import h46_b3_accelerated_run as accelerated
import h46_b3_trace_recovery as traced
from h46_b3_durable import save

UNIT='prt-h46-b3-20261026-switch-fast.service'
STATUS=b.PRIVATE/'acceleration/handoff_status.json'


def process(pid):
    try:
        p=Path('/proc',str(pid));fields=(p/'stat').read_text().rsplit(')',1)[1].split()
        uid=int(next(x.split()[1] for x in (p/'status').read_text().splitlines() if x.startswith('Uid:')))
        return dict(state=fields[0],start=fields[19],exit_status=int(fields[49]),uid=uid)
    except (FileNotFoundError,ProcessLookupError):return None


def science_children(pid):
    result=[]
    for token in Path('/proc',str(pid),'task',str(pid),'children').read_text().split():
        child=int(token);info=process(child)
        if info is None:continue
        try:cmd=Path('/proc',token,'cmdline').read_bytes()
        except FileNotFoundError:continue
        if b'h46_b3_archive.py' in cmd or b'multiprocessing.resource_tracker' in cmd:continue
        if b'multiprocessing.spawn' not in cmd and info['state']!='Z':
            raise ValueError('UNEXPECTED_CONTROLLER_CHILD:'+token)
        if info['uid']!=os.getuid():raise ValueError('CHILD_UID_MISMATCH')
        result.append(child)
    return result


def rss(pid):
    try:return 1024*int(next(x.split()[1] for x in Path('/proc',str(pid),'status').read_text().splitlines() if x.startswith('VmRSS:')))
    except (FileNotFoundError,StopIteration):return 0


def pause_and_drain(pid,expected_script,*,minimum_startup_rss=128*2**20,observe=None,poll_seconds=5):
    parent=process(pid)
    if parent is None or parent['uid']!=os.getuid():raise ValueError('CONTROLLER_UID_OR_LIVENESS')
    if expected_script.encode() not in Path('/proc',str(pid),'cmdline').read_bytes():
        raise ValueError('CONTROLLER_SOURCE_BINDING')
    if (Path('/proc',str(pid))/'cwd').resolve()!=b.ROOT:raise ValueError('CONTROLLER_CWD_BINDING')
    interrupted=False
    try:
        os.kill(pid,signal.SIGSTOP)
        deadline=time.monotonic()+2
        while process(pid)['state'] not in ('T','t'):
            if time.monotonic()>deadline:raise ValueError('CONTROLLER_NOT_PAUSED')
            time.sleep(.01)
        children=science_children(pid)
        deadline=time.monotonic()+20
        while True:
            states={c:process(c) for c in children}
            if any(v is None for v in states.values()):raise ValueError('WORKER_DISAPPEARED')
            if all(v['state']=='Z' or rss(c)>=minimum_startup_rss for c,v in states.items()):break
            if time.monotonic()>deadline:raise ValueError('WORKER_STARTUP_NOT_DRAINABLE')
            time.sleep(.1)
        while True:
            info=process(pid)
            if info is None or info['start']!=parent['start'] or info['state'] not in ('T','t'):
                raise ValueError('CONTROLLER_CHANGED_DURING_HANDOFF')
            states={c:process(c) for c in children}
            if any(v is None for v in states.values()):raise ValueError('WORKER_DISAPPEARED_BEFORE_JOIN')
            alive=[c for c,v in states.items() if v['state']!='Z']
            if observe:observe(alive,children)
            if not alive:break
            time.sleep(poll_seconds)
        codes={str(c):os.waitstatus_to_exitcode(v['exit_status']) for c,v in states.items()}
        os.kill(pid,signal.SIGINT);os.kill(pid,signal.SIGCONT);interrupted=True
        return dict(controller_pid=pid,controller_start=parent['start'],worker_exitcodes=codes,
            healthy_workers_signalled=0,all_workers_finished_before_controller_interrupt=True)
    finally:
        info=process(pid)
        if not interrupted and info is not None and info['start']==parent['start'] and info['state'] in ('T','t'):
            os.kill(pid,signal.SIGCONT)


def main():
    def termination(signum,frame):raise SystemExit('HANDOFF_TERMINATED')
    signal.signal(signal.SIGTERM,termination)
    accelerated.validation_summary();traced.check_sources()
    result=subprocess.run(['systemctl','--user','show',traced.OLD_UNIT,'-p','MainPID','--value'],capture_output=True,text=True,check=True)
    pid=int(result.stdout.strip())
    before=b.read(b.PRIVATE/'B3_EXPLORATION_progress.json');traced.technical_failures(before)
    save(b.PRIVATE/'acceleration/controller_progress_before_handoff.json',before)
    rows=b.read(b.PUBLIC/'native_selection.json')['candidates']
    def observe(alive,children):
        count=sum((b.PRIVATE/'cost/B3_EXPLORATION'/r['id']/str(i)/'result.json').exists() for r in rows for i in range(9))
        save(STATUS,dict(stage='DRAINING_HEALTHY_WORKERS_FOR_ACCELERATION',updated_epoch=time.time(),
            old_controller_pid=pid,old_science_workers=len(children),running_old_workers=len(alive),
            observed_completed_result_files=count,original_counter_frozen=before['complete'],
            healthy_workers_signalled=0,next='Resume queued work and technical holes using exact faster fingerprint'))
    observation=pause_and_drain(pid,'scripts/h46_b3_spawn_continue.py',observe=observe)
    save(b.PRIVATE/'acceleration/drained_worker_exits.json',observation)
    while traced.is_active(traced.OLD_UNIT):time.sleep(2)
    if any(code not in (0,-7) for code in observation['worker_exitcodes'].values()):
        raise ValueError('NON_SIGBUS_WORKER_EXIT; no automatic recovery')
    accelerated.UNIT=UNIT;accelerated.configure()
    with (b.PRIVATE/'B3_EXPLORATION_runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        traced.preserve_and_adopt(before)
        path=b.PUBLIC/'interruption_run03.json';record=b.read(path)
        completed=sum((b.PRIVATE/'cost/B3_EXPLORATION'/r['id']/str(i)/'result.json').exists()
            for r in rows for i in range(9))
        record.update(graceful_acceleration_handoff=observation,handoff_source='scripts/h46_b3_handoff.py',
            handoff_source_sha256=b.signal.sha(Path(__file__)),old_progress_counters_frozen_during_drain=True,
            completed_at_controller_pause=before['complete'],
            completed_before_recovery=completed-len(record['adopted_diagnostic_trials']),new_controller_unit=UNIT)
        save(path,record)
    traced.recovery.mp=SimpleNamespace(get_context=lambda _:SimpleNamespace(Process=traced.TracedProcess))
    traced.recovery.cached_context=b.native_context
    b.save=save;b.run_cost=traced.recovery.run_cost
    save(STATUS,dict(stage='ACCELERATED_RECOVERY_RUNNING',updated_epoch=time.time(),
        worker_exitcodes=observation['worker_exitcodes'],validation='acceleration_validation.json'))
    with (b.PRIVATE/'acceleration/archive_fast.log').open('a') as output:
        archive=subprocess.Popen([sys.executable,'-B',str(b.ROOT/'scripts/h46_b3_archive.py')],stdout=output,stderr=output)
        try:
            traced.recovery.run_cost('B3_EXPLORATION','native_selection.json')
            import h46_b3_finish as finish
            finish.pipeline()
        finally:
            if (b.PUBLIC/'stages_complete.json').exists():
                if archive.wait()!=0:raise ValueError('EVENT_ARCHIVE_EXIT')
            else:archive.terminate();archive.wait()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['continue','complete'])
    if parser.parse_args().command=='continue':main()
    else:
        accelerated.UNIT=UNIT;accelerated.configure();traced.completion()
