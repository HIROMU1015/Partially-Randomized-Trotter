"""Fresh-interpreter continuation for technical SIGBUS holes, after run02 exits.

Uses the same isolated-task scheduler and unchanged scientific kernels. The
spawn context avoids inheriting live native-library state from the controller.
No successful trajectory is repeated and native scientific STOPs stay closed.
"""
import faulthandler
import multiprocessing as real_mp
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import h46_b3 as b
import h46_b3_recover as recovery

original_context=b.native_context


def fresh_trial(job):
    faulthandler.enable(all_threads=True)
    recovery.NATIVE=b.setup()
    recovery.ORIGINAL_COMPILE=recovery.NATIVE.compile_cached
    recovery.NATIVE.compile_cached=recovery.compiled_with_representative_action
    system=job[0]['system']
    recovery.NATIVE.CONTEXT[system]=original_context(system,[0])
    recovery.NATIVE.cost_trial(job)
    r,cohort,index=job;directory=b.PRIVATE/'cost'/cohort/r['id']/str(index)
    # Byte-check any earlier preserved unfinished event record, without counting
    # a replay as an additional random observation.
    preserved=list(directory.parent.glob(str(index)+'_unfinished_*/events_PRIVATE.json'))
    diagnostic=b.PRIVATE/'diagnostic_SIGBUS_H4/cost'/cohort/r['id']/str(index)/'events_PRIVATE.json'
    if diagnostic.exists(): preserved.append(diagnostic)
    matched=[]
    for old in preserved:
        try: actual=recovery.event_digest(directory/'events_PRIVATE.json')
        except FileNotFoundError: actual=recovery.event_digest(directory/'events_PRIVATE.json')
        if recovery.event_digest(old)!=actual:
            b.save(directory/'STOP.json',dict(reason='EVENT_REPLAY_MISMATCH',preserved=str(old)))
            raise ValueError('EVENT_REPLAY_MISMATCH')
        matched.append(str(old))
    if matched: b.save(directory/'event_replay_check.json',dict(matched_sources=matched,sha256=actual))


def ensure_no_prior_scientific_process():
    names=('scripts/h46_b3_recover.py','scripts/h46_b3_sigbus_probe.py','scripts/h46_b3_spawn_continue.py')
    for name in names:
        result=subprocess.run(['pgrep','-f',name],capture_output=True,text=True)
        for token in result.stdout.split():
            if int(token)==os.getpid(): continue
            p=Path('/proc')/token
            try:
                args=(p/'cmdline').read_bytes().split(b'\0')
                if len(args)>=3 and args[1]==name.encode() and (p/'cwd').resolve()==b.ROOT:
                    raise ValueError('PRIOR_SCIENTIFIC_PROCESS_STILL_PRESENT:'+token)
            except FileNotFoundError: pass


def main(recover_lost=False):
    progress=b.read(b.PRIVATE/'B3_EXPLORATION_progress.json')
    ensure_no_prior_scientific_process()
    if (progress.get('active') or progress.get('pending')) and not recover_lost:
        raise ValueError('PREVIOUS_COHORT_STILL_ACTIVE; do not interrupt it')
    if not progress['errors'] and not recover_lost: raise ValueError('PREVIOUS_RUN_ALREADY_SUCCEEDED')
    if any(e.get('native_STOP') or e.get('exitcode')!=-7 for e in progress['errors']):
        raise ValueError('ONLY_SIGBUS_TECHNICAL_HOLES; scientific STOPs are preserved')
    preserved=b.PRIVATE/'interruptions/run02_SIGBUS_external_loss';preserved.mkdir(exist_ok=False)
    for path in [b.PRIVATE/'B3_EXPLORATION_progress.json',b.PUBLIC/'b3_exploration_cost_summary.json',b.PUBLIC/'b3_exploration_costs.json.gz']:
        shutil.copy2(path,preserved/path.name)
    renamed=0
    for r in b.read(b.PUBLIC/'native_selection.json')['candidates']:
        parent=b.PRIVATE/'cost/B3_EXPLORATION'/r['id']
        for directory in parent.glob('*'):
            if not directory.is_dir() or not directory.name.isdecimal(): continue
            if (directory/'STOP.json').exists(): raise ValueError('PRESERVED_SCIENTIFIC_STOP')
            if (directory/'result.json').exists(): continue
            directory.rename(directory.with_name(directory.name+'_unfinished_recovery02'));renamed+=1
    b.save(b.PUBLIC/'interruption_run02.json',dict(status='PRESERVED_CONTROLLER_AND_WORKERS_ABSENT',
        completed_trials=progress['complete'],recorded_active_but_no_process=progress.get('active'),
        technical_SIGBUS_tasks=len(progress['errors']),unfinished_directories_preserved=renamed,
        cause_of_controller_loss='UNKNOWN',completed_results_repeated=False,
        preserved=str(preserved),recovery='Fresh Python spawn; same law, inputs, candidates, n, cohort and seeds'))
    # Replace only this module's multiprocessing context and worker entry.
    # The currently running/previous study's processes are never modified.
    recovery.mp=SimpleNamespace(get_context=lambda _:real_mp.get_context('spawn'))
    recovery.trial=fresh_trial
    recovery.cached_context=original_context
    b.run_cost=recovery.run_cost
    with (b.PRIVATE/'archive_run03.log').open('a') as output:
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


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--recover-lost-controller',action='store_true')
    main(parser.parse_args().recover_lost_controller)
