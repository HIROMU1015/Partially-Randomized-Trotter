"""Fresh-interpreter continuation for technical SIGBUS holes, after run02 exits.

Uses the same isolated-task scheduler and unchanged scientific kernels. The
spawn context avoids inheriting live native-library state from the controller.
No successful trajectory is repeated and native scientific STOPs stay closed.
"""
import faulthandler
import multiprocessing as real_mp
from pathlib import Path
import shutil
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
    for old in directory.parent.glob(str(index)+'_unfinished_*/events_PRIVATE.json'):
        try: actual=recovery.event_digest(directory/'events_PRIVATE.json')
        except FileNotFoundError: actual=recovery.event_digest(directory/'events_PRIVATE.json')
        if recovery.event_digest(old)!=actual: raise ValueError('EVENT_REPLAY_MISMATCH')


def main():
    progress=b.read(b.PRIVATE/'B3_EXPLORATION_progress.json')
    if progress.get('active') or progress.get('pending'):
        raise ValueError('PREVIOUS_COHORT_STILL_ACTIVE; do not interrupt it')
    summary=b.read(b.PUBLIC/'b3_exploration_cost_summary.json')
    if not summary['errors']: raise ValueError('PREVIOUS_RUN_ALREADY_SUCCEEDED')
    if any(e.get('native_STOP') or e.get('exitcode')!=-7 for e in summary['errors']):
        raise ValueError('ONLY_SIGBUS_TECHNICAL_HOLES; scientific STOPs are preserved')
    preserved=b.PRIVATE/'interruptions/run02_SIGBUS';preserved.mkdir(exist_ok=False)
    for path in [b.PRIVATE/'B3_EXPLORATION_progress.json',b.PUBLIC/'b3_exploration_cost_summary.json',b.PUBLIC/'b3_exploration_costs.json.gz']:
        shutil.copy2(path,preserved/path.name)
    for error in summary['errors']:
        r=error['candidate'];directory=b.PRIVATE/'cost/B3_EXPLORATION'/r['id']/str(error['index'])
        if (directory/'result.json').exists(): continue
        if (directory/'STOP.json').exists(): raise ValueError('PRESERVED_SCIENTIFIC_STOP')
        directory.rename(directory.with_name(directory.name+'_unfinished_recovery01'))
    # Replace only this module's multiprocessing context and worker entry.
    # The currently running/previous study's processes are never modified.
    recovery.mp=SimpleNamespace(get_context=lambda _:real_mp.get_context('spawn'))
    recovery.trial=fresh_trial
    recovery.cached_context=original_context
    b.run_cost=recovery.run_cost
    recovery.run_cost('B3_EXPLORATION','native_selection.json')
    import h46_b3_finish as finish
    finish.pipeline()


if __name__=='__main__': main()
