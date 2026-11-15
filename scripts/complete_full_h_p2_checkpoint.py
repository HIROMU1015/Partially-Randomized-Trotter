"""Authorized zero-compile completion: saved native path0 and finished path1.

Reuse the existing P2 executor, SOURCE gates, phase checks and record producer.
The historical handoff preserves its stopped SOURCE. Explicitly authorized
recovery uses the same saved inputs in a fresh namespace; no automatic retry.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_full_h_kappa2_p2 as P
import full_h_p2_acceleration as A
from continue_full_h_p2_checkpoint import identity

CALLER='scripts/complete_full_h_p2_checkpoint.py'
FROZEN='docs/research/d_full_h_p2_completion_20261115/FROZEN_COMPLETION.json'
RECOVERY='docs/research/d_full_h_p2_completion_20261115/FROZEN_RECOVERY.json'


def finished_path(receipt,row,cohort,index,seeds,compiler):
    if A.file_hash(receipt['path'])!=receipt['sha256']:raise ValueError('Finished receipt changed')
    saved=json.loads(Path(receipt['path']).read_text())
    if saved['status']!='PASS':raise ValueError('Finished path is not PASS')
    result=saved['path'];trial=result['trial'];facts=result['checks']
    expected={'low_control','high_control_leakage','independent_occupation_vector',
        'Hadamard_cosine','Hadamard_sine','compiled_vector_cosine','compiled_vector_sine'} if index==0 else set()
    if (result['row']!=row or result['cohort']!=cohort or result['index']!=index or result['seeds']!=seeds
            or trial['candidate']!=row or trial['cohort']!=cohort or trial['index']!=index or trial['status']!='PASS'
            or {g['check'] for g in trial['checks']}!=expected
            or any(not g['error']<g['tolerance'] or g['phase_alignment'] for g in trial['checks'])
            or facts['builder_lowering_id']!=P.P1.LOWERING_ID or facts['parity']['status']!='PASS'
            or facts['scalar_event_phases']['status']!='PASS' or result['xy_phase_gate']['status']!='PASS'):
        raise ValueError('Finished path identity/correctness differs')
    for axis in ('cosine','sine'):
        rec=result['compile_receipts'][axis];record=rec['record'];locator=rec['locator']
        if (record['compiler']!=compiler or record['fingerprint']!=result['fingerprints'][axis]
                or record['metrics']!=trial['prices'][axis] or rec['fingerprint']!=record['fingerprint']
                or A.file_hash(locator['path'])!=locator['sha256']
                or json.loads(Path(locator['path']).read_text())!=record):
            raise ValueError('Finished compile record differs')
        if rec.get('producer_commit',receipt['source_commit'])!=receipt['source_commit']:
            raise ValueError('Finished compile producer differs')
        rec.update(producer_commit=receipt['source_commit'],reused=True,completed_path_reused=True)
    result['completed_receipt']=receipt
    return result


def inputs(job):
    P.validate(job['plan'],execution=True)
    cell=P.selected(job['plan'],job['stage'],job['cell_id']);row=P.physical_row(cell)
    if cell['cell_id']!='G36223' or cell['n_paths']!=2 or job['new_compile_maximum']!=0:
        raise ValueError('Only remaining G36223 checks; zero new compiles')
    old=job['old_run'];previous=json.loads(Path(old['job']).read_text())
    if A.file_hash(old['job'])!=old['job_sha256']:raise ValueError('Old job changed')
    if any(previous[k]!=job[k] for k in ('plan','stage','cell_id','step_seeds')):
        raise ValueError('Fixed plan/cohort/seed differs')
    if job['checkpoint0']!=previous['resume']['0']:raise ValueError('Native checkpoint lineage differs')
    cp=job['checkpoint0']
    for path,digest in cp['files'].items():
        if A.file_hash(path)!=digest:raise ValueError('Native checkpoint changed')
    for private in (Path(cp['private']),Path(previous['private'])):
        if any((private/'cost'/cell['cohort']/row['id']/str(i)/'STOP.json').exists() for i in (0,1)):
            raise ValueError('Scientific STOP forbids continuation')
    receipt=job['finished1']
    if (receipt['source_commit']!=old['source_commit']
            or Path(receipt['path'])!=Path(previous['private'])/'path_result_1.json'):
        raise ValueError('Finished path SOURCE/locator differs')
    finished_path(receipt,row,cell['cohort'],1,job['step_seeds'][1],cell['profile']['compiler'])
    return previous,cell


def stopped(job):
    for who in job['old_run']['processes']:
        current=identity(who['pid'])
        if current and current['state']!='Z':raise ValueError('Old worker is still active')
    terminal=json.loads(Path(job['old_run']['result']).read_text())
    if (terminal['source']['commit']!=job['old_run']['source_commit'] or terminal['status']!='STOP_CELL'
            or terminal['child'] is not None or terminal['error'] is not None
            or job.get('handoff_reason')!='CONTROLLED_SAVED_PATH_HANDOFF'):
        raise ValueError('Controlled terminal record required; no failure retry')


def child(path):
    job=json.loads(Path(path).read_text());private=Path(job['private']);started=time.monotonic();last=[None,started]
    with (private/'PROGRESS.jsonl').open('x') as log:
        def progress(stage):
            now=time.monotonic()
            log.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),stage=stage,
                elapsed_seconds=now-started,previous_completed=last[0],previous_seconds=now-last[1]))+'\n')
            log.flush();last[:]=[stage,now]
        results=[]
        try:
            import h46_long_time_signal as S
            S.install();P.P1.provider_checks()
            full=P.P1.full_h_module(job['output'],job['private'])
            cell=P.selected(job['plan'],job['stage'],job['cell_id'])
            native=full.setup()
            with patch.object(native,'sample_requests',side_effect=RuntimeError('ZERO_COMPILE_COMPLETION_NO_SAMPLING')), \
                 patch.object(native,'compile_cached',side_effect=RuntimeError('ZERO_COMPILE_COMPLETION_NO_COMPILE')):
                results.append(A.resume_saved_path(P,full,P.physical_row(cell),cell['cohort'],0,
                    job['step_seeds'][0],job['checkpoint0'],progress=progress))
                progress('reuse_completed_path1')
                results.append(finished_path(job['finished1'],P.physical_row(cell),cell['cohort'],1,
                    job['step_seeds'][1],cell['profile']['compiler']))
            progress('serialize_complete_receipts')
            P.P1.write(Path(job['result']),dict(status='PASS',paths=results,new_sampling=0,new_compiles=0))
            progress('COMPLETE')
            return 0
        except BaseException:
            progress('STOP')
            P.P1.write(Path(job['result']),dict(status='STOP',error=traceback.format_exc(),paths=results))
            return 2


def observe_child(command, *, cwd, private, interval=30):
    """Observe from the parent; never walk concurrently changing Python frames.

    Monitoring delays/errors do not terminate a worker. wait() reaps it and
    retains its actual return code/signal even when no child receipt was made.
    """
    private=Path(private);started=time.monotonic()
    process=subprocess.Popen(command,cwd=cwd)
    with (private/'OBSERVATIONS.jsonl').open('x') as log:
        while True:
            observation=dict(utc=datetime.now(timezone.utc).isoformat(),pid=process.pid,
                elapsed_seconds=time.monotonic()-started)
            try:
                observation['proc_status']=Path('/proc',str(process.pid),'status').read_text()
                observation['proc_stat']=Path('/proc',str(process.pid),'stat').read_text()
                progress=private/'PROGRESS.jsonl'
                observation['progress']=progress.read_text().splitlines()[-1] if progress.exists() and progress.stat().st_size else None
                observation['path_progress']={p.name:p.read_text().splitlines()[-1]
                    for p in private.glob('PATH_*_PROGRESS.jsonl') if p.stat().st_size}
            except OSError as exc:observation['observation_error']=str(exc)
            log.write(json.dumps(observation)+'\n');log.flush()
            try:code=process.wait(timeout=interval);break
            except subprocess.TimeoutExpired:pass
    P.P1.write(private/'CHILD_EXIT.json',dict(pid=process.pid,returncode=code,
        signal=signal.Signals(-code).name if code<0 else None,elapsed_seconds=time.monotonic()-started))
    return code


def recovery_inputs(job):
    """Reuse the original saved inputs; preserve the failed verification run."""
    inputs(job);stopped(job)
    failed=job['recovery_of']
    for name in ('job','result'):
        if A.file_hash(failed[name])!=failed[name+'_sha256']:raise ValueError('Failed run evidence changed')
    old=json.loads(Path(failed['job']).read_text())
    if any(old[k]!=job[k] for k in ('plan','stage','cell_id','step_seeds','checkpoint0','finished1','old_run')):
        raise ValueError('Recovery scientific inputs differ')
    terminal=json.loads(Path(failed['result']).read_text())
    if (terminal['source']['commit']!=failed['source_commit'] or terminal['status']!='STOP_CELL'
            or terminal['child'] is not None or terminal['error'] is not None):
        raise ValueError('Recovery cannot waive a scientific failure')


def owned(who):
    current=identity(who['pid'])
    if (not current or current['uid']!=os.getuid() or current['starttime']!=who['starttime']
            or current['command']!=who['command']):raise ValueError('Old process identity differs')
    return current


def old_finished(job,previous):
    result=Path(job['old_run']['result'])
    if result.exists():
        try:value=json.loads(result.read_text())
        except json.JSONDecodeError:return True
        if value['source']['commit']!=job['old_run']['source_commit']:raise ValueError('Old terminal SOURCE differs')
        if value['status']!='FINITE_CELL_COMPLETE':raise ValueError('Old terminal STOP; no retry')
        return True
    return (Path(previous['private'])/'path_result_0.json').exists() or (Path(previous['private'])/'child.json').exists()


def handoff(job):
    previous,cell=inputs(job)
    if old_finished(job,previous):print('OLD_FINISHED_OR_FINALIZING_USE_EXISTING',flush=True);return 0
    out=ROOT/job['transfer_output'];out.mkdir(parents=True,exist_ok=False)
    processes=job['old_run']['processes'];paused=[];terminated=[]
    for who in processes:owned(who)
    try:
        # Manager first: it cannot turn the controlled worker exit into a gate failure.
        for who in processes:
            owned(who);os.kill(who['pid'],signal.SIGSTOP);paused.append(who)
            for _ in range(100):
                if owned(who)['state']=='T':break
                time.sleep(.01)
            else:raise ValueError('Stable saved boundary unavailable')
        if old_finished(job,previous):print('OLD_FINALIZING_USE_EXISTING',flush=True);return 0
        P.P1.write(out/'TRANSFER_READY.json',dict(reason='CONTROLLED_SAVED_PATH_HANDOFF',old=job['old_run'],
            checkpoint0=job['checkpoint0'],finished1=job['finished1'],new_sampling=0,new_compiles=0))
        for who in reversed(processes):
            owned(who);os.kill(who['pid'],signal.SIGTERM);terminated.append(who['pid']);os.kill(who['pid'],signal.SIGCONT)
            for _ in range(300):
                current=identity(who['pid'])
                if not current or current['state']=='Z':break
                time.sleep(.1)
            else:raise ValueError('Old process still active; no new launch')
        for _ in range(300):
            if Path(job['old_run']['result']).exists():break
            time.sleep(.1)
        else:raise ValueError('Old terminal receipt missing')
        job['handoff_reason']='CONTROLLED_SAVED_PATH_HANDOFF';stopped(job)
        source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        P.P1.write(out/'TRANSFER.json',dict(reason=job['handoff_reason'],source=source,old=job['old_run'],
            utc=datetime.now(timezone.utc).isoformat(),new_sampling=0,new_compiles=0))
        runtime=out/'JOB_COMPLETION.json';P.P1.write(runtime,job)
        with (out/'G36223.console.txt').open('x') as console:
            return subprocess.call(['/usr/bin/time','-v','-o',str(out/'G36223.time.txt'),sys.executable,'-B',
                str(ROOT/CALLER),'--execute-approved-job',str(runtime)],cwd=ROOT,stdout=console,stderr=subprocess.STDOUT)
    finally:
        for who in reversed(paused):
            if who['pid'] not in terminated:
                current=identity(who['pid'])
                if current and current['starttime']==who['starttime']:os.kill(who['pid'],signal.SIGCONT)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handoff');parser.add_argument('--execute-approved-job');parser.add_argument('--recover-approved-job');parser.add_argument('--child')
    args=parser.parse_args();P.CALLER=CALLER
    if args.child:return child(args.child)
    path=args.handoff or args.execute_approved_job or args.recover_approved_job
    if not path:parser.error('Fixed job required')
    job=json.loads(Path(path).read_text());source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for filename in (CALLER,'scripts/full_h_p2_acceleration.py','scripts/run_full_h_kappa2_p2.py'):
        if (ROOT/filename).read_bytes()!=P.P1.blob(source,filename):raise ValueError('Completion SOURCE not committed')
    frozen=json.loads(P.P1.blob(source,RECOVERY if args.recover_approved_job else FROZEN))
    actual=job if args.recover_approved_job else {k:v for k,v in job.items() if k!='handoff_reason'}
    if actual!=frozen:
        raise ValueError('Completion inputs differ from fixed SOURCE')
    if args.handoff:return handoff(job)
    if args.recover_approved_job:
        recovery_inputs(job)
        from functools import partial
        return P.execute(job,run_child=partial(observe_child,private=job['private']))
    inputs(job);stopped(job)
    return P.execute(job)


if __name__=='__main__':
    try:sys.exit(main())
    except Exception:
        traceback.print_exc();sys.exit(2)
