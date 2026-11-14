"""Explicitly authorized, one-shot transfer at a completed native checkpoint.

Never interrupt in-flight sampling/compile. Preserve old raw and terminal results;
continue only unperformed work with the same fixed D7 plan/cohort/seeds.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import json
import os
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_full_h_kappa2_p2 as P
import full_h_p2_acceleration as A


def identity(pid):
    base=Path('/proc')/str(pid)
    if not base.exists():return None
    fields=(base/'stat').read_text().rsplit(')',1)[1].split()
    return dict(starttime=fields[19],state=fields[0],uid=base.stat().st_uid,
                command=(base/'cmdline').read_bytes().replace(b'\0',b' ').decode())


def completed_paths(previous):
    cell=P.selected(previous['plan'],'CALIBRATION',previous['cell_id']);row=P.physical_row(cell)
    private=Path(previous['private']);completed=[]
    if (private/'child.json').exists():return 'FINALIZING',[]
    for index in range(cell['n_paths']):
        directory=private/'cost'/cell['cohort']/row['id']/str(index)
        if (directory/'STOP.json').exists():raise ValueError('Old scientific STOP; no retry')
        if not directory.exists():continue
        if not (directory/'result.json').exists():return 'NATIVE_WORK_IN_PROGRESS',[]
        trial=json.loads((directory/'result.json').read_text())
        if (trial['status']!='PASS' or trial['candidate']!=row or trial['cohort']!=cell['cohort'] or trial['index']!=index
                or any(not g['error']<g['tolerance'] or g['phase_alignment'] for g in trial['checks'])):
            raise ValueError('Old checkpoint binding/gate failed')
        expected={'low_control','high_control_leakage','independent_occupation_vector',
            'Hadamard_cosine','Hadamard_sine','compiled_vector_cosine','compiled_vector_sine'} if index==0 else set()
        if {g['check'] for g in trial['checks']}!=expected:raise ValueError('Old numerical gates incomplete')
        events=json.loads((directory/'events_PRIVATE.json').read_text())
        if events['cohort']!=cell['cohort'] or events['seeds']!=previous['step_seeds'][index]:
            raise ValueError('Old seed/cohort binding differs')
        completed.append(index)
    return ('READY' if completed else 'NO_CHECKPOINT'),completed


def snapshot(previous,source,indices):
    cell=P.selected(previous['plan'],'CALIBRATION',previous['cell_id']);row=P.physical_row(cell)
    private=Path(previous['private']);cache=private/'compiled_cache';result={}
    cache_files=[*cache.glob('*.json'),*cache.glob('*.qpy.gz')]
    if len(list(cache.glob('*.json')))!=2*len(indices):raise ValueError('Unexpected compiled checkpoint count')
    for index in indices:
        directory=private/'cost'/cell['cohort']/row['id']/str(index)
        files=[directory/'result.json',directory/'events_PRIVATE.json',*cache_files]
        result[str(index)]=dict(private=str(private),trial=str(directory/'result.json'),events=str(directory/'events_PRIVATE.json'),
            source_commit=source,files={str(path):A.file_hash(path) for path in files})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--job',required=True)
    args=parser.parse_args();job=json.loads(Path(args.job).read_text());old=job['previous_run']
    previous=json.loads(Path(old['job']).read_text());pid=old['worker_pid']
    P.validate(job['plan'],execution=True)
    if any(previous[k]!=job[k] for k in ('plan','cell_id','step_seeds','stage')):raise ValueError('Fixed job differs')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for name in ('scripts/continue_full_h_p2_checkpoint.py','scripts/run_full_h_kappa2_p2_accelerated.py','scripts/full_h_p2_acceleration.py'):
        if (ROOT/name).read_bytes()!=P.P1.blob(source,name):raise ValueError('Transfer SOURCE is not committed')
    out=ROOT/'docs/research/d_full_h_p2_acceleration_20261115/run01';out.mkdir(parents=True,exist_ok=True)
    def owner(value):
        if (not value or value['starttime']!=old['worker_starttime'] or value['uid']!=os.getuid()
                or '--child '+old['job'] not in value['command']):
            raise ValueError('Old worker identity differs; no process action')
    owner(identity(pid));state,indices=completed_paths(previous)
    if state!='READY':
        print('NO_TRANSFER',state,flush=True);return 0
    # User explicitly authorized old-worker exit after this completed native boundary.
    owner(identity(pid));os.kill(pid,signal.SIGSTOP);terminated=False
    try:
        for _ in range(100):
            value=identity(pid)
            if value and value['state']=='T':break
            time.sleep(.01)
        else:raise ValueError('Could not establish a stable checkpoint boundary')
        state,indices=completed_paths(previous)
        if state!='READY':raise ValueError('Native boundary changed; old worker will continue')
        checkpoints=snapshot(previous,old['source_commit'],indices)
        if checkpoints!=job['expected_checkpoint']:raise ValueError('Committed checkpoint inputs changed')
        cell=P.selected(job['plan'],'CALIBRATION',job['cell_id']);remaining=cell['n_paths']-len(indices)
        if 2*remaining>job['maximum_remaining_real_compiles']:raise ValueError('Remaining compile scope differs')
        job.update(resume=checkpoints,new_compile_maximum=2*remaining)
        note=dict(reason='COMPLETED_NATIVE_CHECKPOINT_HANDOFF_NOT_SCIENTIFIC_FAILURE',
            utc=datetime.now(timezone.utc).isoformat(),old=old,acceleration_source=source,completed_paths=indices,
            new_compile_maximum=2*remaining,checkpoints=checkpoints,old_SOURCE_and_raw_unchanged=True)
        P.P1.write(out/'TRANSFER_READY.json',note)
        owner(identity(pid));os.kill(pid,signal.SIGTERM);terminated=True
        try:os.kill(pid,signal.SIGCONT)
        except ProcessLookupError:pass
        for _ in range(300):
            value=identity(pid)
            if not value or value['state']=='Z':break
            time.sleep(.1)
        else:raise ValueError('Old worker did not exit; do not start a second worker')
        original=Path(old['result'])
        for _ in range(300):
            if original.exists():
                try:
                    saved=json.loads(original.read_text())
                    if saved['source']['commit']!=old['source_commit']:raise ValueError('Old SOURCE differs')
                    break
                except json.JSONDecodeError:pass
            time.sleep(.1)
        else:raise ValueError('Old parent terminal receipt missing')
        old['termination_reason']='CONTROLLED_AFTER_NATIVE_CHECKPOINT'
        P.P1.write(out/'TRANSFER.json',dict(note,old_terminal_status=saved['status'],old_terminal_path=str(original)))
        runtime=out/'JOB_CONTINUATION.json';P.P1.write(runtime,job)
        print('Checkpoint preserved; starting the remaining fixed work.',flush=True)
        with (out/'G36223.console.txt').open('x') as stream:
            return subprocess.call(['/usr/bin/time','-v','-o',str(out/'G36223.time.txt'),sys.executable,'-B',
                str(ROOT/'scripts/run_full_h_kappa2_p2_accelerated.py'),'--execute-approved-job',str(runtime)],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    finally:
        if not terminated:
            value=identity(pid)
            if value and value['starttime']==old['worker_starttime']:os.kill(pid,signal.SIGCONT)


if __name__=='__main__':
    try:sys.exit(main())
    except Exception as exc:
        print('STOP_NO_RETRY',type(exc).__name__,str(exc),flush=True);sys.exit(2)
