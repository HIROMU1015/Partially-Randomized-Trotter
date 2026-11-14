"""The fixed P2 cell contract, with process-parallel paths/axes and checkpoint reuse."""
from pathlib import Path
import argparse
import json
import multiprocessing as mp
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_full_h_kappa2_p2 as P
import full_h_p2_acceleration as A

CALLER='scripts/run_full_h_kappa2_p2_accelerated.py'
HELPER='scripts/full_h_p2_acceleration.py'


def path_worker(job,index):
    destination=Path(job['private'])/('path_result_'+str(index)+'.json')
    try:
        full=P.P1.full_h_module(job['output'],job['private'])
        cell=P.selected(job['plan'],job['stage'],job['cell_id'])
        args=(P,full,P.physical_row(cell),cell['cohort'],index,job['step_seeds'][index])
        checkpoint=job.get('resume',{}).get(str(index))
        if checkpoint:
            result=A.resume_saved_path(*args,checkpoint)
        else:
            result=A.accelerated_path(*args,parents=job.get('parents'))
        if cell.get('composed_of'):
            result['paired_check']=P.check_composition(result,job['parents'])
        P.P1.write(destination,dict(status='PASS',path=result))
    except BaseException:
        P.P1.write(destination,dict(status='STOP',error=traceback.format_exc()))
        raise


def child(path):
    job=json.loads(Path(path).read_text());P.validate(job['plan'],execution=True)
    results=[];errors=[];workers=[]
    try:
        import h46_long_time_signal as signal
        signal.install();P.P1.provider_checks()
        import qiskit
        if qiskit.__version__!='1.3.0':raise ValueError('Qiskit 1.3.0 required')
        for index in range(len(job['step_seeds'])):
            worker=mp.get_context('fork').Process(target=path_worker,args=(job,index),name='FullH_path_'+str(index))
            worker.start();workers.append((index,worker))
        for index,worker in workers:
            worker.join()
            p=Path(job['private'])/('path_result_'+str(index)+'.json')
            value=json.loads(p.read_text()) if p.exists() else dict(status='STOP',error='Missing path worker record')
            if worker.exitcode!=0 or value['status']!='PASS':errors.append(dict(index=index,error=value.get('error'),exit=worker.exitcode))
            else:results.append(value['path'])
        P.P1.write(Path(job['result']),dict(status='STOP' if errors else 'PASS',paths=results,errors=errors))
        return 2 if errors else 0
    except BaseException:
        for _,worker in workers:worker.join()
        P.P1.write(Path(job['result']),dict(status='STOP',error=traceback.format_exc(),paths=results))
        return 2


def check_continuation(job):
    """No duplicate scientific execution of a path already owned by the old worker."""
    if not job.get('resume'):return
    old=job['previous_run'];proc=Path('/proc')/str(old['worker_pid'])
    if proc.exists():
        fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
        if fields[0]!='Z':raise ValueError('Previous scientific worker is still active')
    previous=json.loads(Path(old['job']).read_text())
    original=json.loads(Path(old['result']).read_text())
    if (previous['plan']!=job['plan'] or previous['cell_id']!=job['cell_id']
            or previous['step_seeds']!=job['step_seeds'] or original['source']['commit']!=old['source_commit']):
        raise ValueError('Continuation SOURCE/plan/seed binding differs')
    if (old.get('termination_reason')!='CONTROLLED_AFTER_NATIVE_CHECKPOINT'
            or original.get('child') is not None or original.get('error') is not None
            or original.get('status')!='STOP_CELL'):
        raise ValueError('Only a controlled checkpoint handoff is supported; no failure retry')
    cell=P.selected(job['plan'],job['stage'],job['cell_id']);row=P.physical_row(cell)
    for index in range(cell['n_paths']):
        directory=Path(previous['private'])/'cost'/cell['cohort']/row['id']/str(index)
        if (directory/'STOP.json').exists():raise ValueError('Previous native STOP forbids continuation')
        checkpoint=job['resume'].get(str(index))
        if checkpoint:
            if checkpoint['source_commit']!=old['source_commit']:raise ValueError('Checkpoint SOURCE differs')
            for path,digest in checkpoint['files'].items():
                if A.file_hash(path)!=digest:raise ValueError('Checkpoint changed before continuation')
            trial=json.loads(Path(checkpoint['trial']).read_text())
            expected={'low_control','high_control_leakage','independent_occupation_vector',
                'Hadamard_cosine','Hadamard_sine','compiled_vector_cosine','compiled_vector_sine'} if index==0 else set()
            if (trial['status']!='PASS' or trial['candidate']!=row or trial['cohort']!=cell['cohort'] or trial['index']!=index
                    or {g['check'] for g in trial['checks']}!=expected
                    or any(not g['error']<g['tolerance'] or g['phase_alignment'] for g in trial['checks'])):
                raise ValueError('Checkpoint numerical gates incomplete before continuation')
        elif directory.exists():
            raise ValueError('Unclaimed old path exists; sampling/compile cannot be repeated')
    remaining=cell['n_paths']-len(job['resume'])
    if job['new_compile_maximum']!=2*remaining:raise ValueError('Remaining fixed compile scope differs')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-approved-job')
    parser.add_argument('--child')
    args=parser.parse_args()
    P.CALLER=CALLER
    if args.child:return child(args.child)
    if not args.execute_approved_job:parser.error('Explicit approved cell job required')
    job=json.loads(Path(args.execute_approved_job).read_text())
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for path in (CALLER,HELPER):
        if (ROOT/path).read_bytes()!=P.P1.blob(commit,path):raise ValueError('Acceleration SOURCE differs from HEAD')
    check_continuation(job)
    return P.execute(job)


if __name__=='__main__':sys.exit(main())
