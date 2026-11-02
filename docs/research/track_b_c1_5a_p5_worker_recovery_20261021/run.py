"""Independent P5-only one-shot. Exec producer once, exec lightweight batches once."""
import os
import resource
import sys
import time
import traceback
from common import HERE,ROOT,read,save,require,digest,git,new_marker,snapshot,verify
from supervisor import supervise

def main():
    require(len(sys.argv)==1,'NO_RUNTIME_OVERRIDES')
    auth=read(HERE/'authorization_receipt.json');contract=read(HERE/'execution_contract.json')
    require(git('rev-parse','HEAD^')==auth['source_commit'],'AUTHORIZATION_DIRECT_SOURCE_CHILD')
    require(git('diff','HEAD^','HEAD','--name-only')==str((HERE/'authorization_receipt.json').relative_to(ROOT)),'AUTHORIZATION_ONLY')
    require(not git('status','--porcelain'),'CLEAN_BEFORE_EXECUTION')
    verify()
    require(contract['methods']==['closed_P5_full'] and contract['trials']==1024 and contract['microsteps']==4 and contract['retry']==0,'FIXED_P5_ONLY')
    new_marker(HERE/'attempt_marker.json',dict(schema='P5R_CONSUMED_NEW_ONE_SHOT',source=auth['source_commit'],
        authorization_HEAD=git('rev-parse','HEAD'),contract_SHA256=auth['contract_SHA256'],scientific_attempts=1,retry=0,pid=os.getpid()))
    out=HERE/'result';out.mkdir();save(out/'resources_before.json',snapshot());start=time.perf_counter()
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def task(identifier,args):return dict(id=identifier,cmd=[sys.executable,'-B',str(HERE/'job.py'),*args],cwd=str(ROOT),env=env)
    try:
        p=supervise([task('producer',['producer'])],out/'producer_supervision',1)
        require(p['complete'],'PRODUCER_FAILED')
        workers=supervise([task('batch_'+str(i).zfill(3),['provider',str(i)]) for i in range(32)],
            out/'worker_supervision',contract['logical_workers'])
        require(workers['complete'],'LOGICAL_PROVIDER_FAILED')
        from summarize import summarize
        summary=summarize()
        verify();save(out/'resources_after.json',snapshot())
        own=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN)
        save(out/'terminal.json',dict(status='C1_5A_P5R_COMPLETE',scientific_attempts=1,retry=0,complete=True,
            trials=1024,microsteps=4096,wall_seconds=time.perf_counter()-start,mandatory_STOP=True,
            all_job_hierarchy_CPU_seconds=own.ru_utime+own.ru_stime+children.ru_utime+children.ru_stime,
            root_max_RSS_KiB=own.ru_maxrss,
            old_C1_5A_status_unchanged=True,old_Ordinary_Full_rerun=False,native_calls=0,shots=0,sufficient_N=None))
    except BaseException as exc:
        save(out/'resources_after.json',snapshot())
        save(out/'terminal.json',dict(status='C1_5A_P5R_TECHNICAL_STOP',scientific_attempts=1,retry=0,
            complete=False,exception=repr(exc),traceback=traceback.format_exc(),mandatory_STOP=True))
        raise

if __name__=='__main__':main()
