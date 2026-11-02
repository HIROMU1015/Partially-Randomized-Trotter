#!/usr/bin/env python3
"""Same frozen price workers; adapt only submission to observed spare CPU."""
import importlib.util,json,math,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('adaptive_hybrid_worker',ROOT/'scripts/run_h6_hybrid_fairness_extension.py');W=importlib.util.module_from_spec(s);s.loader.exec_module(W)
def main():
    W.check();plan=W.read(W.OUT/'confirmation_plan.json');stage=W.OUT/'stages/confirmation';stage.mkdir(parents=True,exist_ok=False)
    W.save(W.OUT/'CONFIRMATION_DISPATCH_SOURCE.json',dict(code_sha256=W.sha(Path(__file__)),confirmation_plan_sha256=W.sha(W.OUT/'confirmation_plan.json'),price_worker_SOURCE_unchanged=True,only_resource_submission_adaptive=True))
    jobs=[(cid,i) for cid,x in plan['candidates'].items() for i in range(x['n_C'])]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    active={};results=[];next_job=0;peak=0;failed=False;initial=None
    while active or (next_job<len(jobs) and not failed):
        for p,(cid,i,start,stdout,stderr) in list(active.items()):
            rc=p.poll()
            if rc is None:continue
            p.wait();stdout.close();stderr.close();del active[p]
            results.append(dict(candidate_id=cid,index=i,pid=p.pid,exit_code=rc,reaped=True,elapsed_seconds=time.time()-start))
            failed|=rc!=0
        if next_job<len(jobs) and not failed:
            capacity=W.D.idle_capacity();initial=initial or capacity
            slots=max(0,min(capacity['affinity']-len(active),math.floor(capacity['idle_core_equivalents'])))
            if not active:slots=max(1,slots)
            for _ in range(min(slots,len(jobs)-next_job)):
                cid,i=jobs[next_job];next_job+=1;name=stage/'native'/f'{cid}_{i:05d}';name.parent.mkdir(exist_ok=True)
                stdout=name.with_suffix('.stdout.log').open('w');stderr=name.with_suffix('.stderr.log').open('w')
                cmd=['nice','-n','10',W.R.COMP,str(ROOT/'scripts/run_h6_hybrid_fairness_extension.py'),'price',cid,'confirmation',str(i)]
                p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
                active[p]=(cid,i,time.time(),stdout,stderr)
            peak=max(peak,len(active))
        else:time.sleep(.2)
    W.save(stage/'resources.json',dict(initial_capacity=initial,peak_workers=peak,adaptive_to_spare_CPU=True,no_worker_or_global_setting_changed=True))
    W.save(stage/'native.json',dict(workers=peak,results=results,all_exit_zero=not failed,all_reaped=True))
    if failed:raise RuntimeError('FAILED_CONFIRMATION_PRESERVED_NO_RETRY')
    assert next_job==len(jobs)==len(results)
    print('ADAPTIVE_CONFIRMATION_EXIT0_REAPED',len(results),'peak_workers',peak,flush=True)
if __name__=='__main__':main()
