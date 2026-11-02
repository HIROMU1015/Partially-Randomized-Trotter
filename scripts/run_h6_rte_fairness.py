#!/usr/bin/env python3
"""New H6-only batches. Reuse frozen target, sampler, compiler and statistics."""
import concurrent.futures, csv, gzip, importlib, importlib.util, json, math, os
from pathlib import Path
import shutil, subprocess, sys, time
from fractions import Fraction as Q

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/h6_rte_fairness/2026-10-25'
OLD=ROOT/'artifacts/pauli_h2_h6_comparison/2026-10-25'
BASE=ROOT/'artifacts/pauli_hybrid_rte_comparison/2026-10-25'
PREP=ROOT/'artifacts/pauli_hybrid_rte_preparation/2026-10-25'
s=importlib.util.spec_from_file_location('h6_fair_original',ROOT/'scripts/pauli_h2_h6_study.py')
R=importlib.util.module_from_spec(s);s.loader.exec_module(R);R.OUT=OUT
F=importlib.import_module('hx_study.h6_rte_fairness')
R.CODE+=['src/trotterlib/pauli_vector_tight_enclosure.py','scripts/refine_pauli_h2_h6_numeric.py',
         'src/trotterlib/h6_rte_fairness.py','scripts/run_h6_rte_fairness.py']
read,save,sha=R.read,R.save,R.sha
idle_capacity=R.A.A.extract('scripts/run_pauli_hybrid_rte_comparison.py','idle_capacity',
                          dict(os=os,Path=Path,time=time,math=math))

def csvrows(p):
    with Path(p).open() as f:return [{k:None if v in ('','null') else v for k,v in r.items()} for r in csv.DictReader(f)]

def source_check():
    if (OUT/'STUDY_STOP.json').exists():raise ValueError('COMPLETED_STUDY_NO_RESUME')
    s=read(OUT/'SOURCE.json')
    for p,h in {**s['code_sha256'],**s['input_sha256']}.items():
        if sha(ROOT/p)!=h:raise ValueError('SOURCE_OR_INPUT_CHANGED '+p)
    return s

def stats_fn():
    import numpy as np
    return R.A.A.extract('src/trotterlib/fixed_signal_evaluation.py','_statistics',dict(np=np,math=math,Fraction=Q))

def account(c,t,st):
    out=[];prices={a:st['mean_exact'][a] if st else None for a in R.AXES}
    for e in R.EPS:
        x=F.account(t,prices,e,scale=F.scale(t['B_saved_exact']))
        x.update(candidate_id=c['candidate_id'],config=c['config'],cohort=c['cohort'],molecule=6,
                 eligibility=R.A.joint_status(t,e),price_statistics=st,target_path=c.get('target_path'),
                 B_exact=t['B_saved_exact'],bias={a:t['axes'][a] for a in R.AXES},G_SE=None)
        if st and x['N_total'] is not None:
            p=[{a:st['values'][a][i] for a in R.AXES} for i in range(st['n'])]
            z=R.A.A.paired(p,dict(zip(R.AXES,[x['N_cos'],x['N_sin']])),stats_fn())
            assert Q(z['mean_exact'])==Q(x['G_exact'])
            x['G_SE']=z['standard_error'];x['G_nominal95_halfwidth']=z['nominal95_halfwidth']
        out.append(x)
    return out

def prepare():
    import numpy as np,scipy,qiskit
    assert (np.__version__,scipy.__version__,qiskit.__version__)==('1.26.4','1.14.1','1.3.0')
    if (OUT/'SOURCE.json').exists():raise ValueError('ALREADY_INITIALIZED')
    dest=OUT/'H6/inputs';dest.mkdir(parents=True,exist_ok=False)
    for name in ['model.json','reference.json']:shutil.copyfile(OLD/'H6/inputs'/name,dest/name)
    inputs=[OLD/'H6/inputs/model.json',OLD/'H6/inputs/reference.json',
            OLD/'H6/candidate_results.json.gz',PREP/'candidate_evidence_inventory.csv',
            BASE/'comparison_by_precision.json',BASE/'new_candidate_accounting.csv',BASE/'new_price_statistics.json',
            OUT/'study_plan.json',dest/'model.json',dest/'reference.json']
    save(OUT/'SOURCE.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         code_sha256={p:sha(ROOT/p) for p in R.CODE},input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},
         old_result_commits=['8fd1ceb283309a7e515ff4a533a253260bdd0f78','90c1c60820d52ca7ddcd4fb5315e9d21bb4a86c3']))
    save(OUT/'runtime.json',dict(qiskit=qiskit.__version__,numpy=np.__version__,scipy=scipy.__version__,
         compiler_python=R.COMP,numeric_python=R.ARB,affinity=len(os.sched_getaffinity(0))))
    old=json.loads(gzip.decompress((OLD/'H6/candidate_results.json.gz').read_bytes()))
    renewed=[];audit=[]
    for cid in sorted({x['candidate_id'] for x in old if x['config']['method']=='PR' and x['config']['L_D']==0}):
        previous=[x for x in old if x['candidate_id']==cid];first=previous[0]
        rows=account(dict(candidate_id=cid,config=first['config'],cohort='exploration',target_path=first['case_path']+'/target_refined.json'),first['target'],first['price_statistics'])
        for row,prior in zip(rows,previous):
            assert row['epsilon_sig']==prior['epsilon_sig']
            if prior['N_total'] is not None:assert (row['N_cos'],row['N_sin'],row['G_exact'])==(prior['N_cos'],prior['N_sin'],prior['G_exact'])
            audit.append(dict(candidate_id=cid,epsilon_sig=row['epsilon_sig'],old_N_cos=prior['N_cos'],old_N_sin=prior['N_sin'],old_G_exact=prior['G_exact'],
                         old_reasons={a:prior['axes'][a]['rho_zero']['reason'] for a in R.AXES},
                         new_N_cos=row['N_cos'],new_N_sin=row['N_sin'],new_G_exact=row['G_exact'],
                         new_reasons={a:row['axes'][a]['rho_zero']['reason'] for a in R.AXES},old_evidence_unchanged=True))
        renewed+=rows
    save(OUT/'old_rte_accounting_extended.json',renewed);save(OUT/'integer_limit_investigation.json',audit)
    print(json.dumps(dict(old_rte_candidates=len(renewed)//5,newly_finite_rows=sum(x['old_G_exact'] is None and x['new_G_exact'] is not None for x in audit),remaining_null_rows=sum(x['new_G_exact'] is None for x in audit))),flush=True)

def dispatch(jobs,stage):
    stage.mkdir(parents=True,exist_ok=False)
    resources=idle_capacity();workers=min(len(jobs),resources['workers_from_actual_free_CPU'])
    save(stage/'resources.json',resources)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def one(job):
        cmd,name=job;base=stage/'native'/name;base.parent.mkdir(parents=True,exist_ok=True);start=time.time()
        with base.with_suffix('.stdout.log').open('w') as stdout,base.with_suffix('.stderr.log').open('w') as stderr:
            p=subprocess.Popen(['nice','-n','10']+cmd,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
            rc=p.wait()
        return dict(command=cmd,pid=p.pid,exit_code=rc,reaped=True,elapsed_seconds=time.time()-start,log_base=str(base.relative_to(ROOT)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,workers)) as pool:results=list(pool.map(one,jobs))
    save(stage/'native.json',dict(workers=workers,results=results,all_exit_zero=all(x['exit_code']==0 for x in results)))
    if any(x['exit_code'] for x in results):
        save(stage/'FAILED_STOP.json',dict(no_automatic_retry=True,all_workers_reaped=True))
        raise RuntimeError('FAILED_NATIVE_CALLS_PRESERVED')

def registry():
    result=[]
    for batch in sorted((OUT/'H6').glob('batch*')):
        for c in read(batch/'plan.json')['candidates']:
            d=batch/'cases'/c['candidate_id'];c=dict(c,case_path=str(d.relative_to(ROOT)),target_path=str((d/'target.json').relative_to(ROOT)))
            result.append(c)
    return result

def fresh_rows():
    result=[]
    for c in registry():
        p=ROOT/c['target_path']
        if not p.exists():continue
        for cohort in ['exploration','confirmation']:
            st=R.statistics(c,cohort)
            if st is None and cohort=='confirmation':continue
            result+=account(dict(c,cohort=cohort),read(p),st)
    return result

def snapshot():
    save(OUT/'new_candidate_results.json',fresh_rows())

def batch(name,triples):
    source_check();existing={c['candidate_id'] for c in registry()}
    old={r['candidate_id'] for r in read(OUT/'old_rte_accounting_extended.json')}
    candidates=[]
    for triple in triples:
        q,r,K=map(int,triple.split(':'));assert q in (1,2,4,8) and r in (2,4,8) and K in (2,4)
        c=R.spec(6,'PR',0,q,r,K)
        if c['candidate_id'] in existing|old:raise ValueError('DUPLICATE_SCIENTIFIC_CANDIDATE')
        candidates.append(c)
    path=R.register(6,name,candidates,kind='NEW_H6_RTE_FAIRNESS')
    save(path/'scope.json',dict(target_actions_max=len(candidates),exploration_trajectories_max=8*len(candidates),axis_compiles_max=16*len(candidates),no_retry=True))
    jobs=[]
    for c in candidates:
        d=path/'cases'/c['candidate_id']
        jobs.append(([R.ARB,str(ROOT/'scripts/refine_pauli_h2_h6_numeric.py'),str(OUT/'H6/inputs/model.json'),str(OUT/'H6/inputs/reference.json'),str(d/'spec.json'),str(d/'target.json')],c['candidate_id']))
    dispatch(jobs,path/'target_execution')
    jobs=[]
    for c in candidates:
        t=read(path/'cases'/c['candidate_id']/'target.json')
        if not any(R.A.joint_status(t,e)=='CONDITIONAL_GUARD_ELIGIBLE' for e in R.EPS):continue
        for i in range(8):jobs.append(([R.COMP,str(Path(__file__).resolve()),'price',str(path),c['candidate_id'],'exploration',str(i)],c['candidate_id']+f'_{i:04d}'))
    dispatch(jobs,path/'price_execution');snapshot()
    save(path/'BATCH_STOP.json',dict(completed=True,no_resume=True,target_actions=len(candidates),trajectories=len(jobs),axis_compiles=2*len(jobs)))
    print(json.dumps(dict(batch=name,new_candidates=len(candidates),exploration_trajectories=len(jobs))),flush=True)

def confirm(ids):
    source_check();candidates=[next(x for x in registry() if x['candidate_id']==cid) for cid in ids]
    prior_ids={cid for p in (OUT/'confirmations').glob('cohort_*/plan.json') for cid in read(p)['candidate_ids']}
    if len(prior_ids|set(ids))>7:raise ValueError('CONFIRMATION_SCOPE_MAX_SEVEN')
    if len(set(ids))!=len(ids):raise ValueError('DUPLICATE_CONFIRMATION')
    for c in candidates:
        if R.statistics(c,'confirmation') is not None:raise ValueError('SUCCESSFUL_CONFIRMATION_ALREADY_EXISTS')
    root=OUT/'confirmations'/('cohort_'+str(len(list((OUT/'confirmations').glob('cohort_*')))+1))
    root.mkdir(parents=True,exist_ok=False)
    save(root/'plan.json',dict(candidate_ids=ids,counts=[48,96,192,384],SOURCE=read(OUT/'SOURCE.json'),
         rule='extend iff matched/family G gap <= 2 combined SE AND nominal t95 halfwidth >1%G AND new SE > old comparator SE; unresolved at384',
         trajectory_max=384*len(ids),axes_per_trajectory=2,no_retry=True))
    pending=candidates
    for n in [48,96,192,384]:
        if not pending:break
        jobs=[]
        for c in pending:
            st=R.statistics(c,'confirmation');start=st['n'] if st else 0
            batch=ROOT/c['case_path'];batch=batch.parents[1]
            for i in range(start,n):jobs.append(([R.COMP,str(Path(__file__).resolve()),'price',str(batch),c['candidate_id'],'confirmation',str(i)],c['candidate_id']+f'_{i:04d}'))
        dispatch(jobs,root/f'n{n:03d}');snapshot()
        # Selection-independent matched hybrid reference for the nominated epsilon;
        # assess all five epsilons to avoid extending only a favorable cell.
        rows=fresh_rows();decisions=[];next_pending=[]
        prior=read(BASE/'comparison_by_precision.json')
        for c in pending:
            extend=False
            for x in [r for r in rows if r['candidate_id']==c['candidate_id'] and r['cohort']=='confirmation']:
                e=x['epsilon_sig'];h=next(t for t in prior if t['molecule']=='H6' and t['epsilon_sig']==e)['confirmation_Hybrid']['selected']
                if x['G_exact'] is None:continue
                from scipy.stats import t
                gap=abs(float(Q(x['G_exact'])-Q(h['G_exact'])));a=x['G_SE'];b=float(h['G_SE'])
                half=float(t.ppf(.975,n-1))*a
                more=gap<=2*math.hypot(a,b) and half>.01*float(Q(x['G_exact'])) and a>b
                decisions.append(dict(candidate_id=c['candidate_id'],epsilon_sig=e,n=n,extend=more,G_gap=gap,combined_SE=math.hypot(a,b),nominal_t95_halfwidth=half,nominal_not_population_guarantee=True))
                extend|=more
            if extend and n<384:next_pending.append(c)
        save(root/f'n{n:03d}'/'decisions.json',decisions);pending=next_pending
    save(root/'COHORT_STOP.json',dict(completed=True,no_resume=True,candidate_ids=ids))
    print(json.dumps(dict(confirmed=ids)),flush=True)

if __name__=='__main__':
    mode,*args=sys.argv[1:]
    if mode=='prepare':prepare()
    elif mode=='batch':batch(args[0],args[1:])
    elif mode=='price':source_check();R.price_job(Path(args[0]),args[1],args[2],int(args[3]))
    elif mode=='confirm':confirm(args)
    elif mode=='snapshot':source_check();snapshot()
    else:raise ValueError(mode)
