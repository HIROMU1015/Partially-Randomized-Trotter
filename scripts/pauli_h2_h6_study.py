#!/usr/bin/env python3
"""Finite batch registration, process-isolated compiles and conditional tables."""
import concurrent.futures,dataclasses,gzip,hashlib,importlib,io,json,math,os,subprocess,sys,time,types
from fractions import Fraction as Q
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/pauli_h2_h6_comparison/2026-10-25'
COMP='/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python'
ARB='/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/non-df-h4-c-reference-fix-20261023/.c-reference-fix-venv/bin/python'
pkg=types.ModuleType('hx_study');pkg.__path__=[str(ROOT/'src/trotterlib')];sys.modules[pkg.__name__]=pkg
A=importlib.import_module(pkg.__name__+'.h4_pauli_b0_comparison')
S=importlib.import_module(pkg.__name__+'.h4_pauli_pr_search')
EPS=A.EPSILONS;AXES=A.AXES
CODE=['scripts/pauli_h2_h6_study.py','scripts/pauli_h2_h6_numeric.py','src/trotterlib/pauli_vector_enclosure.py',
      'src/trotterlib/h4_pauli_pr_search.py','src/trotterlib/pauli_signal_adapter.py','src/trotterlib/rte.py',
      'src/trotterlib/h4_d2_pauli_control.py','src/trotterlib/rte_compiled_cost.py','src/trotterlib/fixed_signal_evaluation.py',
      'src/trotterlib/h4_conditional_cng.py','src/trotterlib/h4_pauli_b0_comparison.py','src/trotterlib/h4_pauli_b0_execution.py',
      'src/trotterlib/h4_ball_evidence.py']
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n');t.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def spec(n,method,L,q,r=4,K=4):
 return dict(candidate_id=f'{method}_p{L:04d}_q{q}'+(f'_r{r}_K{K}' if method=='PR' else ''),molecule=n,
             config=dict(method=method,L_D=L,q=q,r=r,K=K,T=.8))
def initial(n):
 model=read(OUT/f'H{n}/inputs/model.json');m=model['nonidentity_terms'];z=model['all_Z_prefix']
 prefixes=range(1,m+1) if n==2 else sorted(set([1,z,z+1,round(.25*m),round(.5*m),round(.75*m),round(.9*m),m-8,m-4,m]))
 b=[spec(n,'PF' if L==m else 'B0',L,q) for L in prefixes for q in [1,2,4,8]]
 pp=sorted(set(max(0,min(m-1,x)) for x in [0,round(.25*m),z,z+1,round(.5*m),round(.75*m)]))
 pr=[spec(n,'PR',L,q) for L in pp for q in [1,2,4,8]]
 pr +=[spec(n,'PR',L,q,K=2) for L in pp for q in [1,4,8]]
 pr +=[spec(n,'PR',L,q,r=r) for L in sorted(set([z,round(.5*m)])) for q in [1,4,8] for r in [2,8]]
 return list({x['candidate_id']:x for x in b+pr}.values())
def source(n):
 return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             code_sha256={p:sha(ROOT/p) for p in CODE},model_sha256=sha(OUT/f'H{n}/inputs/model.json'),
             reference_sha256=sha(OUT/f'H{n}/inputs/reference.json'),study_plan_sha256=sha(OUT/'study_plan.json'))
def verify(batch):
 p=read(batch/'plan.json');s=p['SOURCE'];n=p['molecule']
 for file,h in s['code_sha256'].items():
  if sha(ROOT/file)!=h:raise ValueError('SOURCE_CHANGED '+file)
 for file,h in [(OUT/f'H{n}/inputs/model.json',s['model_sha256']),(OUT/f'H{n}/inputs/reference.json',s['reference_sha256'])]:
  if sha(file)!=h:raise ValueError('INPUT_CHANGED')
 return p
def register(n,name,candidates,kind='initial'):
 import numpy as np
 from hx_study.pauli_signal_adapter import PauliHamiltonian
 model=read(OUT/f'H{n}/inputs/model.json');h=model['hamiltonian'];ham=PauliHamiltonian(model['n_qubits'],h['constant'],tuple(map(tuple,h['terms'])),json.dumps(h.get('metadata',{})))
 batch=OUT/f'H{n}'/name;batch.mkdir(exist_ok=False)
 for c in candidates:
  if c['config']['method']=='PR':
   x=c['config'];c['law']=S.law_descriptor(ham,S.config(x['L_D'],x['q'],x['r'],x['K']))
  save(batch/'cases'/c['candidate_id']/'spec.json',c)
 plan=dict(molecule=n,batch=name,kind=kind,candidates=candidates,SOURCE=source(n),
           calls=dict(target_max=len(candidates),exploration_trajectory_max=8*sum(c['config']['method']=='PR' for c in candidates),
                      compiled_axes_per_trajectory=2,compile_only_if_joint_eligible=True),
           no_automatic_retry=True,terminal_stage='WAIT_FOR_ADAPTIVE_OR_CONFIRMATION_WITHIN_FIXED_STUDY')
 save(batch/'plan.json',plan);return batch
def seeds(c,cohort,index):
 import numpy as np
 x=c['config'];tag={'exploration':101,'confirmation':202,'deterministic':0}[cohort]
 return int(np.random.SeedSequence([20261025,c['molecule'],x['L_D'],x['q'],x['r'],x['K'],tag,index]).generate_state(1,dtype=np.uint64)[0])
def qpy_saved(circuit,path):
 from qiskit import qpy
 a=io.BytesIO();qpy.dump(circuit,a);raw=a.getvalue();compressed=gzip.compress(raw,mtime=0)
 Path(path).write_bytes(compressed)
 return dict(raw_sha256=hashlib.sha256(raw).hexdigest(),gzip_sha256=hashlib.sha256(compressed).hexdigest(),raw_bytes=len(raw),gzip_bytes=len(compressed))
def price_job(batch,cid,cohort,index):
 import numpy as np,qiskit
 from hx_study.pauli_signal_adapter import PauliHamiltonian,PauliSignalAdapter,append_pauli_evolution,append_pauli_event
 from hx_study.rte import sample_rte_events
 from hx_study.h4_d2_pauli_control import central_control_body
 from hx_study.fixed_signal_evaluation import hadamard_wrapper,default_compiler
 from hx_study.rte_compiled_cost import transpile_and_measure_cost
 plan=verify(batch);c=next(x for x in plan['candidates'] if x['candidate_id']==cid);n=c['molecule'];x=c['config']
 d=batch/'cases'/cid/'prices'/cohort/f'{index:04d}';d.mkdir(parents=True,exist_ok=False)
 model=read(OUT/f'H{n}/inputs/model.json');h=model['hamiltonian'];ham=PauliHamiltonian(model['n_qubits'],h['constant'],tuple(map(tuple,h['terms'])),json.dumps(h.get('metadata',{})))
 a=PauliSignalAdapter(ham,x['method'],x['L_D'],.8/x['q'],x['r'],x['K']);seed=seeds(c,cohort,index)
 events=sample_rte_events(a.components,a.distribution,sample_count=x['q']*x['r'],seed=seed) if a.randomized else []
 it=iter(events);body=qiskit.QuantumCircuit(a.num_qubits)
 for _ in range(x['q']):
  for s,v in a.retained:append_pauli_evolution(body,s,v,a.step_time/2)
  if a.randomized:
   for _ in range(x['r']):append_pauli_event(body,next(it),a.labels)
  for s,v in reversed(a.retained):append_pauli_evolution(body,s,v,a.step_time/2)
  body.global_phase-=ham.constant*a.step_time
 lift=central_control_body(body);result=dict(candidate_id=cid,cohort=cohort,index=index,seed=seed,axes={},
    event_count=len(events),orders=[e.taylor_order for e in events],
    events=[dict(order=e.taylor_order,products=e.product_component_ids,rotation=e.rotation_component_id,
                 angle=e.rotation_angle,phase=[e.phase.real,e.phase.imag],signs=[i.coefficient_sign for i in e.application_sequence]) for e in events],
    body_phase=float(body.global_phase),state_preparation_excluded=True,qiskit_version=qiskit.__version__)
 if qiskit.__version__!='1.3.0':raise ValueError('QISKIT_VERSION')
 for axis in AXES:
  wrapper=hadamard_wrapper(lift.body,axis,already_controlled=True)
  ins=qpy_saved(wrapper,d/f'{axis}_input.qpy.gz')
  cost=transpile_and_measure_cost(wrapper,default_compiler(),actual_circuit_fingerprint=ins['raw_sha256'])
  outs=qpy_saved(cost.transpiled_circuit,d/f'{axis}_compiled.qpy.gz')
  result['axes'][axis]={k:getattr(cost,k) for k in ['rz_count','cx_count','total_depth','rz_depth','cx_depth','circuit_size','global_phase']}
  result['axes'][axis].update(input_qpy=ins,compiled_qpy=outs)
 save(d/'result.json',result)
def run_jobs(jobs,record):
 """Threads only wait for isolated native workers; Qiskit stays in children."""
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 def one(job):
  cmd,base=job;base=Path(base);base.parent.mkdir(parents=True,exist_ok=True);start=time.time()
  with base.with_suffix('.stdout.log').open('w') as a,base.with_suffix('.stderr.log').open('w') as b:
   p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=a,stderr=b);rc=p.wait()
  return dict(command=cmd,pid=p.pid,exit_code=rc,reaped=True,elapsed=time.time()-start,log_base=str(base.relative_to(ROOT)))
 workers=min(len(jobs),len(os.sched_getaffinity(0))) if jobs else 0
 with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,workers)) as pool:rows=list(pool.map(one,jobs))
 save(record,dict(workers=workers,results=rows,all_exit_zero=all(r['exit_code']==0 for r in rows)))
 if any(r['exit_code'] for r in rows):raise RuntimeError('NATIVE_FAILURE_PRESERVED '+str(record))
 return rows
def execute(batch):
 plan=verify(batch);n=plan['molecule'];jobs=[]
 for c in plan['candidates']:
  cid=c['candidate_id'];d=batch/'cases'/cid
  jobs.append(([ARB,str(ROOT/'scripts/pauli_h2_h6_numeric.py'),'target',str(OUT/f'H{n}/inputs/model.json'),str(d/'target.json'),str(d/'spec.json'),str(OUT/f'H{n}/inputs/reference.json')],d/'target'))
 run_jobs(jobs,batch/'targets_native.json')
 jobs=[]
 for c in plan['candidates']:
  cid=c['candidate_id'];target=read(batch/'cases'/cid/'target.json')
  if not any(A.joint_status(target,e)=='CONDITIONAL_GUARD_ELIGIBLE' for e in EPS):continue
  cohort='exploration' if c['config']['method']=='PR' else 'deterministic';count=8 if cohort=='exploration' else 1
  for i in range(count):jobs.append(([COMP,str(Path(__file__).resolve()),'price',str(batch),cid,cohort,str(i)],batch/'native'/f'{cid}_{cohort}_{i}'))
 run_jobs(jobs,batch/'prices_native.json');summarize(n)
def registry(n):
 result=[]
 for batch in sorted((OUT/f'H{n}').glob('batch*')):
  for c in read(batch/'plan.json')['candidates']:
   d=batch/'cases'/c['candidate_id'];target=read(d/'target.json') if (d/'target.json').exists() else None
   result.append(dict(c,batch=str(batch.relative_to(ROOT)),target=target,case_path=str(d.relative_to(ROOT))))
 return result
def statistics(c,cohort):
 import numpy as np
 paths=sorted((ROOT/c['case_path']/'prices'/cohort).glob('*/result.json'))
 rows=[read(p) for p in paths]
 if not rows:return None
 values={a:[r['axes'][a]['rz_count'] for r in rows] for a in AXES}
 mean={a:Q(sum(v),len(v)) for a,v in values.items()}
 return dict(n=len(rows),values=values,mean_exact={a:str(v) for a,v in mean.items()},
             SE={a:float(np.std(v,ddof=1)/math.sqrt(len(v))) if len(v)>1 else None for a,v in values.items()},
             orders={str(k):sum(r['orders'].count(k) for r in rows) for k in [0,2,4]},paths=[str(p.relative_to(ROOT)) for p in paths])
def accounting(c,cohort):
 import numpy as np
 st=statistics(c,cohort);prices={a:st['mean_exact'][a] if st else None for a in AXES};result=[]
 for e in EPS:
  t=c['target'];row=S.account(t,prices,e,c['law']) if c['config']['method']=='PR' else A.account(t,prices,e)
  row.update(candidate_id=c['candidate_id'],config=c['config'],molecule=c['molecule'],cohort=cohort,
    eligibility=A.joint_status(t,e),price_statistics=st,target=t,case_path=c['case_path'])
  if st and row['N_cos'] is not None and row['N_sin'] is not None and st['n']>1:
   v=np.array(st['values']['cosine'],dtype=float)*row['N_cos']+np.array(st['values']['sine'],dtype=float)*row['N_sin']
   row['G_SE']=float(np.std(v,ddof=1)/math.sqrt(len(v)))
  else:row['G_SE']=None
  result.append(row)
 return result
def summarize(n):
 rows=[]
 for c in registry(n):
  if c['target'] is None:continue
  rows.extend(accounting(c,'exploration' if c['config']['method']=='PR' else 'deterministic'))
 save(OUT/f'H{n}/candidate_results.json',rows)
 best=[]
 for e in EPS:
  r=[r for r in rows if r['epsilon_sig']==e];b=A.finite_minimum([x for x in r if x['config']['method']=='B0']);f=A.finite_minimum([x for x in r if x['config']['method']=='PF']);p=A.finite_minimum([x for x in r if x['config']['method']=='PR']);d=A.finite_minimum([x for x in r if x['config']['method']!='PR'])
  best.append(dict(epsilon_sig=e,B0=b,PF=f,PR_exploration=p,deterministic=d,
                   PR_over_D_exact=None if not p or not d else str(Q(p['G_exact'])/Q(d['G_exact']))))
 save(OUT/f'H{n}/best_exploration.json',best)
 print(json.dumps(dict(molecule=n,candidates=len(registry(n)),best=[{k:(v.get('candidate_ids') if isinstance(v,dict) else v) for k,v in x.items()} for x in best])),flush=True)
 return best
def adapt(n,name,round_no):
 old=registry(n);known={c['candidate_id'] for c in old};rows=read(OUT/f'H{n}/candidate_results.json');m=read(OUT/f'H{n}/inputs/model.json')['nonidentity_terms'];new=[]
 for method,cap in [('B0',48),('PR',32)]:
  pool=[]
  for e in EPS:
   rr=sorted([x for x in rows if x['epsilon_sig']==e and x['G_exact'] is not None and (x['config']['method']=='PR')==(method=='PR')],key=lambda x:(Q(x['G_exact']),x['candidate_id']))[:2]
   for row in rr:
    x=row['config'];L=x['L_D'];q=x['q'];r=x['r'];K=x['K'];qgrid=[1,2,4,8]
    choices=[(j,q,r,K) for j in [L-4,L-1,L+1,L+4]]
    evaluated=sorted(set(c['config']['L_D'] for c in old if (c['config']['method']=='PR')==(method=='PR')))
    if method=='B0':choices +=[(int((j+L)//2),q,r,K) for j in evaluated if j!=L and not any(min(j,L)<k<max(j,L) for k in evaluated)]
    choices +=[(L,qq,r,K) for qq in qgrid if abs(qgrid.index(qq)-qgrid.index(q))==1]
    if method=='PR':choices +=[(L,q,rr,K) for rr in [2,4,8] if rr!=r]+[(L,q,r,6-K)]
    for LL,qq,rr,kk in choices:
     if not 0<=(LL if method=='PR' else LL-1)<(m if method=='PR' else m):continue
     c=spec(n,'PF' if LL==m else method,LL,qq,rr,kk)
     if c['candidate_id'] not in known:pool.append(c);known.add(c['candidate_id'])
  new+=pool[:cap]
 if not new:return None
 return register(n,name,new,f'adaptive_round_{round_no}')
def confirm(n,name='confirmation01'):
 rows=read(OUT/f'H{n}/candidate_results.json');reg={c['candidate_id']:c for c in registry(n)};final=[];nominees={}
 for e in EPS:
  rr=sorted([x for x in rows if x['epsilon_sig']==e and x['config']['method']=='PR' and x['G_exact'] is not None],key=lambda x:(Q(x['G_exact']),x['candidate_id']))[:2]
  nominees[e]=[r['candidate_id'] for r in rr]
  for r in rr:
   if r['candidate_id'] not in final:final.append(r['candidate_id'])
 d=OUT/f'H{n}'/name;d.mkdir(exist_ok=False)
 plan=dict(molecule=n,finalists=final,nominees=nominees,initial_n=48,extension_n=[96,192,384],SOURCE=source(n),
           selection='frozen using exploration only; independent seed tag202; no pooling',prior_results_sha256=sha(OUT/f'H{n}/candidate_results.json'))
 save(d/'plan.json',plan)
 jobs=[]
 for cid in final:
  c=reg[cid];batch=ROOT/c['batch']
  for i in range(48):jobs.append(([COMP,str(Path(__file__).resolve()),'price',str(batch),cid,'confirmation',str(i)],d/'native'/f'{cid}_{i}'))
 run_jobs(jobs,d/'native.json')
 results=[]
 for cid in final:results.extend(accounting(reg[cid],'confirmation'))
 save(d/'results.json',results)
 print('confirmation',n,len(final),'finalists',flush=True)
if __name__=='__main__':
 mode=sys.argv[1]
 if mode=='price':price_job(Path(sys.argv[2]),sys.argv[3],sys.argv[4],int(sys.argv[5]))
 elif mode=='initial':
  n=int(sys.argv[2]);execute(register(n,'batch01',initial(n)))
 elif mode=='adapt':
  n=int(sys.argv[2]);r=int(sys.argv[3]);b=adapt(n,f'batch{r+1:02d}',r)
  if b:execute(b)
 elif mode=='confirm':confirm(int(sys.argv[2]))
 elif mode=='summary':summarize(int(sys.argv[2]))
 else:raise ValueError(mode)
