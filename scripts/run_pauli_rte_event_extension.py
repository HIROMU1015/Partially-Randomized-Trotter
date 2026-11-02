#!/usr/bin/env python3
"""Fresh bounded H4/H6 event-extension batches using prepared kernels."""
import concurrent.futures,csv,gzip,hashlib,importlib,io,json,math,os,shutil,subprocess,sys,time,types
from pathlib import Path
from fractions import Fraction as Q
ROOT=Path(__file__).resolve().parents[1]
PREP=ROOT/'artifacts/pauli_rte_event_extension_preparation/2026-10-26'
OUT=ROOT/'artifacts/pauli_rte_event_extension/2026-10-26'
COMP='/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python'
ARB='/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/non-df-h4-c-reference-fix-20261023/.c-reference-fix-venv/bin/python'
pkg=types.ModuleType('event_run');pkg.__path__=[str(ROOT/'src/trotterlib')];sys.modules[pkg.__name__]=pkg
E=importlib.import_module(pkg.__name__+'.pauli_rte_event_extension');A=E.A
B=importlib.import_module(pkg.__name__+'.h4_pauli_b0_comparison')
AXES=A.AXES;EPS=B.EPSILONS
save=A.extract('scripts/pauli_h2_h6_study.py','save',dict(Path=Path,json=json))
idle_capacity=A.extract('scripts/run_pauli_hybrid_rte_comparison.py','idle_capacity',dict(os=os,Path=Path,time=time,math=math))
qpy_saved=A.extract('scripts/pauli_h2_h6_study.py','qpy_saved',dict(io=io,gzip=gzip,Path=Path,hashlib=hashlib))
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvrows(p):
    with Path(p).open() as f:return [{k:None if v in ('','null') else v for k,v in x.items()} for x in csv.DictReader(f)]
def stats_fn():
    import numpy as np
    return A.extract('src/trotterlib/fixed_signal_evaluation.py','_statistics',dict(np=np,math=math,Fraction=Q))
def current_source():return OUT/read(OUT/'CURRENT_SOURCE.json')['path']
def check(source=None):
    if (OUT/'STUDY_STOP.json').exists():raise ValueError('COMPLETED_SCIENTIFIC_STUDY_NO_RESUME')
    p=Path(source) if source else current_source();s=read(p)
    for f,h in {**s['code_sha256'],**s['input_sha256']}.items():
        if sha(ROOT/f)!=h:raise ValueError('SOURCE_OR_INPUT_CHANGED '+f)
    return s
def seal(reason):
    if (OUT/'STUDY_STOP.json').exists():raise ValueError('MANDATORY_STOP')
    prepared=read(PREP/'prospective_SOURCE_inputs.json');roles=read(OUT/'input_roles.json')
    paths=list(prepared['code_sha256'])+['scripts/run_pauli_rte_event_extension.py','scripts/run_pauli_hybrid_rte_comparison.py']
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    source=dict(commit=commit,reason=reason,code_sha256={f:sha(ROOT/f) for f in paths},
        input_sha256={**prepared['input_sha256'],**{f:sha(ROOT/f) for f in roles.values()},
                      **{str((OUT/name).relative_to(ROOT)):sha(OUT/name) for name in ['active_scope.json','input_roles.json','runtime.json']}})
    path=OUT/('SOURCE_'+commit+'.json')
    with path.open('x') as f:json.dump(source,f,indent=2);f.write('\n')
    save(OUT/'CURRENT_SOURCE.json',dict(path=path.name));return path
def initialize():
    import numpy as np,scipy,qiskit
    assert (np.__version__,scipy.__version__,qiskit.__version__)==('1.26.4','1.14.1','1.3.0')
    prepared=read(PREP/'prospective_SOURCE_inputs.json')
    for f,h in {**prepared['code_sha256'],**prepared['input_sha256']}.items():assert sha(ROOT/f)==h,f
    OUT.mkdir(parents=True,exist_ok=False);(OUT/'inputs').mkdir()
    roles=read(PREP/'inputs.json')['roles'];newroles={}
    for role in ['H4_model','H4_reference','H6_model','H6_reference']:
        p=OUT/'inputs'/(role+'.json');shutil.copyfile(ROOT/roles[role],p);newroles[role]=str(p.relative_to(ROOT))
    for role in ['H4_comparison','H6_comparison','prior_inventory','H6_prior_registry']:newroles[role]=roles[role]
    newroles['baseline_quotes']=str((PREP/'saved_baseline_quotes.csv').relative_to(ROOT))
    newroles['H6_RTE_inventory']='artifacts/h6_rte_fairness/2026-10-25/expanded_RTE_inventory.csv'
    save(OUT/'input_roles.json',newroles)
    active=read(PREP/'execution_plan_DRAFT_DISABLED.json');active.update(status='ACTIVE_APPROVED_FIXED_SCOPE',execute_enabled=True,
        preparation_commit='c6d42d925b258939768a8c6a34ee3f48db27b964',
        technical_retry_policy='User permits correcting unintended technical interruptions and limited fresh attempts. Preserve failed evidence/SOURCE; never repeat successful science, alter scientific conditions, or exceed total limits. Old STOPs stay closed.')
    save(OUT/'active_scope.json',active)
    save(OUT/'runtime.json',dict(qiskit=qiskit.__version__,numpy=np.__version__,scipy=scipy.__version__,compiler=COMP,numeric=ARB,affinity=len(os.sched_getaffinity(0))))
    save(OUT/'usage.json',dict(law=0,target_reserved=0,exploration_trajectories=0,confirmation_trajectories=0))
    seal('Initial approved event-extension implementation; no old run resumed')
def spec(cid):return next(x for x in read(OUT/'active_scope.json')['candidates'] if x['candidate_id']==cid)
def case(cid):return OUT/'cases'/cid
def model_ref(molecule):
    r=read(OUT/'input_roles.json');return read(ROOT/r[molecule+'_model']),read(ROOT/r[molecule+'_reference'])
def ham(model):
    from event_run.pauli_signal_adapter import PauliHamiltonian
    h=model['hamiltonian'];return PauliHamiltonian(model['n_qubits'],h['constant'],tuple(map(tuple,h['terms'])),'{}')
def reserve(**increments):
    u=read(OUT/'usage.json');limits=read(OUT/'active_scope.json')['limits']
    for k,v in increments.items():u[k]+=v
    assert u['law']<=limits['law_descriptor_evaluations_max'] and u['target_reserved']<=limits['total_target_state_actions_max']
    assert u['exploration_trajectories']<=limits['exploration_trajectories_max'] and u['confirmation_trajectories']<=limits['confirmation_trajectories_max']
    assert 2*(u['exploration_trajectories']+u['confirmation_trajectories'])<=limits['total_new_axis_compiles_max']
    save(OUT/'usage.json',u)
def dispatch(jobs,folder):
    folder.mkdir(parents=True,exist_ok=False);capacity=idle_capacity();workers=min(len(jobs),capacity['workers_from_actual_free_CPU'])
    save(folder/'resources.json',capacity)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def one(job):
        cmd,name=job;base=folder/'native'/name;base.parent.mkdir(parents=True,exist_ok=True);start=time.time()
        with base.with_suffix('.stdout.log').open('w') as a,base.with_suffix('.stderr.log').open('w') as b:
            p=subprocess.Popen(['nice','-n','10']+cmd,cwd=ROOT,env=env,stdout=a,stderr=b,start_new_session=True);rc=p.wait()
        return dict(command=cmd,pid=p.pid,exit_code=rc,reaped=True,elapsed_seconds=time.time()-start,log_base=str(base.relative_to(ROOT)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,workers)) as pool:native=list(pool.map(one,jobs))
    save(folder/'native.json',dict(workers=workers,results=native,all_exit_zero=all(x['exit_code']==0 for x in native)))
    if any(x['exit_code'] for x in native):
        save(folder/'TECHNICAL_INTERRUPTION.json',dict(failures_preserved=True,all_workers_reaped=True,automatic_retry=False))
        raise RuntimeError('NATIVE_FAILURE_PRESERVED_REPAIR_BEFORE_NEW_ATTEMPT')
def target_worker(cid,source):
    import importlib.metadata
    assert importlib.metadata.version('numpy')=='1.26.4' and importlib.metadata.version('python-flint')=='0.9.0'
    check(source);c=spec(cid);d=case(cid);model,reference=model_ref(c['molecule']);law=read(d/'law.json')
    marker=d/'target_call_starts.json'
    with marker.open('x') as f:json.dump([],f)
    bits_used=[]
    for bits in ([64,96,128] if c['molecule']=='H4' else [128]):
        bits_used.append(bits);save(marker,bits_used)
        t=E.h4_mean(law,model['state'],reference,bits) if c['molecule']=='H4' else E.h6_mean(model,reference,law)
        save(d/f'target_{bits}.json',t)
        if c['molecule']=='H6' or not any(x['status']=='WIDTH_AMBIGUITY' for x in B.E.precision_decisions(t)):break
    t.update(state_actions_this_candidate=len(bits_used),attempted_bits=bits_used,SOURCE=str(Path(source).relative_to(ROOT)))
    save(d/'target.json',t)
def price_worker(cid,cohort,index,source):
    import qiskit
    from event_run.fixed_signal_evaluation import default_compiler
    from event_run.rte_compiled_cost import transpile_and_measure_cost
    check(source);assert qiskit.__version__=='1.3.0';c=spec(cid);model,_=model_ref(c['molecule'])
    d=case(cid)/'prices'/cohort/f'{index:04d}';d.mkdir(parents=True,exist_ok=False)
    body,wrapped,meta=E.trajectory(ham(model),c['molecule'],c['r'],c['K'],cohort,index)
    out=dict(candidate_id=cid,molecule=c['molecule'],cohort=cohort,index=index,seed=meta['seed'],
        events=[e.to_dict() for e in meta['events']],orders=[e.taylor_order for e in meta['events']],event_count=meta['event_count'],
        body_phase=float(body.global_phase),step_seeds=meta['step_seeds'],control_sidecar=meta['control_sidecar'],
        qiskit_version=qiskit.__version__,state_preparation_excluded=True,SOURCE=str(Path(source).relative_to(ROOT)),axes={})
    for axis in AXES:
        inp=qpy_saved(wrapped[axis],d/(axis+'_input.qpy.gz'))
        cost=transpile_and_measure_cost(wrapped[axis],default_compiler(),actual_circuit_fingerprint=inp['raw_sha256'])
        compiled=qpy_saved(cost.transpiled_circuit,d/(axis+'_compiled.qpy.gz'))
        out['axes'][axis]={k:getattr(cost,k) for k in ['rz_count','cx_count','total_depth','rz_depth','cx_depth','circuit_size','global_phase']}
        out['axes'][axis].update(input_qpy=inp,compiled_qpy=compiled)
    save(d/'result.json',out)
def registry():return [read(p) for p in sorted((OUT/'cases').glob('*/spec.json'))]
def price_stats(cid,cohort):
    import numpy as np
    raw=[read(p) for p in sorted((case(cid)/'prices'/cohort).glob('*/result.json'))]
    if not raw:return None
    values={a:[x['axes'][a]['rz_count'] for x in raw] for a in AXES}
    return dict(n=len(raw),values=values,mean_exact={a:str(Q(sum(v),len(v))) for a,v in values.items()},
        SE={a:float(np.std(v,ddof=1)/math.sqrt(len(v))) if len(v)>1 else None for a,v in values.items()})
def rows():
    out=[]
    for c in registry():
        d=case(c['candidate_id'])
        if not (d/'target.json').exists():continue
        t=read(d/'target.json');law=read(d/'law.json')
        for cohort in ['exploration','confirmation']:
            st=price_stats(c['candidate_id'],cohort)
            if st is None and cohort=='confirmation':continue
            prices={a:st['mean_exact'][a] if st else None for a in AXES}
            for e in EPS:
                x=E.account_target(t,prices,e,law);x.update(candidate_id=c['candidate_id'],molecule=c['molecule'],r=c['r'],q=8,K=c['K'],s=c['s'],L_D=0,
                    cohort=cohort,eligibility=B.joint_status(t,e),B_exact=str(Q(law['B_saved_binary64'])),price_statistics=st,G_SE=None,target_path=str((d/'target.json').relative_to(ROOT)))
                if st and x['N_total'] is not None:
                    z=A.paired([{a:st['values'][a][i] for a in AXES} for i in range(st['n'])],dict(zip(AXES,[x['N_cos'],x['N_sin']])),stats_fn())
                    assert Q(z['mean_exact'])==Q(x['G_exact']);x.update(G_SE=z['standard_error'],G_nominal95_halfwidth=z['nominal95_halfwidth'])
                out.append(x)
    return out
def snapshot():save(OUT/'candidate_results.json',rows())
def explore(name,ids,reason):
    check();source=current_source();stage=OUT/'stages'/name;stage.mkdir(parents=True,exist_ok=False)
    cs=[spec(cid) for cid in ids];assert len(set(ids))==len(ids)
    assert not any(case(cid).exists() for cid in ids)
    save(stage/'plan.json',dict(candidates=cs,reason=reason,SOURCE=str(source.relative_to(ROOT)),law_max=len(cs),target_max=sum(3 if c['molecule']=='H4' else 1 for c in cs),exploration_max=8*len(cs)))
    reserve(law=len(cs),target_reserved=sum(3 if c['molecule']=='H4' else 1 for c in cs))
    for c in cs:
        d=case(c['candidate_id']);d.mkdir(parents=True,exist_ok=False);save(d/'spec.json',c)
        model,_=model_ref(c['molecule']);save(d/'law.json',E.law_descriptor(ham(model),c['r'],c['K']))
    dispatch([([ARB,str(Path(__file__).resolve()),'target',c['candidate_id'],str(source)],c['candidate_id']) for c in cs],stage/'targets')
    jobs=[]
    for c in cs:
        t=read(case(c['candidate_id'])/'target.json')
        if any(B.joint_status(t,e)=='CONDITIONAL_GUARD_ELIGIBLE' for e in EPS):
            jobs.extend(([COMP,str(Path(__file__).resolve()),'price',c['candidate_id'],'exploration',str(i),str(source)],c['candidate_id']+f'_{i:04d}') for i in range(8))
    reserve(exploration_trajectories=len(jobs));dispatch(jobs,stage/'prices');snapshot()
    save(stage/'STAGE_STOP.json',dict(completed=True,no_resume=True,candidates=ids,trajectories=len(jobs)))
    print(json.dumps(dict(stage=name,candidates=ids,trajectories=len(jobs))),flush=True)

def finite(x):return x.get('G_exact') is not None and x.get('eligibility','CONDITIONAL_GUARD_ELIGIBLE')=='CONDITIONAL_GUARD_ELIGIBLE'
def nearest(a,b):
    return finite(a) and finite(b) and abs(float(Q(a['G_exact'])-Q(b['G_exact'])))<=2*math.hypot(float(a['G_SE']),float(b['G_SE']))
def minimum(rs):
    good=[x for x in rs if finite(x)];return min(good,key=lambda x:(Q(x['G_exact']),x['candidate_id'])) if good else None
def proposal(data,baselines,universe):
    selected={x['candidate_id'] for x in data};choices={};observations=[]
    for m in ['H6','H4']:
        for e in EPS:
            rr=[x for x in data if x['molecule']==m and x['epsilon_sig']==e and x['cohort']=='exploration']
            best=minimum(rr)
            h=next(x for x in baselines if x['molecule']==m and x['epsilon_sig']==e and x['group']=='Hybrid')
            old=next(x for x in baselines if x['molecule']==m and x['epsilon_sig']==e and x['group']=='RTE')
            if not best:continue
            if m=='H6':
                k2=[x for x in rr if x['K']==2];b2=minimum(k2)
                edge=next((x for x in k2 if x['r']==64),None)
                if edge and b2 and (edge['candidate_id']==b2['candidate_id'] or nearest(edge,b2)):
                    choices['H6_p000_q8_r128_K2']='H6_S1024: s512 minimum or within2SE'
                for x in k2:
                    promising=b2 and (x['candidate_id']==b2['candidate_id'] or nearest(x,b2))
                    material=any((abs(Q(x['axes'][a]['signed_bias_exact']))+Q(x['axes'][a]['u_enc_exact']))>=Q(x['axes'][a]['epsilon_axis_exact'])/20 for a in AXES)
                    if promising and (x['eligibility'] in ('BIAS_INELIGIBLE','NUMERICALLY_UNDETERMINED') or material or nearest(x,h)):
                        choices[f'H6_p000_q8_r{x["r"]}_K4']='H6_K4: promising K2 plus bias/ambiguity/Hybrid proximity'
            else:
                new=minimum([x for x in rr if x['r']==16])
                if new and finite(old):
                    improved=Q(new['G_exact'])<Q(old['G_exact']);changed=(Q(new['G_exact'])<Q(h['G_exact']))!=(Q(old['G_exact'])<Q(h['G_exact']))
                    if improved or nearest(new,old) or changed or nearest(new,h):
                        for K in (2,4):choices[f'H4_p000_q8_r32_K{K}']='H4_S256: improvement/proximity/rank effect at s128'
            observations.append(dict(molecule=m,epsilon_sig=e,exploration_min=best['candidate_id'],G_exact=best['G_exact']))
    allowed={x['candidate_id'] for x in universe}
    return dict(candidates=[dict(candidate_id=k,reason=v) for k,v in sorted(choices.items()) if k not in selected and k in allowed],observations=observations)
def decide():
    check();data=rows();base=csvrows(PREP/'saved_baseline_quotes.csv')
    value=proposal(data,base,read(OUT/'active_scope.json')['candidates'])
    path=OUT/'decisions'/f'round_{len(list((OUT/"decisions").glob("round_*.json")))+1:02d}.json';save(path,value);print(json.dumps(value,indent=2))

def confirm(name,ids):
    check();source=current_source();stage=OUT/'confirmations'/name;stage.mkdir(parents=True,exist_ok=False)
    assert len(set(ids))==len(ids) and len(ids)<=12
    assert not any(price_stats(cid,'confirmation') is not None for cid in ids)
    save(stage/'plan.json',dict(frozen_candidate_ids=ids,SOURCE=str(source.relative_to(ROOT)),counts=[48,96,192,384],rule=read(OUT/'active_scope.json')['pricing']['confirmation_extension']))
    pending=ids;baselines=csvrows(PREP/'saved_baseline_quotes.csv')
    for n in [48,96,192,384]:
        if not pending:break
        jobs=[]
        for cid in pending:
            st=price_stats(cid,'confirmation');start=st['n'] if st else 0
            jobs.extend(([COMP,str(Path(__file__).resolve()),'price',cid,'confirmation',str(i),str(source)],cid+f'_{i:04d}') for i in range(start,n))
        reserve(confirmation_trajectories=len(jobs));dispatch(jobs,stage/f'n{n:03d}');snapshot();data=rows();decisions=[];next_pending=[]
        from scipy.stats import t
        for cid in pending:
            more=False
            for x in [r for r in data if r['candidate_id']==cid and r['cohort']=='confirmation' and finite(r)]:
                others=[z for z in baselines if z['molecule']==x['molecule'] and z['epsilon_sig']==x['epsilon_sig'] and z['group']=='Hybrid']
                rr=[z for z in data if z['molecule']==x['molecule'] and z['epsilon_sig']==x['epsilon_sig'] and z['candidate_id']!=cid and finite(z)]
                lower=max((z['s'] for z in rr if z['s']<x['s']),default=None);upper=min((z['s'] for z in rr if z['s']>x['s']),default=None)
                for s in [lower,upper]:
                    if s is not None:
                        candidates=[z for z in rr if z['s']==s];confirmed=[z for z in candidates if z['cohort']=='confirmation'];others.append(minimum(confirmed or candidates))
                half=float(t.ppf(.975,n-1))*x['G_SE']
                extend=any(nearest(x,z) and x['G_SE']>float(z['G_SE']) for z in others) and half>.01*float(Q(x['G_exact']))
                decisions.append(dict(candidate_id=cid,epsilon_sig=x['epsilon_sig'],n=n,extend=extend,nominal_t95_halfwidth=half,comparators=[z['candidate_id'] for z in others],not_population_or_sequential_coverage=True));more|=extend
            if more and n<384:next_pending.append(cid)
        save(stage/f'n{n:03d}'/'decisions.json',decisions);pending=next_pending
    save(stage/'COHORT_STOP.json',dict(completed=True,no_resume=True,candidates=ids));print(json.dumps(dict(confirmed=ids)),flush=True)

if __name__=='__main__':
    mode,*args=sys.argv[1:]
    if mode=='init':initialize()
    elif mode=='explore':explore(args[0],args[1:],'Frozen candidate batch within approved event-extension scope')
    elif mode=='target':target_worker(args[0],args[1])
    elif mode=='price':price_worker(args[0],args[1],int(args[2]),args[3])
    elif mode=='decide':decide()
    elif mode=='confirm':confirm(args[0],args[1:])
    elif mode=='snapshot':check();snapshot()
    elif mode=='seal-repair':seal(' '.join(args))
    else:raise ValueError(mode)
