#!/usr/bin/env python3
"""Fresh H2-only confirmations and bounded pure-RTE extension; reuse H2 science."""
import csv,copy,gzip,hashlib,importlib.util,json,math,os,shutil,subprocess,sys
from dataclasses import asdict
from fractions import Fraction as Q
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/h2_pauli_rte_validation/2026-10-26'
PRIOR=ROOT/'artifacts/pauli_h2_h6_comparison/2026-10-25'
def module(name,path):
    s=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
H=module('h2_original_runner','scripts/pauli_h2_h6_study.py');H.OUT=OUT
D=module('h2_dispatch_reuse','scripts/run_pauli_rte_event_extension.py')
read,save,sha=H.read,H.save,H.sha
A=H.A.A;EPS=H.EPS;AXES=H.AXES;COMP,ARB=H.COMP,H.ARB
dispatch=D.dispatch
def finite(x):return x.get('G_exact') is not None and x['eligibility']=='CONDITIONAL_GUARD_ELIGIBLE'
def minimum(rows):return min((x for x in rows if finite(x)),key=lambda x:(Q(x['G_exact']),x['candidate_id']),default=None)
def near(x,y):return finite(x) and finite(y) and abs(float(Q(x['G_exact'])-Q(y['G_exact'])))<=2*math.hypot(float(x['G_SE']),float(y['G_SE']))
def validate(q,r,K):
    if any(type(x) is not int for x in (q,r,K)) or q not in (1,2,4,8) or r not in (2,4,8,16,32) or K not in (2,4) or (r>8 and q!=8):raise ValueError('H2_SCOPE')
    return f'p000_q{q}_r{r}_K{K}'
law_fn=A.extract('src/trotterlib/h4_pauli_pr_search.py','law_descriptor',dict(__package__='hx_study',asdict=asdict,math=math,candidate_id=lambda p,q,r,K:validate(q,r,K) if p==0 else (_ for _ in ()).throw(ValueError('PURE_RTE_ONLY')),probability_masses=H.S.probability_masses))
def check():
    if (OUT/'STUDY_STOP.json').exists():raise ValueError('MANDATORY_STOP_NO_SCIENCE_RESUME')
    s=read(OUT/'SOURCE.json')
    for f,h in {**s['code_sha256'],**s['input_sha256']}.items():assert sha(ROOT/f)==h,f
    return s
def spec(cid):return next(x for x in read(OUT/'study_plan.json')['candidates'] if x['candidate_id']==cid)
def case(cid):
    c=spec(cid);return OUT/'H2'/('phase_A' if c['phase']=='A' else 'phase_B')/'cases'/cid
def oldrows():return json.loads(gzip.decompress((PRIOR/'H2/candidate_results.json.gz').read_bytes()))
def prior_quotes():return read(OUT/'prior_inventory.json')
def init():
    import numpy,scipy,qiskit
    assert (numpy.__version__,scipy.__version__,qiskit.__version__)==('1.26.4','1.14.1','1.3.0')
    p=OUT/'H2/inputs';p.mkdir(parents=True,exist_ok=False)
    for n in ['model.json','reference.json']:shutil.copyfile(PRIOR/'H2/inputs'/n,p/n)
    plan=read(OUT/'study_plan.json');prior=read(OUT/'prior_evidence_selection.json')
    for rel,meta in prior['selected_members'].items():assert sha(OUT/'prior_evidence'/rel)==meta['sha256']
    paths=list(dict.fromkeys(H.CODE+['scripts/run_h2_pauli_rte_validation.py','scripts/run_pauli_rte_event_extension.py','scripts/run_pauli_hybrid_rte_comparison.py','src/trotterlib/pauli_vector_tight_enclosure.py']))
    ipaths=[str((p/n).relative_to(ROOT)) for n in ['model.json','reference.json']]+[str(x.relative_to(ROOT)) for x in [OUT/'study_plan.json',OUT/'prior_inventory.json',PRIOR/'H2/candidate_results.json.gz']]
    save(OUT/'runtime.json',dict(qiskit=qiskit.__version__,numpy=numpy.__version__,scipy=scipy.__version__,compiler=COMP,numeric=ARB))
    save(OUT/'SOURCE.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),code_sha256={x:sha(ROOT/x) for x in paths},input_sha256={x:sha(ROOT/x) for x in ipaths},model_sha256=sha(p/'model.json'),reference_sha256=sha(p/'reference.json'),study_plan_sha256=sha(OUT/'study_plan.json')))
    save(OUT/'usage.json',dict(new_targets=0,new_laws=0,exploration_trajectories=0,confirmation_trajectories=0))
def reserve(**kw):
    u=read(OUT/'usage.json');lim=read(OUT/'study_plan.json')['limits']
    for k,v in kw.items():u[k]+=v;assert u[k]<=lim[k],k
    assert u['exploration_trajectories']+u['confirmation_trajectories']<=lim['total_trajectories']
    save(OUT/'usage.json',u)
def register(ids):
    import hx_study.pauli_signal_adapter as P
    from hx_study.fixed_signal_evaluation import FixedSignalConfig
    source=check();cs=[spec(x) for x in ids];assert cs and len({c['phase'] for c in cs})==1
    phase=cs[0]['phase'];batch=OUT/'H2'/('phase_A' if phase=='A' else 'phase_B');batch.mkdir(parents=True,exist_ok=True)
    model=read(OUT/'H2/inputs/model.json');h=model['hamiltonian'];ham=P.PauliHamiltonian(model['n_qubits'],h['constant'],tuple(map(tuple,h['terms'])),'{}')
    batch_candidates=[]
    for c in cs:
        cid=c['candidate_id'];d=case(cid);d.mkdir(parents=True,exist_ok=False);q,r,K=c['q'],c['r'],c['K'];validate(q,r,K)
        cfg=dict(method='PR',L_D=0,q=q,r=r,K=K,T=.8)
        if phase=='A':
            old=OUT/'prior_evidence'/c['old_case'];s=read(old/'spec.json');assert s['config']==cfg
            shutil.copyfile(old/'target.json',d/'target.json');target_origin=dict(path=str((old/'target.json').relative_to(ROOT)),sha256=sha(old/'target.json'),new_state_actions=0)
        else:
            reserve(new_laws=1);law=law_fn(ham,FixedSignalConfig('pauli_jw',method='PR',T=.8,L_D=0,q=q,r=r,K=K,cost_samples=8,seed=20261025))
            s=dict(candidate_id=cid,molecule=2,config=cfg,law=law);target_origin=dict(new_state_actions_max=1)
        save(d/'spec.json',s);save(d/'target_origin.json',target_origin);batch_candidates.append(s)
    # Fresh immutable batch plan, compatible with the original H2 price_job.
    name='registration_'+str(len(list(batch.glob('registration_*.json')))+1)
    plan=dict(molecule=2,candidates=batch_candidates,SOURCE=source,phase=phase)
    save(batch/(name+'.json'),plan)
    existing=read(batch/'plan.json')['candidates'] if (batch/'plan.json').exists() else []
    save(batch/'plan.json',dict(molecule=2,candidates=existing+batch_candidates,SOURCE=source,append_only_registration_refs=sorted(x.name for x in batch.glob('registration_*.json'))))
    if phase=='B':
        reserve(new_targets=len(cs));dispatch([([ARB,str(Path(__file__).resolve()),'target',c['candidate_id']],c['candidate_id']) for c in cs],OUT/'stages'/name)
        jobs=[]
        for c in cs:
            if any(H.A.joint_status(read(case(c['candidate_id'])/'target.json'),e)=='CONDITIONAL_GUARD_ELIGIBLE' for e in EPS):jobs +=jobs_for(c['candidate_id'],'exploration',0,8)
        reserve(exploration_trajectories=len(jobs));dispatch(jobs,OUT/'stages'/(name+'_prices'));snapshot()
def jobs_for(cid,cohort,start,end):return [([COMP,str(Path(__file__).resolve()),'price',cid,cohort,str(i)],cid+f'_{i:04d}') for i in range(start,end)]
def target(cid):
    import importlib.metadata
    assert importlib.metadata.version('numpy')=='1.26.4' and importlib.metadata.version('python-flint')=='0.9.0'
    check();c=spec(cid);assert c['phase']=='B';d=case(cid);s=read(d/'spec.json')
    with (d/'target_call_started.json').open('x') as f:json.dump(dict(state_actions_max=1),f)
    from hx_study.pauli_vector_tight_enclosure import evaluate
    t=evaluate(read(OUT/'H2/inputs/model.json'),read(OUT/'H2/inputs/reference.json'),s['config'],s['law']);save(d/'target.json',t)
def price(cid,cohort,index):
    check();c=spec(cid);assert cohort in ('exploration','confirmation')
    assert 0<=index<(8 if cohort=='exploration' else 384)
    batch=case(cid).parents[1];H.price_job(batch,cid,cohort,index)
def stats(cid,cohort):return H.statistics(dict(case_path=str(case(cid).relative_to(ROOT))),cohort)
def rows():
    prior=oldrows();result=[]
    for c in read(OUT/'study_plan.json')['candidates']:
        cid=c['candidate_id'];d=case(cid)
        if not (d/'target.json').exists():continue
        s=read(d/'spec.json');t=read(d/'target.json')
        for cohort in ['exploration','confirmation']:
            st=stats(cid,cohort)
            if not st:continue
            prices=st['mean_exact']
            for e in EPS:
                if c['phase']=='A':
                    original=next(x for x in prior if x['candidate_id']==cid and x['epsilon_sig']==e)
                    x=copy.deepcopy({k:v for k,v in original.items() if k not in ['target','price_statistics']})
                    x['G_exact']=str(sum(Q(x['N_cos' if a=='cosine' else 'N_sin'])*Q(prices[a]) for a in AXES)) if x['N_total'] is not None else None
                    x['G_guard_range_exact']=[x['G_exact'],str(sum(Q(x['axes'][a]['guard_face']['N'])*Q(prices[a]) for a in AXES))] if x['G_exact'] is not None else None
                    x['N_origin']='SAVED_EXACT_REUSE_NO_SHOT_RECALCULATION'
                else:x=H.S.account(t,prices,e,s['law']);x['N_origin']='SAME_ORIGINAL_SHOT_ACCOUNTING'
                x.update(candidate_id=cid,molecule='H2',cohort=cohort,phase=c['phase'],config=s['config'],s=c['q']*c['r'],eligibility=H.A.joint_status(t,e),price_statistics=st,B_exact=t['B_saved_exact'],G_SE=None,target_path=str((d/'target.json').relative_to(ROOT)))
                if x['N_total'] is not None:
                    z=A.paired([{a:st['values'][a][i] for a in AXES} for i in range(st['n'])],dict(zip(AXES,[x['N_cos'],x['N_sin']])),D.stats_fn())
                    assert Q(z['mean_exact'])==Q(x['G_exact']);x.update(G_SE=z['standard_error'],G_nominal95_halfwidth=z['nominal95_halfwidth'])
                result.append(x)
    return result
def snapshot():save(OUT/'candidate_results.json',rows())
def confirmed_comparators(x,data):
    p=[r for r in prior_quotes() if r['epsilon_sig']==x['epsilon_sig'] and r['cohort']=='confirmation'];p +=[r for r in data if r['epsilon_sig']==x['epsilon_sig'] and r['cohort']=='confirmation' and r['candidate_id']!=x['candidate_id']]
    return [r for r in p if finite(r)]
def extend_rule(x,others,n):
    from scipy.stats import t
    half=float(t.ppf(.975,n-1))*x['G_SE']
    return any(near(x,z) and x['G_SE']>float(z['G_SE']) for z in others) and half>.01*float(Q(x['G_exact']))
def confirm(name,ids):
    source=check();stage=OUT/'confirmations'/name;stage.mkdir(parents=True,exist_ok=False)
    assert not any(stats(cid,'confirmation') for cid in ids)
    save(stage/'plan.json',dict(candidate_ids=ids,SOURCE=source,counts=[96,192,384],rule=read(OUT/'study_plan.json')['confirmation_rule'],frozen_before_confirmation=True))
    pending=ids
    for n in [96,192,384]:
        if not pending:break
        jobs=[]
        for cid in pending:
            st=stats(cid,'confirmation');jobs+=jobs_for(cid,'confirmation',st['n'] if st else 0,n)
        reserve(confirmation_trajectories=len(jobs));dispatch(jobs,stage/f'n{n:03d}');snapshot();data=rows();decisions=[];next_pending=[]
        for cid in pending:
            more=False
            for x in [r for r in data if r['candidate_id']==cid and r['cohort']=='confirmation' and finite(r)]:
                others=confirmed_comparators(x,data);extend=extend_rule(x,others,n)
                decisions.append(dict(candidate_id=cid,epsilon_sig=x['epsilon_sig'],n=n,extend=extend,comparator_ids=[z['candidate_id'] for z in others],not_population_or_sequential_coverage=True));more|=extend
            if more and n<384:next_pending.append(cid)
        save(stage/f'n{n:03d}'/'decisions.json',decisions);pending=next_pending
    save(stage/'COHORT_STOP.json',dict(completed=True,no_resume=True,candidates=ids))
def adaptive():
    check();data=rows();plan=read(OUT/'study_plan.json');choices={};observations=[]
    for e in EPS:
        b=[x for x in data if x['phase']=='B' and x['cohort']=='exploration' and x['epsilon_sig']==e];best=minimum(b)
        old=minimum([x for x in prior_quotes() if x['group']=='RTE' and x['cohort']=='confirmation' and x['epsilon_sig']==e]+[x for x in data if x['phase']=='A' and x['cohort']=='confirmation' and x['epsilon_sig']==e])
        edge=next((x for x in b if x['config']['r']==16 and x['config']['K']==2),None)
        if edge and best and finite(edge) and (edge['candidate_id']==best['candidate_id'] or near(edge,best)) and (Q(edge['G_exact'])<Q(old['G_exact']) or near(edge,old)):
            for K in [2,4]:choices[f'PR_p0000_q8_r32_K{K}']='s128 remains competitive minimum/near-min; test s256'
        if edge and old and finite(edge) and near(edge,old):choices['PR_p0000_q8_r16_K4']='s128 near confirmed current minimum'
        if edge and any(x['eligibility']!='CONDITIONAL_GUARD_ELIGIBLE' for x in b if x['config']['r']==16):choices['PR_p0000_q8_r16_K4']='s128 K2 precision/bias ambiguity'
        observations.append(dict(epsilon_sig=e,phase_B_exploration_min=best['candidate_id'] if best else None,current_confirmed_min=old['candidate_id'] if old else None))
    choices={k:v for k,v in choices.items() if not case(k).exists()}
    out=dict(add_candidates=choices,observations=observations);path=OUT/'decisions'/f'adaptive_{len(list((OUT/"decisions").glob("adaptive_*.json"))):02d}.json';save(path,out);print(json.dumps(out,indent=2))
if __name__=='__main__':
    mode,*args=sys.argv[1:]
    if mode=='init':init()
    elif mode=='register':register(args)
    elif mode=='target':target(args[0])
    elif mode=='price':price(args[0],args[1],int(args[2]))
    elif mode=='confirm':confirm(args[0],args[1:])
    elif mode=='snapshot':check();snapshot()
    elif mode=='adaptive':adaptive()
    else:raise ValueError(mode)
