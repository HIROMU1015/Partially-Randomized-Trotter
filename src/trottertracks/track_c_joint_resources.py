"""Fixed PF/step policies, new alpha allocation, paired independent prices.

RZ/CX/size are additive for the actual concatenated native implementation;
serial depth is an upper bound. There is no whole-m17 fit or model transfer.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from fractions import Fraction
import hashlib
import json
import math
import os
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator,Statevector
from .track_c_joint_isometry import OUT,PRIOR,PUBLIC,source,configuration,catalogue,build_joint,build_singletons,runs,append_run,decode,circuit
from .track_c_bottleneck import allocate,account,identity
from .track_c_bottleneck_select import new_point
from .track_c_h4 import ROOT,common,save_json
from .track_c_d_resources import r_part,r_key,metrics,assemble,modular_parts
from .track_c_compiler_a import compile_reference
from .track_c_h4_price import wrapper
from trotterlib.rpe_four_round_phase_validation import reconstruct_rpe_phase


def records(directory):
    rs={}
    for path in sorted(directory.glob('*.json')):
        r=json.loads(path.read_text());rs.setdefault(r_key(r['t'],r['n']),{}).setdefault(str(r['rare']),[]).append(r)
    for strata in rs.values():
        for rows in strata.values():rows.sort(key=lambda r:r['j']);assert len(rows)==16
    return rs


def points(policy,rs,arm='selected',new_certificate=True):
    certificate=json.loads(((OUT/'CERTIFICATE.json') if new_certificate else (PRIOR/'REPLACEMENT_CERTIFICATE.json')).read_text())
    rr={k:{rare:[{**r,'selected':r[arm]} for r in values] for rare,values in strata.items()} for k,strata in rs.items()}
    result=[]
    for s in policy['selected']:
        q=s['qualification'];base={**q,'rho_lower':q.get('rho_original',q['rho_lower']),'beta_stat_lower':q.get('beta_original',q['beta_stat_lower'])}
        result.append(new_point(s['signal'],base,rr,certificate))
    return result


def freeze():
    if (OUT/'POLICY.json').exists():raise RuntimeError('Do not overwrite frozen policy')
    assert not any(r['improves'] for r in json.loads((OUT/'LAYOUTS_RESUMED.json').read_text())['full']), 'Price any improved Full representation before comparison'
    rs=records(OUT/'training_final');old=json.loads((PUBLIC/'OPTIMIZATION.json').read_text());new=[]
    for p in old['optimized']:
        if p['family']!='PR':continue
        primary=('RZ','CX').index(p['primary']);pp=points(p,rs);alpha,ns=allocate(pp,primary)
        new.append(dict(primary=p['primary'],selected=[dict(signal=x['signal'],qualification=x['qualification']) for x in pp],
                        alpha=alpha.tolist(),**account(pp,ns)))
    save_json(OUT/'POLICY.json',dict(source_commit=source(),catalogue_source=json.loads((OUT/'CATALOGUE_FINAL.json').read_text())['source_commit'],
        policies=new,old_policies=old['optimized'],rule='D case, BCH coefficients, h, resolution and logical signals frozen to previous PR primary policies. Reallocate alpha using training prices and new certified rho/beta. Holdout never retunes. Full retains same alpha optimization; all four QR layouts offered separately.'))
    print('POLICY_FROZEN',flush=True)


def price(task):
    t,n,rare,j=task;target=OUT/'holdout_prices'/f'{r_key(t,n)}_{int(rare)}_{j}.json'
    if target.exists():return json.loads(target.read_text())
    seed=int.from_bytes(hashlib.sha256(f'TrackC_joint_isometry_HOLDOUT:{t}:{n}:{rare}:{j}'.encode()).digest()[:8],'big')
    full,p,choices=r_part(t,n,rare,seed);singleton,_=build_singletons(t,n,choices);joint,used=build_joint(t,n,choices)
    fm,sm,jm=[metrics(compile_reference(q)) for q in (full,singleton,joint)]
    baseline=sm if sm[0]<fm[0] else fm;selected=jm if jm[0]<baseline[0] else baseline
    r=dict(t=t,n=n,rare=rare,j=j,seed=seed,probability=p if rare else 1-p,choices=choices,
           full=fm,singleton=sm,baseline=baseline,joint_proposal=jm,selected=selected,choice='joint' if jm[0]<baseline[0] else 'baseline',used_runs=used)
    save_json(target,r);return r


def weights(policy):
    w={}
    for s,r in zip(policy['selected'],policy['rounds']):
        for recipe in s['signal']['recipe']:
            k=r_key(recipe['time'],recipe['r']);w[k]=w.get(k,0)+sum(r['shots'])*s['signal']['steps']
    return w


def covariance(rs,weighted):
    result=np.zeros((4,4))
    for key,strata in rs.items():
        for rows in strata.values():
            v=sum((weight.get(key,0)*np.array([r[arm] for r in rows],float) for arm,weight in weighted),np.zeros((16,4)))*rows[0]['probability']
            result+=np.cov(v,rowvar=False,ddof=1)/16
    return result


def result(policy,old,rs):
    pp=points(policy,rs);ns=np.array([r['shots'] for r in policy['rounds']]);new=account(pp,ns)
    baseline=account(points(old,rs,'baseline',False),np.array([r['shots'] for r in old['rounds']]))
    wn,wo=weights(policy),weights(old);cov=covariance(rs,[('selected',wn)]);paired=covariance(rs,[('selected',wn),('baseline',{k:-v for k,v in wo.items()})])
    fixed=covariance(rs,[('selected',wo),('baseline',{k:-v for k,v in wo.items()})])
    phases=[math.atan2(s['signal']['raw']['imag'],s['signal']['raw']['real']) for s in policy['selected']]
    decoder=reconstruct_rpe_phase(phases,[1<<m for m in range(18)]);energy=-decoder['final_phase_estimate']/.02
    error=abs(energy-common()['energy']);branch=all(abs(r['base_phase_estimate']+.02*common()['energy'])<=.4/(1<<m)+1e-10 for m,r in enumerate(decoder['rounds']))
    assert branch and error<1.5936e-4 and all(p['qualification']['closed'] for p in pp)
    alpha=sum(Fraction.from_float(a) for row in policy['alpha'] for a in row);assert alpha<=Fraction(1,20)
    fixed_saving=sum((wo.get(k,0)*rows[0]['probability']*np.mean([np.array(r['baseline'])-r['selected'] for r in rows],axis=0)
                      for k,strata in rs.items() for rows in strata.values()),np.zeros(4))
    return dict(primary=policy['primary'],new={**new,'MC_SE':np.sqrt(np.diag(cov)).tolist(),'covariance_MC':cov.tolist()},
                baseline=baseline,paired_total_saving=(np.array(baseline['totals']['total'])-new['totals']['total']).tolist(),
                paired_total_saving_MC_SE=np.sqrt(np.diag(paired)).tolist(),fixed_old_shot_basis_saving=fixed_saving.tolist(),
                fixed_shot_difference_MC_SE=np.sqrt(np.diag(fixed)).tolist(),alpha_sum_exact=str(alpha),
                decoder=decoder,mean_branch_pass=branch,mean_energy_error=error,energy_statistical_bound=.4/(.02*(1<<17)),
                scope='Canonical mean decoded without E0. Uniform source/native replacement disk encloses new mean; supplemented rho/beta and exact alpha give conditional fresh-bit success guarantee.')


def initial_state():
    c=common();v=np.zeros(512,complex);v[[int(f'{int(i):08b}'[::-1],2) for i in c['sector'].basis_indices]]=c['state'];return v


def representative(task):
    path,start,stop=task;target=OUT/'rep_checks'/(hashlib.sha256(f'{path.name}:{start}:{stop}'.encode()).hexdigest()+'.json')
    if target.exists():return json.loads(target.read_text())
    r=json.loads(path.read_text());_,flat,groups=runs(r['t'],r['n'],r['choices'])
    _,_,basis,union=next(g for g in groups if g[:2]==(start,stop));entry=catalogue()[basis+'|'+','.join(map(str,union))]
    old=QuantumCircuit(9);new=QuantumCircuit(9);oldops=common()['preps'][3].rte_preparation.basis_registry.definition(basis).runtime_operations
    p=append_run(old,flat[start:stop],oldops);pnew=append_run(new,flat[start:stop],decode(entry['operations']),dict(zip(entry['support'],entry['targets'])))
    assert p==pnew
    # Include a nontrivial ancilla-relative minus sign as an explicit stress.
    old.p(p+math.pi,8);new.p(p+math.pi,8);native=compile_reference(new)
    a,b,c=[Operator(q).data for q in (old,new,native)]
    phase=np.vdot(b,c);phase/=abs(phase);aligned=c/phase
    errors=[float(np.linalg.norm(a-b,'fro')),float(np.linalg.norm(b-aligned,'fro'))]
    if max(errors)>=1e-9:
        save_json(OUT/'REP_FAILURE.json',dict(file=path.name,start=start,stop=stop,errors=errors));raise AssertionError(errors)
    inactive=float(np.linalg.norm(aligned[:256,:256]-np.eye(256),'fro'));assert inactive<1e-9
    means=[]
    for axis in ('X','Y'):
        values=[]
        for q in (old,new,native):
            state=Statevector(initial_state()).evolve(wrapper([q],axis).remove_final_measurements(inplace=False)).data
            values.append(float(np.vdot(state[:256],state[:256]).real-np.vdot(state[256:],state[256:]).real))
        assert max(abs(x-values[0]) for x in values)<1e-9;means.append(dict(axis=axis,values=values))
    value=dict(file=path.name,start=start,stop=stop,key=entry['key'],support=entry['support'],targets=entry['targets'],
                joint_operator_Fro_errors=errors,inactive_identity_Fro=inactive,means=means,
                native_absolute_Fro_difference=float(np.linalg.norm(b-c,'fro')),joint_common_phase=float(np.angle(phase)),
                phase_scope='One common phase for entire already-controlled 512x512 joint operator, as specified by compiler A. Both branches share this same alignment.')
    save_json(target,value);return value


def short_check(task):
    x,steps=task;case=x['case'];h=x['h'];rate=x['rate'];oldtails=[];tails=[]
    for cycle in range(steps):
        full,_,choices=r_part(h,max(1,math.ceil(h*rate/.02)),cycle==0,981171+cycle)
        single,_=build_singletons(h,max(1,math.ceil(h*rate/.02)),choices);joint,_=build_joint(h,max(1,math.ceil(h*rate/.02)),choices)
        baseline=min((full,single),key=lambda q:metrics(compile_reference(q))[0]);selected=min((baseline,joint),key=lambda q:metrics(compile_reference(q))[0])
        oldtails.append(baseline);tails.append(selected)
    old=assemble(case,h,steps,oldtails);new=assemble(case,h,steps,tails);parts=modular_parts(case,h,steps,tails)
    p=common()['preps'][3];scalar=QuantumCircuit(9);scalar.p(-steps*h*(p.constant_coefficient+p.extracted_identity_coefficient),8)
    pre=QuantumCircuit(9);pre.h(8);pre=compile_reference(pre);rows=[]
    for axis in ('X','Y'):
        read=QuantumCircuit(9,1);read.compose(scalar,inplace=True)
        if axis=='Y':read.sdg(8)
        read.h(8);read.measure(8,0);read=compile_reference(read)
        modular=QuantumCircuit(9,1);modular.compose(pre,inplace=True)
        for part in parts:modular.compose(part,inplace=True)
        modular.compose(read,inplace=True);whole=compile_reference(wrapper([new],axis));state_error=None
        assert all(metrics(modular)[i]==metrics(whole)[i] for i in (0,1,3))
        if steps==1:
            vv=[Statevector(initial_state()).evolve(q.remove_final_measurements(inplace=False)).data for q in (wrapper([old],axis),whole,modular)]
            readout=lambda a:float(np.vdot(a[:256],a[:256]).real-np.vdot(a[256:],a[256:]).real)
            state_error=max(abs(readout(v)-readout(vv[0])) for v in vv[1:]);assert state_error<1e-8
        rows.append(dict(axis=axis,whole_DAG=metrics(whole),modular_DAG=metrics(modular),
                         serial_depth_upper=sum(q.depth() for q in [pre,*parts,read]),readout_error=state_error))
    value=dict(name=x['name'],h=h,rate=rate,steps=steps,rows=rows);save_json(OUT/'short_checks'/f'{x["name"]}_{steps}.json',value);return value


def representatives():
    examples={}
    for path in sorted((PRIOR/'holdout_prices').glob('*.json')):
        r=json.loads(path.read_text());_,flat,groups=runs(r['t'],r['n'],r['choices'])
        for start,stop,basis,union in groups:
            k=basis+'|'+','.join(map(str,union));entry=catalogue().get(k)
            if entry is None or not entry['adopt'] or stop-start<2 or stop-start>8:continue
            sig=(len(union),any(a.role=='product' for _,a in flat[start:stop]),entry['reflected'])
            examples.setdefault(sig,(path,start,stop))
    return [examples[k] for k in sorted(examples)]


def run():
    if (OUT/'RESOURCES.json').exists():raise RuntimeError('Do not overwrite resources')
    configuration();catalogue()
    if not (OUT/'POLICY.json').exists():freeze()
    for d in ('holdout_prices','short_checks','rep_checks'):(OUT/d).mkdir(exist_ok=True)
    frozen=json.loads((OUT/'POLICY.json').read_text());cells=sorted({(r['time'],r['r']) for p in frozen['policies'] for s in p['selected'] for r in s['signal']['recipe']})
    reps=representatives();checks=[];rep_results=[]
    signals=[frozen['policies'][0]['selected'][0]['signal'],frozen['policies'][0]['selected'][-1]['signal']]
    with ProcessPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
        for f in as_completed(pool.submit(price,(*cell,rare,j)) for cell in cells for rare in (False,True) for j in range(16)):f.result()
        for f in as_completed(pool.submit(representative,t) for t in reps):rep_results.append(f.result())
        save_json(OUT/'REPRESENTATIVES.json',dict(source_commit=source(),rows=rep_results))
        for f in as_completed(pool.submit(short_check,(x,n)) for x in signals for n in (1,2)):checks.append(f.result())
    rs=records(OUT/'holdout_prices');results=[]
    for p in frozen['policies']:
        old=next(x for x in frozen['old_policies'] if x['family']=='PR' and x['primary']==p['primary']);results.append(result(p,old,rs))
    save_json(OUT/'RESOURCES.json',dict(source_commit=source(),policies=results,Full=[p for p in frozen['old_policies'] if p['family']=='Full'],
        new_holdout_samples=sum(len(v) for strata in rs.values() for v in strata.values()),cells=cells,representative_checks=len(reps),short_checks=checks,
        statistics='Independent cell/stratum/index seeds; shared cell reused across rounds. Sum cell covariance after all cache weights; paired covariance includes differing old/new shot counts. Selection frozen before holdout.',
        price_model='Concrete modular native implementation, exact additive RZ/CX/size; serial depth upper bound. Direct occurrence compile, no whole-m17 fit or unverified extrapolation.'))
    print('JOINT_RESOURCES_COMPLETE',flush=True)


if __name__=='__main__':run()
