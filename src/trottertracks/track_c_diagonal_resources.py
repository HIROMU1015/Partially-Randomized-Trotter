"""Fixed scientific policies, fair Full synthesis, independent price holdout."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
import os
import numpy as np
import mpmath as mp
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator,Statevector
from .track_c_diagonal_fusion import OUT,PRIOR,MODES,build,d_fused,best,configuration,catalogue,source
from .track_c_h4 import ROOT,common,save_json
from .track_c_bottleneck import saved,allocate,account
from .track_c_joint_resources import records,weights,covariance,initial_state
from .track_c_joint_isometry import build_joint,build_singletons
from .track_c_d_resources import split,d_piece,r_part,r_key,metrics,assemble
from .track_c_compiler_a import compile_reference
from .track_c_h4_price import wrapper
from . import track_c_source_guarantee as sg
from trotterlib.df_trotter.ops import DiagonalEvolutionPrimitives


@lru_cache(None)
def dets():
    return {(r['case']['name'],r['h']):r for p in (OUT/'det_prices').glob('*.json') for r in [json.loads(p.read_text())]}


def qualify(q,x):
    mp.iv.prec=192
    # Each event has one rotation with |theta|<pi<4. Fraction sums are exact;
    # each grouped coefficient is rounded once. Sum of absolute polynomial
    # coefficients <=sum |theta|. ||Delta U|| <= 1/2 sum |Delta theta|.
    # The tiny second term also covers subnormal rounding of <=2N coefficients.
    n=x['RTE_events'];op=2*sg.I(2)**(-53)*n+sg.I(2)**(-1074)*n
    extra=sg.up(2*mp.iv.sqrt(sg.I(2))*op) if n else 0.
    rho=sg.lower(sg.I(q['rho_lower'])-sg.I(extra));ratio=sg.I(extra)/sg.I(q['rho_lower'])
    angle=sg.up(mp.iv.atan2(ratio,mp.iv.sqrt(1-ratio**2)))
    beta=sg.lower(sg.I(q['beta_stat_lower'])-sg.I(angle))
    return {**q,'fusion_parent_rho':q['rho_lower'],'fusion_parent_beta':q['beta_stat_lower'],
            'fusion_raw_disk_upper':extra,'rho_lower':rho,'beta_stat_lower':beta,'closed':bool(q['closed'] and rho>0 and beta>0)}


def point(s,rs,mode='selected',legacy=False):
    x=s['signal'];n=x['steps'];old=saved()[2][(x['name'],x['h'])];new=dets()[(x['name'],x['h'])]
    # D source half-angles are exactly representable in these fixed inputs;
    # no duplicate support exists in the original DF primitive producer.
    assert all(e==0 for e in new['errors'].values())
    g=old['groups'] if legacy else new['groups'] if mode=='selected' else {k:v['metrics'][mode] for k,v in new['variants'].items()}
    g={k:np.array(v,float) for k,v in g.items()}
    if x['case']['kind'].startswith('full_'):
        bulk=n*(g['prefix']+g['suffix']);boundary=(n-1)*(g['middle']-g['prefix']-g['suffix'])
    else:
        bulk=n*(g['first']+g['last']+g['inside']);boundary=(n-1)*(g['junction']-g['first']-g['last'])
    rmean=np.zeros(4);reps={str(v):np.zeros((16,4)) for v in (False,True)}
    for recipe in x['recipe']:
        for rare,rows in rs[r_key(recipe['time'],recipe['r'])].items():
            vectors=[r['baseline'] if legacy else r['selected'] if mode=='selected' else r['variants'][mode] for r in rows]
            v=n*rows[0]['probability']*np.array(vectors);rmean+=v.mean(axis=0);reps[rare]+=v
    over=np.array([old['overhead'][a] for a in ('X','Y')]);q=s['qualification'] if legacy else qualify(s['qualification'],x)
    return dict(signal=x,qualification=q,D_bulk=bulk,D_boundary=boundary,D=bulk+boundary,R=rmean,overhead=over,mean=bulk+boundary+rmean+over,reps=reps)


def old_policies():
    data=json.loads((PRIOR/'POLICY.json').read_text())
    return [dict(family='PR',**p) for p in data['policies']]+[p for p in data['old_policies'] if p['family']=='Full']


def freeze():
    if (OUT/'POLICY.json').exists():raise RuntimeError('Do not overwrite policy')
    rs=records(OUT/'training_prices');new=[]
    for p in old_policies():
        pp=[point(s,rs) for s in p['selected']];alpha,ns=allocate(pp,('RZ','CX').index(p['primary']))
        new.append(dict(family=p['family'],primary=p['primary'],selected=[dict(signal=x['signal'],qualification=x['qualification']) for x in pp],
                        alpha=alpha.tolist(),**account(pp,ns)))
    save_json(OUT/'POLICY.json',dict(source_commit=source(),policies=new,old_policies=old_policies(),
        rule='D coefficients/h/resolution/signals unchanged. Same synthesis/fallback candidates for PR and Full; training-only alpha, exact sum <=.05. Per trajectory/group choose one circuit by actual RZ then CX/depth/size.',
        qualification='Retain parent saved-H native/source/isometry/finite-bit certificates. New diagonal output is only native RZ/CX plus retained CZ/P. Existing all128 basis contexts apply. Exact GF2 ending identity; rational angle round bound <=2u*N+N*2^-1074. All D rounding errors exactly0 for the fixed source.'))


def held(task):
    t,n,rare,j=task;target=OUT/'holdout_prices'/f'{r_key(t,n)}_{int(rare)}_{j}.json'
    if target.exists():return json.loads(target.read_text())
    seed=int.from_bytes(hashlib.sha256(f'TrackC_diagonal_fusion_HOLDOUT:{t}:{n}:{rare}:{j}'.encode()).digest()[:8],'big')
    full,p,choices=r_part(t,n,rare,seed);single,_=build_singletons(t,n,choices);joint,_=build_joint(t,n,choices)
    previous=[metrics(compile_reference(q)) for q in (full,single,joint)]
    # Preserve the previous strictly-RZ fallback tie rule.
    baseline=previous[0]
    for v in previous[1:]:
        if v[0]<baseline[0]:baseline=v
    variants={'reference':baseline};stats=[]
    for mode in MODES[1:]:
        q,s=build(t,n,choices,mode);variants[mode]=metrics(compile_reference(q))
        if mode=='network':stats=s
    chosen=best(variants);r=dict(t=t,n=n,rare=rare,j=j,seed=seed,probability=p if rare else 1-p,choices=choices,
           baseline=baseline,variants=variants,selected=variants[chosen],fusion_choice=chosen,stats=stats)
    save_json(target,r);return r


def weightvec(p):return weights(p)


def result(p,old,rs):
    # Reuse unsupplemented-to-fusion parent rows, not the newly supplemented
    # rows, when reconstructing means. The new rounding disk is added once.
    pp=[point(s,rs) for s in old['selected']];ns=np.array([r['shots'] for r in p['rounds']]);new=account(pp,ns)
    base=account([point(s,rs,legacy=True) for s in old['selected']],np.array([r['shots'] for r in old['rounds']]))
    wn,wo=weightvec(p),weightvec(old);cov=covariance(rs,[('selected',wn)]);bcov=covariance(rs,[('baseline',wo)])
    paired=covariance(rs,[('selected',wn),('baseline',{k:-v for k,v in wo.items()})])
    new.update(MC_SE=np.sqrt(np.diag(cov)).tolist(),covariance_MC=cov.tolist());base.update(MC_SE=np.sqrt(np.diag(bcov)).tolist(),covariance_MC=bcov.tolist())
    phases=[math.atan2(s['signal']['raw']['imag'],s['signal']['raw']['real']) for s in old['selected']]
    dec=reconstruct_rpe_phase(phases,[1<<m for m in range(18)]);energy=-dec['final_phase_estimate']/.02
    branch=all(abs(r['base_phase_estimate']+.02*common()['energy'])<=.4/(1<<m)+1e-10 for m,r in enumerate(dec['rounds']))
    assert branch and abs(energy-common()['energy'])<1.5936e-4 and all(x['qualification']['closed'] for x in pp)
    a=sum(Fraction.from_float(v) for row in p['alpha'] for v in row);assert a<=Fraction(1,20)
    ablations=[]
    for mode in MODES:
        value=account([point(s,rs,mode=mode) for s in old['selected']],ns)
        ablations.append(dict(mode=mode,totals=value['totals']))
    expanded={k:{rare:[{**r,**r['variants']} for r in rows] for rare,rows in strata.items()} for k,strata in rs.items()}
    differences=[]
    for before,after in zip(MODES,MODES[1:]):
        a=next(x['totals']['total'] for x in ablations if x['mode']==before);b=next(x['totals']['total'] for x in ablations if x['mode']==after)
        cv=covariance(expanded,[(before,wn),(after,{k:-v for k,v in wn.items()})])
        differences.append(dict(before=before,after=after,saving=(np.array(a)-b).tolist(),paired_MC_SE=np.sqrt(np.diag(cv)).tolist()))
    return dict(family=p['family'],primary=p['primary'],new=new,baseline=base,
        saving=(np.array(base['totals']['total'])-new['totals']['total']).tolist(),paired_MC_SE=np.sqrt(np.diag(paired)).tolist(),
        fixed_new_shot_ablations=ablations,ablation_differences=differences,alpha_sum_exact=str(sum(Fraction.from_float(v) for row in p['alpha'] for v in row)),mean_branch_pass=branch,mean_energy_error=abs(energy-common()['energy']),decoder=dec)


@lru_cache(None)
def group(case_json,h,role):
    case=json.loads(case_json);seq=split(case,h)[role];ld=12 if case['kind'].startswith('full_') else 3
    mode=dets()[(case['name'],h)]['variants'][role]['selected']
    return d_piece(tuple(seq),ld) if mode=='reference' else d_fused(seq,ld,mode)[0]


def short_check(task):
    s,steps=task;x=s['signal'];case=x['case'];h=x['h'];tails=[];oldtails=[];g=split(case,h)
    for cycle in range(0 if g['full'] else steps):
        full,_,choices=r_part(h,max(1,math.ceil(h*x['rate']/.02)),cycle==0,501781+cycle)
        single,_=build_singletons(h,max(1,math.ceil(h*x['rate']/.02)),choices);joint,_=build_joint(h,max(1,math.ceil(h*x['rate']/.02)),choices)
        baseline=min((full,single,joint),key=lambda q:metrics(compile_reference(q))[0]);variants={'reference':baseline}
        for mode in MODES[1:]:variants[mode]=build(h,max(1,math.ceil(h*x['rate']/.02)),choices,mode)[0]
        chosen=min(variants,key=lambda k:tuple(metrics(compile_reference(variants[k]))));tails.append(variants[chosen]);oldtails.append(baseline)
    old=assemble(case,h,steps,oldtails);cj=json.dumps(case,sort_keys=True);g=split(case,h)
    if g['full']:parts=[group(cj,h,'prefix')]+[group(cj,h,'middle')]*(steps-1)+[group(cj,h,'suffix')]
    else:
        parts=[group(cj,h,'first')]
        for k,tail in enumerate(tails):parts.extend([tail,group(cj,h,'last' if k==steps-1 else 'junction')])
    nativeparts=[compile_reference(q) for q in parts];ld=12 if g['full'] else 3;p=common()['preps'][ld]
    scalar=QuantumCircuit(9);scalar.p(-steps*h*(p.constant_coefficient+p.extracted_identity_coefficient),8)
    pre=QuantumCircuit(9);pre.h(8);pre=compile_reference(pre);rows=[]
    for axis in ('X','Y'):
        sourceq=wrapper([*parts,scalar],axis);whole=compile_reference(sourceq)
        read=QuantumCircuit(9,1);read.compose(scalar,inplace=True)
        if axis=='Y':read.sdg(8)
        read.h(8);read.measure(8,0);read=compile_reference(read)
        modular=QuantumCircuit(9,1);modular.compose(pre,inplace=True)
        for q in nativeparts:modular.compose(q,inplace=True)
        modular.compose(read,inplace=True);expected=np.sum([metrics(q) for q in [pre,*nativeparts,read]],axis=0)
        assert all(metrics(modular)[i]==expected[i] for i in (0,1,3))
        error=None
        if steps==1:
            vv=[Statevector(initial_state()).evolve(q.remove_final_measurements(inplace=False)).data for q in (wrapper([old],axis),whole,modular)]
            f=lambda v:float(np.vdot(v[:256],v[:256]).real-np.vdot(v[256:],v[256:]).real)
            error=max(abs(f(v)-f(vv[0])) for v in vv);assert error<1e-8
        rows.append(dict(axis=axis,whole_DAG=metrics(whole),modular_DAG=metrics(modular),serial_depth_upper=int(expected[2]),readout_error=error,
                         whole_minus_modular=(np.array(metrics(whole))-metrics(modular)).tolist()))
    r=dict(name=x['name'],h=h,steps=steps,rows=rows);save_json(OUT/'short_checks'/f'{x["name"]}_{steps}.json',r);return r


def d_check(task):
    index,t=task;old=d_piece((('D',index,t),),12);a=Operator(old).data;rows=[]
    for mode in MODES[1:]:
        q,e=d_fused([('D',index,t)],12,mode);b=Operator(q).data;c=Operator(compile_reference(q)).data;phase=np.vdot(b,c);phase/=abs(phase)
        errors=[float(np.linalg.norm(a-b,'fro')),float(np.linalg.norm(b-c/phase,'fro'))];assert max(errors)<1e-9 and e==0
        rows.append(dict(mode=mode,source_Fro_error=errors[0],native_projective_Fro_error=errors[1],rounding_error=e))
    return dict(index=index,t=t,rows=rows)


def run():
    if (OUT/'RESOURCES.json').exists():raise RuntimeError('Do not overwrite resources')
    configuration();catalogue();freeze();(OUT/'holdout_prices').mkdir();(OUT/'short_checks').mkdir()
    f=json.loads((OUT/'POLICY.json').read_text());cells=sorted({(r['time'],r['r']) for p in f['policies'] for s in p['selected'] for r in s['signal']['recipe']})
    checks=[];ds=[]
    with ProcessPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
        for v in as_completed(pool.submit(d_check,t) for t in ((0,-.1),(3,.32),(12,.08))):ds.append(v.result())
        save_json(OUT/'D_CORRECTNESS.json',dict(source_commit=source(),rows=ds))
        for v in as_completed(pool.submit(held,(*c,rare,j)) for c in cells for rare in (False,True) for j in range(16)):v.result()
        signals=[next(p for p in f['old_policies'] if p['family']==family and p['primary']=='RZ')['selected'][-1] for family in ('PR','Full')]
        for v in as_completed(pool.submit(short_check,(s,n)) for s in signals for n in (1,2)):checks.append(v.result())
    rs=records(OUT/'holdout_prices');results=[]
    for p in f['policies']:
        old=next(v for v in f['old_policies'] if (v['family'],v['primary'])==(p['family'],p['primary']));results.append(result(p,old,rs))
    save_json(OUT/'RESOURCES.json',dict(source_commit=source(),policies=results,short_checks=checks,holdout_samples=384,
        qualification='Saved-H4 exact ground, ideal native, fresh bits. Parent source/native/isometry certificates plus new uniform rational aggregation disk. No RTE law/normalization/return change.',
        statistical='Independent cell/stratum/index seeds, common trajectories across all methods; combine shared-cache round weights before cell covariance. No holdout retuning.',
        depth='Global totals are serial depth upper bounds; short/occurrence metrics are actual DAG depths. RZ/CX/size sum exactly for the concrete modular implementation.'))
    print('DIAGONAL_RESOURCES_COMPLETE',flush=True)


from trotterlib.rpe_four_round_phase_validation import reconstruct_rpe_phase

if __name__=='__main__':run()
