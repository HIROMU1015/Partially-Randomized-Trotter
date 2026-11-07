"""Source-qualified finite-menu step/alpha optimization, with direct prices."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import math
import os
import subprocess
import numpy as np
from .track_c_bottleneck import OUT,ROOT,PRIOR,PUBLIC,saved,identity,components,allocate,account,rigorous_shots,NAMES
from .track_c_bottleneck_certificate import supplement
from . import track_c_d_qualify as ql
from . import track_c_compiler_a_18round as ca
from .track_c_h4 import common,save_json
from .track_c_d_resources import r_key

def replacement_prices():
    samples={}
    for directory in ('runwise_prices','joint_prices'):
        for path in (OUT/directory).glob('*.json'):
            r=json.loads(path.read_text());samples.setdefault(r_key(r['t'],r['n']),{}).setdefault(str(r['rare']),[]).append(r)
    for strata in samples.values():
        for rs in strata.values():rs.sort(key=lambda r:r['j']);assert len(rs)==16
    return samples

def new_point(x,q,rs,certificate):
    if x['case']['kind'].startswith('full_'):
        old=saved()[0];record=next(r for p in old.values() for r in p['selected'] if identity(r['signal'])==identity(x));return components(record)
    d=saved()[2][(x['name'],x['h'])];g={k:np.array(v,float) for k,v in d['groups'].items()};n=x['steps'];bulk=n*(g['first']+g['last']+g['inside']);boundary=(n-1)*(g['junction']-g['first']-g['last'])
    rmean=np.zeros(4);rep={str(rare):np.zeros((16,4)) for rare in (False,True)}
    for recipe in x['recipe']:
        for rare,records in rs[r_key(recipe['time'],recipe['r'])].items():
            value=n*records[0]['probability']*np.array([r['selected'] for r in records]);rmean+=value.mean(axis=0);rep[rare]+=value
    overhead=np.array([d['overhead'][axis] for axis in ('X','Y')]);cq=supplement(q,x,certificate)
    return dict(signal=x,qualification=cq,D_bulk=bulk,D_boundary=boundary,D=bulk+boundary,R=rmean,overhead=overhead,mean=bulk+boundary+rmean+overhead,reps=rep)

def optimize(points,primary):
    byround={m:[p for p in points if p['signal']['m']==m] for m in range(18)}
    score=lambda p,a:sum(rigorous_shots(p['qualification']['rho_lower'],p['qualification']['beta_stat_lower'],alpha)*p['mean'][axis,primary] for axis,alpha in enumerate(a))
    chosen=[min(byround[m],key=lambda p:score(p,[1/720]*2)) for m in range(18)];seen=set();history=[];best=None
    while True:
        ids=tuple(identity(p['signal']) for p in chosen)
        if ids in seen:break
        seen.add(ids);alpha,ns=allocate(chosen,primary);result=account(chosen,ns);cost=result['totals']['total'][primary]
        history.append(dict(cost=cost,points=ids))
        if best is None or cost<best[0]:best=(cost,chosen[:],alpha.copy(),ns.copy(),result)
        chosen=[min(byround[m],key=lambda p:score(p,alpha[m])) for m in range(18)]
    _,chosen,alpha,ns,result=best
    return dict(primary=['RZ','CX'][primary],alpha=alpha.tolist(),selected=[dict(signal=p['signal'],qualification=p['qualification'],price_mean=p['mean'].tolist()) for p in chosen],history=history,**result)

def native_new_signals(signals):
    logical=saved()[1];rows=[x for x in signals if identity(x) not in logical]
    bounds=json.loads((OUT/'REPLACEMENT_LOCAL_INTERVALS.json').read_text())['primitive_bounds'].copy()
    for r in json.loads((PRIOR/'QUALIFICATION_EXTENSION.json').read_text())['checked']:bounds[r['source_key']]=max(bounds.get(r['source_key'],0.),r['operator_error_upper'])
    catalogue={}
    for spec in ca.inventory()['catalogue'].values():
        if spec['family'] not in {s['family'] for s in catalogue.values()}:catalogue[spec['key']]=spec
    for x in rows:
        from .track_c_d_internal import schedule
        seq=schedule(x['case'],x['h'])
        for k,i,t in seq:
            if k=='D':
                for key,s in ql.d_counts(i,t)[1].items():
                    if key not in bounds:catalogue[key]=s
        for key,s in ql.d_counts(seq[0][1],2*seq[0][2])[1].items():
            if key not in bounds:catalogue[key]=s
        for recipe in x['recipe']:
            tau=common()['preps'][3].exact_rte_lambda_r*recipe['time']/recipe['r']
            for order in (0,2):
                for sign in (-1,1):
                    for family in ('crz','crzz'):
                        s=ca.spec(family,[sign*2*math.atan(tau/(order+1))]);catalogue[s['key']]=s
    contexts=[];unique={};new=[]
    with ProcessPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
        for f in as_completed(pool.submit(ca.context_compile,(mask,catalogue)) for mask in range(128)):
            r=f.result();contexts.append({k:r[k] for k in ('mask','context','post_unroll')})
            for record in r['records']:unique[record['lowering_sha256']]=record
        checked=list(pool.map(ca.interval_check,unique.values()))
        for r in checked:bounds[r['source_key']]=max(bounds.get(r['source_key'],0.),r['operator_error_upper'])
        for f in as_completed(pool.submit(ql.qualify,(x,bounds)) for x in rows):new.append(f.result())
    save_json(OUT/'JOINT_NATIVE_SOURCE.json',dict(contexts=contexts,checked=checked,rows=new,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()))
    return {**logical,**{identity(x):x for x in new}}

def run():
    if (OUT/'OPTIMIZATION.json').exists():raise RuntimeError('Do not overwrite optimization')
    raw={identity(x):x for x in json.loads((PRIOR/'SIGNALS.json').read_text())['rows']}
    for policy in saved()[0].values():
        for r in policy['selected']:raw[identity(r['signal'])]=r['signal']
    added=json.loads((OUT/'JOINT_CANDIDATES.json').read_text())['new_signals']
    for name in ('tuned_0123','tuned_0321'):
        for m in (16,17):added.append(next(x for x in raw.values() if x['name']==name and x['m']==m and x['requested_h']==.32 and x['rate']==32))
    qs=native_new_signals(added)
    for x in added:raw[identity(x)]=x
    rs=replacement_prices();certificate=json.loads((OUT/'REPLACEMENT_CERTIFICATE.json').read_text());points=[]
    for key,q in qs.items():
        if key not in raw:continue
        x=raw[key]
        p=q['dominant_overlap'];leak=math.asin(min(1.,(1-p)/p)) if p>.5 else math.inf
        if not q['closed'] or q['dominant_accumulated_bias_abs']+leak>=.4:continue
        if not all(r_key(r['time'],r['r']) in rs for r in x['recipe']):continue
        point=new_point(x,q,rs,certificate)
        if point['qualification']['closed']:points.append(point)
    result=[]
    for family,full in [('PR',False),('Full',True)]:
        pool=[p for p in points if p['signal']['case']['kind'].startswith('full_')==full]
        for primary in (0,1):result.append(dict(family=family,**optimize(pool,primary)))
    fixed=[]
    for name in NAMES:
        pp=[new_point(r['signal'],qs[identity(r['signal'])],rs,certificate) for r in sorted(saved()[0][name]['selected'],key=lambda r:r['signal']['m'])]
        ns=np.array([[rigorous_shots(p['qualification']['rho_lower'],p['qualification']['beta_stat_lower'],1/720)]*2 for p in pp])
        alpha,an=allocate(pp,0);fixed.append(dict(name=name,uniform=account(pp,ns),allocated=dict(alpha=alpha.tolist(),**account(pp,an))))
    save_json(OUT/'OPTIMIZATION.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),optimized=result,fixed_points=fixed,
        point_count=len(points),qualification='All selected source/native budgets closed, including uniform replacement disk. Exact alpha sum <=.05. Nonresonant float diagnostic filter. Conditional saved-matrix model, ideal native, fresh bits.',
        limits='Alternating finite-menu step/alpha selection, not globally optimal. Price MC training selection needs a fixed-policy holdout. Depth is serial upper bound, not full DAG depth.'))
    print('OPTIMIZATION_COMPLETE',len(points),flush=True)

if __name__=='__main__':run()
