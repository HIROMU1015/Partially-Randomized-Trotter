"""Saved-data-only optimization of the registered C2-B precision pairs.

No acquisition, Hamiltonian, circuit, or synthesis entry point is called.
"""
from __future__ import annotations
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction as F
from functools import reduce
from itertools import product
from pathlib import Path
import hashlib, json, lzma, math, os, platform, tarfile, time, warnings
import mpmath as mp
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp, minimize_scalar
from .track_c_h4 import ROOT, save_json
from .track_c2_b_pilot import exponent, allocate, phase_disk_upper, shots
from .track_c2_b_verify import allocation_lp_lower
from .track_c_compiler_a_18round import I, upper, lower

BASE='78004943ed5b718c12841e80f55fddf1d1644364'
INPUT=ROOT/'artifacts/track_c2_b_pilot/2026-11-06'
OUT=ROOT/'artifacts/track_c2_b_strong_baseline/2026-11-06'
RECORDS={}


def load_inputs():
    global RECORDS
    checked=[]
    for b in json.loads((INPUT/'BUNDLES.json').read_text()):
        if b['path'] not in ('raw_synthesis.tar.xz','PRECISION_REQUESTS.json.xz','verified_final/RESULTS.json.xz'):continue
        raw=(INPUT/b['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==b['sha256'],b['path']
        if 'decompressed_sha256' in b:assert hashlib.sha256(lzma.decompress(raw)).hexdigest()==b['decompressed_sha256']
        checked.append(b)
    records={}
    with tarfile.open(INPUT/'raw_synthesis.tar.xz') as tar:
        for member in tar:
            assert member.isfile() and member.name.startswith('synthesis/') and member.name.endswith('.json')
            v=json.load(tar.extractfile(member));key=(v['theta_hex'],v['precision_exponent'])
            assert key not in records and v['status']=='PASS'
            assert hashlib.sha256(v['sequence'].encode()).hexdigest()==v['sequence_sha256']
            assert 0<=F(v['strict_error_upper'])<=F(10)**v['precision_exponent']
            assert v['T']==v['sequence'].count('T')+v['sequence'].count('t')
            records[key]=v
    assert len(records)==7603
    req=json.loads(lzma.decompress((INPUT/'PRECISION_REQUESTS.json.xz').read_bytes()))
    old=json.loads(lzma.decompress((INPUT/'verified_final/RESULTS.json.xz').read_bytes()))
    assert len(req['cases'])==72
    assert all(len(x['rounds'])==18 for x in old.values())
    # Exactly the adopted gamma_4096 correction, once; old geometry is replayed below.
    factor=F(1)/(1-F(4096,2**53))
    for c in req['cases']:
        c['additional_raw_compiler_disk']=math.nextafter(float(F.from_float(c['additional_raw_compiler_disk'])*factor),math.inf)
        e=exponent(F(c['budget']),sum((F(x['weight']) for x in c['items']),F(0)))
        assert all((x['angle'],j) in records for x in c['items'] for j in (e,e+1))
        assert all(x['count']>0 and F(x['weight'])>0 for x in c['items'])
    RECORDS=records
    return req['cases'],old,dict(status='PASS',base_commit=BASE,records=len(records),registered_cases=72,
        input_bundles=checked,all_registered_precision_pairs_present=True,adopted_results='verified_final',
        initial_results_not_adopted=True,old_uniform_alpha_LP_total_not_reused=True)


def offers_for(c,records):
    items=c['items'];budget=F(c['budget']);total=sum((F(x['weight']) for x in items),F(0));e=exponent(budget,total)
    base_error=total*F(10)**e;base_T=sum(x['count']*records[x['angle'],e]['T'] for x in items)
    room=budget-base_error;offers=[]
    for i,x in enumerate(items):
        saving=x['count']*(records[x['angle'],e]['T']-records[x['angle'],e+1]['T'])
        weight=F(x['weight'])*9*F(10)**e
        if saving>0:offers.append(dict(i=i,saving=saving,weight=weight))
    return e,base_T,base_error,room,offers


def exact_fractional_price_lower(base_T,room,offers):
    remain=room;savings=F(0)
    for o in sorted(offers,key=lambda o:F(o['saving'])/o['weight'],reverse=True):
        take=min(remain,o['weight']);savings+=F(o['saving'])*take/o['weight'];remain-=take
        if remain==0:break
    # The integer saving lies on this exact gcd lattice.
    g=reduce(math.gcd,(o['saving'] for o in offers),0) or 1
    lower_T=base_T-g*(savings//g)
    return int(lower_T),str(F(base_T)-savings)


def optimize_case(c,records=None):
    records=RECORDS if records is None else records;started=time.monotonic()
    e,base_T,base_error,room,offers=offers_for(c,records)
    lower_T,continuous_lower=exact_fractional_price_lower(base_T,room,offers)
    old=allocate(c['items'],F(c['budget']),records,True)
    oldchosen={i for i,x in enumerate(old['allocation']) if x['exponent']==e+1}
    selected=set(oldchosen);solver=dict(status='NO_POSITIVE_RELAXATIONS',message='')
    if offers and room>0:
        # Normalize the one-row knapsack. Float solves only propose a binary vector;
        # exact rational feasibility and the lower bound never depend on its tolerance.
        aa=np.array([float(o['weight']/room) for o in offers],float)
        profit=np.array([o['saving'] for o in offers],float)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='Unrecognized options detected')
            r=milp(-profit,integrality=np.ones(len(offers)),bounds=Bounds(0,1),
                constraints=LinearConstraint(aa[None,:],-np.inf,1.),
                options=dict(presolve=True,mip_rel_gap=0.,threads=1,parallel=False))
        solver=dict(status=int(r.status),message=r.message,node_count=getattr(r,'mip_node_count',None),
            reported_gap=getattr(r,'mip_gap',None),reported_negative_saving_dual=getattr(r,'mip_dual_bound',None))
        if r.x is not None:
            candidate={o['i'] for o,z in zip(offers,r.x) if z>.5};byid={o['i']:o for o in offers}
            used=sum((byid[i]['weight'] for i in candidate),F(0));repairs=0
            for i in sorted(candidate,key=lambda i:F(byid[i]['saving'])/byid[i]['weight']):
                if used<=room:break
                candidate.remove(i);used-=byid[i]['weight'];repairs+=1
            for o in sorted(offers,key=lambda o:F(o['saving'])/o['weight'],reverse=True):
                if o['i'] not in candidate and used+o['weight']<=room:candidate.add(o['i']);used+=o['weight']
            if sum(byid[i]['saving'] for i in candidate)>=base_T-old['T']:selected=candidate
            solver['exact_feasibility_repairs']=repairs
    chosen=[];nominal=F(0);actual=F(0);price=0
    for i,x in enumerate(c['items']):
        j=e+int(i in selected);r=records[x['angle'],j]
        nominal+=F(x['weight'])*F(10)**j;actual+=F(x['weight'])*F(r['strict_error_upper']);price+=x['count']*r['T']
        chosen.append(dict(role=x['role'],angle=x['angle'],count=x['count'],weight=x['weight'],exponent=j,key=r['key']))
    assert actual<=nominal<=F(c['budget']) and lower_T<=price<=old['T']
    # Reuse/check the original exact LP function, without importing its old total.
    assert lower_T>=allocation_lp_lower(c['items'],F(c['budget']),records)
    return dict(m=c['m'],q=c['q'],cap=c['cap'],structural=c['structural'],T=price,T_lower=lower_T,
        fractional_price_lower=continuous_lower,rigorous_T_gap=price-lower_T,
        rigorous_relative_T_gap=(price-lower_T)/price,exact_optimality_proved=price==lower_T,
        old_greedy_T=old['T'],exponents=[e,e+1],solver=solver,allocation=chosen,
        nominal_error=str(nominal),actual_strict_error=str(actual),budget=c['budget'],
        geometry=geometry(c),wall_seconds=time.monotonic()-started)


def geometry(c):
    mp.iv.prec=192;d=c['synthesis_raw_disk_cap'];extra=c['additional_raw_compiler_disk']
    rho=lower(I(c['rho_base'])-I(d)-I(extra))
    phase=phase_disk_upper(upper(I(d)+I(extra)),c['rho_base']);beta=lower(I(c['beta_base'])-I(phase))
    assert rho>0 and beta>0
    coeff=I(4)/(I(rho)**2*mp.iv.sin(I(beta))**2)
    return dict(rho=rho,beta=beta,C_lower=lower(coeff),C_upper=upper(coeff),
        synthesis_raw_disk_cap=d,additional_raw_compiler_disk=extra,total_phase_extra=phase,normalizer=1.,CX=c['cx'])


def safe_alpha(rows):
    weights=[2*r['T']*4/(r['geometry']['rho']**2*math.sin(r['geometry']['beta'])**2) for r in rows]
    alpha=[.025*w/sum(weights) for w in weights]
    while 2*sum((F.from_float(x) for x in alpha),F(0))>F('.05'):
        alpha=[math.nextafter(x,0.) for x in alpha]
    return alpha


def evaluate(rows,alpha):
    assert len(rows)==len(alpha)==18
    assert 2*sum((F.from_float(x) for x in alpha),F(0))<=F('.05')
    rounds=[]
    for r,a in zip(rows,alpha):
        g=r['geometry'];n=shots(g['rho'],g['beta'],a)
        rounds.append(dict(m=r['m'],q=r['q'],cap=r['cap'],T_per_shot=r['T'],T_per_shot_lower=r['T_lower'],
            alpha_per_axis=a,shots_per_axis=n,round_T=2*n*r['T'],round_T_lower=2*n*r['T_lower'],
            round_CX=2*n*g['CX'],rho_lower=g['rho'],beta_stat=g['beta'],
            synthesis_operator_error_nominal=r['nominal_error'],synthesis_operator_error_actual=r['actual_strict_error'],
            **{k:v for k,v in g.items() if k not in ('rho','beta')}))
    return dict(total_T=sum(r['round_T'] for r in rounds),price_LP_lower_at_fixed_schedule=sum(r['round_T_lower'] for r in rounds),
        total_shots=2*sum(r['shots_per_axis'] for r in rounds),total_CX=sum(r['round_CX'] for r in rounds),rounds=rounds)


def best_caps(cases,alpha):
    rows=[]
    for m,a in enumerate(alpha):
        eligible=[r for r in cases if r['m']==m]
        rows.append(min(eligible,key=lambda r:shots(r['geometry']['rho'],r['geometry']['beta'],a)*r['T']))
    return rows


def compare_old(cases,old,records):
    out={}
    for structural,name in [(False,'gate_sum_nonuniform'),(True,'structural_nonuniform')]:
        rows=old[name]['rounds'];cs=[r for r in cases if r['structural']==structural]
        selected=[next(r for r in cs if r['m']==x['m'] and r['cap']==x['cap']) for x in rows]
        alpha=[x['alpha_per_axis'] for x in rows]
        out[name]=evaluate(selected,alpha)
        # Inherited geometry and integer ceilings must agree exactly with adopted data.
        for r,x in zip(out[name]['rounds'],rows):
            assert r['rho_lower']==x['rho_lower'] and r['beta_stat']==x['beta_stat']
            assert r['shots_per_axis']==x['cost_sensitive_shots_per_axis']
    return out


def common_alpha_comparisons(cases,old):
    comparisons={}
    for source in ('gate_sum_nonuniform','structural_nonuniform'):
        alpha=[r['alpha_per_axis'] for r in old[source]['rounds']];arms={}
        for structural,name in [(False,'ordinary'),(True,'structural')]:
            cs=[r for r in cases if r['structural']==structural]
            arms[name]=evaluate(best_caps(cs,alpha),alpha)
            arms[name]['global_price_lower_for_this_alpha']=sum(min(2*shots(r['geometry']['rho'],r['geometry']['beta'],a)*r['T_lower'] for r in cs if r['m']==m) for m,a in enumerate(alpha))
        comparisons[source]=arms
    return comparisons


def reoptimize_alpha(cases,old,structural):
    cs=[r for r in cases if r['structural']==structural];best=None;trace=[]
    for start in ('gate_sum_nonuniform','structural_nonuniform'):
        alpha=[r['alpha_per_axis'] for r in old[start]['rounds']];seen=set()
        while True:
            rows=best_caps(cs,alpha);value=evaluate(rows,alpha)
            signature=tuple(r['cap'] for r in rows)
            trace.append(dict(start=start,total_T=value['total_T'],caps=list(signature)))
            if best is None or value['total_T']<best['total_T']:best=value
            proposed=safe_alpha(rows);pv=evaluate(rows,proposed)
            if pv['total_T']<best['total_T']:best=pv
            if signature in seen:break
            seen.add(signature);alpha=proposed
    best['selection_trace']=trace
    return best


def global_alpha_lower(cases,structural):
    """Fresh lower bound: rational price LP + continuous-shot Lagrange dual.

    For each registered cap, use rigorous C_lower and T_lower. For lambda>=0,
    inf_N>=0 [2*T*N +4*lambda*exp(-N/C)] is explicit. Sum the per-round
    cap minima and subtract .05*lambda. This relaxes integer shots and all
    alpha choices; it is never an executable baseline cost.
    """
    cs=[r for r in cases if r['structural']==structural]
    grouped=[[r for r in cs if r['m']==m] for m in range(18)]
    sums=[sum(min(r['T_lower']*r['geometry']['C_lower'] for r in group) for group in grouped),
          sum(max(r['T_lower']*r['geometry']['C_lower'] for r in group) for group in grouped)]
    def dual_float(loglam):
        lam=math.exp(loglam);terms=[]
        for group in grouped:
            vals=[]
            for r in group:
                tc=r['T_lower']*r['geometry']['C_lower']
                vals.append(4*lam if 2*lam<=tc else 2*tc*(math.log(2*lam/tc)+1))
            terms.append(min(vals))
        return math.fsum(terms)-.05*lam
    lo,hi=math.log(40*sums[0]),math.log(40*sums[1])
    opt=minimize_scalar(lambda x:-dual_float(x),bounds=(lo-1,hi+1),method='bounded',options=dict(xatol=1e-13))
    lam=float(math.exp(opt.x));mp.iv.prec=192;terms=[]
    for group in grouped:
        vals=[]
        for r in group:
            tc=I(r['T_lower'])*I(r['geometry']['C_lower'])
            z=I(4)*I(lam) if 2*F.from_float(lam)<=r['T_lower']*F.from_float(r['geometry']['C_lower']) else 2*tc*(mp.iv.log(2*I(lam)/tc)+1)
            vals.append(z)
        # An outward lower endpoint of the cap minimum suffices for a lower bound.
        terms.append(min(lower(v) for v in vals))
    dual=sum((I(v) for v in terms),I(0))-I(1)/20*I(lam)
    bound=math.ceil(lower(dual))
    return dict(total_T_lower=bound,lambda_binary64_hex=lam.hex(),continuous_shots=True,
        alpha_scope='all allocations with total .05 and equal X/Y sufficient-budget model',
        caps='registered .01/.02 per round',precision='registered pair, rational LP relaxation',
        directed_bits=192,numeric_dual_search_success=bool(opt.success),executable_price=False)


def run():
    if (OUT/'STOP.json').exists():raise RuntimeError('Stopped validation; no automatic restart')
    OUT.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    cases,old,check=load_inputs();save_json(OUT/'INPUT_CHECK.json',check)
    print('Inputs PASS: 7603 records, 72 registered cases',flush=True)
    # fork shares the read-only record dictionary; independent single-thread MILPs
    # use all available independent-task/CPU-affinity slots without oversubscription.
    results=[];workers=min(len(cases),len(os.sched_getaffinity(0)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(optimize_case,c) for c in cases]
        for future in as_completed(futures):
            r=future.result();results.append(r)
            if len(results)%6==0:print('MILP cases',len(results),'/72',flush=True)
    results.sort(key=lambda r:(r['structural'],r['m'],r['cap']))
    raw=json.dumps(results,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    (OUT/'ALLOCATIONS.json.xz').write_bytes(lzma.compress(raw))
    fixed=compare_old(results,old,RECORDS);common=common_alpha_comparisons(results,old)
    reopt={};bounds={}
    for structural,name in [(False,'ordinary'),(True,'structural')]:
        reopt[name]=reoptimize_alpha(results,old,structural)
        bounds[name]=global_alpha_lower(results,structural)
        reopt[name]['global_total_T_lower']=bounds[name]['total_T_lower']
        reopt[name]['global_relative_gap']=(reopt[name]['total_T']-bounds[name]['total_T_lower'])/reopt[name]['total_T']
        assert bounds[name]['total_T_lower']<=reopt[name]['total_T']
    oldN=old['gate_sum_nonuniform']['total_T_cost_sensitive_alpha'];oldS=old['structural_nonuniform']['total_T_cost_sensitive_alpha']
    n=fixed['gate_sum_nonuniform']['total_T'];s=fixed['structural_nonuniform']['total_T']
    newN=reopt['ordinary']['total_T'];newS=reopt['structural']['total_T']
    status=('CONDITIONAL_IMPROVEMENT_REMAINS' if newS<bounds['ordinary']['total_T_lower'] else
        'IMPROVEMENT_DISAPPEARS_CERTIFIED' if newN<=bounds['structural']['total_T_lower'] else
        'OPTIMALITY_UNDETERMINED')
    summary=dict(status=status,base_commit=BASE,old_policy=dict(ordinary_T=oldN,structural_T=oldS,reduction=1-oldS/oldN),
        fixed_original_alpha_and_caps=dict(ordinary_optimized_T=n,old_structural_T=oldS,structural_optimized_T=s,
            old_structural_vs_stronger_ordinary=1-oldS/n,reoptimized_structural_vs_stronger_ordinary=1-s/n),
        reoptimized_cost_sensitive_alpha=dict(ordinary_T=newN,structural_T=newS,reduction=1-newS/newN,bounds=bounds),
        exact_case_optima=sum(r['exact_optimality_proved'] for r in results),
        solver_reported_optima=sum(r['solver']['status']==0 for r in results),cases=72,workers=workers,
        new_synthesis=0,new_compiles=0,new_sampling=0,new_precision_candidates=0,
        old_STOPs_preserved=True,wall_seconds=time.monotonic()-start)
    save_json(OUT/'COMPARISONS.json',dict(fixed_original=fixed,common_stored_alpha=common,reoptimized_alpha=reopt,global_lower_bounds=bounds))
    save_json(OUT/'SUMMARY.json',summary)
    table=[]
    for arm,v in reopt.items():
        for row in v['rounds']:table.append(dict(arm=arm,**row))
    cols=list(table[0]);(OUT/'ROUND_TABLE.tsv').write_text('\t'.join(cols)+'\n'+'\n'.join('\t'.join(str(r[k]) for k in cols) for r in table)+'\n')
    save_json(OUT/'ENVIRONMENT.json',dict(host=platform.node(),cpu_affinity=len(os.sched_getaffinity(0)),workers=workers,
        solver='SciPy milp / HiGHS; one thread per independent MILP',options=dict(presolve=True,mip_rel_gap=0.,threads=1,parallel=False),
        artificial_RAM_limit=None,subagents=0,shared_settings_changed=False))
    save_json(OUT/'STOP.json',dict(status='MANDATORY_STOP',automatic_resume=False,workers_remaining=0))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':run()
