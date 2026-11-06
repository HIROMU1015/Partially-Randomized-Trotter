"""Synthetic optimizer checks; --saved checks the completed validation only."""
import argparse,json,lzma,sys
from fractions import Fraction as F
from itertools import product
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from trottertracks.track_c2_b_strong_baseline import optimize_case,exponent,global_alpha_lower,evaluate

def synthetic():
    tests=[([10,20,30],[60,100,120],F(51,100)),
           ([7,9,13,19,23,31,37,41],[5,9,3,17,7,19,11,13],F(54,100)),
           ([F(7,3),F(9,7),F(13,11),F(19,5)],[11,7,13,3],F(4,100)),
           ([1,2,3],[3,5,7],F(6,1000))]
    for weights,savings,budget in tests:
        e=exponent(budget,sum(map(F,weights)));records={};items=[]
        for i,(w,s) in enumerate(zip(weights,savings)):
            angle=float(i+.125).hex();items.append(dict(role='B' if i%2 else 'D',angle=angle,count=1,weight=str(w)))
            for j,t in ((e,300),(e+1,300-s)):records[angle,j]=dict(T=t,key=f'synthetic_{i}_{j}',strict_error_upper=str(F(10)**j/4))
        c=dict(m=0,q=1,structural=False,cap=.01,items=items,budget=str(budget),rho_base=.999,beta_base=.38,synthesis_raw_disk_cap=.001,additional_raw_compiler_disk=1e-12,cx=3)
        result=optimize_case(c,records)
        feasible=[]
        for bits in product((0,1),repeat=len(items)):
            error=sum((F(x['weight'])*F(10)**(e+b) for x,b in zip(items,bits)),F(0))
            if error<=budget:feasible.append(sum(300-s*b for s,b in zip(savings,bits)))
        assert result['T']==min(feasible) and result['T_lower']<=min(feasible)
        if len(items)==3 and weights[0]==10:assert result['T']<result['old_greedy_T']
    # Fresh global-alpha dual cannot exceed a known feasible 18-round policy.
    cases=[]
    for m in range(18):
        for cap in (.01,.02):
            cases.append(dict(m=m,q=2**m,cap=cap,structural=False,T=10,T_lower=10,nominal_error='1/1000',actual_strict_error='1/4000',
                geometry=dict(rho=1.,beta=.3,C_lower=4/(0.2955202066613396**2)-1e-10,C_upper=46.,CX=0)))
    alpha=[float(F(1,1000))]*18;v=evaluate([next(x for x in cases if x['m']==m) for m in range(18)],alpha)
    bound=global_alpha_lower(cases,False);assert bound['total_T_lower']<=v['total_T']
    print(json.dumps(dict(status='PASS',brute_force_knapsack_cases=4,greedy_counterexample=True,global_alpha_dual_synthetic=True,new_scientific_calls=0)))

def saved():
    p=root/'artifacts/track_c2_b_strong_baseline/2026-11-06'
    rows=json.loads(lzma.decompress((p/'ALLOCATIONS.json.xz').read_bytes()));assert len(rows)==72
    for r in rows:
        assert F(r['actual_strict_error'])<=F(r['nominal_error'])<=F(r['budget'])
        assert r['T_lower']<=r['T']<=r['old_greedy_T']
        assert all(a['exponent'] in r['exponents'] for a in r['allocation'])
    compare=json.loads((p/'COMPARISONS.json').read_text())
    for arm,v in compare['reoptimized_alpha'].items():
        assert len(v['rounds'])==18
        assert 2*sum((F.from_float(r['alpha_per_axis']) for r in v['rounds']),F(0))<=F('.05')
        assert sum(r['round_T'] for r in v['rounds'])==v['total_T']
        assert v['global_total_T_lower']<=v['total_T']
    stop=json.loads((p/'STOP.json').read_text());assert stop['status']=='MANDATORY_STOP' and not stop['automatic_resume']
    print(json.dumps(dict(status='PASS',registered_cases=72,rounds_per_arm=18,exact_feasibility=True,alpha_and_T_totals=True,old_STOPs_preserved=True)))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--saved',action='store_true');args=a.parse_args();saved() if args.saved else synthetic()
