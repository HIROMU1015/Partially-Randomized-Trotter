"""Verify public scalar extraction with separate multinomial/Fraction arithmetic.

No external runtime, NPZ, quantum library or new trajectory is needed.
"""
from collections import Counter,defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import resource

HERE=Path(__file__).resolve().parent


def read(name):return json.loads((HERE/name).read_text())


def main():
    resource.setrlimit(resource.RLIMIT_AS,(2**30,2**30))
    data=read('selected_trajectory_cost_evidence_v1.json')['records']
    analysis=read('cost_reliability_analysis_v1.json')
    audit=read('analysis_audit_v1.json')
    for p,h in audit['generated_file_hashes'].items():
        assert hashlib.sha256((HERE/p).read_bytes()).hexdigest()==h,p
    by=defaultdict(list)
    for r in data:
        assert len(r['orders'])==r['cell']['R']
        assert r['n2']==r['orders'].count(2) and r['n4']==r['orders'].count(4)
        assert r['max_order']==max(r['orders'])
        by[(r['cell_id'],r['phase'])].append(r)
    assert len(data)==320 and len(by)==16
    for (cid,phase),rs in by.items():
        n=32 if phase=='confirmation' else 8
        assert {r['replica'] for r in rs}==set(range(n)) and len(rs)==n
    tested=0
    for row in analysis['epsilon_cell_phase_rows']:
        rs=by[(row['cell_id'],row['phase'])];R=rs[0]['cell']['R']
        law=rs[0]['finite_distribution'];p=dict(zip(law['orders'],law['order_probabilities']))
        p2,p4=p[2],p.get(4,0.);p0=1-math.fsum((p2,p4))
        histogram={}
        for n4 in range(R+1):
            for n2 in range(R-n4+1):
                n0=R-n2-n4
                coefficient=math.factorial(R)//(math.factorial(n0)*math.factorial(n2)*math.factorial(n4))
                histogram[(n2,n4)]=coefficient*p0**n0*p2**n2*p4**n4
        assert math.isclose(math.fsum(histogram.values()),1.,abs_tol=2e-14)
        N=row['shot_counts'];costs=[sum(N[a]*r['metrics'][a]['RZ'] for a in N) for r in rs]
        assert math.isclose(float(Fraction(sum(costs),len(costs))),row['G']['mean'],rel_tol=2e-15)
        present=Counter((r['n2'],r['n4']) for r in rs)
        missing=math.fsum(v for k,v in histogram.items() if not present[k])
        hm=row['histogram_mixture']
        assert math.isclose(missing,hm['unobserved_probability_mass'],rel_tol=1e-13,abs_tol=1e-15)
        known=0.
        for k,prob in histogram.items():
            values=[D for r,D in zip(rs,costs) if (r['n2'],r['n4'])==k]
            if values:known+=prob*float(Fraction(sum(values),len(values)))
        assert math.isclose(known,hm['known_mass_plugin'],rel_tol=2e-15)
        if missing:
            boundary=hm['unknown_common_conditional_mean_break_even']
            assert math.isclose(known+missing*boundary,row['PF_G'],rel_tol=2e-15)
        assert hm['cost_tail_bound'] is None and not hm['population_mean_certified']
        tested+=1
    result=dict(status='LOCAL_PUBLIC_EXTRACTION_FRACTION_MULTINOMIAL_MATCH',selected_trajectories=320,
        scalar_cost_comparison_rows=tested,public_scalar_analysis_reproducible=True,
        external_raw_read=False,new_science_calls=0,new_transpiles=0,scientific_imports=0,CI=False,
        scientific_review_repeated=False,cost_population_bound_proven=False,
        verification_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with (HERE/'public_consistency_v1.json').open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
