"""Independent saved-result structural/integer audit, no new science calls."""
import csv
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
FINAL=ROOT/'artifacts/resource_applicability/track_a_h246_pf_common_final/2026-10-22'
BASE='0b1886f53f7c3a5e99e0c13cb5f7090581c5e38c'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cid(r):return '_'.join(str(r[k]) for k in ('system','method','prefix','q','formula'))


def check():
    checks=[]
    def must(ok,label):
        if not ok:raise ValueError(label)
        checks.append(label)
    rows=read(FINAL/'candidate_status_588.json');comps=read(FINAL/'comparisons_60.json');expected=set()
    for system,rank,prefixes in [('H2',3,range(3)),('H4',10,range(1,10)),('H6',19,range(5,16))]:
        for f in ('2nd','4th','4th(m5_best)'):
            for q in (1,2,4,8):
                for method,p in [('FULL',rank)]+[(m,p) for p in prefixes for m in ('B0','B0I')]:expected.add(f'{system}_{method}_{p}_{q}_{f}')
    must(len(rows)==588 and {r['id'] for r in rows}==expected,'EXACT588_REGISTERED_GRID')
    must(all(r['status']=='COMPLETE' for r in rows),'ALL588_EVALUATED_NOT_ALL_COMPILED')
    prior=list(csv.DictReader((ROOT/'artifacts/resource_applicability/track_a_h246_saved_cost_results/2026-10-21/candidate_precision_costs_1000.csv').open()))
    lookup={(r['system'],float(r['epsilon']),r['candidate_id']):r for r in prior if r['method']=='PR' and r['mode']=='USUAL_SIGNED'}
    allgrants=[]
    for system,n in [('H2',84),('H4',228),('H6',276)]:
        b=ROOT/f'artifacts/resource_applicability/track_a_pf_common_{system}_results/2026-10-22'
        with gzip.open(b/'PF_endpoint_costs.json.gz','rt') as f:endpoints=json.load(f)
        must(len(endpoints)==5*n,'ENDPOINT_COUNT:'+system)
        for r in endpoints:
            must(r['status'] in ('ELIGIBLE','INELIGIBLE'),'NO_UNKNOWN:'+system+':'+cid(r['candidate'])+':'+str(r['epsilon']))
            must(r['u']>=r['old_u'] and r['log_B_margin']>=r['old_margin'],'SAVED_ALLOWANCE_NOT_REDUCED:'+system)
            if r['status']=='ELIGIBLE':
                p=r['price'];must(Fraction(r['N_cos'])*p['cosine']+Fraction(r['N_sin'])*p['sine']==Fraction(r['total_RZ_exact']),'EXACT_INTEGER_COST:'+system)
            else:must(r['total_RZ_exact'] is None,'INELIGIBLE_COST_NULL:'+system)
        for r in [r for r in comps if r['system']==system]:
            s=r['PR_saved'];must(s==lookup[system,r['epsilon'],s['candidate_id']],'PR_SAVED_VALUES_VERBATIM:'+system)
            subset=[v for v in endpoints if v['epsilon']==r['epsilon'] and (r['formula']=='ALL_FORMULAS' or v['candidate']['formula']==r['formula'])]
            must(len(subset)==(n if r['formula']=='ALL_FORMULAS' else n//3),'FULL_MINIMUM_SCOPE:'+system)
            pf=r['PF']['best'];available=[v for v in subset if v['status']=='ELIGIBLE']
            must(pf in available and Fraction(pf['total_RZ_exact'])==min(Fraction(v['total_RZ_exact']) for v in available),'COMPLETE_GRID_BEST:'+system)
            ratio=Fraction(int(s['total_RZ_numerator']),int(s['total_RZ_denominator']))/Fraction(pf['total_RZ_exact'])
            must(str(ratio)==r['PR_over_PF_exact'] and abs(float(100*(1-ratio))-r['PR_reduction_percent'])<1e-10,'EXACT_RATIO:'+system)
        # H2/H4 preserve original control export format; H6 uses same original dictionaries.
        with gzip.open(b/'controls_and_ledgers.json.gz','rt') as f:controls=json.load(f)
        for stage,data in controls.items():
            g=data['grant.json'];allgrants.append(g['grant_id']);counts={k:0 for k in g['caps']}
            for item in data['ledger']:must(item['entered_nonrefundable'],'NONREFUNDABLE:'+system);counts[item['kind']]+=item['count']
            must(counts==data['reserved_counts.json']==data['completion.json']['actual_calls'],'LEDGER_COUNTER_RECEIPT:'+system+stage)
            must(all(counts[k]<=g['caps'][k] for k in counts),'CAPS:'+system+stage)
            must(data['CONSUMED.json']['grant']==g,'MARKER_GRANT:'+system+stage)
    meta=read(ROOT/'artifacts/resource_applicability/track_a_h6_pf_metadata_completion/2026-10-22/grant_and_ledger.json')
    allgrants.append(meta['grant']['grant_id']);must(len(allgrants)==len(set(allgrants))==13,'DISTINCT13_SUCCESSFUL_GRANTS')
    for r in read(FINAL/'prior_results_preserved.json')['files']:must(sha(ROOT/r['path'])==r['sha256'],'PRIOR_RESULT_BYTES:'+r['path'])
    for row in read(FINAL/'nested_PF_cost_change_decomposition_10.json'):
        product=Fraction(row['policy_and_saved_arithmetic_factor_exact'])*Fraction(row['m5_prefix_expansion_factor_exact'])*Fraction(row['formula_expansion_factor_exact'])
        must(product==Fraction(row['exact_product']),'EXACT_NESTED_COST_FACTOR_PRODUCT')
    must(len(comps)==60,'COMPARISON_COUNT60')
    result=dict(status='PASS',checks_passed=len(checks),checks=checks,new_scientific_calls=0,old_PR_literal_preserved=True,old_H2_H4_recomputed=False,full_grid_minima_checked=True)
    with (FINAL/'independent_saved_result_checks.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    return result


if __name__=='__main__':
    result=check();print('PASS',result['checks_passed'],'saved-result checks')
