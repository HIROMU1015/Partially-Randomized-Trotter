"""Read-only reaggregation of already-qualified points; no new simulation."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/track_c_d_internal/2026-11-08'

def main():
    if sys.argv[1:]==['--branch']:
        from trottertracks.track_c_h4 import common
        from trotterlib.rpe_four_round_phase_validation import reconstruct_rpe_phase
        cap=json.loads((OUT/'SHOT_CAPPED_G.json').read_text());raw=json.loads((OUT/'G_RESULTS.json').read_text())
        source={(r['signal']['name'],r['signal']['m'],r['signal']['h'],r['signal']['rate']):r['signal'] for p in raw['policies'] for r in p['selected']}
        points=[source[(r['name'],r['m'],r['h'],r['rate'])] for r in cap['selected']]
        phases=[float(np.angle(complex(r['raw']['real'],r['raw']['imag']))) for r in points]
        result=reconstruct_rpe_phase(phases,[1<<m for m in range(18)])
        energy=-result['final_phase_estimate']/.02;error=abs(energy-common()['energy'])
        branch=all(abs(r['base_phase_estimate']+.02*common()['energy'])<=.4/(1<<m)+1e-10 for m,r in enumerate(result['rounds']))
        assert branch and error<=1.5936e-4
        target=OUT/'CAPPED_BRANCH.json'
        if target.exists():raise RuntimeError('Do not overwrite branch')
        target.write_text(json.dumps(dict(decoder=result,energy=energy,error_Ha=error,all_rounds_correct_branch=branch,E0_supplied_to_decoder=False),indent=2)+'\n')
        print('CAPPED_BRANCH_PASS');return
    target=OUT/'SHOT_CAPPED_G.json'
    if target.exists():raise RuntimeError('Do not overwrite capped-shot result')
    raw=json.loads((OUT/'G_RESULTS.json').read_text());closed=json.loads((OUT/'QUALIFIED_G.json').read_text())
    base=next(p for p in closed['results'] if p['name']=='original');cap={x['m']:x['shots_per_axis'] for x in base['selected']}
    qualified={}
    for file,field in [('QUALIFICATION.json','rows'),('QUALIFICATION_EXTENSION.json','rows')]:
        for q in json.loads((OUT/file).read_text())[field]:qualified[(q['name'],q['m'],q['h'],q['rate'])]=q
    allowed={'original','order_0321','vertex_j0','tuned_0123','tuned_0321'};available=[]
    for policy in raw['policies']:
        if policy['name'] not in allowed:continue
        for r in policy['selected']:
            x=r['signal'];q=qualified[(x['name'],x['m'],x['h'],x['rate'])]
            if q['closed'] and q['shots_per_axis']<=cap[x['m']] and q['dominant_accumulated_bias_abs']<.4:available.append((r,q))
    chosen=[min((v for v in available if v[0]['signal']['m']==m),key=lambda v:v[1]['shots_per_axis']*sum(x[0] for x in v[0]['price_mean'])) for m in range(18)]
    samples={}
    for path in (OUT/'rte_price').glob('*.json'):
        r=json.loads(path.read_text());key=(r['t'],r['n'],r['rare']);samples.setdefault(key,[]).append(r)
    for rs in samples.values():rs.sort(key=lambda r:r['j'])
    total=np.zeros(4);reps={str(rare):np.zeros((16,4)) for rare in (False,True)};selected=[]
    for r,q in chosen:
        x=r['signal'];N=q['shots_per_axis'];total+=N*np.sum(r['price_mean'],axis=0)
        for recipe in x['recipe']:
            for rare in (False,True):
                rs=samples[(recipe['time'],recipe['r'],rare)];reps[str(rare)]+=2*N*x['steps']*rs[0]['probability']*np.array([r['metrics'] for r in rs])
        selected.append(dict(m=x['m'],name=x['name'],h=x['h'],rate=x['rate'],shots_per_axis=N,original_cap=cap[x['m']],case=x['case']))
    covariance=sum((np.cov(x,rowvar=False,ddof=1)/16 for x in reps.values()),np.zeros((4,4)))
    delta_se=np.sqrt(sum(np.var(reps[s]-np.array(base['reps'][s]),axis=0,ddof=1)/16 for s in reps))
    payload=dict(status='QUALIFIED_POINT_REAGGREGATION',rule='Lowest expected RZ among already-qualified five PR families, per-round shots <= original and no dominant-bias fold. Same original physical/RPE conditions.',
        G=total.tolist(),MC_SE=np.sqrt(np.maximum(0,np.diag(covariance))).tolist(),paired_delta_MC_SE=delta_se.tolist(),ratio_to_original=(total/np.array(base['G'])).tolist(),
        selected=selected,total_shots=sum(2*x['shots_per_axis'] for x in selected),original_total_shots=base['total_shots'],all18_closed=True,
        qualification='Conditional fixed-matrix native success guarantee inherited pointwise. Finite-MC policy choice is not globally optimal; point estimator selection bias is not removed. Depth field is serial upper bound; T not evaluated.')
    target.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    print('SHOT_CAPPED_COMPLETE',payload['total_shots'],payload['G'][:2],payload['ratio_to_original'][:2])

if __name__=='__main__':main()
