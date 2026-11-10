"""Descriptive error/structure analysis of the completed fixed H6 V1 only.

No prediction, correction fit, bootstrap, circuit, sampling, N or G call.
"""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'docs/research/h6_df19_price_v1_20261110'


def digest(b):return hashlib.sha256(b).hexdigest()


def main():
    output=PUBLIC/'STRUCTURE_ANALYSIS.json'
    if output.exists():raise FileExistsError('Preserve completed analysis')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    name=Path(__file__).relative_to(ROOT).as_posix()
    if subprocess.check_output(['git','show',source+':'+name],cwd=ROOT)!=Path(__file__).read_bytes():raise ValueError('Commit analysis source first')
    completed=json.loads((PUBLIC/'COMPLETION.json').read_text());groups=[];inputs={}
    for filename in ['DEVELOPMENT.json','HOLDOUT.json','AUXILIARY.json']:
        raw=(PUBLIC/filename).read_bytes();data=json.loads(raw);inputs[filename]=digest(raw)
        if data['model_source']!=completed['model_source']:raise ValueError('Frozen model differs')
        rows=data['rows']
        for family in dict.fromkeys(r['family'] for r in rows):
            for axis in ('cosine','sine'):
                rs=[r for r in rows if r['family']==family and r['axis']==axis];n=len(rs)
                keys=set().union(*(r['prediction']['features'] for r in rs))
                roles=set().union(*(r['prediction']['native_rz_by_role'] for r in rs))
                def summary(values):return dict(minimum=min(values),maximum=max(values),mean=sum(values)/len(values))
                order_groups=[]
                for rare in (False,True):
                    subset=[r for r in rs if bool(r['prediction']['features'].get('event_order_2',0))==rare]
                    order_groups.append(dict(order2_present=rare,n_paths=len(subset),residual=None if not subset else summary([r['residual_rz'] for r in subset]),absolute_relative_error=None if not subset else summary([r['absolute_relative_error'] for r in subset])))
                groups.append(dict(stage=filename,family=family,axis=axis,n_paths=n,
                    expanded_rz=summary([r['prediction']['expanded_rz_after_basis_cancellation'] for r in rs]),
                    folded_rz=summary([r['prediction']['structural_rz'] for r in rs]),
                    all_underpredicted=all(r['residual_rz']>0 for r in rs),
                    feature_ranges={k:summary([r['prediction']['features'].get(k,0) for r in rs]) for k in sorted(keys)},
                    expanded_role_ranges={k:summary([r['prediction']['native_rz_by_role'].get(k,0) for r in rs]) for k in sorted(roles)},
                    order2_groups=order_groups,
                    causal_attribution='Not identified: analytic decomposition/Z-fold and original compiler are different; no causal fit or ablation performed'))
    payload=dict(model_source=completed['model_source'],analysis_source=source,analysis_script_sha256=digest(Path(__file__).read_bytes()),input_sha256=inputs,groups=groups,
        no_cohort_or_axis_pooling=True,new_predictions=0,fit_bootstrap=0,compiled_circuit_decoded=False,N_or_G=0,rare_tail_bound=None,causal_correction_model=None,mandatory_stop=True)
    with output.open('x') as f:json.dump(payload,f,ensure_ascii=False,indent=2);f.write('\n')
    print('Descriptive structure/error analysis completed; no price prediction rerun or fit.')


if __name__=='__main__':main()
