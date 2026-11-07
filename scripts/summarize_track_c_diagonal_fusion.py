#!/usr/bin/env python3
"""Saved-only quantitative opportunity, ablation and all-round summaries."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/track_c_diagonal_fusion/2026-11-08'
METRICS=('RZ','CX','serial_depth_upper','size')


def read(n):return json.loads((OUT/n).read_text())
def write(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    if (OUT/'FINAL.json').exists():raise RuntimeError('Do not overwrite FINAL')
    resource=read('RESOURCES.json');frozen=read('POLICY.json');cells={}
    for path in sorted((OUT/'holdout_prices').glob('*.json')):
        r=json.loads(path.read_text());key=f'{float(r["t"]).hex()}_{r["n"]}'
        c=cells.setdefault(key,dict(t=r['t'],n=r['n'],rotations=0.,unique_rotations=0.,product_wires=0.,odd_product_wires=0.,runs=0.,metrics={k:[0.]*4 for k in ('reference','term_parity','merge','network','selected')}))
        w=r['probability']/16
        for s in r['stats']:
            c['runs']+=w
            for k in ('rotations','unique_rotations'):c[k]+=w*s[k]
            c['product_wires']+=w*s['products'];c['odd_product_wires']+=w*s['odd_product_wires']
        for k in c['metrics']:
            values=r['selected'] if k=='selected' else r['variants'][k]
            for i in range(4):c['metrics'][k][i]+=w*values[i]
    write('OPPORTUNITY.json',dict(cells=cells,interpretation='Probability-weighted independent holdout moments. Raw rare-stratum frequency is not a physical frequency. Diagonal duplicate counts are logical; total compiled prices already include existing A cancellation. Known single-term control is separate from new multi-operator fusion.'))
    pr=next(x for x in resource['policies'] if x['family']=='PR' and x['primary']=='RZ');full=next(x for x in resource['policies'] if x['family']=='Full' and x['primary']=='RZ')
    term=next(x['totals']['total'] for x in pr['fixed_new_shot_ablations'] if x['mode']=='term_parity')
    main_new=pr['new']['totals']['total'];base=pr['baseline']['totals']['total']
    final=dict(base='2745d57bea5c28604739bf9c01338e14e649c467',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        scientific_source_commits=['62fe68b07cd638aad12ded90bc97922ba94a5589','87c2f29f5526d66e0d1444ece271d639366bfd9a'],
        scope='Linear H4/1 Angstrom/STO-3G/DF12, Partial LD3 Full LD12, saved original-H exact ground,18 rounds delta0 .02,epsilon1.5936e-4Ha,failure.05,phase.4,ideal native,fresh independent bits,preparation excluded.',
        metrics=list(METRICS),policies=resource['policies'],PR_primary_reductions=[(base[i]-main_new[i])/base[i] for i in range(4)],
        PR_ratio_to_matched_Full=[main_new[i]/full['new']['totals']['total'][i] for i in range(4)],
        PR_collective_fusion_reduction_from_known_term=[(term[i]-main_new[i])/term[i] for i in range(4)],
        training_samples=384,independent_holdout_samples=384,deterministic_cells=29,run_joint_operator_checks=33,D_joint_operator_checks=9,short_axis_checks=8,artificial_tests='3 passed',
        source_rounding_proof='Exact binary64 rational aggregation. Controlled RZ(theta)=RZ_S(theta/2)RZ_{aS}(-theta/2). All |theta|<4; total per-trajectory error<=2*2^-53*N+N*2^-1074. D half-angle sums have exactly0 rounding error in all fixed groups. Retain prior all128-context basis/native/isometry/source/finite-bit certificate; add only new rounding raw disk.',
        reused_qualification='artifacts/track_c_joint_isometry/2026-11-08/CERTIFICATE.json plus POLICY and earlier D qualification; all older evidence preserved.',
        invariants=['RTE sampling law/return aggregation unchanged','D BCH coefficients/h/resolution and canonical signals unchanged','only diagonal operations inside same source basis reordered algebraically','joint common global phase only; controlled relative phases retained','both PR and Full receive identical synthesis/fallback and cost-sensitive alpha opportunities'],
        input_cache_sha256=hashlib.sha256((ROOT/'.venv/common.pkl').read_bytes()).hexdigest(),
        limitations=['Conditional saved Hermitian-H model, not unconditional continuum-H inclusion','RZ/CX/size exact additive for concrete modular execution; total depth serial upper, not m17 full DAG','Price MC SE distinct from source/native rounding disk; no whole-m17 extrapolation model','Known single-Pauli control and known GraySynth are not new inventions; primary novelty/performance evidence is isolated in ablations','No T synthesis, quantum backend shots, external reproduction or detailed GPT scientific review'],mandatory_STOP=True)
    write('FINAL.json',final)
    rows=[]
    for p in resource['policies']:
        f=next(x for x in frozen['policies'] if (x['family'],x['primary'])==(p['family'],p['primary']))
        for s,row in zip(f['selected'],p['new']['rounds']):
            x=s['signal'];q=s['qualification'];r=dict(family=p['family'],primary=p['primary'],m=x['m'],name=x['name'],h=x['h'],resolution=x['rate'],shots_X=row['shots'][0],shots_Y=row['shots'][1],rho_lower=q['rho_lower'],beta_stat_lower=q['beta_stat_lower'],fusion_disk=q['fusion_raw_disk_upper'])
            for k in ('D','R','wrapper','total'):
                for i,metric in enumerate(METRICS):r[k+'_'+metric]=row[k][i]
            rows.append(r)
    with (OUT/'ROUND_RESOURCES.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    print('DIAGONAL_SAVED_SUMMARY_COMPLETE')


if __name__=='__main__':main()
