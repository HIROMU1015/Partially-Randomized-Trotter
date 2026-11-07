#!/usr/bin/env python3
"""Saved-only resource/run summaries; no Hamiltonian or circuit computation."""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/track_c_joint_isometry/2026-11-08'
METRICS=('RZ','CX','serial_depth_upper','size')


def read(name):return json.loads((OUT/name).read_text())
def write(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    if (OUT/'FINAL.json').exists():raise RuntimeError('Do not overwrite FINAL')
    resource=read('RESOURCES.json');policy=read('POLICY.json');scans=read('PILOT.json')['scans'];cat=read('CATALOGUE_FINAL.json')['entries']
    bycell={}
    for scan in scans:
        key=f'{float(scan["t"]).hex()}_{scan["n"]}';cell=bycell.setdefault(key,dict(t=scan['t'],n=scan['n'],histogram=defaultdict(float),full_basis=[0.]*4,joint_basis=[0.]*4))
        w=scan['probability']/16
        for r in scan['runs']:
            cell['histogram'][(r['length'],len(r['support']))]+=w;e=cat[r['key']]
            for i in range(4):
                cell['full_basis'][i]+=w*e['full_basis_pair'][i]
                cell['joint_basis'][i]+=w*(e['joint_basis_pair'][i] if e['adopt'] else e['full_basis_pair'][i])
    for cell in bycell.values():
        cell['histogram']=[dict(length=k[0],columns=k[1],expected_runs=v) for k,v in sorted(cell['histogram'].items())]
        cell['expected_runs']=sum(r['expected_runs'] for r in cell['histogram'])
        cell['expected_applications']=sum(r['length']*r['expected_runs'] for r in cell['histogram'])
    held=defaultdict(list)
    for path in (OUT/'holdout_prices').glob('*.json'):
        r=json.loads(path.read_text());held[(f'{float(r["t"]).hex()}_{r["n"]}',r['rare'])].append(r)
    corrected=[]
    for r in resource['policies']:
        old=next(p for p in policy['old_policies'] if p['family']=='PR' and p['primary']==r['primary']);w=defaultdict(int)
        for s,row in zip(old['selected'],old['rounds']):
            for recipe in s['signal']['recipe']:w[f'{float(recipe["time"]).hex()}_{recipe["r"]}']+=sum(row['shots'])*s['signal']['steps']
        cov=[[0.]*4 for _ in range(4)]
        for (key,rare),rows in held.items():
            values=[[w[key]*rows[0]['probability']*x['baseline'][i] for x in rows] for i in range(4)]
            for i in range(4):
                for j in range(4):cov[i][j]+=statistics.covariance(values[i],values[j])/len(rows)
        # The legacy account helper pools j across cells as a CRN estimator.
        # These new seeds are independent by cell: use their sum covariance.
        base={**r['baseline'],'MC_SE':[math.sqrt(cov[i][i]) for i in range(4)],'covariance_MC':cov}
        corrected.append({**r,'baseline':base})
    unknown={}
    for (key,rare),rows in held.items():
        unknown.setdefault(key,0.)
        for r in rows:unknown[key]+=r['probability']/16*sum(x['key'] not in cat for x in r['used_runs'])
    write('RUN_ANALYSIS.json',dict(cells=bycell,holdout_expected_unregistered_runs=unknown,
        scope='Histogram uses exact stored stratum probabilities, not raw oversampled rare frequencies. Basis-only potential is isolated-template accounting; actual selected circuit gains are separate direct paired measurements. Unseen unions fall back, catalogue never extended on holdout.'))
    fp=next(x for x in resource['Full'] if x['primary']=='RZ');pr=corrected[0]
    final=dict(base='10bf5a4813559cea807fcd8c50f3b966407747d1',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        scope='Linear H4 1 Angstrom STO-3G DF12 LD3/Full12, saved exact ground,18 rounds delta0 .02, energy1.5936e-4Ha, failure.05 phase.4; ideal native/fresh bits conditional saved-H model; preparation excluded.',
        metrics=list(METRICS),policies=corrected,Full=resource['Full'],
        RZ_primary_reduction=[pr['paired_total_saving'][i]/pr['baseline']['totals']['total'][i] for i in range(4)],
        RZ_primary_ratio_to_Full=[pr['new']['totals']['total'][i]/fp['totals']['total'][i] for i in range(4)],
        ablation=read('ABLATION.json')['results'],certificate={k:v for k,v in read('CERTIFICATE.json').items() if k!='columns'},
        catalogue_entries=len(cat),catalogue_adopted=sum(e['adopt'] for e in cat.values()),
        training_samples=384,independent_holdout_samples=resource['new_holdout_samples'],representative_joint_checks=resource['representative_checks'],short_axis_checks=8,
        artificial_tests='3 passed',
        input_cache_sha256=hashlib.sha256((ROOT/'.venv/common.pkl').read_bytes()).hexdigest(),
        scientific_source_commits=['2f3ddd5566d62b99c38de79e85b77dd438e06068','bce0168fcd9b4f7a25ef9854416e05c9106c4117','b1f59fd50ffbbdd297a3fbc1d72751a509f94e24','2a5649cadffcb5cae54fff3551ce148d054a666f','30aa3a8f56eaab816251549b264ef31d2d063523'],
        reused_input_commits=['10bf5a4813559cea807fcd8c50f3b966407747d1','b9dacb639eabdd3cf499e89f8807ff50b68e23ef'],
        statistical_note='FINAL baseline covariance corrects legacy pooled-j helper to independent-cell covariance. New-policy and paired SE already use sum of cell covariance in RESOURCES; G, policies and shots unchanged.',
        invariants=['D BCH coefficients, h and resolution unchanged','RTE law, signs, event order, normalization, return aggregation unchanged','new catalogue frozen before independent holdout','joint common global phase allowed; controlled relative phase and X/Y maintained'],
        external_reproduction=False,T_synthesis=0,mandatory_STOP=True)
    write('FINAL.json',final)
    rows=[]
    for p in corrected:
        frozen=next(x for x in policy['policies'] if x['primary']==p['primary'])
        for s,row,brow in zip(frozen['selected'],p['new']['rounds'],p['baseline']['rounds']):
            x=s['signal'];q=s['qualification'];r=dict(primary=p['primary'],m=row['m'],name=x['name'],h=x['h'],resolution=x['rate'],shots_X=row['shots'][0],shots_Y=row['shots'][1],rho_lower=q['rho_lower'],beta_stat_lower=q['beta_stat_lower'])
            for i,metric in enumerate(METRICS):r['new_'+metric]=row['total'][i];r['baseline_'+metric]=brow['total'][i]
            rows.append(r)
    with (OUT/'ROUND_RESOURCES.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    print('SAVED_SUMMARY_COMPLETE')


if __name__=='__main__':main()
