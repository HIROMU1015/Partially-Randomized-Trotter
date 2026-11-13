"""Finite Full PF H_eff menu: metadata only; no scientific execution entry."""
import argparse
from copy import deepcopy
import csv
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import h4_pr_m1_rmse as existing

C_API='86f37d25ffd5eee3daf7f25ce001b090b1f2cf33'
B_API='437ba4acdb82708b6fc98e3cd119a2a734f7cf64'
D_API='e7dca5f7301c8633b17a59b6a7161331303aef2d'
BASE=dict(commit='c5466946ab0ef9b50511210ca1e58fb7147bb0f0',
    path='docs/research/baseline_finite_signal_plan_20261113/BASELINE_FINITE_MANIFEST.json')
CATALOG=dict(commit=D_API,path='docs/research/d_full_pf_full_h_v2_final_receipts_20261113/COVERAGE_V2.json')
ANCHOR='69ee0ff8898960d5f9959af8d34d89f223caec87'
ANCHOR_DIR='docs/research/h4_full_pf_heff_results_20261113/run01/'
FORMULAS=('2nd','4th','4th(m5_best)')
WIDTHS=(.05,.1,.2,.4,.8)


def read(ref):return json.loads(existing.blob(ROOT,ref['commit'],ref['path']))
def exact(v):return Fraction(v['numerator'],v['denominator'])
def encoded(v):return dict(numerator=v.numerator,denominator=v.denominator)


def api():
    # Reuse C's existing private B/D loader, not another canonical ID rule.
    scope=dict(__package__='trottertracks',hashlib=hashlib,types=types,sys=sys,Path=Path,
        blob=existing.blob,B_SOURCE=B_API,D_SOURCE=D_API,lru_cache=lru_cache)
    return existing.function(ROOT,C_API,'src/trottertracks/baseline_signal_pipeline.py',
        'published_api',scope)(str(ROOT))


def build():
    published=read(BASE);base=published['candidates'][0]['configuration'];a=api()
    catalog=read(CATALOG)['rows'];indexed={r['physical_candidate_id']:(i,r) for i,r in enumerate(catalog)}
    # These five widths come from the saved H4 Full PF q-grid, not fitted errors.
    for f in FORMULAS:
        domain={r['actual_width'] for r in catalog if r['system']=='H4' and r['formula']==f and r['T']==.8}
        if domain!=set(WIDTHS):raise ValueError('Published width-grid SOURCE differs')
    configs=[];cells={};unique_physical=set()
    for f in FORMULAS:
        for width in WIDTHS:
            spec=a.D.full_pf_spec(base['spec'],formula=f,delta_pf=encoded(Fraction(width)),M=12,rank=10)
            config=a.B.generate(spec);a.B._validate(config)
            physical=a.D.physical_schedule(spec);a.D._check_against_b(physical,config)
            label=f+' / '+str(width);rows=[]
            for r,pr in zip(config['rounds'],physical['rounds']):
                c=pr['physical_candidate'];actual=exact(r['actual_delta_pf'])
                if exact(r['T'])!=r['q_PF']*actual:raise ValueError('Actual step/power time differs')
                cell=f+' / '+str(float(actual));unique_physical.add(c['id'])
                hit=indexed.get(c['id']);price=dict(X='OUTSIDE_CHECKED_CATALOG',Y='OUTSIDE_CHECKED_CATALOG',source=None)
                if hit is not None:
                    i,record=hit
                    if not existing.p.same(record['physical_candidate'],c):raise ValueError('Same ID, different physical candidate')
                    price=dict(X=record['X'],Y=record['Y'],source=dict(**CATALOG,selector=['rows',i]),
                        binding_status=record['binding']['status'],new_consumer_acceptance=False)
                row=dict(m=r['m'],T=r['T'],T_float=c['T'],q_RPE=r['q_RPE'],q_PF=r['q_PF'],
                    actual_delta_pf=r['actual_delta_pf'],canonical_candidate_id=r['candidate']['id'],
                    physical_candidate=deepcopy(c),step_cell=cell,price_coverage=price,
                    signed_S2_piece_count=r['q_PF']*len(pr['signed_weights']),N=None,C=None,G=None)
                rows.append(row)
                if cell not in cells:
                    saved=f=='2nd' and actual==Fraction(.1)
                    cells[cell]=dict(label=cell,formula=f,actual_delta_pf=r['actual_delta_pf'],
                        signed_weights=pr['signed_weights'],signed_piece_count=len(pr['signed_weights']),
                        source_status='SAVED_DIAGNOSTIC_REFERENCE_ONLY' if saved else 'NEW_SCIENCE_REQUIRES_PERMISSION',
                        saved_source=dict(commit=ANCHOR,path=ANCHOR_DIR+'EIGENSYSTEM.json') if saved else None,
                        first_check=deepcopy(row),last_check=None,uses=[],delta_E_ha=None,rpe_bias_ha=None,qualified=False)
                cell_data=cells[cell];cell_data['uses'].append(dict(configuration=label,m=r['m']))
                if r['m']==12:cell_data['last_check']=deepcopy(row)
            configs.append(dict(label=label,formula=f,requested_delta_pf=spec['delta_pf'],configuration_id=config['id'],
                constructor=dict(api='D.full_pf_spec -> B.generate',base=BASE,base_selector=['candidates',0,'configuration','spec'],
                    formula=f,delta_pf=spec['delta_pf'],M=12,rank=10),rounds=rows,
                sufficient_N=None,global_RMSE_bound=None,G=None,selected=False))
    anchor=next(c for c in configs if c['formula']=='2nd' and exact(c['requested_delta_pf'])==Fraction(.1))
    if anchor['configuration_id']!=base['id']:raise ValueError('Existing canonical Full PF identity changed')
    new=[c for c in cells.values() if c['saved_source'] is None]
    assert len(configs)==15 and len(cells)==15 and len(new)==14
    return dict(status='PROPOSED_FINITE_HEFF_MENU_NOT_EXECUTED',sources=dict(C_API=C_API,B_API=B_API,D_API=D_API,
        base=BASE,price_catalog=CATALOG,C_price_evidence='3ae2e605dd6efcc63ed3742fe3076a8be3e6afae',
        numerical_PF='dd24dd329949c255d5e37b90ff8993fa8df7159b',existing_H4_anchor=ANCHOR),
        shared=dict(binding=base['spec']['binding'],profile=base['spec']['profile'],h=base['spec']['h'],M=12,P=0,
            rmse_policy=base['spec']['rmse_policy'],target=published['shared']['target'],
            numerical_policy_source=dict(commit='9af1ac8ed8adcaef059f439ce690b22b7a320de0',
                path='docs/research/h4_full_pf_heff_preparation_20261113/PLAN.json',selector=['numerical_policy']),
            cached_reference_eigensystem=dict(commit=ANCHOR,path=ANCHOR_DIR+'EIGENSYSTEM.json',selector=['reference_comparison'])),
        configurations=configs,step_cells=list(cells.values()),
        metadata_counts=dict(configurations=15,round_configuration_rows=195,unique_physical_round_candidates=len(unique_physical),
            unique_formula_actual_width_cells=15,saved_step_cells=1,new_step_cells=14),
        proposed_scientific_scope=dict(saved_context_load=1,new_PF_steps=14,new_effective_Schur=14,
            original_H_eigh=0,supplied_reference_eigensystem_with_residual_gate=True,
            direct_PF_complex_means=28,full_effective_spectral_complex_means=28,
            new_full_H_reference_actions=0,old_saved_signal_reanalysis=0,normalizers=0,N_MC_C_G=0,
            automatic_grid_extension=False,automatic_retry=False),
        actual_scientific_calls=0,saved_numerical_reanalysis=0,selected=None,global_RMSE_bound=None,G=None,mandatory_stop=True)


def write(out):
    plan=build();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with (out/'PLAN.json').open('x') as f:f.write(json.dumps(plan,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    with (out/'ROUND_SCHEDULE.csv').open('x',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(('configuration','m','T','q_RPE','q_PF','actual_delta','step_cell',
            'signed_S2_piece_count_NOT_compiled_RZ','canonical_candidate_id','physical_candidate_id','X_catalog_status','Y_catalog_status'))
        for c in plan['configurations']:
            for r in c['rounds']:
                writer.writerow((c['label'],r['m'],r['T_float'],r['q_RPE'],r['q_PF'],float(exact(r['actual_delta_pf'])),
                    r['step_cell'],r['signed_S2_piece_count'],r['canonical_candidate_id'],r['physical_candidate']['id'],
                    r['price_coverage']['X'],r['price_coverage']['Y']))
    return plan


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    value=write(p.parse_args().out)
    print(json.dumps(value['metadata_counts']),flush=True)
