"""New, exclusive grant for selected current-S2 comparisons; no legacy grant use."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'artifacts/resource_applicability/track_a_current_s2_theory_compiled/2026-10-22/run01'
OLD=ROOT/'docs/progress/track_a/figures/pr_pf_theory_compiled_ratio_20261022'
SPEC=importlib.util.spec_from_file_location('current_s2_runtime',ROOT/'src/trotterlib/current_s2_theory.py')
a=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(a)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def execute():
    grant=read(RAW/'grant.json'); plan=read(RAW/'execution_plan.json')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert grant['SOURCE']==source and grant['plan_sha256']==sha(RAW/'execution_plan.json')
    assert grant['user_instruction_sha256']==sha(RAW/'user_instruction.txt')
    assert grant['approved_by_user'] is True and grant['one_shot'] is True and grant['automatic_retry'] is False
    for e in plan['inputs']+plan['source_closure']:
        assert sha(ROOT/e['path'])==e['sha256'],e['path']
    out=RAW/'output';out.mkdir(exist_ok=False)
    write(RAW/'one_shot.marker.json',dict(SOURCE=source,grant_sha256=sha(RAW/'grant.json'),nonrefundable=True))
    start=time.monotonic();terminal=dict(status='STOP_FAILURE',SOURCE=source,automatic_retry=False)
    def reserve(kind,**kw):
        with (out/'ledger.jsonl').open('a') as f:
            f.write(json.dumps(dict(kind=kind,**kw))+'\n');f.flush()
            import os;os.fsync(f.fileno())
    try:
        previous=read(OLD/'figure_data.json'); selected=[]; selection=[]
        for r in previous['current']:
            row=dict(system=r['system'],epsilon=r['epsilon'],PR=r['PR_saved_record'],PF=r['PF_saved_record'],
                     PR_shots=r['PR_shots'],PF_shots=r['PF_shots'],PR_selection_source=r['PR_selection_source'],
                     PF_selection_source=r['PF_selection_source'],PR_shot_status=r['PR_shot_status'])
            if r['system']=='H4' and r['epsilon']=='0.001':
                reserve('H4_EPS_0p001_S2_PF_SELECTION',scope='Existing registered eligible Full/B0/B0I S2 only')
                src=read(OLD/'sources/h4_d.json');choices=[(i,c) for i,c in enumerate(src['rows'])
                    if str(c['epsilon'])=='0.001' and c['cell_id'].endswith('_2nd') and c['eligible'] is True and c['G_RZ_exact'] is not None]
                ci,winner=min(choices,key=lambda x:(F(x[1]['G_RZ_exact']),x[1]['cell_id']))
                chosen=next(x for x in r['PF_method_minima'] if x['candidate_id']==winner['cell_id'])
                assert chosen['cost_exact']==winner['G_RZ_exact']
                row['PF']=chosen;row['PF_shots']=winner['shots']
                row['PF_selection_source']=dict(label='h4_d',locator=f'/rows/{ci}',rule='EXISTING_REGISTERED_ELIGIBLE_S2_MINIMUM_COMPILED_G_THEN_ID',
                    commit='406f70570323366da185355550a3900388717fbb',sha256=sha(OLD/'sources/h4_d.json'))
                selection.append(dict(system='H4',epsilon='0.001',selected=winner,source=row['PF_selection_source'],
                    eligible_S2_candidates=len(choices),eligible_records=[dict(locator=f'/rows/{i}',**c) for i,c in choices],
                    exact_ties=[c['cell_id'] for _,c in choices if F(c['G_RZ_exact'])==F(winner['G_RZ_exact'])],new_candidate_generation=0))
            assert row['PR']['PF_order']==row['PF']['PF_order']=='S2'
            for field in ['hamiltonian_hash','state_hash','DF_rank','snapshot_hash','compiler_settings_hash','control','T','P','bias_mode','allowance_factor']:
                assert row['PR'][field]==row['PF'][field],field
            for role in ['PR','PF']:
                row[role+'_compiled_G']=row[role]['cost_exact']
                row[role+'_theory_G']=None
            row.update(a.fixed_pair_display(dict(compiled=row['PR_compiled_G'],theory=None),dict(compiled=row['PF_compiled_G'],theory=None)))
            row['status']='MISSING_NATIVE_PREPARATION_METADATA'
            row['reasons']=['ORIGINAL_ORDERED_BASIS_AND_DIAGONAL_METADATA_NOT_AVAILABLE_ON_THIS_HOST']
            if row['system'] in ['H6','H8']:
                row['reasons'].append('ORIGINAL_NEW_A_N8_AXIS_SHOTS_NOT_AVAILABLE_IN_PUBLIC_CSV')
            selected.append(row)
        write(out/'candidate_selection_audit.json',selection)
        write(out/'selected_conditions.json',selected)
        # Actual saved-reference accounting regression (not new current-G data).
        units=read(OLD/'sources/stage3_theory_rows.json')
        totals=read(OLD/'sources/stage3_shot_weighted_totals.json')
        shots=read(OLD/'sources/stage3_preserved_shots.json');regression=[]
        reserve('SAVED_S2_TOTAL_ACCOUNTING_REGRESSION',conditions=17,model='NATIVE_MIXED',factor='1.0')
        for i,t in enumerate(totals):
            if t['factor']!='1.0' or t['metric']!='RZ' or t['model']!='NATIVE_MIXED' or '_2nd' not in t['candidate_id'] or t['status']!='ELIGIBLE_SAVED_EMPIRICAL_SHOTS':continue
            u={r['axis']:{b:r[b]['rz'] for b in ['lower','upper']} for r in units if r['system']==t['system'] and r['candidate_id']==t['candidate_id'] and r['model']=='NATIVE_MIXED'}
            ns=next(x for x in shots if x['candidate_id']==t['candidate_id'] and x['system']==t['system'] and x['epsilon']==t['epsilon'] and x['factor']=='1.0')
            G=a.total_from_saved_shots(u,ns['shots'])
            assert F(G['lower'])==F(t['theory_G_lower']) and F(G['upper'])==F(t['theory_G_upper'])
            regression.append(dict(locator=f'/stage3_shot_weighted_totals/{i}',system=t['system'],candidate_id=t['candidate_id'],epsilon=t['epsilon'],
                model='NATIVE_MIXED',units=u,shots=ns['shots'],recomputed_theory_G=G,status='PASS',use_in_current_main=False))
        assert len(regression)==17
        write(out/'saved_theory_regression.json',regression)
        unavailable=[]
        unique={}
        for row in selected:
            for role in ['PR','PF']:
                c=row[role];key=(row['system'],c['candidate_id'])
                if key not in unique:
                    unique[key]=dict(system=key[0],candidate_id=key[1],role=role,DF_rank=c['DF_rank'],
                        hamiltonian_hash=c['hamiltonian_hash'],state_hash=c['state_hash'],
                        scientific_SOURCE=c['scientific_SOURCE'],status='NOT_RUN_MISSING_FIXED_INPUT',
                        missing_fields=['ordered original Gaussian basis operations','original block diagonal coefficients and their binary64 evaluation profile',
                                        'original saved scalar receipt' if c['method']=='B0+I' else 'original scalar/identity coefficients']+
                                       (['complete native finite-RTE component law and order probabilities'] if role=='PR' else []),
                        native_theory_counter_calls=0)
        unavailable=list(unique.values())
        write(out/'theory_tasks.json',unavailable)
        write(out/'theory_totals.json',dict(status='NO_CURRENT_THEORY_G_CREATED_MISSING_INPUT',current_theory_totals=[],
              unique_candidate_tasks=unavailable,actual_counter_evaluations=0,not_zero_cost=True))
        terminal.update(status='AVAILABLE_AUTHORIZED_WORK_COMPLETE_STOP_MISSING_FIXED_INPUT',selected_conditions=20,
            H4_S2_selection_passes=1,saved_reference_G_regression_conditions=17,complete_current_four_total_pairs=0,
            new_current_theory_G=0,new_real_counter_evaluations=0,compiled_ratio_display_conditions=20,
            new_sampling=0,new_signal=0,new_compile=0,new_Hamiltonian_DF=0,independent_confirmation=0,
            unique_current_candidate_tasks=len(unique),automatic_extra_stage=False)
    except BaseException as e:
        terminal['error']=repr(e);raise
    finally:
        terminal['wall_seconds']=time.monotonic()-start
        write(out/'terminal.json',terminal)
    return terminal


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true',required=True);p.parse_args()
    print(json.dumps(execute()))
