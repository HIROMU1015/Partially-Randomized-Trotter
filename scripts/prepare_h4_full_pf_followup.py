"""Exact h-relative follow-up menu; metadata only, no numerical science leaf."""
import argparse
from copy import deepcopy
import csv
from fractions import Fraction
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import prepare_h4_full_pf_heff_scan as previous

PREVIOUS_RESULT='3ae055fa7a696b13320e00ec79a5ba15c4ed182e'
OLD_PLAN=dict(commit='878000d8d5d956b1d2f5513986ae40d0612e0bdd',
    path='docs/research/h4_full_pf_heff_scan_preparation_20261114/PLAN.json')
DIRECTORY='docs/research/h4_full_pf_heff_scan_results_20261114/run01/'
MENU=(('S2_16h_over_7','2nd',16,7,'PROPOSED_FIRST_BATCH'),
      ('S2_8h_over_3','2nd',8,3,'PROPOSED_FIRST_BATCH'),
      ('Y4_16h_over_3','4th',16,3,'PROPOSED_FIRST_BATCH'),
      ('Y4_32h_over_5','4th',32,5,'PROPOSED_FIRST_BATCH'),
      ('M5_16h','4th(m5_best)',16,1,'PROPOSED_FIRST_BATCH_WITH_EXPLICIT_WINDOW'),
      ('M5_32h','4th(m5_best)',32,1,'HOLD_BRANCH_POLICY_UNRESOLVED'))


def build():
    old=previous.read(OLD_PLAN);published=previous.read(previous.BASE)
    base=published['candidates'][0]['configuration'];h=previous.exact(base['spec']['h']);api=previous.api()
    known={}
    for i,cell in enumerate(old['step_cells']):
        source=cell['saved_source'] or dict(commit=PREVIOUS_RESULT,path=DIRECTORY+'cell_'+str(i).zfill(2)+'/EIGENSYSTEM.json')
        known[(cell['formula'],previous.exact(cell['actual_delta_pf']))]=source
    catalog=previous.read(previous.CATALOG)['rows'];prices={r['physical_candidate_id']:(i,r) for i,r in enumerate(catalog)}
    configurations=[];cells={}
    for label,formula,numerator,denominator,stage in MENU:
        width=h*Fraction(numerator,denominator)
        spec=api.D.full_pf_spec(base['spec'],formula=formula,delta_pf=previous.encoded(width),M=12,rank=10)
        config=api.B.generate(spec);api.B._validate(config)
        physical=api.D.physical_schedule(spec);api.D._check_against_b(physical,config)
        rows=[]
        for r,pr in zip(config['rounds'],physical['rounds']):
            delta=previous.exact(r['actual_delta_pf']);key=(formula,delta);c=pr['physical_candidate']
            times=[(c['T']*w/c['q']).hex() for w in pr['signed_weights']]
            hit=prices.get(c['id']);coverage=dict(X='OUTSIDE_CHECKED_CATALOG',Y='OUTSIDE_CHECKED_CATALOG',source=None)
            if hit is not None:
                i,record=hit
                if not previous.existing.p.same(record['physical_candidate'],c):raise ValueError('Same ID, different physical candidate')
                coverage=dict(X=record['X'],Y=record['Y'],source=dict(**previous.CATALOG,selector=['rows',i]),new_acceptance=False)
            if key not in cells:
                source=known.get(key);new=source is None
                branch=(dict(mode='ENERGY_WINDOW',purpose='EFFECTIVE_SPECTRUM_WINDOW',window=[-2.5,1.0],
                             status='PROPOSED_NOT_SCIENTIFICALLY_ADOPTED') if formula=='4th(m5_best)' and delta==16*h else
                        dict(mode=None,status='HOLD_NO_FULL_SPECTRUM_BRANCH_POLICY') if formula=='4th(m5_best)' and delta==32*h else
                        dict(mode='PRINCIPAL',status='CONVENTION_ONLY_ALIAS_UNQUALIFIED'))
                cells[key]=dict(label=formula+' @ '+str(delta/h)+'h',formula=formula,actual_delta_pf=r['actual_delta_pf'],
                    actual_delta_display=float(delta),signed_weights=pr['signed_weights'],signed_piece_times_hex=times,
                    saved_source=source,new_science_required=new,branch=branch,
                    execution_readiness='BRANCH_POLICY_REQUIRED' if branch['mode'] is None else 'AWAITING_USER_SCIENCE_PERMISSION' if new else 'SAVED_SOURCE_REFERENCE_ONLY',
                    uses=[],first_check=None,last_check=None,delta_E_ha=None,C=None,N=None,G=None,qualified=False)
            cell=cells[key]
            if cell['signed_piece_times_hex']!=times:raise ValueError('Same exact delta has a different backend signed-time recipe')
            row=dict(m=r['m'],T=r['T'],q_RPE=r['q_RPE'],q_PF=r['q_PF'],actual_delta_pf=r['actual_delta_pf'],
                physical_candidate=c,canonical_candidate_id=r['candidate']['id'],step_cell=cell['label'],
                signed_S2_piece_count_NOT_compiled_RZ=r['q_PF']*len(pr['signed_weights']),price_coverage=coverage)
            rows.append(row);cell['uses'].append(dict(configuration=label,m=r['m']))
            if cell['first_check'] is None:cell['first_check']=deepcopy(row)
            if r['m']==12:cell['last_check']=deepcopy(row)
        configurations.append(dict(label=label,formula=formula,h_multiplier=dict(numerator=numerator,denominator=denominator),
            requested_delta_pf=previous.encoded(width),display_only_delta=float(width),stage=stage,configuration_id=config['id'],
            constructor=dict(base=previous.BASE,base_selector=['candidates',0,'configuration','spec'],api='D.full_pf_spec -> B.generate',
                formula=formula,delta_pf=spec['delta_pf'],M=12,rank=10),rounds=rows,selected=False,sufficient_N=None,global_RMSE_bound=None,G=None))
    values=list(cells.values());ready=[c for c in values if c['new_science_required'] and c['branch']['mode'] is not None]
    hold=[c for c in values if c['new_science_required'] and c['branch']['mode'] is None]
    if len(ready)!=5 or len(hold)!=1:raise ValueError('Fixed follow-up scope differs')
    return dict(status='FOLLOWUP_PLAN_ONLY_NO_SCIENCE',sources=dict(previous_result=PREVIOUS_RESULT,previous_plan=OLD_PLAN,
        base=previous.BASE,C_API=previous.C_API,B_API=previous.B_API,D_API=previous.D_API,price_catalog=previous.CATALOG,
        latest_C_context='367dc98d832ecb4d769f2a3da3d174bc697331a5'),
        shared=old['shared'],configurations=configurations,step_cells=values,
        metadata_counts=dict(configurations=6,first_batch_configurations=5,round_rows=78,total_actual_step_cells=len(values),
            first_batch_new_actual_steps=5,held_new_actual_steps=1),
        proposed_first_science_batch=dict(context_load=1,new_PF_steps=5,new_effective_Schur=5,new_original_H_eigh=0,
            direct_PF_means=10,full_spectral_means=10,new_full_H_reference_actions=0,
            saved_M5_08_spectrum_window_compatibility_check=1,old_saved_signal_reanalysis=0,
            normalizers_N_MC_compiled_C_G=0,automatic_retry=False,automatic_extension=False),
        held_M5_32h=dict(status='TECHNICAL_BRANCH_POLICY_REQUIRED_NOT_IN_FIRST_SCIENCE_BATCH',
            physical_schedule_defined=True,energy_bias_policy_adopted=False,delta_E_ha=None,
            additional_operators_if_later_approved=1,additional_Schur_if_later_approved=1,
            possible_additional_branch_continuation_calls='UNSPECIFIED_MUST_BE_SEPARATELY_FIXED'),
        actual_scientific_calls=0,saved_numerical_reanalysis=0,new_compiled_price_calculations=0,
        sufficient_N=None,global_RMSE_bound=None,G=None,winner=None,mandatory_stop=True)


def write(out):
    plan=build();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with (out/'PLAN.json').open('x') as f:f.write(json.dumps(plan,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    with (out/'ROUND_SCHEDULE.csv').open('x',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(('label','stage','m','q_RPE','q_PF','actual_delta_numerator','actual_delta_denominator',
            'step_cell','signed_S2_pieces_NOT_compiled_RZ','canonical_candidate_id','physical_candidate_id','X_catalog_status','Y_catalog_status'))
        for c in plan['configurations']:
            for r in c['rounds']:
                writer.writerow((c['label'],c['stage'],r['m'],r['q_RPE'],r['q_PF'],r['actual_delta_pf']['numerator'],r['actual_delta_pf']['denominator'],
                    r['step_cell'],r['signed_S2_piece_count_NOT_compiled_RZ'],r['canonical_candidate_id'],r['physical_candidate']['id'],
                    r['price_coverage']['X'],r['price_coverage']['Y']))
    return plan


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    value=write(p.parse_args().out);print(json.dumps(value['metadata_counts']))
