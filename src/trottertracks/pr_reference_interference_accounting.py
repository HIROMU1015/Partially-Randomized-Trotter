"""Saved-shot accounting only, called iff the finite A3 gate passes."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
from .pr_reference_interference_pilot import OUT, METRICS, stored, save_json

OLD_ROUNDS='artifacts/track_c_guarantee_cn_price/2026-11-06/A_guarantee/verified/ROUNDS.json'


def account():
    price=json.loads((OUT/'A3_PRICES.json').read_text())
    if not price['pass_to_A4']:raise RuntimeError('A3 did not authorize A4')
    original=next(x for x in json.loads(stored(OLD_ROUNDS)) if x['name']=='CN_S2_h0.02_ld3')
    observed={x['q']:x for x in price['rows']}
    rows=[];subtotals={name:np.zeros(4) for name in ('existing_direct','Simon_Love','reference')}
    for old in original['rows']:
        q=old['q'];r=old['r'];n=old['shots_per_axis']
        known=r==1 and q in observed
        costs=None
        if known:
            costs={name:(n*np.sum(values,axis=0)).tolist() for name,values in observed[q]['means'].items()}
            for name,values in costs.items():subtotals[name]+=values
        rows.append(dict(m=old['m'],q=q,r=r,
            conditional_ideal_shots_per_axis=n,
            saved_raw_radius_lower=old['raw_radius_lower'],
            saved_phase_residual_lower=old['statistical_phase_lower'],
            saved_raw_extra_disk_upper=old['raw_extra_disk_upper'],
            old_disk_not_added_again=True,
            signal_units='raw attenuated Hadamard signal; unchanged normalization',
            native_reference_disk_upper=None,native_Simon_Love_disk_upper=None,
            new_phase_correction_rounding_disk_upper=None,
            native_qualified_shots_per_axis=None,
            total_cost_if_prices_observed=costs,
            price_status='measured conditional expectation (MC)' if known else 'NOT_ACQUIRED'))
    raw=[json.loads(p.read_text()) for p in sorted((OUT/'prices').glob('*.json'))]
    differences={}
    for baseline in ('existing_direct','Simon_Love'):
        variance=np.zeros(4);diff=np.zeros(4)
        for row in rows:
            if row['total_cost_if_prices_observed'] is None:continue
            q=row['q'];n=row['conditional_ideal_shots_per_axis']
            for rare in (False,True):
                cohort=[x for x in raw if x['q']==q and x['rare']==rare]
                p=cohort[0]['probability']
                d=np.array([n*np.sum(np.array(x['values']['reference'])-x['values'][baseline],axis=0) for x in cohort])
                diff+=p*d.mean(axis=0);variance+=p*p*d.var(axis=0,ddof=1)/len(cohort)
        differences[baseline]=dict(reference_minus_baseline=diff.tolist(),
            paired_MC_SE=np.sqrt(variance).tolist(),
            diagnostic_95_upper=(diff+1.96*np.sqrt(variance)).tolist())
    result=dict(status='UNDETERMINED_ALL18_NATIVE_COST',
        scope='same saved S2/LD3/h=.02 finite-law model and old shot schedule; no new energy/shot success certification',
        original_rounds_ref='602ca4b67b1037e221e910a2733444630c5add62:'+OLD_ROUNDS,
        ideal_signal_identity='same trajectory and law; phase-corrected vacuum reference; fresh shots',
        conditional_total_shots=original['total_shots'],
        old_failure_allocation='2/1440 per axis, 36 axes; sum .05; inherited only for old model',
        phase_budget=.4,energy_accuracy_Ha=1.5936e-4,
        extra_reference_resets_if_all_rounds_use_same_gadget=8*original['total_shots'],
        source_CSWAPs_if_all_rounds_use_same_gadget=16*original['total_shots'],
        rows=rows,metrics=list(METRICS),
        measured_round_indices=[x['m'] for x in rows if x['total_cost_if_prices_observed'] is not None],
        measured_round_subtotals={k:v.tolist() for k,v in subtotals.items()},
        measured_round_paired_differences=differences,
        all18_native_G={name:None for name in METRICS},
        all18_T=None,
        missing='15 round prices incl q4; r>1 and long q; all-support new compiler/phase disk; T synthesis prices',
        guarantee='No missing value is zero. Measured subtotal is not a feasible complete RPE baseline price.',
        conclusion='Short-round CX/depth opportunity, if observed, does not establish total RPE improvement.')
    save_json(OUT/'A4_RPE_ACCOUNTING.json',result)
    with (OUT/'A4_ROUNDS.tsv').open('w') as f:
        f.write('m\tq\tr\tconditional_shots_per_axis\told_raw_radius_lower\told_phase_residual_lower\tprice_status\tnew_native_disk_upper\tfull_G\n')
        for row in rows:
            f.write('\t'.join(map(str,[row['m'],row['q'],row['r'],row['conditional_ideal_shots_per_axis'],
                row['saved_raw_radius_lower'],row['saved_phase_residual_lower'],row['price_status'],'UNKNOWN','UNKNOWN']))+'\n')
    return result
