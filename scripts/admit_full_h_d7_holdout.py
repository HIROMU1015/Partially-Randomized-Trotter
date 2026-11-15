"""Copy saved D7 holdouts and join the fixed Q3 allocation; no price arithmetic."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from trottertracks import price_implementation_guard as G
from admit_pr_df_ld1_q3_prices import equal, require

ROOT = Path(__file__).resolve().parents[1]
BASE = 'docs/research/full_h_d7_holdout_admission_plan_20261115'
D1 = '06956c64380107392f1b24fe54b9273f0f5ae1c0'
D_DIR = 'docs/research/d_full_h_d7_holdout_20261115/'
EXEC = '9cbabc617e95a81810924413c1b2b06ced329514'
RAW = 'b5fac0ee4eac83f027f883e5ea5630623a98bf29'
HANDOFF = dict(commit=D1, path=D_DIR+'HANDOFF_C.json')
COMPARISON = dict(commit=D1, path=D_DIR+'COMPARISON.json')
FROZEN = dict(commit='b5165f11bb63f9eb2a6bfe385ba75d8003b7e724',
              path='docs/research/full_h_d7_model_freeze_20261115/run01/FROZEN_MODELS.json')
PRIOR = dict(commit='f46fa120b4cddcaac3edcf21e3731a0d9020b2b8',
             path='docs/research/full_h_d7_model_freeze_20261115/INPUTS.json')
B = '6c5569c3479013b080b2ba53842585f1d670d3c9'
B_DIR = 'docs/research/full_h_kappa2_scale3o2_q3_results_20261115/'
B_JOIN = dict(commit=B, path=B_DIR+'C_RESULT_JOIN.json')
B_INPUT = dict(commit=B, path=B_DIR+'INPUTS.json')
ALIGNMENT = dict(commit='0b4b2d941cd669aaa45231e1236bc2d3158a9d12',
                 path='docs/research/h4_five_method_candidate_alignment_20261115/ALIGNMENT.json')
SHOTS = [96,90,84,78,71,65,57,51,45,38,32,26,18]
VERDICT = 'NOT_ASSESSED_NO_ADOPTED_TOLERANCE'
EXACT_KEYS = ('P', 'O', 'signed_error', 'signed_relative_error')


def at(ref, *selector):
    return dict(ref, selector=list(selector))


def check_receipt(row, price, evidence, compiled, manifest, publication, prediction, domain, prior):
    """Check saved producer evidence; never regenerate a circuit or a path."""
    equal({k:v for k,v in row.items() if k!='locators'}, price, 'Original axis receipt')
    equal(evidence['price'], price, 'Implementation/price binding')
    equal(compiled, price['compile_receipt']['record'], 'Original compiler record')
    equal((compiled['metrics']['rz_count'], compiled['fingerprint'], compiled['compiler']),
          (price['rz_count'], price['fingerprint'], domain['profile']['compiler']), 'RZ/fingerprint/compiler')
    for key in ('cell_id', 'axis', 'cohort', 'physical'):
        equal(row[key], prediction[key], 'Frozen physical/axis/cohort '+key)
    equal(row['seeds'], prediction['step_seeds'][row['index']], 'Pre-observation path seeds')
    equal(row['source']['commit'], EXEC, 'Execution SOURCE')
    equal(row['source']['step_seeds'], prediction['step_seeds'], 'Original step seed table')
    equal(row['locators']['price']['commit'], RAW, 'Raw publication SOURCE')
    equal(row['source'], row['locators']['execution_source'], 'Receipt/producer SOURCE')
    equal(row['raw_path_selector'], row['locators']['decoded_path_selector'], 'Decoded raw path')
    equal(row['gate_status'], 'PASS', 'Recorded acquisition gates')
    equal(row['correctness_scope'], 'NUMERIC_7_ON_INDEX0;PARITY_SCALAR_EVENT_XY_ALL_PATHS', 'Original gate scope')
    for key in ('scalar', 'preparation_hash', 'sampling'):
        equal(evidence[key], prior[key], 'Previous preparation binding '+key)
    equal(evidence['measurement'], dict(axis=row['axis'], include_measurement=True,
          state_preparation_cost_included=False), 'Measured axis and P0')
    checks=evidence['checks']
    equal(checks['builder_lowering_id'], domain['implementation']['lowering_id'], 'New ZZ lowering')
    equal(checks['legacy_rzz_control_instructions'], 0, 'No legacy ZZ substitution')
    for gate in (checks['parity'], checks['scalar_event_phases'], evidence['xy_phase_gate']):
        equal(gate['status'], 'PASS', 'Recorded parity/phase/XY gate')
    equal(checks['scalar'], checks['scalar_expected'], 'Recorded scalar gate')
    equal(checks['scalar'], evidence['scalar'], 'Scalar evidence')
    require(evidence['xy_phase_gate']['same_controlled_evolution'] is True, 'Same controlled evolution')
    equal(len(evidence['trial_checks']), 7 if row['index']==0 else 0, 'Index0-only numerical gate scope')
    equal(publication['status'], 'PASS', 'D1 lossless publication validation')
    require(publication['actual_path_and_compile_evidence_checked'] is True, 'D1 raw/compile linkage')
    equal(manifest['raw_sha256'], publication['raw_sha256'], 'Raw digest in publication check')
    require(manifest['roundtrip_bytes_identical'] is True, 'Original lossless raw publication')


def check_comparison(unit, handoff_unit, prediction, receipts):
    """Return supplied exact quantities unchanged; no averaging/subtraction."""
    equal(unit, handoff_unit, 'D1 comparison and handoff')
    for key in ('cell_id', 'axis', 'physical', 'cohort', 'step_seeds', 'n_paths'):
        equal(unit[key], prediction[key], 'Frozen comparison '+key)
    equal(unit['P'], prediction['predicted_RZ'], 'Frozen exact prediction')
    equal(unit['status'], 'VALID', 'D1 descriptive unit valid')
    equal(unit['model_verdict'], VERDICT, 'No model verdict')
    require(unit['tolerance'] is None, 'No adopted numerical tolerance')
    equal(sorted(r['original_record']['index'] for r in receipts), list(range(prediction['n_paths'])), 'Original path count')
    require(all(r['status']=='ACCEPTED_SAVED_RECEIPT' for r in receipts), 'All axis receipts accepted')
    equal(unit['observation_sources'], [r['original_record']['locators'] for r in receipts], 'Comparison observation locators')
    return {k:deepcopy(unit[k]) for k in EXACT_KEYS}


def join_rows(case, joined, alignment, frozen, comparisons, admissions):
    equal(joined['model_status'], 'MODEL_BOUND_MEETS_ADOPTED_THRESHOLD', 'B conditional Q3 status')
    equal(len(joined['rows']), 26, '26 B axes')
    equal({(r['m'],r['axis']) for r in joined['rows']}, {(m,a) for m in range(13) for a in ('X','Y')}, 'Unique B schedule')
    rows=[]
    for i,b in enumerate(joined['rows']):
        m,axis=b['m'],b['axis']; source=next(r for r in case['rows'] if r['m']==m)
        p=source['physical_candidate']
        equal(p['id'], b['physical_id'], 'B signal/Q3 physical candidate')
        equal((p['family'],p['L_D'],p['K']), ('FULL_H_PURE_RTE',0,2), 'Full-H, not DF prefix0')
        equal(b['N'], SHOTS[m], 'Fixed 1.5x shot allocation')
        equal(joined['allocation']['shots'][m], dict(m=m,N_X=SHOTS[m],N_Y=SHOTS[m]), 'B allocation vector')
        ai=next(k for k,r in enumerate(alignment['rows']) if r.get('physical_id')==p['id'] and r['axis']==axis)
        old=alignment['rows'][ai]
        equal(old['physical_candidate'], p, 'Original alignment physical descriptor')
        equal(old['configuration_id'], joined['original_configuration_id'], 'Original configuration')
        equal(old['original_canonical_id'], b['original_candidate_id'], 'Original canonical ID')
        equal(old['decoder']['total_canonical_id'], b['total_candidate_id'], 'Total decoder-only bridge')
        model=frozen['models'][axis]
        current=[k for k,u in enumerate(comparisons) if u['m']==m and u['axis']==axis]
        for k in current:
            equal(comparisons[k]['physical_id'], p['id'], 'Holdout and B physical candidate')
        receipt_refs=[['receipt_admissions',k] for k,r in enumerate(admissions)
                      if r['original_record']['physical']['id']==p['id'] and r['original_record']['axis']==axis]
        rows.append(dict(m=m,axis=axis,physical_id=p['id'],physical_candidate=deepcopy(p),
            T_exact=deepcopy(source['T']),q_grouping=p['q'],q_PF=None,R_outer=p['R_outer'],event_positions=p['s'],
            original_configuration_id=joined['original_configuration_id'],total_configuration_id=joined['total_configuration_id'],
            original_candidate_id=b['original_candidate_id'],total_candidate_id=b['total_candidate_id'],binding_id=joined['binding_id'],
            allocation_id=joined['allocation']['allocation_id'],N=b['N'],sufficient_N=None,
            Q3_status=joined['model_status'],B_join=at(B_JOIN,'candidates',0,'rows',i),
            signal_round_source=deepcopy(source['raw_record_source']),law_selectors=deepcopy({k:b[k] for k in ('original_law_selector','total_law_selector')}),
            B_physical_source=at(B_INPUT,'cases',0,'rows',m),C_alignment=at(ALIGNMENT,'rows',ai),
            frozen_model=at(FROZEN,'models',axis),calibration_sources=deepcopy(model['input_locators']),
            calibration_role='FIVE_NONCONFIGURATION_CELLS_NOT_THIS_ROUND',
            holdout_admissions=[['comparisons',k] for k in current],receipt_admissions=receipt_refs,
            previous_price_evidence=deepcopy(old['price']),
            model_application='SAVED_FROZEN_PREDICTION_DESCRIPTIVE_ONLY' if current else 'UNVALIDATED_TRANSFER_NOT_EVALUATED',
            qualification='MATCHED_BUT_UNQUALIFIED',qualified_for_candidate=False,E_C=None,CI=None,G=None))
    return rows


def build(root):
    h=G.evidence(root,HANDOFF);cmp=G.evidence(root,COMPARISON)
    frozen=G.evidence(root,FROZEN);prior=G.evidence(root,PRIOR)
    case=G.evidence(root,B_INPUT)['cases'][0];joined=G.evidence(root,B_JOIN)['candidates'][0]
    alignment=G.evidence(root,ALIGNMENT)
    equal(h['frozen_model_source'], FROZEN, 'Original frozen SOURCE')
    equal(h['status'], 'DESCRIPTIVE_COMPLETE', 'D1 campaign complete')
    for obj in (h,cmp):
        equal(obj['model_verdict'],VERDICT,'Model remains unassessed')
        require(obj['tolerance'] is None,'Tolerance remains null')
    equal(case['scalar']['total'], prior['path_axes'][0]['scalar'], 'B signal and D1 scalar')
    equal(case['finite_K'],2,'B finite K2')
    equal(len(h['individual_axis_receipts']),6,'Six saved axis receipts')
    equal({(r['cell_id'],r['index'],r['axis']) for r in h['individual_axis_receipts']},
          {(cell,i,a) for cell,n in (('M5_ROUND',2),('M6_ROUND',1)) for i in range(n) for a in ('X','Y')}, 'Three shared paths, six axes')
    admissions=[]
    for i,r in enumerate(h['individual_axis_receipts']):
        item=dict(source=at(HANDOFF,'individual_axis_receipts',i),original_record=deepcopy(r),status='HELD',reason=None)
        try:
            loc=r['locators'];raw=G.C.blob(root,loc['compile_record']['commit'],loc['compile_record']['path'])
            equal(hashlib.sha256(raw).hexdigest(),loc['compile_record_sha256'],'Published compiler receipt hash')
            pred=next(p for p in frozen['predictions'] if p['cell_id']==r['cell_id'] and p['axis']==r['axis'])
            check_receipt(r,G.evidence(root,loc['price']),G.evidence(root,loc['implementation_evidence']),
                json.loads(raw),G.evidence(root,loc['raw_receipt']),
                next(p for p in h['publication_checks'] if p['cell_id']==r['cell_id']),pred,
                frozen['models'][r['axis']]['domain'],prior['path_axes'][0])
            item['status']='ACCEPTED_SAVED_RECEIPT'
            item['numerical_gate_scope']='SEVEN_ORIGINAL_GATES_INDEX0' if r['index']==0 else 'NO_NUMERICAL_GATE_INDEX1'
        except (KeyError,ValueError,OSError) as error:
            item['reason']=str(error)
        admissions.append(item)
    comparisons=[]
    for i,u in enumerate(cmp['rows']):
        pi=next(k for k,p in enumerate(frozen['predictions']) if p['cell_id']==u['cell_id'] and p['axis']==u['axis'])
        pred=frozen['predictions'][pi]
        item=dict(m=pred['m'],axis=u['axis'],physical_id=u['physical']['id'],status='HELD',reason=None,
                  frozen_prediction=at(FROZEN,'predictions',pi),comparison=at(COMPARISON,'rows',i),
                  model_verdict=VERDICT,tolerance=None)
        try:
            equal(u['prediction_source'],item['frozen_prediction'],'D1/C prediction locator')
            item['saved_exact_comparison']=check_comparison(u,h['comparison_rows'][i],pred,
                [r for r in admissions if r['original_record']['cell_id']==u['cell_id'] and r['original_record']['axis']==u['axis']])
            item['display_signed_percent']=u['display_signed_percent']
            item['status']='ACCEPTED_SAVED_DESCRIPTIVE_COMPARISON'
        except (KeyError,ValueError) as error:
            item['reason']=str(error)
        comparisons.append(item)
    rows=join_rows(case,joined,alignment,frozen,comparisons,admissions)
    nr=sum(r['status']=='ACCEPTED_SAVED_RECEIPT' for r in admissions)
    nc=sum(r['status']=='ACCEPTED_SAVED_DESCRIPTIVE_COMPARISON' for r in comparisons)
    return dict(schema='FULL_H_D7_C_HOLDOUT_ADMISSION_AND_Q3_JOIN_V1',
        sources=dict(D1_final=HANDOFF,D1_execution=EXEC,D1_raw=RAW,comparison=COMPARISON,frozen=FROZEN,
                     prior_calibration=PRIOR,B_join=B_JOIN,B_input=B_INPUT,alignment=ALIGNMENT),
        status='DESCRIPTIVE_COMPLETE' if (nr,nc)==(6,4) else 'PARTIAL_ADMISSION',
        receipt_counts=dict(requested=6,accepted=nr,held=6-nr),comparison_counts=dict(requested=4,accepted=nc,held=4-nc),
        shared_fresh_paths=3,joined_axes=len(rows),model_verdict=VERDICT,tolerance=None,
        receipt_admissions=admissions,comparisons=comparisons,rows=rows,
        calibration_input_locators={a:deepcopy(frozen['models'][a]['input_locators']) for a in ('X','Y')},
        paired_calibration_dependence=deepcopy(frozen['models']['X']['paired_dependence']),
        B_law=deepcopy(case['law']),B_scalar=deepcopy(case['scalar']),B_physical_binding=deepcopy(case['physical_binding']),
        D1_publication_checks=deepcopy(h['publication_checks']),D1_resource_observations=deepcopy(h['resources']),
        qualified_E_C_axes=0,E_C=None,CI=None,law_distance=None,N_sufficient=None,G=None,winner=None,
        new_scientific_calculations=0,mean_or_error_or_prediction_recomputed=False,refit=False,
        next_work='PROPOSED_ZERO_COMPILE_FIXED_SHOT_MODEL_DIAGNOSTIC_REQUIRES_SEPARATE_EXECUTION_PERMISSION',mandatory_stop=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    require(not args.out.exists(),'Reuse published admission; do not overwrite')
    result=build(ROOT);args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'ADMISSION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','receipt_counts','comparison_counts','joined_axes','new_scientific_calculations')}))


if __name__=='__main__':
    main()
