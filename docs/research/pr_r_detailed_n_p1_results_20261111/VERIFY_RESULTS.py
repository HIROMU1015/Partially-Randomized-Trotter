"""Read-only P1 result/identity/provenance checks. No action/N/C computation."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent

def verify():
    raw=json.loads((OUT/'run01/result.json').read_bytes())
    hand=json.loads((OUT/'run01/N_HANDOFF.json').read_bytes())
    plan=json.loads(subprocess.check_output(['git','show',raw['plan_source']['commit']+':'+raw['plan_source']['path']],cwd=ROOT))
    expected=[r for r in plan['rows'] if r['package']=='P1_SHORT_MIDDLE']
    assert raw['status']=='P1_COMPLETE_CONDITIONAL_N' and raw['errors']==[]
    assert (OUT/'EXIT_CODE.txt').read_text().strip()=='0'
    assert raw['source_preserved'] and raw['input_preserved']
    assert raw['records']==hand['shots'] and hand['status']==raw['status']
    assert hand['comparison_policy']=='R_AWARE_EMPIRICAL_ACTION_COUNT_V1'
    assert hand['prices']==[] and hand['total_resources'] is None
    assert len(raw['records'])==len(expected)==12
    s=raw['scope_observed']
    assert s['saved_baseline_scalar']==3 and s['scalar_candidates_completed']==12
    assert s['finite_mean_attempts']==s['exact_tail_controls_completed']==9
    assert s['new_reference_actions_attempted']==s['new_reference_actions_completed']==2
    assert sorted(s['context_systems'])==['H4','H6']
    assert set(raw['references'])=={'H4:0.1','H4:6.4','H6:1.6','H6:25.6'}
    assert raw['references']['H6:1.6']['source']['commit']==raw['scientific_source']
    assert raw['price_calls']==raw['G_evaluations']==0
    assert not raw['actual_N_qualified'] and not raw['full_RPE_qualified'] and not raw['automatic_retry']
    for n,row in zip(raw['records'],expected):
        assert n['identity']==row['N_request']['identity'] and n['law_binding']==row['N_request']['law_binding']
        assert n['identity']['m']!=12 and n['plan']['status']=='CONDITIONAL_ROUND_PASS'
        assert n['N_comparison_policy']==n['diagnostics']['allowance']['policy']=='R_AWARE_EMPIRICAL_ACTION_COUNT_V1'
        assert n['source']['commit']==raw['scientific_source']
        assert n['actual']['N_X'] is None and n['actual']['N_Y'] is None
        for axis in ('X','Y'):
            x=n['plan']['axes'][axis];N=n['plan']['N_'+axis]
            assert type(N) is int and N>0 and N==x['shots']
            assert str(N)==n['diagnostics']['N_'+axis+'_decimal']
            assert x['radius_lower']>0 and x['beta_total']<=row['N_request']['identity']['configuration']['beta_total']
            assert x['alpha_axis']==row['N_request']['identity']['configuration']['alpha_axis']
    csv_rows=list(csv.DictReader((OUT/'run01/ROUND_N.csv').open()))
    assert len(csv_rows)==12
    for n,r in zip(raw['records'],csv_rows):
        assert r['candidate_id']==n['identity']['candidate']['id']
        assert r['N_X']==str(n['plan']['N_X']) and r['N_Y']==str(n['plan']['N_Y'])
        assert r['rho_lower']==str(n['plan']['axes']['X']['radius_lower'])
        assert r['beta_det']==str(n['plan']['axes']['X']['beta_deterministic'])
    for name,sha in raw['source_files'].items():
        data=(ROOT/name).read_bytes()
        assert hashlib.sha256(data).hexdigest()==sha
        assert data==subprocess.check_output(['git','show',raw['scientific_source']+':'+name],cwd=ROOT)
    protected=['docs/research/pr_r_detailed_n_preparation_20261111',
        'docs/research/pr_r_normalization_phase0_results_20261111',
        'docs/research/h4_df10_rpe_batch_results_20261110',
        'docs/research/h6_df19_rpe_m8_12_results_20261110']
    assert not subprocess.check_output(['git','diff',raw['scientific_source'],'--',*protected],cwd=ROOT)
    return dict(status='PASS_P1_RECORD_SOURCE_HANDOFF_CHECKS',scientific_source=raw['scientific_source'],
        records=12,conditional_round_PASS=12,actual_N_qualified=False,full_RPE_qualified=False,
        finite_means=9,exact_tail_controls=9,saved_baseline_scalar=3,new_reference_actions=2,
        source_files_preserved=len(raw['source_files']),input_hashes_preserved=len(raw['input_hashes']),
        canonical_identity_law_allocation_match=True,matched_allowance_policy=True,
        JSON_CSV_handoff_match=True,old_results_and_STOP_preserved=True,
        price_G_calls=0,scientific_retries=0,independent_scientific_review=False,mandatory_stop=True)

if __name__=='__main__':print(json.dumps(verify(),ensure_ascii=False,indent=2))
