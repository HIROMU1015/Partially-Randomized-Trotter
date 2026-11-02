"""Read-only V3 metadata projection/check. No scientific module is imported.

Run: python3 validate_metadata.py --repo PATH_TO_REPOSITORY
All numeric scientific fields are copied from fixed JSON, never recomputed.
The optional --write-map writes only this preflight's CSV, not source evidence.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ENV = dict(os.environ, GIT_OPTIONAL_LOCKS='0', GIT_NO_LAZY_FETCH='1')
AXES = ('cosine', 'sine')


def packed(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True)


def a0_scope(ld, delta, r, k, q, axis, policy):
    """Classify the fixed A0 domain; not a numerical holdout evaluation."""
    if (ld, delta, r, k, policy) != (3, .02, 32, 2, 'support_run_le_1') or axis not in AXES:
        return 'NOT_APPLICABLE_TUPLE_OR_POLICY_MISMATCH'
    if q in (16, 32):
        return 'MATCHED_FIXED_Q16_Q32_HOLDOUT'
    return 'MATCHED_FAMILY_ONLY_NO_DIRECT_FINAL_ROUND_PRICE'


def classify(q, direct_q):
    if q in direct_q:
        return 'DIRECT_CALIBRATION_ANCHOR' if q in (1, 2) else 'DIRECT_FIXED_HOLDOUT'
    return 'UNMEASURED_WITHIN_TESTED_Q_SPAN' if q < max(direct_q) else 'BEYOND_MATCHED_FIXED_HOLDOUTS'


def project(inputs):
    a = inputs['OPT2JSON']
    physics = {k: a['configuration'][k] for k in
               ('molecule', 'geometry_angstrom', 'basis', 'n_qubits', 'df_rank')}
    records = []
    for ld, candidate, policy in ((3, 'Partial', 'support_run_le_1'),
                                  (12, 'Full', 'deterministic_tail_free')):
        best = a['coherent_reoptimization']['best_by_ld'][str(ld)]['best']
        for row in best['rounds']:
            q, r, k, m = (row[n] for n in ('q_m', 'r_m', 'K_m', 'round_index'))
            cell = f'ld{ld}:delta0.02:r{r}'
            direct = a['direct_compiled_rz_measurements'][cell]
            direct_q = tuple(int(x) for x in direct['q'])
            diag = (a['proxy_diagnostics'][f'delta0.02:r{r}'] if ld == 3
                    else a['deterministic_diagnostics']['delta0.02'])
            manifest_id = 'EXTMAN' if direct['source_partition'] == 'fresh32_extension' else 'INITIALMAN' if ld == 3 else 'LEGACY'
            cal_tasks = []
            if manifest_id != 'LEGACY':
                for t in inputs[manifest_id]['tasks']:
                    if (t['ld'], t['delta'], t['r']) == (ld, .02, r) and t['q'] in (1, 2):
                        cal_tasks.append({n:t[n] for n in ('ld','delta','r','k','q','seed')})
            for axis in AXES:
                ar = row['axes'][axis]
                dbase = f'/direct_compiled_rz_measurements/{cell}'
                rbase = f'/coherent_reoptimization/best_by_ld/{ld}/best/rounds/{m}'
                holdout_rows = []
                for hq in sorted(direct_q):
                    if hq in (1,2):
                        continue
                    d = direct['q'][str(hq)]
                    hr = {'q':hq, 'role':d['role'], 'sample_count':d['sample_count'],
                          'record':d['compiled_rz'][policy][axis],
                          'selector':f'{dbase}/q/{hq}/compiled_rz/{policy}/{axis}'}
                    if ld == 3:
                        hr['saved_relative_error'] = next(v['absolute_relative_error'] for v in diag['holdout_rows']
                            if (v['q'],v['axis'],v['metric'],v['policy']) == (hq,axis,'rz_count',policy))
                    else:
                        hr['saved_maximum_rz_holdout_error'] = diag['maximum_rz_holdout_error']
                    holdout_rows.append(hr)
                matched = direct['q'].get(str(q))
                related = []
                if ld == 3 and r == 32:
                    related = [dict(v) for v in inputs['A0']['fixed_holdout_rows']
                               if v['policy'] == policy and v['axis'] == axis]
                out = {
                    'row_id': f'{candidate}_m{m:02d}_{axis}', 'environment': 'LOCAL',
                    'candidate': candidate, 'ld': ld, 'round_index':m, 'q':q, 'r':r, 'K':k,
                    'delta_time':.02, 'saved_tau_m':row['tau_m'],
                    'physical_identity':packed(physics),
                    'snapshot_sha256':'13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b',
                    'state_price_scope':'state_preparation_excluded;V2_physical_H_ground_state_condition_retained',
                    'pf_and_control':'controlled_partial_S2;boundary_optimized;extract_identity_phase',
                    'wrapper':'complete_Hadamard_interrogation;include_measurement=True;8_system_plus_1_ancilla',
                    'compiler':packed(a['configuration']['compiler']), 'axis':axis, 'price_policy':policy,
                    'metric':'compiled_rz_count', 'saved_N':ar['shots'], 'saved_alpha':ar['alpha'],
                    'saved_model_unit_rz':ar['predicted_rz_count_per_interrogation'],
                    'saved_model_unit_calibration_SE':ar['propagated_calibration_standard_error'],
                    'saved_round_cost_BOTH_AXES_DO_NOT_SUM_AXIS_ROWS':row['compiled_rz_point_estimate'],
                    'saved_round_calibration_SE_conservative_sum_BOTH_AXES':row['calibration_standard_error_conservative_sum'],
                    'model_formula':'(2-q)*mean_q1+(q-1)*mean_q2',
                    'model_source':'DECISIONSRC:_pair_model;ABLATIONSRC:AffineAxisCostModel;PRESRC:affine_prediction_with_standard_error',
                    'source_partition':direct['source_partition'], 'calibration_source_id':manifest_id,
                    'calibration_tasks':packed(cal_tasks) if cal_tasks else 'NOT_APPLICABLE_DETERMINISTIC_LEGACY_PROVIDER',
                    'calibration_q1':packed(direct['q']['1']), 'calibration_q2':packed(direct['q']['2']),
                    'calibration_selector':f'{dbase}/q/{{1,2}}/compiled_rz/{policy}/{axis}',
                    'matched_family_holdouts':packed(holdout_rows),
                    'exact_tuple_direct_status':classify(q,direct_q),
                    'exact_tuple_direct_record':packed({'sample_count':matched['sample_count'], 'role':matched['role'],
                        'record':matched['compiled_rz'][policy][axis]}) if matched else 'NOT_FOUND_IN_FIXED_DIRECT_PRICE_EVIDENCE',
                    'exact_tuple_direct_selector':f'{dbase}/q/{q}/compiled_rz/{policy}/{axis}' if matched else 'NOT_FOUND',
                    'formula_domain':'CALIBRATION_ANCHOR' if q in (1,2) else 'EXTRAPOLATION_FROM_Q1_Q2_EVEN_IF_HOLDOUT_EXISTS',
                    'long_q_gt32':str(q > 32).lower(),
                    'saved_family_maximum_selected_RZ_discrepancy':diag['maximum_selected_rz_holdout_error'] if ld==3 else diag['maximum_rz_holdout_error'],
                    'A0_applicability':a0_scope(ld,.02,r,k,q,axis,policy),
                    'A0_related_saved_rows':packed(related) if related else 'NOT_APPLICABLE',
                    'shared_calibration_group':f'{cell}:{policy}:q1_q2_shared_across_rounds;cos_sin_paired',
                    'pairing':'same_trajectory_cos_sin_and_policies;not_independent_experiments' if ld==3 else 'deterministic_provider;sample_count_1_is_not_a_population_CI',
                    'unverified':'long_q_systematic_error;covariance_model;backend_noise;state_preparation;external_transfer',
                    'result_source_id':'OPT2JSON','result_commit':'4591b8d75a2a6a55c74a51225f0c61648023c8ee',
                    'result_url':'https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_full_opt2/2026-09-25/wp11_all_r_opt2_coherent_analysis_20260925_065827.json',
                    'result_selector':f'{rbase}/axes/{axis}',
                    'source_ids':'OPT2JSON;AUDIT;'+manifest_id+';OPT2SRC;OPT2EXTSRC;DECISIONSRC;ABLATIONSRC;PRESRC;COMPSRC6;LEGACY;A0',
                }
                records.append({n:str(v) for n,v in out.items()})
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--write-map', action='store_true')
    args = p.parse_args()
    index = json.loads((HERE/'EVIDENCE_INDEX.json').read_text())
    def read(item):
        name, e = item
        raw = subprocess.check_output(['git','-C',str(args.repo),'show',e['commit']+':'+e['path']],env=ENV)
        assert len(raw)==e['bytes']
        assert hashlib.sha256(raw).hexdigest()==e['sha256'], name
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==e['blob'], name
        return name,json.loads(raw) if e['path'].endswith('.json') else raw.decode()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        inputs=dict(pool.map(read,index.items()))
    expected=project(inputs)
    assert len(expected)==72
    assert len({r['row_id'] for r in expected})==72
    assert len({(r['candidate'],r['round_index'],r['axis']) for r in expected})==72
    assert inputs['OPT2JSON']['direct_compiled_rz_measurements']==inputs['AUDIT']['direct_compiled_rz_measurements']
    for r in expected:
        assert r['A0_applicability']!='MATCHED_FIXED_Q16_Q32_HOLDOUT', 'No final selected round is an exact A0 holdout tuple'
    mock_checks=[
        a0_scope(3,.02,1,2,16,'cosine','support_run_le_1').startswith('NOT_APPLICABLE'),
        a0_scope(3,.02,32,2,16,'cosine','support_run_le_1')=='MATCHED_FIXED_Q16_Q32_HOLDOUT',
        a0_scope(3,.02,32,2,131072,'sine','support_run_le_1')=='MATCHED_FAMILY_ONLY_NO_DIRECT_FINAL_ROUND_PRICE',
        a0_scope(12,.02,0,0,32,'sine','deterministic_tail_free').startswith('NOT_APPLICABLE'),
        a0_scope(3,.01,32,2,32,'cosine','support_run_le_1').startswith('NOT_APPLICABLE'),
        a0_scope(3,.02,32,2,32,'cosine','full_basis_shared').startswith('NOT_APPLICABLE'),
        classify(4,(1,2,8))=='UNMEASURED_WITHIN_TESTED_Q_SPAN',
        classify(16,(1,2,8))=='BEYOND_MATCHED_FIXED_HOLDOUTS',
    ]
    assert all(mock_checks)
    path=HERE/'PRICE_EVIDENCE_MAP.csv'
    if args.write_map:
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(expected[0]),lineterminator='\n');w.writeheader();w.writerows(expected)
    with path.open(newline='',encoding='utf-8') as f:
        actual=list(csv.DictReader(f))
    assert actual==expected, 'CSV must be a literal metadata projection, with no new prices or SEs'
    print(packed({'status':'TECHNICAL_METADATA_PASS','fixed_files':len(index),'price_axis_rows':len(actual),
                  'mock_scope_checks':len(mock_checks),'scientific_modules_imported':0,
                  'new_compile_sampling_fit_cost_evaluations':0}))


if __name__=='__main__':
    main()
