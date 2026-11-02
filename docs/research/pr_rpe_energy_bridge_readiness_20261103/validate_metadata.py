#!/usr/bin/env python3
"""Validate this documentary addendum using fixed Git objects and stdlib only.

No scientific module import, phase/error/shot arithmetic, sampling, or compile.
Numeric JSON literals are read as strings and compared without recomputation.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
STATES = {'DIRECT_SAVED_VALIDATION','EMPIRICAL_PROXY','CONDITIONAL_MODEL',
          'EXTRAPOLATED','ASSUMED_NOT_VALIDATED','UNRESOLVED','NOT_APPLICABLE'}

def load(b):
    return json.loads(b, parse_int=str, parse_float=str)

def pointer(value, path):
    for part in path.strip('/').split('/') if path else []:
        part = part.replace('~1', '/').replace('~0', '~')
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value

def key(row):
    return tuple(row[k] for k in ('q_m','r_m','K_m'))

def self_checks():
    # Artificial metadata labels only: no physical signal or numeric evaluation.
    a = dict(q_m='Q', r_m='R', K_m='K')
    assert key(a) == key(dict(a))
    for field in a:
        assert key(a) != key({**a, field:'different'})
    assert pointer({'x':[{'a/b':'null'}]}, '/x/0/a~1b') == 'null'
    assert load('{"x":0.125,"y":null}') == {'x':'0.125','y':None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evidence-repo', required=True,
                    help='Existing local repository containing fixed objects; no fetch')
    args = ap.parse_args()
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0', GIT_NO_LAZY_FETCH='1')
    index = json.loads((HERE/'EVIDENCE_INDEX.json').read_text())
    data = {}
    for ident, item in index['artifacts'].items():
        raw = subprocess.check_output(
            ['git','-C',args.evidence_repo,'cat-file','blob',item['blob']], env=env)
        spec = item['commit']+':'+item['path']
        oid = subprocess.check_output(
            ['git','-C',args.evidence_repo,'rev-parse',spec], env=env).decode().strip()
        assert oid == item['blob'], ident
        assert len(raw) == item['bytes'], ident
        assert hashlib.sha256(raw).hexdigest() == item['sha256'], ident
        if item['path'].endswith('.json'):
            data[ident] = load(raw)
    for check in index['source_file_hash_checks']:
        prov = data[check['result_id']].get('provenance', data[check['result_id']])
        assert prov['source_sha256'][check['source_path']] == check['recorded_sha256']
        assert check['match'] and check['recorded_sha256'] == check['carrier_sha256']
    model = data['OPT2JSON']; grid = data['ABLGRID']; abl = data['ABLJSON']
    def rows(name):
        with (HERE/name).open(newline='') as f:
            result = list(csv.DictReader(f))
        assert all(None not in row and all(v != '' for v in row.values()) for row in result)
        return result
    rounds = rows('ROUND_EVIDENCE_MAP.csv'); assumptions = rows('ASSUMPTION_EVIDENCE_MATRIX.csv')
    assert len(rounds) == 36 and len({r['row_id'] for r in rounds}) == 36
    assert len(assumptions) == 33 and len({r['item_id'] for r in assumptions}) == 33
    for row in rounds:
        c = model['coherent_reoptimization']['best_by_ld'][row['ld']]
        src = pointer(model,row['model_selector']); best = c['best']
        assert row['model_content_fingerprint'] == model['content_fingerprint']
        assert row['model_commit'] == index['artifacts']['OPT2JSON']['commit']
        assert load(row['compiler_json']) == model['configuration']['compiler']
        for field in ('q_m','r_m','K_m','round_index'):
            assert row[field] == src[field]
        for dst,field in [('model_attenuation','attenuation'),
                          ('model_radius_lower_bound','conservative_radius_lower_bound'),
                          ('model_finite_RTE_phase_bound','finite_rte_phase_bound'),
                          ('model_epsilon_coordinate','epsilon_coordinate'),
                          ('model_PF_phase_proxy','empirical_pf_phase_proxy'),
                          ('model_round_compiled_rz','compiled_rz_point_estimate'),
                          ('model_tau_m','tau_m')]:
            assert row[dst] == src[field], (row['row_id'],dst)
        for axis in ('cosine','sine'):
            assert row[axis+'_shots'] == src['axes'][axis]['shots']
            assert row['alpha_'+axis] == src['axes'][axis]['alpha']
            assert row['predicted_rz_'+axis] == src['axes'][axis]['predicted_rz_count_per_interrogation']
        for field in ('beta_pf_budget','beta_rte_budget','beta_stat_budget'):
            assert row[field] == best[field]
        assert row['saved_alpha_total_allocated'] == best['alpha_total_allocated']
        assert row['PF_coefficient'] == c['pf_coefficient']
        assert row['delta_time'] == c['delta_time'] == '0.02'
        pref = json.loads(row['physical_signal_refs'])[0]
        physical = pointer(data[pref['id']], pref['selector'])
        assert row['physical_signal_status'] == 'DIRECT_SAVED_VALIDATION'
        if row['ld'] == '3':
            found = [v for v in grid['points'] if key(v)==key(row)]
            assert len(found) == 1
            point = found[0]
            assert physical['state_label'] == 'physical_df_ground_state'
            assert load(row['physical_signal_json']) == physical['attenuated_event_mean_signal']
            for dst,field in [('physical_PF_radius','reference_signal_radius'),
                              ('physical_observed_radius','observed_attenuated_radius'),
                              ('physical_PF_vs_exact_phase_error','pf_vs_exact_phase_error'),
                              ('physical_finite_RTE_phase_error','finite_rte_phase_error'),
                              ('physical_finite_RTE_phase_bound','finite_rte_phase_error_bound'),
                              ('physical_radius_lower_bound','conservative_radius_lower_bound')]:
                assert row[dst] == physical[field]
            assert row['physical_B_normalization'] == point['finite_distribution_normalization']
            assert row['physical_epsilon_Z'] == point['round_signal_error_bound']
            assert row['physical_operator_error'] == point['corrected_operator_error_spectral_norm']
            old = next(v for v in data['SCHEDULEJSON']['selected_schedule_matrix_validations'] if v['delta_time']=='0.02')
            old_point = next(v for v in old['points'] if v['q_m']==row['q_m'])
            assert key(load(row['old_56_point_schedule_tuple'])) == key(old_point)
            assert row['old_56_point_exact_RTE_tuple_match'] == str(key(old_point)==key(row)).lower()
            assert row['PF_direct_phase_status'] == 'DIRECT_SAVED_VALIDATION'
        else:
            assert row['physical_observed_radius'] == physical['observed_radius']
            assert load(row['physical_signal_json']) == physical['signal']
            assert row['PF_direct_phase_status'] == 'UNRESOLVED'
            assert row['finite_RTE_direct_status'] == 'NOT_APPLICABLE'
        cell = f"ld{row['ld']}:delta0.02:r{row['r_m']}"
        prices = model['direct_compiled_rz_measurements'][cell]['q']
        direct = prices.get(row['q_m'])
        assert (row['direct_price_status']=='DIRECT_SAVED_VALIDATION') == (direct is not None)
        if direct is not None:
            assert load(row['direct_price_record']) == direct
        assert row['price_model_status'] == ('EXTRAPOLATED' if int(row['q_m']) > max(map(int,prices)) else 'EMPIRICAL_PROXY')
        assert row['statistics_status'] == 'CONDITIONAL_MODEL'
        assert row['branch_status'] == 'ASSUMED_NOT_VALIDATED'
        for k,v in row.items():
            if k.endswith('_status'): assert v in STATES, (k,v)
    for ld in ('3','12'):
        rr = [r for r in rounds if r['ld']==ld]
        assert [r['q_m'] for r in rr] == [r['q_m'] for r in model['coherent_reoptimization']['best_by_ld'][ld]['best']['rounds']]
        assert len(rr)==18 and rr[-1]['q_m']=='131072'
    for row in assumptions:
        assert row['evidence_state'] in STATES
    for row in rounds+assumptions:
        for k,v in row.items():
            if k.endswith('_refs'):
                for r in json.loads(v):
                    orig = index['artifacts'][r['id']]
                    assert all(r[x]==orig[x] for x in ('commit','path','url'))
                    if r['selector'] and r['id'] in data: pointer(data[r['id']], r['selector'])
    # Match the two recorded snapshot identities without reading/decoding the NPZ.
    snap = abl['source_evidence']['snapshot']
    assert snap['sha256'] == data['INITPROV']['input_sha256'][snap['path']]
    assert snap['sha256'] == data['EXTPROV']['input_sha256'][snap['path']]
    assert abl['source_evidence']['ld3_selected_schedule_signal_grid']['sha256'] == index['artifacts']['ABLGRID']['sha256']
    for field, ident in [('latest_fresh_proxy','OPT2JSON'),('legacy_fixed_holdouts','LEGACY')]:
        assert data['A0']['source_evidence'][field]['sha256'] == index['artifacts'][ident]['sha256']
    assert data['INITPROV']['source_sha256'] == data['EXTPROV']['source_sha256']
    assert all(data['AUDIT']['integrity_checks'].values())
    assert model['scope']['final_total_cost_evaluation_performed'] is False
    assert model['scope']['scientific_superiority_claimed'] is False
    for path in HERE.iterdir():
        if path.is_file():
            content = path.read_text()
            assert not re.search(r'/h[o]me/|/t[m]p/|(?:10|192\.168)\.\d+\.\d+\.\d+|BEGIN (?:RSA |OPENSSH )?PRIVATE KEY|gh[pousr]_[A-Za-z0-9]{20,}', content), path.name
    self_checks()
    print(json.dumps(dict(result='PASS',fixed_objects=len(index['artifacts']),
        reported_source_file_hash_comparisons=len(index['source_file_hash_checks']),
        round_rows=len(rounds),assumption_rows=len(assumptions),
        partial_tuple_matches=18,full_saved_radius_matches=18,
        scope='metadata/string comparisons only; not a scientific PASS')))

if __name__ == '__main__':
    main()
