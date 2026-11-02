#!/usr/bin/env python3
"""Read/validate saved H4/H6 metadata. No numerical kernels or execution grants.

Only Git bytes, JSON/CSV parsing, exact value equality, keys, and row counts are
used. Existing scientific values are never recomputed. Default action is check.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = Path('artifacts/resource_applicability/track_a_h46_higher_pf_dataset_v1/2026-10-20')
SIGNAL_SOURCE = 'bec018c64e7a8458dbf13b1c0e9655d628292857'
COST_SOURCE = '635ba563766875d688ad6912499b7f90397d0cc5'
BASE = '5764aacb0714c6be42cb6dc7c7d8fdf1c8d98bdb'
EPS = ('0.05', '0.01', '0.005', '0.001', '0.0005')
AXES = {'real': 'cosine', 'imag': 'sine'}
METRICS = ('circuit_size', 'cx_count', 'cx_depth', 'rz_count', 'rz_depth', 'total_depth')
COUNTS = {'candidates': 88, 'signal_new': 82, 'signal_reused': 6,
          'eligibility_rows': 880, 'precision_rows': 440, 'candidate_axes': 176,
          'measured_axes': 96, 'measured_new_axes': 92, 'measured_reused_axes': 4,
          'measured_DET_candidates': 48, 'inactive_DET_candidates': 16,
          'pending_PR_groups': 24, 'native_gate_rows': 368}


class DatasetError(ValueError):
    pass


def require(condition, label):
    if not condition:
        raise DatasetError(label)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(data):
    return (json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True,
                       allow_nan=False) + '\n').encode()


def csv_bytes(rows):
    require(bool(rows), 'EMPTY_TABLE')
    out = io.StringIO(newline='')
    w = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator='\n')
    w.writeheader()
    for row in rows:
        w.writerow({k: json.dumps(v, ensure_ascii=False, separators=(',', ':'))
                    if isinstance(v, (list, dict)) else v for k, v in row.items()})
    return out.getvalue().encode()


def unique(rows, key, label):
    result = {}
    for row in rows:
        k = key(row)
        require(k not in result, 'DUPLICATE_' + label)
        result[k] = row
    return result


def cell(v):
    return '' if v is None else str(v)


def get_sources(root=ROOT):
    catalog = json.loads((root / BUNDLE / 'source_catalog.json').read_bytes())

    def read(item):
        key, e = item
        require(len(e['commit']) == 40 and '..' not in Path(e['path']).parts
                and not Path(e['path']).is_absolute(), 'UNSAFE_SOURCE_REFERENCE')
        p = subprocess.run(['git', 'show', e['commit'] + ':' + e['path']],
                           cwd=root, check=True, capture_output=True)
        b = p.stdout
        require(len(b) == e['bytes'] and digest(b) == e['sha256'], 'SOURCE_BYTES_' + key)
        # Git blob identity is independently checked, not inferred from filename.
        obj = subprocess.run(['git', 'rev-parse', e['commit'] + ':' + e['path']],
                             cwd=root, check=True, capture_output=True, text=True).stdout.strip()
        require(obj == e['git_blob'], 'SOURCE_BLOB_' + key)
        if e['path'].endswith('.json'):
            value = json.loads(b)
        elif e['path'].endswith('.csv'):
            value = list(csv.DictReader(io.StringIO(b.decode())))
        else:
            value = b
        return key, value

    # Independent, small Git metadata reads; one cache per command. No science
    # workers, BLAS threads, package imports, or private input access.
    with ThreadPoolExecutor() as pool:
        parsed = dict(pool.map(read, catalog.items()))
    return catalog, parsed


def assemble(s, reused_projection, conditions):
    """Exact structural joins; no eligibility, shot, cost-mean or rank formulas."""
    domain = s['domain']['rows']
    dm = unique(domain, lambda r: r['id'], 'CANDIDATE')
    signal = unique(s['signal']['records'], lambda r: r['candidate']['id'], 'SIGNAL')
    ready = unique(s['readiness']['candidate_table'], lambda r: r['candidate']['id'], 'READINESS')
    coverage = unique(s['coverage'], lambda r: r['candidate_id'], 'COVERAGE')
    require(set(dm) == set(signal) == set(ready) == set(coverage), 'CANDIDATE_SETS')
    require(s['signal']['SOURCE'] == SIGNAL_SOURCE, 'SIGNAL_SOURCE')
    require(s['signal']['strict_bias_certificate'] is None, 'STRICT_CERTIFICATE')
    signal_results = unique(s['signal_H4_complete']['results'] + s['signal_H6_complete']['results'],
                            lambda r: r['id'], 'SIGNAL_RESULT_HASH')
    require(set(signal_results) == set(dm), 'SIGNAL_RESULT_KEYS')
    require(s['readiness']['shot_policy_adopted'] is False, 'SHOT_POLICY')
    require(s['readiness']['AD03'] == conditions['AD03'] == 'DEFERRED', 'AD03')
    require(conditions['formal_shot_policy_adopted'] is False, 'FORMAL_SHOTS')
    require(conditions['compiler'] == s['readiness']['compiler'], 'COMPILER_CONDITIONS')
    require(conditions['compiler_settings_hash'] == s['readiness']['compiler_settings_hash'], 'COMPILER_HASH')
    for sys, row in conditions['systems'].items():
        require(row['binding'] == s['inputs']['systems'][sys]['binding'], 'INPUT_BINDING')
    e_index = unique(s['eligibility'], lambda r: (r['id'], r['axis'], r['epsilon']), 'ELIGIBILITY')
    p_index = unique(s['precision_status'], lambda r: (r['id'], r['epsilon']), 'PRECISION')
    e_offsets = {(r['id'], r['axis'], r['epsilon']): i for i, r in enumerate(s['eligibility'])}
    p_offsets = {(r['id'], r['epsilon']): i for i, r in enumerate(s['precision_status'])}
    signal_offsets = {r['candidate']['id']: i for i, r in enumerate(s['signal']['records'])}
    expected_e = {(cid, ax, ep) for cid in dm for ax in AXES for ep in EPS}
    expected_p = {(cid, ep) for cid in dm for ep in EPS}
    require(set(e_index) == expected_e and set(p_index) == expected_p, 'ENDPOINT_DOMAIN')
    new = unique(s['new_cost'], lambda r: (r['candidate_id'], r['axis']), 'NEW_COST')
    stages = unique(s['completion'], lambda r: r['stage'], 'STAGE_COMPLETION')
    require(set(stages) == {'H4_DET', 'H6_DET'}, 'COMPLETION_STAGES')
    completed_results = {stage: unique(r['completion']['results'], lambda x: x['id'], 'COMPLETED_RESULT')
                         for stage, r in stages.items()}
    active = unique(s['active_cost'], lambda r: (r['candidate_id'], r['axis']), 'ACTIVE_COST')
    reused = unique(reused_projection['rows'], lambda r: (r['candidate_id'], r['axis']), 'REUSED_PROJECTION')
    old = unique([r for r in s['readiness']['reuse_axes'] if r['eligible_in_current_five_eps']],
                 lambda r: (r['candidate_id'], r['axis']), 'ACTIVE_REUSE')
    require(set(new).isdisjoint(reused) and set(new) | set(reused) == set(active), 'COST_PARTITION')
    require(set(reused) == set(old), 'SAVED_ACTIVE_SET')
    require(len(s['readiness']['reuse_axes']) == 12, 'SAVED_ALL_12_PRESERVED')
    gates = unique(s['native_gates'], lambda r: (r['candidate_id'], r['check']), 'NATIVE_GATE')
    for row in gates.values():
        require(row['PASS'] == 'True' and row['phase_alignment'] == 'False', 'SAVED_NATIVE_PASS')
    candidates, endpoints, axes, measured = [], [], [], []
    for cid, candidate in dm.items():
        sr, rr, cr = signal[cid], ready[cid], coverage[cid]
        require(sr['candidate'] == rr['candidate'] == candidate, 'CANDIDATE_IDENTITY')
        require(candidate['q'] in (1, 2, 4, 8) and candidate['system'] in ('H4', 'H6'), 'FIXED_DOMAIN')
        require(sr['SOURCE'] == SIGNAL_SOURCE and sr['status'] == 'SIGNAL_DIAGNOSTIC_PASS', 'SIGNAL_RECORD_SOURCE_STATUS')
        require(sr['preparation_hash'] == rr['preparation_hash'], 'SIGNAL_PREPARATION')
        require(sr['shot_allowance_formally_adopted'] is False and sr['strict_bias_certificate'] is None,
                'SIGNAL_POLICY_UNCHANGED')
        require(sr['signal_origin'] == rr['signal_origin'], 'SIGNAL_ORIGIN')
        require(json.loads(cr['eligible_epsilons']) == rr['eligible_epsilons'], 'SAVED_ELIGIBLE_EPS')
        for field in ('system', 'method', 'formula', 'prefix', 'q'):
            require(cell(candidate[field]) == cr[field], 'COVERAGE_IDENTITY')
        sr_e = unique(sr['empirical_eligibility']['rows'],
                      lambda r: (r['axis'], cell(r['epsilon'])), 'RECORD_ELIGIBILITY')
        require(set(sr_e) == {(ax, ep) for ax in AXES for ep in EPS}, 'RECORD_ENDPOINT_DOMAIN')
        for ax in AXES:
            for ep in EPS:
                er = e_index[(cid, ax, ep)]
                require(all(er[k] == cell(v) for k, v in sr_e[(ax, ep)].items()), 'ELIGIBILITY_VALUE_TRANSCRIPTION')
                require(all(er[k] == cell(candidate[k]) for k in candidate if k in er), 'ELIGIBILITY_IDENTITY')
                require(er['shots'] == er['cost'] == er['strict_bias_certificate'] == '', 'NULL_SCIENTIFIC_RESULTS')
        has_axes = (cid, 'cosine') in active and (cid, 'sine') in active
        require(((cid, 'cosine') in active) == ((cid, 'sine') in active), 'MISSING_PAIRED_AXIS')
        if candidate['method'] == 'PR':
            require(not has_axes and cr['n_E'] == '', 'PR_UNACQUIRED')
            require(candidate['K'] == 4 and candidate['r_policy'] == 'ABS_TIME_PER_OUTER'
                    and candidate['r_value'] == 32, 'PR_OUTER32_LAW')
        if has_axes:
            require(cr['status'] == 'BOTH_AXES_DIRECT_MEASURED' and rr['eligible_epsilons'], 'ACTIVE_COVERAGE')
        elif candidate['method'] != 'PR':
            require(cr['status'] == 'INACTIVE_DOMAIN_PRESERVED' and not rr['eligible_epsilons'], 'INACTIVE_COVERAGE')
        candidates.append({**candidate, 'conditions_id': candidate['input_series'],
                           'signal_source_id': 'signal', 'signal_record_index': signal_offsets[cid],
                           'signal_result_sha256': signal_results[cid]['sha256'],
                           'reused_original_signal_sha256': sr['source_signal_sha256'],
                           'signal_origin': sr['signal_origin'], 'preparation_hash': sr['preparation_hash'],
                           'eligible_epsilons': rr['eligible_epsilons'], 'cost_status': cr['status'],
                           'n_E': None, 'shots': None, 'total_RZ': None, 'AD03': 'DEFERRED'})
        for ep in EPS:
            pr = p_index[(cid, ep)]
            for ax in AXES:
                require(pr[ax + '_status'] == e_index[(cid, ax, ep)]['status_under_proposed_empirical_policy'],
                        'PRECISION_STATUS_TRANSCRIPTION')
            require((float(ep) in rr['eligible_epsilons']) ==
                    (pr['real_status'] == pr['imag_status'] == 'ELIGIBLE'), 'SAME_EPS_BOTH_AXES_LABEL_JOIN')
            endpoints.append({'candidate_id': cid, 'epsilon': ep, 'real_status': pr['real_status'],
                              'imag_status': pr['imag_status'], 'source_id': 'precision_status',
                              'source_record_index': p_offsets[(cid, ep)],
                              'real_eligibility_record_index': e_offsets[(cid, 'real', ep)],
                              'imag_eligibility_record_index': e_offsets[(cid, 'imag', ep)],
                              'shots': None, 'total_RZ': None})
        for signal_axis, circuit_axis in AXES.items():
            key = (cid, circuit_axis)
            evidence = active.get(key)
            axes.append({'candidate_id': cid, 'system': candidate['system'], 'signal_axis': signal_axis,
                         'circuit_axis': circuit_axis, 'signal_source_id': 'signal',
                         'signal_json_pointer': f'/records/{signal_offsets[cid]}/signal',
                         'signal_SOURCE': sr['SOURCE'], 'signal_result_sha256': signal_results[cid]['sha256'],
                         'eligibility_source_id': 'eligibility',
                         'eligibility_record_indices': [e_offsets[(cid, signal_axis, ep)] for ep in EPS],
                         'cost_key': list(key) if evidence else None, 'cost_status': cr['status'],
                         'measurement': evidence['measurement'] if evidence else None,
                         'measurement_SOURCE': evidence['measurement_SOURCE'] if evidence else None,
                         'acceptance_SOURCE': evidence['acceptance_SOURCE'] if evidence else None,
                         'receipt_sha256': evidence['receipt_sha256'] if evidence else None,
                         'result_sha256': evidence['result_sha256'] if evidence else None,
                         'shots': None, 'total_RZ': None})
    for key, ar in active.items():
        cid, axis = key
        require(cid in dm and axis in AXES.values() and dm[cid]['method'] != 'PR', 'COST_FOREIGN_KEY')
        require(ar['preparation_hash'] == ready[cid]['preparation_hash'], 'COST_PREPARATION')
        require(ar['compiler_settings_hash'] == conditions['compiler_settings_hash'], 'COST_COMPILER')
        require(ar['shots'] == ar['total_shot_RZ'] == '', 'NO_COST_ACCOUNTING')
        if key in new:
            nr = new[key]
            require(nr['stage'] == dm[cid]['system'] + '_DET', 'NEW_STAGE')
            stage = stages[nr['stage']]
            require(nr['plan_sha256'] == stage['plan_sha256'] == stage['completion']['plan_sha256'], 'NEW_COST_PLAN')
            require(nr['native_policy'] == conditions['native_policy'], 'NEW_NATIVE_POLICY')
            require(nr['result_sha256'] == completed_results[nr['stage']][cid + '__0']['result_sha256'],
                    'COST_COMPLETED_RESULT_SHA')
            require(ar['measurement'] == nr['cost_type'] == 'DIRECT_MEASURED_NEW_RUN01', 'NEW_MEASUREMENT_LABEL')
            require(ar['measurement_SOURCE'] == ar['acceptance_SOURCE'] == nr['SOURCE'] == COST_SOURCE, 'COST_SOURCE')
            require(nr['preparation_hash'] == ar['preparation_hash'] and nr['compiler_settings_hash'] == ar['compiler_settings_hash'], 'NEW_COST_IDENTITY')
            require(nr['cost_receipt_sha256'] == ar['receipt_sha256'] and nr['result_sha256'] == ar['result_sha256'], 'NEW_COST_RECEIPT')
            metrics = {m: int(nr[m]) for m in METRICS}
            fingerprint, compiled = nr['fingerprint'], nr['compiled_SHA']
            gate_count = len([g for g in gates if g[0] == cid])
            require(gate_count == 8, 'NEW_PAIR_GATE_COUNT')
            validation = 'SAVED_8_PAIR_CHECKS_PASS'
            old_stop = True
        else:
            projection, prior = reused[key], old[key]
            require(ar['measurement'] == ('DIRECT_MEASURED_SAVED_REUSE_CT01_SEPARATE_ACCEPTANCE'
                    if prior['CT01_separate_acceptance'] else 'DIRECT_MEASURED_SAVED_REUSE'), 'REUSE_LABEL')
            require(projection['private_receipt_byte_verified'] is True, 'REUSE_RECEIPT_NOT_VERIFIED')
            require(projection['receipt_sha256'] == prior['receipt_sha256'] == ar['receipt_sha256'], 'REUSE_RECEIPT_HASH')
            require(projection['fingerprint'] == prior['fingerprint'], 'REUSE_FINGERPRINT')
            require(ar['measurement_SOURCE'] == prior['cost_SOURCE'], 'REUSE_SOURCE')
            require(ar['acceptance_SOURCE'] == ('885bbd7199c9ada26704b7db75b99767734e92e7'
                    if prior['CT01_separate_acceptance'] else prior['cost_SOURCE']), 'REUSE_ACCEPTANCE_SOURCE')
            require(prior['identity_join'] == 'MATCHED_WITHIN_M5_NATIVE_SERIES', 'REUSE_IDENTITY')
            metrics = copy.deepcopy(projection['metrics'])
            fingerprint, compiled = projection['fingerprint'], projection['compiled_SHA']
            require(set(metrics) == set(METRICS), 'REUSE_METRIC_FIELDS')
            gate_count = None
            validation = 'SAVED_CT01_SEPARATE_ACCEPTANCE' if prior['CT01_separate_acceptance'] else 'SAVED_PILOT_ACCEPTANCE'
            old_stop = prior['old_STOP_preserved']
            require(old_stop is True, 'OLD_STOP_PRESERVED')
        require(all(type(metrics[m]) is int and metrics[m] >= 0 for m in METRICS), 'METRIC_TYPE')
        require(str(metrics['rz_count']) == ar['compiled_RZ'], 'RZ_TRANSCRIPTION')
        for k in ('system', 'method', 'formula', 'prefix', 'q'):
            require(ar[k] == cell(dm[cid][k]), 'MEASURED_CANDIDATE_IDENTITY')
            if key in new: require(new[key][k] == ar[k], 'NEW_CANDIDATE_IDENTITY')
        measured.append({'candidate_id': cid, 'system': dm[cid]['system'], 'method': dm[cid]['method'],
                         'formula': dm[cid]['formula'], 'prefix': dm[cid]['prefix'], 'q': dm[cid]['q'],
                         'axis': axis, **metrics, 'measurement': ar['measurement'],
                         'measurement_SOURCE': ar['measurement_SOURCE'], 'acceptance_SOURCE': ar['acceptance_SOURCE'],
                         'compiler_settings_hash': ar['compiler_settings_hash'], 'preparation_hash': ar['preparation_hash'],
                         'fingerprint': fingerprint, 'compiled_SHA': compiled,
                         'receipt_sha256': ar['receipt_sha256'], 'result_sha256': ar['result_sha256'],
                         'metric_source_id': 'new_cost' if key in new else 'saved_reuse_metric_projection_4.json',
                         'native_validation': validation, 'saved_pair_gate_rows': gate_count,
                         'old_STOP_preserved': old_stop, 'shots': None, 'total_shot_RZ': None})
    observed = {'candidates': len(candidates),
                'signal_new': len([c for c in candidates if c['signal_origin'] == 'NEW_FINITE_MEAN_SIGNAL']),
                'signal_reused': len([c for c in candidates if c['signal_origin'] != 'NEW_FINITE_MEAN_SIGNAL']),
                'eligibility_rows': len(e_index), 'precision_rows': len(endpoints), 'candidate_axes': len(axes),
                'measured_axes': len(measured), 'measured_new_axes': len(new), 'measured_reused_axes': len(reused),
                'measured_DET_candidates': len({cid for cid, ax in active}),
                'inactive_DET_candidates': len([c for c in candidates if c['cost_status'] == 'INACTIVE_DOMAIN_PRESERVED']),
                'pending_PR_groups': len([c for c in candidates if c['method'] == 'PR']),
                'native_gate_rows': len(gates)}
    require(observed == COUNTS, 'EXPECTED_STRUCTURAL_COUNTS')
    # Separate already-published end-of-run proof from new compilation claims.
    for stage in s['completion']:
        require(stage['SOURCE'] == COST_SOURCE and stage['native_exit'] == 0
                and stage['all_workers_reaped'] is True, 'SAVED_COMPLETION')
        r = stage['completion']
        require(r['status'] == 'UNIT_COST_STAGE_COMPLETE_MANDATORY_STOP' and r['errors'] == {}
                and r['mandatory_STOP'] is True and r['retry'] is False
                and r['next_phase_started'] is False, 'SAVED_STOP')
    sc = s['signal_completion']
    require(sc['SOURCE'] == SIGNAL_SOURCE and sc['status'] == 'SIGNAL_COMPLETE_MANDATORY_STOP'
            and sc['errors'] == {} and sc['retry'] is False and sc['next_phase_started'] is False
            and all(w['exit'] == 0 and w['reaped'] for w in sc['workers'].values()), 'SIGNAL_COMPLETION')
    timing = unique(s['timing'], lambda r: r['stage'], 'RUN_TIMING')
    runs = [{'stage': 'SIGNAL_RUN02', 'SOURCE': sc['SOURCE'], 'result_commit': 'a960a41da5ebc564b0306e798572257091c8509c',
             'plan_sha256': sc['plan_sha256'], 'status': sc['status'], 'native_exit': s['signal_summary']['native_exit'],
             'all_workers_exit0_reaped': s['signal_summary']['workers_exit0_reaped'],
             'wall_seconds': sc['wall_seconds'], 'external_wall_seconds': s['signal_summary']['external_wall_seconds'],
             'origin': 'SAVED_RESULT_TRANSCRIPTION', 'mandatory_STOP': True}]
    for stage in s['completion']:
        t = timing[stage['stage']]
        require(t['native_exit'] == stage['native_exit'], 'TIMING_EXIT')
        runs.append({'stage': stage['stage'], 'SOURCE': stage['SOURCE'], 'result_commit': BASE,
                     'plan_sha256': stage['plan_sha256'], 'status': stage['completion']['status'],
                     'completion_sha256': stage['completion_sha256'], 'external_exit_sha256': stage['external_exit_sha256'],
                     'native_exit': stage['native_exit'], 'all_workers_exit0_reaped': stage['all_workers_reaped'],
                     'new_compile_axes': stage['compile_axes'], 'completed_pairs': len(stage['completion']['results']),
                     'native_wall_seconds': t['native_wall_seconds'], 'launcher_wall_seconds': t['launcher_wall_seconds'],
                     'completion_written_JST': t['completion_file_written_JST'],
                     'external_exit_written_JST': t['external_exit_file_written_JST'],
                     'origin': 'SAVED_RESULT_TRANSCRIPTION', 'mandatory_STOP': True})
    return {'schema': 'H46_HIGHER_PF_INTEGRATED_DATASET_V1', 'base_commit': BASE,
            'kind': 'POST_EXPERIMENT_EXACT_TRANSCRIPTION_AND_STRUCTURAL_JOIN',
            'counts': observed, 'candidates': candidates, 'candidate_precision_links': endpoints,
            'axis_provenance': axes, 'measured_gate_costs': measured,
            'run_provenance': runs,
            'referenced_tables': {'signal': {'source_id': 'signal', 'json_pointer': '/records', 'rows': 88},
                                  'axis_eligibility': {'source_id': 'eligibility', 'rows': 880},
                                  'native_checks': {'source_id': 'native_gates', 'rows': 368},
                                  'PR_sampling_design': {'source_id': 'sampling_design', 'rows': 24}},
            'null_policy': 'JSON_NULL_AND_CSV_EMPTY_ARE_UNKNOWN_OR_NOT_ADOPTED;NEVER_ZERO_IMPUTED',
            'AD03': 'DEFERRED', 'formal_shot_policy_adopted': False, 'new_scientific_calculations': 0,
            'final_evidence_freeze': False, 'mandatory_STOP': True}


def expected_files(root=ROOT):
    catalog, sources = get_sources(root)
    p = root / BUNDLE
    reuse = json.loads((p / 'saved_reuse_metric_projection_4.json').read_bytes())
    conditions = json.loads((p / 'conditions.json').read_bytes())
    data = assemble(sources, reuse, conditions)
    outputs = {'dataset.json': json_bytes(data),
               'candidates_88.csv': csv_bytes(data['candidates']),
               'candidate_precision_links_440.csv': csv_bytes(data['candidate_precision_links']),
               'axis_provenance_176.csv': csv_bytes(data['axis_provenance']),
               'measured_gate_costs_96.csv': csv_bytes(data['measured_gate_costs']),
               'run_provenance.json': json_bytes(data['run_provenance'])}
    return outputs, data, catalog, sources


def verify(root=ROOT):
    index = json.loads((root / BUNDLE / 'dataset_index.json').read_bytes())
    for name, expected in index['sha256'].items():
        require(Path(name).name == name, 'UNSAFE_INDEX_PATH')
        require(digest((root / BUNDLE / name).read_bytes()) == expected, 'INDEX_CONTENT_SHA_' + name)
    outputs, data, catalog, sources = expected_files(root)
    for name, content in outputs.items():
        require((root / BUNDLE / name).read_bytes() == content, 'MATERIALIZED_FILE_' + name)
    imported = json.loads((root / BUNDLE / 'common_v2_imports.json').read_bytes())
    require(imported['source_commit'] == catalog['common_v2']['commit'], 'LATEST_V2_SNAPSHOT')
    for row in imported['imports']:
        b = (root / row['path']).read_bytes()
        if row['path'] == catalog['common_v2']['path']:
            require(b.startswith(sources['common_v2']), 'V2_ORIGINAL_PREFIX_UNCHANGED')
            require(b[len(sources['common_v2']):].count(b'## 11.') == 1, 'V2_SINGLE_APPEND')
        else:
            require(len(b) == row['bytes'] and digest(b) == row['sha256'], 'IMPORTED_V2_SNAPSHOT')
    return {'status': 'PASS', 'scope': 'STRUCTURE_KEYS_HASHES_EXACT_TRANSCRIPTION_ONLY',
            'counts': data['counts'], 'git_sources_verified': len(catalog),
            'v2_imports_checked': len(imported['imports']), 'new_science_calls': 0,
            'independent_review': False, 'AD03': 'DEFERRED', 'mandatory_STOP': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-index', action='store_true', help='write metadata tables only')
    parser.add_argument('--table', choices=('signal', 'eligibility', 'native_gates', 'measured_gate_costs'))
    args = parser.parse_args()
    if args.write_index:
        for name, b in expected_files()[0].items():
            (ROOT / BUNDLE / name).write_bytes(b)
    checked = verify()
    if args.table:
        _, data, _, sources = expected_files()
        value = data['measured_gate_costs'] if args.table == 'measured_gate_costs' else sources[args.table]
        if args.table == 'signal': value = value['records']
        print(json_bytes(value).decode(), end='')
    else:
        print(json.dumps(checked, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
