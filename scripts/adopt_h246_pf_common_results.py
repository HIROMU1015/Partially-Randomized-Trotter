#!/usr/bin/env python3
"""Document-only adoption of fixed H246 results; default is read-only check.

Only stdlib parsing, hashes, counts and exact structural joins are permitted.
No eligibility/shot/cost/statistics/ranking formula, scientific import, private
input read, execution grant or dispatch is present. Historical CSV prefixes and
all scientific record values are retained. See the supplementary protocol.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
from pathlib import Path
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

from h46_higher_pf_dataset import digest, json_bytes, csv_bytes, require, unique

ROOT = Path(__file__).resolve().parents[1]
RESULT = '90f1cd55a0d963953696bd0a27d9d041ff939bb7'
PAPER = '1923fce8f7461513d1bc47b5061e87cd70917e87'
BUNDLE = Path('artifacts/resource_applicability/track_a_h246_pf_common_adopted/2026-10-22')
FINAL = 'artifacts/resource_applicability/track_a_h246_pf_common_final/2026-10-22'
OLD_PAPER = 'artifacts/resource_applicability/track_a_h468_paper_evidence_integration/2026-10-19'
DISPLAY = 'artifacts/resource_applicability/track_a_h468_paper_display_methods/2026-10-19'
FIGURE = 'artifacts/resource_applicability/track_a_mechanism_figure2_a82_readiness/2026-10-19'
PROTOCOL = 'docs/research/track_a_pr_resource_evaluation_protocol_v2.md'
REPORT = 'docs/research/track_a_h246_pf_common_adoption_20261022.md'
SUPPLEMENT = 'docs/research/track_a_h246_pf_common_protocol_v2_supplement_20261022.md'
EPS = (0.05, 0.01, 0.005, 0.001, 0.0005)
FORMULAS = ('2nd', '4th', '4th(m5_best)')
DOMAINS = {'H2': (3, range(0, 3), 84), 'H4': (10, range(1, 10), 228),
           'H6': (19, range(5, 16), 276)}
URL = 'https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/'
CLAIMS = [
    ('COMMON588_DOMAIN', '588候補・2940 endpoint・未解決0の固定共通域を正式採用する。',
     'summary', '登録完了は不適格候補までcompileした意味ではない。', 'ALL588_EVALUATED_PASS_MANDATORY_STOP'),
    ('COMMON588_H2', 'H2の全5精度で保存PR最良より全公式共通域PF最良が低費用。',
     'comparisons', 'rank3、B0/B0I p0..2、Full3、q≤8に限定。過去q16広域最良とは別。', 'PF_LOWER at all five precisions'),
    ('COMMON588_H4', 'H4はε=.001/.0005でPR低費用、他の3精度でPF低費用。',
     'comparisons', 'tol-only rank10、B0/B0I p1..9、Full10。旧DF12系列を含まない。', 'PR_LOWER at .001/.0005; PF_LOWER at .05/.01/.005'),
    ('COMMON588_H6', 'H6はε=.001/.0005でPR低費用、他の3精度でPF低費用。',
     'comparisons', 'tol-only rank19、B0/B0I p5..15、Full19。中断parentのexit/reapは不明。', 'PR_LOWER at .001/.0005; PF_LOWER at .05/.01/.005'),
    ('COMMON588_WITHIN', '同一公式45比較と全公式15比較を別の主張範囲として保存する。',
     'comparisons', 'm5内のPR優位を全公式PFに対する優位へ移送しない。', '45 within-formula; 15 all-formula saved comparisons'),
    ('COMMON588_H6_DENOMINATOR', 'H6保存結果ではm5 prefix拡張よりS2/Yoshida4を含む公式拡張が分母変更に寄与した。',
     'decomposition', '保存された入れ子集合比較。探索範囲だけの単独因果効果・母平均保証ではない。', 'H6 m5_prefix_expansion_factor_exact=1 at all five precisions'),
    ('COMMON588_SCOPE', '保存探索標本の点費用順位と科学取得・中断履歴を明示して採用する。',
     'summary', '経験的allowance、未観測rare event、選択効果。母平均・大域最適性・energy/RPE全round保証なし。', 'H6_original_parent_exit_reap=null; unobserved_rare_event_bound=null'),
]


def git_bytes(commit, path, root=ROOT):
    require(bool(re.fullmatch(r'[0-9a-f]{40}', commit)), 'FIXED_COMMIT')
    require(not Path(path).is_absolute() and '..' not in Path(path).parts, 'SAFE_PATH')
    return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=root)


def ref_specs(root=ROOT):
    refs = {k: (RESULT, FINAL + '/' + n) for k, n in {
        'summary': 'summary.json', 'candidates': 'candidate_status_588.json',
        'comparisons': 'comparisons_60.json', 'rankings': 'precision_rankings_15.json',
        'old_comparisons': 'old_vs_common_15.json',
        'decomposition': 'nested_PF_cost_change_decomposition_10.json',
        'system_references': 'source_result_references.json',
        'saved_numerical_audit': 'independent_saved_result_checks.json',
        'ending_preservation': 'ending_preservation_checks.json',
        'result_STOP': 'MANDATORY_STOP.json'}.items()}
    refs['result_report'] = RESULT, 'docs/research/track_a_h246_pf_common_final_20261022.md'
    refs['protocol_before_append'] = RESULT, PROTOCOL
    for system in DOMAINS:
        b = f'artifacts/resource_applicability/track_a_pf_common_{system}_results/2026-10-22'
        files = {'status': 'candidate_status.json', 'signal': 'signal_records.json.gz',
                 'endpoints': 'PF_endpoint_costs.json.gz', 'identity': 'source_input_identity.json',
                 'audit': 'end_audit.json', 'receipts': 'stage_receipts.json',
                 'native_inventory': 'native_result_inventory.json',
                 'control_inventory': 'control_inventory.json', 'STOP': 'MANDATORY_STOP.json',
                 'contract': 'execution_contract_projection.json' if system == 'H6'
                 else 'execution_contract_APPROVED_projection.json'}
        for key, name in files.items():
            refs[system + '_' + key] = RESULT, b + '/' + name
    refs['H6_interrupted_controls'] = RESULT, 'artifacts/resource_applicability/track_a_pf_common_H6_results/2026-10-22/interrupted_COST_controls.json'
    refs['H6_metadata_audit'] = RESULT, 'artifacts/resource_applicability/track_a_h6_pf_metadata_completion/2026-10-22/ending_audit.json'
    for key, name in {'prior_paper_manifest': 'paper_manifest.json',
                      'prior_CEM': 'claim_evidence_matrix.csv',
                      'prior_CEM_overlay': 'claim_evidence_matrix_current_overlay.json',
                      'prior_ledger': 'evidence_ledger_v3.csv',
                      'prior_paper_data_manifest': 'paper_data_manifest.csv'}.items():
        refs[key] = PAPER, OLD_PAPER + '/' + name
    for key, name in {'display_manifest': 'publication_manifest.json', 'main_table': 'main_table.csv',
                      'main_table_text': 'main_table.md', 'figure_caption': 'figure2_caption.md',
                      'methods': 'methods.md'}.items():
        refs[key] = PAPER, DISPLAY + '/' + name
    figures = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', PAPER, '--', FIGURE],
                                      cwd=root, text=True).splitlines()
    require(len(figures) == 34, 'ORIGINAL_FIGURE2_34')
    refs.update({f'Figure2_{i:02d}': (PAPER, p) for i, p in enumerate(figures)})
    return refs


def sources(root=ROOT):
    def read(item):
        key, (commit, path) = item
        raw = git_bytes(commit, path, root)
        blob = subprocess.check_output(['git', 'rev-parse', commit + ':' + path], cwd=root, text=True).strip()
        meta = dict(commit=commit, path=path, bytes=len(raw), sha256=digest(raw), git_blob=blob,
                    url=URL + commit + '/' + path)
        decoded = gzip.decompress(raw) if path.endswith('.json.gz') else raw
        value = json.loads(decoded) if path.endswith(('.json', '.json.gz')) else raw
        return key, meta, value, raw
    with ThreadPoolExecutor() as pool:
        rows = list(pool.map(read, ref_specs(root).items()))
    return ({k: m for k, m, _, _ in rows}, {k: v for k, _, v, _ in rows},
            {k: raw for k, _, _, raw in rows})


def candidate_key(c):
    return tuple(c[k] for k in ('system', 'method', 'prefix', 'q', 'formula'))


def assemble(s):
    """Join saved records by identity; never compute a scientific endpoint."""
    summary = s['summary']
    require((summary['registered'], summary['complete'], summary['unresolved']) == (588, 588, 0), 'COMPLETE_588')
    require(summary['H6_original_parent_exit_reap'] is None, 'UNKNOWN_ORIGINAL_EXIT')
    require(summary['unobserved_rare_event_bound'] is None and not summary['population_ranking_guarantee'], 'LIMITS')
    joined, system_data = [], {}
    final_status = unique(s['candidates'], lambda r: r['id'], 'FINAL_CANDIDATE')
    expected = []
    for system, (rank, prefixes, count) in DOMAINS.items():
        statuses = s[system + '_status']
        signals = s[system + '_signal']
        endpoints = s[system + '_endpoints']
        sm = unique(statuses, lambda r: r['id'], 'CANDIDATE_ID')
        em = unique(endpoints, lambda r: (candidate_key(r['candidate']), r['epsilon']), 'ENDPOINT')
        grid = {(system, method, p, q, f) for method in ('B0', 'B0I') for p in prefixes
                for q in (1, 2, 4, 8) for f in FORMULAS}
        grid |= {(system, 'FULL', rank, q, f) for q in (1, 2, 4, 8) for f in FORMULAS}
        require(len(sm) == count and {candidate_key(r) for r in statuses} == grid, 'REGISTERED_GRID_' + system)
        require(set(signals) == set(sm), 'SIGNAL_KEYS_' + system)
        require(set(em) == {(key, ep) for key in grid for ep in EPS}, 'FIVE_PRECISIONS_' + system)
        require(s[system + '_identity']['binding']['rank'] == rank, 'DF_RANK_' + system)
        require(s[system + '_audit']['status'] == 'PASS_MANDATORY_STOP', 'AUDIT_' + system)
        system_data[system] = dict(scientific_identity_saved=s[system + '_identity'],
                                   compiler_saved=s[system + '_contract']['compiler'],
                                   native_saved=s[system + '_contract']['native'],
                                   SOURCE=s[system + '_audit']['SOURCE'],
                                   audit_ref=system + '_audit', contract_ref=system + '_contract',
                                   native_inventory_ref=system + '_native_inventory',
                                   control_inventory_ref=system + '_control_inventory')
        for status in statuses:
            cid = status['id']
            # The final join adds H6 signal_acquisition_lane, preserving the
            # earlier per-system fields. Carry this saved distinction verbatim.
            require(all(final_status[cid].get(k) == v for k, v in status.items()), 'FINAL_JOIN_EQUALITY')
            eps_rows = [em[(candidate_key(status), ep)] for ep in EPS]
            require(status['status'] == 'COMPLETE', 'STATUS_COMPLETE')
            require([r['epsilon'] for r in eps_rows if r['status'] == 'ELIGIBLE'] == status['eligible_epsilons'], 'ELIGIBILITY_TRANSCRIPTION')
            for row in eps_rows:
                if row['status'] == 'ELIGIBLE':
                    require(row['price'] is not None and row['total_RZ_exact'] is not None, 'SAVED_PRICE_PRESENT')
            joined.append(dict(candidate_status_saved=final_status[cid], signal_record_saved=signals[cid],
                               endpoints_saved=eps_rows, identity_ref=system + '_identity',
                               status_ref='candidates', per_system_status_ref=system + '_status',
                               signal_ref=dict(source=system + '_signal', pointer='/' + cid),
                               endpoints_ref=system + '_endpoints'))
        expected.extend(final_status[r['id']] for r in statuses)
    require(expected == s['candidates'], 'FINAL_JOIN_EQUALITY')
    comparisons = s['comparisons']
    require(len(comparisons) == 60, 'COMPARISON_60')
    unique(comparisons, lambda r: (r['system'], r['epsilon'], r['formula']), 'COMPARISON')
    require({(r['system'], r['epsilon'], r['formula']) for r in comparisons} ==
            {(sys, ep, f) for sys in DOMAINS for ep in EPS for f in (*FORMULAS, 'ALL_FORMULAS')}, 'COMPARISON_GRID')
    for row in comparisons:
        require(row['independent_confirmation'] is False and row['PR_saved']['independent_confirmation'] == 'False', 'NO_COHORT_POOLING')
        require(row['PR_saved']['mode'] == 'USUAL_SIGNED' and row['PF']['status'] == 'COMPLETE_REGISTERED_GRID', 'COMPARISON_POLICY')
        require(row['PR_saved']['prefix'] == {'H2': '1', 'H4': '3', 'H6': '5'}[row['system']], 'PR_PREFIX')
    old = s['H6_interrupted_controls']
    require(old['STOP']['original_native_exit_codes'] is None and
            old['STOP']['original_parent_join_reap_receipt_available'] is False and
            old['completion_receipt_not_fabricated'] is True, 'H6_STOP_HISTORY')
    return dict(schema='H246_COMMON_PF_ADOPTED_DATASET_V1', result_commit=RESULT,
                status='FORMALLY_ADOPTED_FIXED_DOMAIN_POINT_RESULTS', systems=system_data,
                candidates=joined, saved_execution_summary=summary,
                scientific_values_recomputed=False, independent_confirmation_included=False)


def append_csv(prefix, rows):
    """Retain the historical CSV byte prefix, including its header/quoting."""
    header = next(csv.reader(io.StringIO(prefix.decode())))
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=header, lineterminator='\n')
    for row in rows:
        require(set(row) <= set(header), 'HISTORICAL_CSV_SCHEMA')
        writer.writerow(row)
    require(prefix.endswith(b'\n'), 'CSV_PREFIX_NEWLINE')
    return prefix + out.getvalue().encode()


def evidence_tables(s, catalog, raw):
    conditions = ('linear H2/H4/H6;1 Angstrom;STO-3G;tol-only DF1e-8 ranks3/10/19;'
                  'B0/B0I p0..2/1..9/5..15,Full3/10/19;q1/2/4/8;delta.8/.4/.2/.1;T=.8;P=0;'
                  'USUAL_SIGNED;factor1;alpha_axis=.025;empirical allowance;'
                  'Qiskit1.3.0/rz,sx,x,cx/opt1/seed17;full cos/sin wrapper;symmetric_directional')
    source_commits = ';'.join(s[sys + '_audit']['SOURCE'] for sys in DOMAINS)
    cem = []
    for cid, claim, ref, limit, literal in CLAIMS:
        cem.append(dict(id=cid, claim_candidate=claim, fixed_commits=RESULT,
                        result_files=catalog[ref]['url'], SOURCE=source_commits,
                        conditions=conditions, existing_value_quote=literal,
                        measured_or_model='Direct compiled RZ prices; saved analytic shots; finite PR sample mean/variance/paired SE unchanged',
                        search_or_confirmation='EXPLORATION_OUTER32_K4; separate from pilot, old cohort and independent confirmation',
                        unresolved=limit, evidence_strength='ADOPTED_FINITE_FIXED_GRID_POINT_RESULT_NOT_FINAL_FREEZE',
                        audit_links=catalog['saved_numerical_audit']['url'] + ';' + catalog['H6_interrupted_controls']['url']))
    ledger = []
    for i, row in enumerate(s['comparisons']):
        ref = catalog['comparisons']
        value = {k: row[k] for k in ('system', 'epsilon', 'formula', 'point_direction',
                                    'PR_over_PF_exact', 'PF_minus_PR_exact', 'PR_reduction_percent', 'PR_saved')}
        value['PF_total_RZ_exact'] = row['PF']['best']['total_RZ_exact']
        value['PF_candidate_saved'] = row['PF']['best']['candidate']
        ledger.append(dict(evidence_row_id=f'COMMON588_C{i+1:02d}', study_id='H246_COMMON_PF_588',
                           metric='SAVED_PR_PF_COMPARISON', value_literal=json.dumps(value, ensure_ascii=False, separators=(',', ':')),
                           unit='compiled_RZ_times_analytic_shots',
                           comparison_scope=f"{row['system']};epsilon={row['epsilon']};formula={row['formula']};q<=8;{conditions}",
                           baseline_denominator='COMMON_REGISTERED_PF_GRID_SAME_FORMULA' if row['formula'] != 'ALL_FORMULAS' else 'COMMON_REGISTERED_PF_GRID_ALL_THREE_FORMULAS',
                           cost_evidence='DIRECT_NATIVE_PRICE_AND_SAVED_EXPLORATION_MEAN',
                           scientific_guarantee='FINITE_POINT_ONLY;EMPIRICAL_ALLOWANCE;NO_POPULATION_OR_GLOBAL_OPTIMUM',
                           evidence_id='COMMON588_' + row['system'], source_line='', source_excerpt='',
                           source_field='/' + str(i), SOURCE_commit=s[row['system'] + '_audit']['SOURCE'],
                           result_commit=RESULT, evidence_path=ref['url'], evidence_file_sha256=ref['sha256'],
                           reported_precision='Exact saved fields; no recomputation or reranking'))
    for i, row in enumerate(s['decomposition']):
        ref = catalog['decomposition']
        value = {k: v for k, v in row.items() if not isinstance(v, dict)}
        ledger.append(dict(evidence_row_id=f'COMMON588_D{i+1:02d}', study_id='H246_COMMON_PF_588',
                           metric='SAVED_NESTED_DENOMINATOR_DECOMPOSITION',
                           value_literal=json.dumps(value, ensure_ascii=False, separators=(',', ':')),
                           unit='saved_exact_ratio', comparison_scope=conditions,
                           baseline_denominator='OLD_SPARSE_M5_TO_COMMON_M5_TO_ALL_FORMULAS',
                           cost_evidence='SAVED_DIRECT_NATIVE_FINITE_POINT',
                           scientific_guarantee='NO_SEARCH_ONLY_CAUSAL_OR_POPULATION_GUARANTEE', evidence_id='COMMON588_H6_DENOMINATOR',
                           source_field='/' + str(i), SOURCE_commit=source_commits, result_commit=RESULT,
                           evidence_path=ref['url'], evidence_file_sha256=ref['sha256'], reported_precision='Exact saved fields'))
    for key in ['summary', 'H2_audit', 'H4_audit', 'H6_audit', 'H6_interrupted_controls']:
        ref = catalog[key]
        value = s[key]
        if key == 'H6_interrupted_controls':
            value = {k: value[k] for k in ('STOP', 'counts', 'old_native_exit_reap', 'completion_receipt_not_fabricated')}
        ledger.append(dict(evidence_row_id='COMMON588_' + key, study_id='H246_COMMON_PF_588',
                           metric='SAVED_STATUS_AND_PROVENANCE', value_literal=json.dumps(value, ensure_ascii=False, separators=(',', ':')),
                           unit='saved_status', comparison_scope=conditions, baseline_denominator='NONE',
                           cost_evidence='SAVED_RECEIPT', scientific_guarantee='H6_ORIGINAL_PARENT_EXIT_REAP_UNKNOWN;NO_EXTERNAL_REPRODUCTION',
                           evidence_id='COMMON588_SCOPE', source_field='/', SOURCE_commit=source_commits,
                           result_commit=RESULT, evidence_path=ref['url'], evidence_file_sha256=ref['sha256'],
                           reported_precision='Literal saved statuses; unknown remains null'))
    return append_csv(raw['prior_CEM'], cem), append_csv(raw['prior_ledger'], ledger)


def expected_files(root=ROOT):
    catalog, s, raw = sources(root)
    dataset = assemble(s)
    cem, ledger = evidence_tables(s, catalog, raw)
    comparisons = s['comparisons']
    files = {'source_catalog.json': json_bytes(catalog),
             'dataset.json.gz': gzip.compress(json_bytes(dataset), mtime=0),
             'candidate_index.csv': csv_bytes([
                 dict(**dict(r['candidate_status_saved'],
                             signal_acquisition_lane=r['candidate_status_saved'].get('signal_acquisition_lane')),
                      signal_provenance=json.dumps(r['signal_record_saved'].get('provenance'), ensure_ascii=False),
                      dataset_pointer='/candidates/' + str(i), scientific_identity_ref=r['identity_ref'],
                      endpoints_ref=r['endpoints_ref']) for i, r in enumerate(dataset['candidates'])]),
             'comparisons_all_formulas_15.json': json_bytes([r for r in comparisons if r['formula'] == 'ALL_FORMULAS']),
             'comparisons_within_formula_45.json': json_bytes([r for r in comparisons if r['formula'] != 'ALL_FORMULAS']),
             'precision_rankings_15.json': raw['rankings'], 'old_vs_common_15.json': raw['old_comparisons'],
             'denominator_decomposition_10.json': raw['decomposition'],
             'claim_evidence_matrix_v2.csv': cem, 'evidence_ledger_v4.csv': ledger}
    files['adoption_record.json'] = json_bytes(dict(
        schema='H246_COMMON_PF_FORMAL_ADOPTION_V1', date='2026-10-22', result_commit=RESULT,
        status='USER_ADOPTED_FIXED_DOMAIN_RESULTS_DOCUMENT_INTEGRATION_COMPLETE',
        authorization='H2/H4/H6共通決定論PF探索588候補・未解決0件の結果を、固定探索域における正式な研究成果として採用します。',
        scientific_execution_authorized=False, new_scientific_calls=0, saved_value_reanalysis=False,
        reranking=False, sampling=False, compile=False, independent_confirmation_modified=False,
        AD03='DEFERRED_SEPARATE_DECISION', final_evidence_freeze=False,
        detailed_scientific_review_started=False, manuscript_main_text_rewritten=False,
        adopted_claim_ids=[c[0] for c in CLAIMS], protocol_supplement=SUPPLEMENT,
        prior_paper_commit=PAPER, main_table_changed=False, Figure2_changed=False,
        H6_old_COST_success_claimed=False, H6_original_parent_exit_reap=None,
        publication_scope='User permits push of verified-public existing results and new adoption documents only; no private originals or executable grants added',
        mandatory_STOP=True))
    manifest_rows = [dict(record_id='COMMON588_' + name, record_type='SUPPLEMENT_SAVED_TRANSCRIPTION',
                          path=str(BUNDLE / name), sha256=digest(data), result_commit=RESULT,
                          scientific_values_recomputed=False, independent_confirmation=False)
                     for name, data in files.items() if name not in ('source_catalog.json', 'adoption_record.json')]
    files['paper_data_manifest.csv'] = csv_bytes(manifest_rows)
    files['paper_manifest.json'] = json_bytes(dict(
        schema='H246_COMMON_PF_PAPER_SUPPLEMENT_V1', status='ADOPTED_SUPPLEMENT_NOT_FINAL_EVIDENCE_FREEZE',
        result_commit=RESULT, protocol=PROTOCOL, protocol_supplement=SUPPLEMENT,
        prior_paper_manifest=catalog['prior_paper_manifest'], prior_display_manifest=catalog['display_manifest'],
        prior_paper_data_manifest=catalog['prior_paper_data_manifest'],
        prior_CEM=catalog['prior_CEM'], prior_CEM_overlay=catalog['prior_CEM_overlay'], prior_ledger=catalog['prior_ledger'],
        main_table=catalog['main_table'], main_table_changed=False,
        Figure2_original_files=[v for k, v in catalog.items() if k.startswith('Figure2_')],
        Figure2_changed=False, independent_confirmation_changed=False,
        paper_data_manifest=str(BUNDLE / 'paper_data_manifest.csv'),
        claim_evidence_matrix=str(BUNDLE / 'claim_evidence_matrix_v2.csv'),
        evidence_ledger=str(BUNDLE / 'evidence_ledger_v4.csv'),
        historical_CEM_prefix_rows=14, appended_CEM_rows=len(CLAIMS),
        historical_ledger_prefix_rows=316, appended_ledger_rows=75,
        comparisons_all_formulas=15, comparisons_within_formula=45,
        original_PR_cohort='SAVED_EXPLORATION_OUTER32_K4_PREFIX1_3_5',
        new_statistics=False, new_ranking=False, new_scientific_calls=0,
        AD03='DEFERRED_SEPARATE_DECISION', final_evidence_freeze=False,
        population_mean_guarantee=False, universal_PR_advantage=False))
    files['MANDATORY_STOP.json'] = json_bytes(dict(
        status='DOCUMENT_ADOPTION_COMPLETE_MANDATORY_STOP', result_commit=RESULT,
        next_stage_authorized=False, scientific_execution=False, new_scientific_calls=0,
        AD03_adopted=False, final_evidence_freeze=False, detailed_scientific_review=False,
        main_table_modified=False, Figure2_modified=False,
        H6_original_parent_exit_reap=None, old_STOP_preserved=True))
    return files, catalog


def check(root=ROOT, pending_receipt=False):
    files, catalog = expected_files(root)
    for name, data in files.items():
        require((root / BUNDLE / name).read_bytes() == data, 'EXACT_TRANSCRIPTION_' + name)
    protocol_before = git_bytes(RESULT, PROTOCOL, root)
    require((root / PROTOCOL).read_bytes().startswith(protocol_before), 'PROTOCOL_APPEND_ONLY')
    # Every tracked scientific/raw/control artifact stays byte-identical. Only
    # explicitly listed documentation/index files may change from result commit.
    allowed = {PROTOCOL, 'PROJECT_MAP.md', 'docs/README.md', 'docs/research/README.md',
               'scripts/README.md', 'src/trotterlib/README.md', 'VALIDATION_STATUS.md',
               'docs/research/研究概要・現状.md', 'docs/research/研究目的・研究課題.md',
               'artifacts/validation_manifest.json', 'docs/research/研究ノート/2026-10-22_pf_common_domain_preparation.md'}
    changed = subprocess.check_output(['git', 'diff', '--name-status', '-z', RESULT, '--'], cwd=root).decode().split('\0')
    additions = {REPORT, SUPPLEMENT, 'scripts/adopt_h246_pf_common_results.py',
                 'tests/test_h246_pf_common_adoption.py'}
    for i in range(0, len(changed) - 1, 2):
        status, path = changed[i:i + 2]
        require((status == 'M' and path in allowed) or
                (status == 'A' and (path in additions or path.startswith(str(BUNDLE) + '/'))),
                'ORIGINAL_SCIENCE_OR_PAPER_PRESERVATION_' + path)
    manifest_before = json.loads(git_bytes(RESULT, 'artifacts/validation_manifest.json', root))
    current = json.loads((root / 'artifacts/validation_manifest.json').read_bytes())
    require(all(current[k] == v for k, v in manifest_before.items()), 'OLD_VALIDATION_ENTRIES_UNCHANGED')
    adoption = current['h246_pf_common_formal_adoption_20261022']
    require(adoption['result_commit'] == RESULT and adoption['bundle'] == str(BUNDLE)
            and adoption['report'] == REPORT and adoption['protocol_supplement'] == SUPPLEMENT
            and adoption['candidates'] == {'H2': 84, 'H4': 228, 'H6': 276, 'total': 588}
            and adoption['saved_endpoint_rows'] == 2940 and adoption['unresolved'] == 0
            and adoption['saved_comparisons'] == {'same_formula': 45, 'all_formulas': 15}
            and adoption['new_scientific_calls'] == 0 and adoption['reranking'] is False
            and adoption['final_evidence_freeze'] is False, 'ADOPTION_MANIFEST_ALIGNMENT')
    for entry in catalog.values():
        p = root / entry['path']
        if entry['commit'] == RESULT and entry['path'] != PROTOCOL:
            require(p.read_bytes() == git_bytes(RESULT, entry['path'], root), 'LOCAL_RESULT_PRESERVATION')
    inventory = json.loads((root / BUNDLE / 'artifact_manifest.json').read_bytes())
    for p, sha in inventory.items():
        require(digest((root / p).read_bytes()) == sha, 'ARTIFACT_SHA_' + p)
    # Check local Markdown targets introduced by this task, not unrelated legacy
    # links in the historical CSV prefixes or cumulative repository indexes.
    documents = {p: (root / p).read_text() for p in
                 (REPORT, SUPPLEMENT, str(BUNDLE / 'README.md'), str(BUNDLE / 'claim_evidence_matrix.md'))}
    for p in allowed - {PROTOCOL, 'artifacts/validation_manifest.json'}:
        before = git_bytes(RESULT, p, root).decode()
        after = (root / p).read_text()
        if p.endswith('2026-10-22_pf_common_domain_preparation.md'):
            require(after.startswith(before), 'NOTE_APPEND_ONLY')
            documents[p] = after[len(before):]
        else:
            require(after.endswith(before), 'DOCUMENT_HISTORY_PRESERVED_' + p)
            documents[p] = after[:len(after) - len(before)]
    documents[PROTOCOL] = (root / PROTOCOL).read_bytes()[len(protocol_before):].decode()
    link_count = 0
    for p, text in documents.items():
        path = root / p
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if target.startswith(('https://', '#')):
                continue
            resolved = (path.parent / target.split('#')[0]).resolve()
            is_pending_self = pending_receipt and resolved == (root / BUNDLE / 'technical_validation.json').resolve()
            require(resolved.exists() or is_pending_self, 'LOCAL_LINK_' + p + ':' + target)
            link_count += 1
    publication = json.loads((root / BUNDLE / 'publication_scope_audit.json').read_bytes())
    require(publication['status'] == 'PASS_PUBLIC_EXISTING_RESULTS_AND_ADOPTION_METADATA_ONLY'
            and publication['result_commit'] == RESULT and publication['paper_commit'] == PAPER
            and publication['new_private_originals'] is False
            and publication['new_execution_grants'] is False, 'PUBLICATION_SCOPE')
    return dict(status='PASS_DOCUMENT_STRUCTURE_SHA_LINKS', result_commit=RESULT,
                source_references_checked=len(catalog), candidate_records=588, endpoint_records=2940,
                all_formula_comparisons=15, within_formula_comparisons=45,
                local_document_links_checked=link_count, original_Figure2_files=34,
                original_CEM_rows_preserved=14, original_ledger_rows_preserved=316,
                new_scientific_calls=0, new_numerical_analysis=False, new_ranking=False,
                validation_kind='PRIMARY_AUTHOR_LOCAL_STRUCTURAL_CHECK_NOT_SCIENTIFIC_REVIEW_OR_CI',
                H6_original_parent_exit_reap=None, AD03='DEFERRED', final_evidence_freeze=False,
                mandatory_STOP=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true', help='write only adoption metadata projections')
    parser.add_argument('--receipt', action='store_true', help='write document validation receipt after PASS')
    args = parser.parse_args()
    if args.build:
        files, _ = expected_files()
        (ROOT / BUNDLE).mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            (ROOT / BUNDLE / name).write_bytes(data)
    if args.build:
        inventory_paths = [p for p in (ROOT / BUNDLE).iterdir()
                           if p.name not in ('artifact_manifest.json', 'technical_validation.json')]
        inventory_paths += [ROOT / p for p in (REPORT, SUPPLEMENT, PROTOCOL,
                            'scripts/adopt_h246_pf_common_results.py',
                            'scripts/h46_higher_pf_dataset.py',
                            'tests/test_h246_pf_common_adoption.py', 'PROJECT_MAP.md',
                            'VALIDATION_STATUS.md', 'artifacts/validation_manifest.json',
                            'docs/README.md', 'docs/research/README.md', 'scripts/README.md',
                            'src/trotterlib/README.md', 'docs/research/研究概要・現状.md',
                            'docs/research/研究目的・研究課題.md',
                            'docs/research/研究ノート/2026-10-22_pf_common_domain_preparation.md')]
        (ROOT / BUNDLE / 'artifact_manifest.json').write_bytes(json_bytes(
            {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in sorted(inventory_paths)}))
    result = check(pending_receipt=args.receipt)
    if args.receipt:
        (ROOT / BUNDLE / 'technical_validation.json').write_bytes(json_bytes(result))
        require(check() == result, 'POST_RECEIPT_LINK_CHECK')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
