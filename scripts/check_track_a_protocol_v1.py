"""Check adopted protocol documents and metadata; never execute science code."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = '9c54d89f6154525e3c7c814a4da5ecdc37453b8a'
STATUS = 'ADOPTED_V1_DOCUMENT_ONLY'
BUNDLE = 'artifacts/resource_applicability/track_a_protocol_v1/2026-10-19'
MATRIX = 'track_a_protocol_compliance_matrix.csv'
DOCS = (
    'docs/research/track_a_pr_resource_evaluation_protocol_v1.md',
    'track_a_protocol_adoption_decisions.md',
    'track_a_protocol_gap_and_exception_report.md',
    'docs/research/track_a_pr_resource_protocol_v1_adoption_20261019.md',
)
BASIC = {'AD01', 'AD02', 'AD04', 'AD06', 'AD07', 'AD12'}
CONDITIONAL = {'AD05', 'AD08', 'AD09', 'AD10', 'AD11'}
EXPECTED = {**dict.fromkeys(BASIC, 'ADOPTED_BASIC_POLICY'),
            **dict.fromkeys(CONDITIONAL, 'CONDITIONALLY_ADOPTED'),
            'AD03': 'DEFERRED_ADDITIONAL_PF'}
STATES = {'COMPLETE', 'PARTIAL', 'PENDING', 'NOT_VERIFIED', 'NOT_APPLICABLE'}
NEW_COMMITS = {'6d290fc8fb656f4db148e2c9f91b757d80ed83c2',
               'bd41b3760ff89e428380f55585d78094a9330a56'}
UNCHANGED = ('system', 'DF_policy', 'observed_rank', 'tier', 'requirement',
             'status', 'target_relation', 'cost_evidence')


def require(value, reason):
    if not value:
        raise ValueError(reason)


def blob(root, commit, path):
    return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=root)


def validate(root, *, verify_git=True, verify_links=True):
    root = Path(root)
    bundle = root / BUNDLE
    load = lambda name: json.loads((bundle / name).read_text())
    adoption = load('adoption_record.json')
    require(adoption['protocol_status'] == STATUS and adoption['draft_commit'] == BASE,
            'FORMAL_DOCUMENT_BASE')
    for key in ('new_science_authorized', 'numerical_reanalysis_authorized',
                'push_authorized', 'automatic_application_to_existing_contracts',
                'certifies_all_series_core_complete'):
        require(adoption[key] is False, 'DOCUMENT_ADOPTION_NOT_EXECUTION:' + key)
    decisions = {r['id']: r for r in adoption['decisions']}
    require(len(decisions) == len(adoption['decisions']) == 12
            and {k: v['status'] for k, v in decisions.items()} == EXPECTED,
            'EXACT_USER_ADOPTION_PARTITION')
    decision_text = (root / DOCS[1]).read_text()
    for key, row in decisions.items():
        require(row['conditions'] and row['adopted_scope'], 'ADOPTION_SCOPE_CONDITIONS')
        require(f'|{key}|{row["status"]}|' in decision_text, 'DECISION_TABLE_BINDING')

    for name in DOCS:
        text = (root / name).read_text()
        require(STATUS in text and 'ユーザー' in text, 'FORMAL_STATUS_NEXT_ACTOR:' + name)
        if verify_links:
            for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)', text):
                if link.startswith(('https:', 'http:', '#')):
                    continue
                path = link.split('#')[0]
                require((root / name).parent.joinpath(path).exists(), 'LOCAL_LINK:' + link)
    protocol = (root / DOCS[0]).read_text()
    for term in ('Core', 'Extended', 'Guarantee', 'symmetric-directional',
                 'MODEL_ESTIMATE', 'POINT_NOT_REPRODUCED', 'R_G=Σ_a',
                 '自動進行しない', 'n=48を全系へ強制しない',
                 'Phase 0完了は独立P実行完了ではない', '説明数値検証は未完了'):
        require(term in protocol, 'REQUIRED_PROTOCOL_BOUNDARY:' + term)

    catalog = load('evidence_catalog.json')
    require(catalog['protocol_status'] == STATUS, 'CATALOG_STATUS')
    evidence = {r['evidence_id']: r for r in catalog['entries']}
    require(len(evidence) == len(catalog['entries']) == 37, 'FIXED_EVIDENCE_COVERAGE')
    require(NEW_COMMITS <= {r['commit'] for r in evidence.values()}, 'NEW_FIXED_COMMITS')
    for row in evidence.values():
        require(re.fullmatch('[0-9a-f]{40}', row['commit'])
                and re.fullmatch('[0-9a-f]{64}', row['sha256']), 'FIXED_HASH_FORMAT')
        path = PurePosixPath(row['path'])
        require(not path.is_absolute() and '..' not in path.parts, 'EVIDENCE_RELATIVE_PATH')
        require(row['retrieved'] is True and row['scientific_module_executed'] is False,
                'STATIC_RETRIEVAL_ONLY')
        if verify_git:
            value = blob(root, row['commit'], row['path'])
            require(hashlib.sha256(value).hexdigest() == row['sha256']
                    and len(value) == row['bytes'], 'FIXED_GIT_BYTES')

    spec = load('requirements.json')
    require(spec['protocol_status'] == STATUS
            and spec['scientific_status_aggregation_performed'] is False, 'NO_CORE_PROMOTION')
    ids = {r['id']: r['tier'] for r in spec['requirements']}
    series = set(spec['series'])
    with (root / MATRIX).open(newline='') as f:
        rows = list(csv.DictReader(f))
    require(len(series) == 4 and len(ids) == 22, 'FOUR_SERIES_TWENTYTWO_REQUIREMENTS')
    keys = {(r['series_id'], r['requirement_id']) for r in rows}
    require(len(rows) == len(keys) == 88
            and keys == {(s, q) for s in series for q in ids}, 'MATRIX_88_UNIQUE_ROWS')
    log = load('matrix_change_log.json')
    require(log['base_commit'] == BASE and log['scientific_status_values_changed'] == 0,
            'NO_SCIENTIFIC_STATUS_CHANGE')
    prior = {(r['series_id'], r['requirement_id']): r for r in log['rows']}
    require(set(prior) == keys, 'CHANGE_LOG_COVERAGE')
    for row in rows:
        key = row['series_id'], row['requirement_id']
        require(row['protocol_status'] == STATUS
                and row['scientific_adoption'] == 'NO_NEW_RESULT_CERTIFICATION'
                and row['new_science_authorized'] == 'FALSE'
                and row['automatic_application_to_existing_contracts'] == 'FALSE',
                'MATRIX_POLICY_NOT_EXECUTION')
        require(row['status'] in STATES and row['tier'] == ids[row['requirement_id']],
                'MATRIX_STATUS_TIER')
        require(row['status'] == prior[key]['prior_status'] == prior[key]['current_status'],
                'SCIENTIFIC_STATUS_UNCHANGED')
        require(set(row['adoption_ids'].split(';')) <= set(decisions), 'ROW_ADOPTION_REFERENCE')
        refs = row['evidence_ids'].split(';')
        require(refs and all(r in evidence for r in refs), 'ROW_EVIDENCE_REFERENCE')
        require(set(row['evidence_commits'].split(';')) == {evidence[r]['commit'] for r in refs},
                'ROW_COMMIT_BINDING')
        require(all(row[k] for k in ('assessed_scope', 'cohort', 'cost_evidence',
                                    'unresolved_or_limit', 'minimal_next_action')), 'ROW_SCOPE')
        if row['series_id'] in ('H6_TOL_ONLY_DF19', 'H8_TOL_ONLY_DF28'):
            require('MODEL' in row['cost_evidence'], 'MODEL_NOT_DIRECT')
        if row['series_id'] == 'H4_LEGACY_DF12' and row['requirement_id'] == 'C_DF':
            require(row['DF_policy'] == 'LEGACY_FIXED_RANK'
                    and row['target_relation'] == 'DIFFERS', 'LEGACY_NOT_TOL_ONLY')
        if row['series_id'] == 'H6_TOL_ONLY_DF19' and row['requirement_id'].startswith('P_'):
            require(row['preparation_status'] == 'PHASE0_COMPLETE_FINAL_SCIENTIFIC_SEAL_PENDING'
                    and row['status'] != 'COMPLETE', 'PHASE0_NOT_CONFIRMATION')
            require('E_H6P0_STOP' in refs and 'E_H6P0_BINDING' in refs, 'H6_PHASE0_EVIDENCE')
        if row['requirement_id'] == 'P_FIXED' and row['status'] == 'COMPLETE':
            require('FIXED_INDEPENDENT' in row['cohort']
                    and any(r in refs for r in ('E_H4P', 'E_H8P')), 'FIXED_P_NOT_UNION')
        if row['series_id'] == 'H8_TOL_ONLY_DF28' and row['requirement_id'] == 'M_EXPL':
            require(row['status'] == 'PARTIAL' and 'E_A82_BINDING' in refs
                    and row['preparation_status'] == 'A82_FIELDS_RECOVERED_NOT_NUMERICALLY_VALIDATED',
                    'A82_RECOVERY_NOT_VALIDATION')

    facts = load('evidence_update_status.json')
    h6, a82 = facts['H6'], facts['A82']
    require(h6['phase0_complete'] is True and h6['independent_confirmation_complete'] is False
            and h6['launch_ready'] is False and h6['scientific_execution_approved'] is False
            and h6['numerical_design_approved'] is False, 'PHASE0_EXECUTION_BOUNDARY')
    require(all(h6[k] is None for k in ('final_scientific_SOURCE', 'final_n',
                                      'final_seed_registry', 'final_scientific_call_budget')),
            'H6_FINAL_SEAL_STILL_NULL')
    require(a82['fields_recovered'] is True and a82['numeric_explanation_validated'] is False
            and a82['historical_complete_sides'] == 63 and a82['historical_total_sides'] == 64
            and a82['new_explanation_SOURCE'] is None, 'A82_NO_NUMERIC_COMPLETION')
    require(a82['cohort'] == 'OLD_CONFIRMATION_UNION_N32'
            and a82['target_side'] == 'A82_OLD_PR.PR'
            and a82['original_signal_SOURCE'] != a82['D0_SOURCE'], 'A82_ROLE_IDENTITY')

    if verify_git:
        baseline = list(csv.DictReader(io.StringIO(blob(root, BASE, MATRIX).decode())))
        original = {(r['series_id'], r['requirement_id']): r for r in baseline}
        for row in rows:
            require(all(row[k] == original[(row['series_id'], row['requirement_id'])][k]
                        for k in UNCHANGED), 'BASELINE_EVIDENCE_LABELS_UNCHANGED')
        # Check only metadata flags/nulls in the two new immutable evidence sets.
        b = evidence['E_H6P0_BINDING']
        binding = json.loads(blob(root, b['commit'], b['path']))
        require(binding['technical_preparation_complete'] is True
                and binding['launch_ready'] is False
                and binding['final_scientific_SOURCE'] is None, 'H6_FIXED_BINDING')
        b = evidence['E_A82_BINDING']
        binding = json.loads(blob(root, b['commit'], b['path']))
        require(binding['numeric_validation'] is False and binding['grant_issued'] is False
                and binding['new_explanation_SOURCE'] is None, 'A82_FIXED_BINDING')

    return dict(status='PASS',validation='ADOPTED_DOCUMENT_METADATA_ONLY',
                protocol_status=STATUS,requirement_rows=88,fixed_evidence_files=37,
                scientific_status_promotions=0,new_science_authorized=False,
                numerical_reanalysis=False,push=False,automatic_application=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    print(json.dumps(validate(parser.parse_args().root), ensure_ascii=False))


if __name__ == '__main__':
    main()
