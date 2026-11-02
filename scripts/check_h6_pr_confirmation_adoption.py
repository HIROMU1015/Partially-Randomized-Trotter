"""Validate document provenance and saved literals; never import scientific code."""
import csv
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = '9036d8a87ad1c38e83b9ff654b412dde7041e9d3'
RESULT = '3cccfcb2051dacf29e91473e2e1152acef1574bd'
SOURCE = 'bf0c26dae4477de5fe7264a7c91e077661ef9b38'
SEAL = 'f79366f68caf8585cc36d2f2b0fefcb1a4bd0d1f'
BUNDLE = 'artifacts/resource_applicability/track_a_h6_pr_confirmation_adoption/2026-10-19'
SAVED = 'artifacts/resource_applicability/track_a_h6_pr_confirmation_science/2026-10-19'
MANIFEST_KEY = 'track_a_h6_pr_confirmation_formal_adoption_20261019'
EXPECTED = {
    'H6_CONFIRM_P4_Q2': ('0.05', 'B2_p4_q2_2nd_r4_K2', 48, 'B0I_p7_q1_2nd', -1, False),
    'H6_CONFIRM_P4_Q8': ('0.0005', 'B2_p4_q8_2nd_r4_K2', 104, 'B0I_p11_q4_4th', 1, True),
}


def require(value, reason):
    if not value:
        raise ValueError(reason)


@lru_cache(maxsize=None)
def blob(commit, path):
    return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=ROOT)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(data):
    # Keep saved numerical tokens as literal strings, with no float conversion.
    return json.loads(data, parse_float=str)


def pointer(value, path):
    for key in path.lstrip('/').split('/'):
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def literal(value):
    return str(value).lower() if isinstance(value, bool) else str(value)


def check_adoption(adoption, completion, saved):
    require(adoption['status'] == 'ADOPTED_COMPLETE_FIXED_SCOPE'
            and completion['status'] == 'FORMALLY_ADOPTED_COMPLETE_FIXED_SCOPE', 'ADOPTION_STATUS')
    for obj in (adoption, completion):
        require(obj['SOURCE'] == SOURCE and obj['result_commit'] == RESULT
                and obj['protocol_commit'] == BASE, 'FIXED_COMMIT_IDENTITY')
    require(adoption['authority'] == 'USER_MESSAGE_CURRENT_SESSION'
            and adoption['completed_scope'] == 'TWO_FIXED_H6_INDEPENDENT_CONFIRMATIONS_ONLY', 'USER_SCOPED_ADOPTION')
    require(set(adoption['scope_ids']) == set(EXPECTED)
            and len(adoption['scope_ids']) == 2, 'TWO_ADOPTED_SCOPES')
    for key in ('new_science_authorized', 'new_numerical_analysis', 'existing_contracts_modified',
                'other_cohorts_pooled', 'new_publication_authorization_confirmed', 'push',
                'final_evidence_freeze', 'next_stage_authorized', 'independent_review'):
        require(adoption[key] is False, 'NO_SCOPE_EXPANSION:' + key)
    require(adoption['new_scientific_calls'] == adoption['new_science_modules_imported']
            == adoption['new_grants_issued'] == 0 and adoption['mandatory_STOP'] is True, 'DOCUMENT_ONLY_STOP')
    scopes = {r['scope']: r for r in completion['scopes']}
    require(set(scopes) == set(EXPECTED) and len(completion['scopes']) == 2, 'EXACT_SCOPE_KEYS')
    original = {r['scope']: r for r in saved['scopes']}
    for key, (eps, candidate, n, pf, direction, cheaper) in EXPECTED.items():
        row, old = scopes[key], original[key]
        require((row['epsilon'], row['candidate'], row['n'], row['fixed_PF'])
                == (eps, candidate, n, pf), 'FIXED_CELL_AND_N')
        require(row['status'] == 'FORMALLY_ADOPTED_COMPLETE_FIXED_SCOPE'
                and row['point_verdict'] == old['primary']['reproduction_status'] == 'REPRODUCED'
                and row['normalized_point_role'] == 'POINT_REPRODUCED', 'REPRODUCTION_ROLE')
        require(row['expected_direction'] == row['point_direction'] == direction
                and row['PR_CHEAPER'] is cheaper, 'DIRECTION_SEPARATE_FROM_CHEAPER')
        require(row['cohort'] == old['primary']['cohort'] == 'H6_INDEPENDENT_UNCONDITIONAL_CONFIRMATION_V1'
                and row['old_samples_pooled'] is False, 'NO_COHORT_POOLING')
        for field in ('mean_exact', 'paired_SE', 'fixed_PF_G_exact', 'PR_reduction_percent_exact',
                      'population_conclusion', 'confidence_interval', 'p_value'):
            require(row[field] == old['primary'][field], 'SAVED_PRIMARY_LITERAL:' + field)
        require(row['confidence_interval'] is None and row['p_value'] is None
                and row['population_conclusion'] == 'UNRESOLVED_WITHOUT_VALID_ALL_SUPPORT_INFERENCE', 'NO_POPULATION_PROMOTION')
        require(row['event_order_counts'] == old['event_order_counts']
                and set(row['event_order_counts']) == {'0', '2'}, 'K2_SAVED_COUNTS')
        for new, old_key in [('raw_result_sha256', 'result_sha256'),
                             ('scope_STOP_sha256', 'scope_STOP_sha256'),
                             ('completion_audit_sha256', 'completion_audit_sha256')]:
            require(row[new] == old[old_key], 'RESULT_AUDIT_SHA_BINDING')
    require(completion['COMMON_CORE_READY'] == 'NOT_CERTIFIED'
            and completion['full_input_generation_equivalence'] == 'NOT_VERIFIED'
            and completion['M_CORE_new_cohort'] == completion['M_EXPL_new_cohort'] == 'NOT_ASSESSED', 'NO_CORE_OR_MECHANISM_PROMOTION')
    for key in ('final_evidence_freeze', 'detailed_scientific_review', 'H4_H8_final_integration',
                'H6_additional_validation_started', 'H4_A50_P_started'):
        require(completion[key] is False, 'EXCLUDED_NEXT_STAGE:' + key)
    require(adoption['tie'] == 'INCONCLUSIVE', 'TIE_NOT_REPRODUCED')


def validate(*, check_inventory=True):
    bundle = ROOT / BUNDLE
    read = lambda name: load((bundle / name).read_bytes())
    adoption, completion = read('adoption_record.json'), read('completion_status.json')
    saved = load(blob(RESULT, SAVED + '/result.json'))
    check_adoption(adoption, completion, saved)

    catalog = read('evidence_catalog.json')
    entries = {e['evidence_id']: e for e in catalog['entries']}
    require(len(entries) == len(catalog['entries']) == 22, 'EXACT_EVIDENCE_CATALOG')
    require(catalog['private_inputs_or_valid_grants_republished'] is False
            and catalog['scientific_modules_executed'] is False, 'PUBLIC_METADATA_ONLY')
    for e in entries.values():
        require(re.fullmatch('[0-9a-f]{40}', e['commit'])
                and re.fullmatch('[0-9a-f]{64}', e['sha256']), 'FIXED_HASH_FORMAT')
        path = PurePosixPath(e['path'])
        require(not path.is_absolute() and '..' not in path.parts, 'REPOSITORY_RELATIVE_PATH')
        data = blob(e['commit'], e['path'])
        require(sha(data) == e['sha256'] and len(data) == e['bytes']
                and e['retrieved'] is True and e['scientific_module_executed'] is False, 'FIXED_GIT_BYTES')
    inventory = load(blob(RESULT, SAVED + '/inventory.json'))
    for name, digest in inventory['files'].items():
        require(sha(blob(RESULT, SAVED + '/' + name)) == digest, 'ORIGINAL_RESULT_INVENTORY')
    contract = load(blob(SEAL, entries['E_H6_SEAL_CONTRACT']['path']))
    require(contract['SOURCE'] == SOURCE and adoption['fixed_endpoints'] == contract['endpoints']
            and adoption['comparison_binding'] == contract['binding'], 'FIXED_SEAL_TRANSCRIPTION')
    preservation = read('preservation.json')
    for row in preservation['original_documents']:
        require((ROOT / row['path']).read_bytes() == blob(BASE, row['path'])
                and sha(blob(BASE, row['path'])) == row['sha256'], 'ORIGINAL_PROTOCOL_OR_TABLE_PRESERVED')
    require(completion['protocol_sha256'] == sha(blob(BASE, completion['protocol_path'])), 'PROTOCOL_V1_SHA')
    source = load(blob(RESULT, SAVED + '/source_and_preservation.json'))
    for path, digest in source['source_hashes'].items():
        require(sha(blob(SOURCE, path)) == digest, 'SEALED_SOURCE_BLOB_SHA')
    require(source['two_final_SHA_gates_passed'] is True
            and source['old_grants_markers_STOP_not_modified'] is True
            and completion['audit']['seed_registry_sha256'] == source['seed_registry_sha256']
            and completion['audit']['runtime_receipt_sha256'] == source['runtime_receipt_sha256'], 'SAVED_POSTRUN_BINDING')
    for row in load(blob(RESULT, SAVED + '/native_completion.json')):
        require(row['driver']['exit_code'] == 0 and row['driver']['reaped'] is True
                and row['raw_audit']['all_native_descendants_reaped'] is True
                and row['all_maps_reaped'] is True and row['cleanup_errors'] == [], 'SAVED_NATIVE_EXIT_AND_REAP')
    for row in load(blob(RESULT, SAVED + '/control_checks.json')):
        require(row['status'] == 'PASS', 'SAVED_NATIVE_CONTROLS')
    grants = load(blob(RESULT, SAVED + '/grant_consumption_receipts.json'))
    require(len(grants) == len({r['grant_sha256'] for r in grants}) == 2, 'TWO_DISTINCT_GRANTS')
    for row in grants:
        require(row['consumed_scope_marker_sha256'] == row['witness_marker_sha256']
                and row['SOURCE'] == SOURCE and row['valid_grant_contents_omitted'] is True, 'SAVED_ONE_SHOT_MARKERS')
    ledger = load(blob(RESULT, SAVED + '/ledger_audit.json'))
    require(ledger['status'] == 'PASS_EXACT_PARENT_AND_TASK_LEDGERS'
            and ledger['statistics_recomputed'] is False, 'SAVED_LEDGER_AUDIT')
    for row in ledger['scopes']:
        old = next(s for s in saved['scopes'] if s['scope'] == row['scope'])
        require(row['parent']['counts'] == old['calls'], 'SAVED_SCOPE_CALL_COUNTS')
    require(completion['total_calls_saved'] == saved['total_calls']
            and saved['total_calls']['trajectory'] == 152
            and saved['total_calls']['wrapper_compile'] == 304
            and saved['total_calls']['RTE_event'] == 3712, 'SAVED_FIXED_BUDGET')
    require(load(blob(RESULT, SAVED + '/STOP.json'))['status'] == 'FINAL_MANDATORY_STOP', 'ORIGINAL_STOP_RETAINED')

    additions = {}
    for table in preservation['prior_tables']:
        raw = (bundle / table['current_file']).read_bytes()
        baseline = blob(BASE, table['base_path'])
        if table.get('prefix_transformation') == 'CRLF_TO_LF_ONLY':
            prefix = baseline.replace(b'\r\n', b'\n')
        else:
            require(table['original_bytes_preserved_as_prefix'] is True, 'DECLARED_PREFIX_PRESERVATION')
            prefix = baseline
        require(raw.startswith(prefix), 'EXACT_ORIGINAL_TABLE_PREFIX')
        rows = list(csv.DictReader(io.StringIO(raw.decode())))
        old = list(csv.DictReader(io.StringIO(baseline.decode())))
        require(rows[:len(old)] == old, 'ALL_ORIGINAL_ROWS_UNCHANGED')
        additions[table['current_file']] = rows[len(old):]
        key = 'record_id' if table['current_file'].startswith('paper_') else 'id' if table['current_file'].startswith('claim_') else 'evidence_row_id'
        require(len(rows) == len({r[key] for r in rows}), 'UNIQUE_TABLE_RECORD_IDS')
    new_manifest = additions['paper_data_manifest.csv']
    require(len(new_manifest) == 4, 'FOUR_NEW_MANIFEST_RECORDS')
    for scope in completion['scopes']:
        old = pointer(saved, scope['result_pointer'])
        for role, key in [('CONFIRMATION_PR', 'mean_exact'), ('FIXED_PF', 'fixed_PF_G_exact')]:
            row = next(r for r in new_manifest if r['source_record'] == scope['result_pointer'] and r['role'] == role)
            require(row['system'] == 'H6' and row['record_type'] == 'H6_FIXED_CONFIRMATION'
                    and row['result_commit'] == RESULT and row['science_SOURCE'] == SOURCE
                    and row['source_file_sha256'] == entries['E_H6_RESULT']['sha256']
                    and row['cost_class'] == 'DIRECT_MEASURED'
                    and row['saved_cost_exact'] == old['primary'][key], 'MANIFEST_FIXED_COMPARISON')
            if role == 'CONFIRMATION_PR':
                require(row['candidate_id'] == scope['candidate'] and row['sample_size'] == str(scope['n'])
                        and row['PR_cohort'] == scope['cohort']
                        and row['saved_SE'] == old['primary']['paired_SE']
                        and row['saved_reduction_percent_exact'] == old['primary']['PR_reduction_percent_exact']
                        and row['saved_cost_status'] == 'REPRODUCED', 'MANIFEST_SAVED_PAIRED_VALUES')
            else:
                require(row['candidate_id'] == scope['fixed_PF'] and row['sample_size'] == 'NOT_APPLICABLE'
                        and row['PR_cohort'] == 'SAVED_DETERMINISTIC_FIXED_CONTROL', 'PF_NOT_NEW_RANDOM_COHORT')
    new_claims = additions['claim_evidence_matrix.csv']
    require(len(new_claims) == 2 and {r['id'] for r in new_claims} == {s['claim_id'] for s in completion['scopes']}, 'TWO_ADOPTED_CLAIMS')
    claims = read('claim_evidence_matrix.json')['rows']
    require(claims == list(csv.DictReader((bundle / 'claim_evidence_matrix.csv').open(newline=''))), 'CLAIM_JSON_CSV_EQUAL')
    for row in new_claims:
        require(row['SOURCE'] == SOURCE and RESULT in row['fixed_commits']
                and 'FORMALLY_ADOPTED_FIXED_SCOPE' in row['evidence_strength'], 'ADOPTED_CLAIM_PROVENANCE')
    require(len(additions['evidence_ledger_v2.csv']) == 34, 'THIRTYFOUR_TRANSCRIBED_LEDGER_ROWS')
    for row in additions['evidence_ledger_v2.csv']:
        evidence = entries[row['evidence_id']]
        require(row['result_commit'] == RESULT and row['SOURCE_commit'] == SOURCE
                and row['evidence_file_sha256'] == evidence['sha256']
                and row['evidence_path'] == evidence['path'], 'LEDGER_SOURCE_BINDING')
        value = pointer(load(blob(evidence['commit'], evidence['path'])), row['source_field'])
        require(row['value_literal'] == literal(value), 'EXACT_SAVED_LEDGER_LITERAL')
    require({r['requirement_id'] for r in completion['protocol_scoped_completion']}
            == {'P_FIXED', 'P_ROLE', 'T_NATIVE', 'T_CLOSE', 'S_RARE'}, 'SCOPED_PROTOCOL_STATUS_ONLY')
    for row in completion['protocol_scoped_completion']:
        require(row['status'] == 'COMPLETE' and row['assessed_scope'] == 'ONLY_H6_CONFIRM_P4_Q2_AND_H6_CONFIRM_P4_Q8'
                and all(e in entries for e in row['evidence_ids']), 'STATUS_EVIDENCE_SCOPE')
    current_manifest = load((ROOT / 'artifacts/validation_manifest.json').read_bytes())
    original_manifest = load(blob(BASE, 'artifacts/validation_manifest.json'))
    require({k: v for k, v in current_manifest.items() if k != MANIFEST_KEY} == original_manifest, 'PRIOR_VALIDATION_STATUS_UNCHANGED')
    require(current_manifest[MANIFEST_KEY]['status'] == completion['status']
            and current_manifest[MANIFEST_KEY]['bundle'] == BUNDLE, 'CURRENT_VALIDATION_POINTER')
    stop = read('STOP.json')
    require(stop['status'] == 'H6_ADOPTION_FINAL_MANDATORY_STOP' and stop['new_scientific_calls'] == 0
            and stop['next_stage_authorized'] is stop['push'] is stop['new_numerical_analysis'] is False, 'ADOPTION_MANDATORY_STOP')
    for path in [ROOT / 'docs/research/track_a_h6_pr_confirmation_adoption_20261019.md', *bundle.glob('*.md')]:
        for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)', path.read_text()):
            if not link.startswith(('https:', 'http:', '#')):
                require((path.parent / link.split('#')[0]).exists(), 'NEW_LOCAL_LINK:' + link)
    if check_inventory:
        inventory = read('inventory.json')
        files = {p.name for p in bundle.iterdir() if p.is_file()} - {'inventory.json'}
        require(set(inventory['files']) == files, 'ADOPTION_INVENTORY_COVERAGE')
        for name, digest in inventory['files'].items():
            require(sha((bundle / name).read_bytes()) == digest, 'ADOPTION_BUNDLE_SHA')
    return dict(status='PASS',validation='DOCUMENT_STRUCTURE_FIXED_GIT_AND_LITERAL_TRANSCRIPTION_ONLY',
                manifest_rows=1572,claim_rows=8,evidence_ledger_rows=256,fixed_evidence_files=22,
                original_manifest_rows_preserved=1568,original_claims_preserved=6,original_ledger_rows_preserved=222,
                new_scientific_calls=0,numerical_reanalysis=False,scientific_modules_imported=0,
                independent_scientific_review=False,push=False,mandatory_STOP=True)


if __name__ == '__main__':
    print(json.dumps(validate(), ensure_ascii=False))
