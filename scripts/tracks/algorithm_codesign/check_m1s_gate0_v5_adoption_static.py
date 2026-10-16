#!/usr/bin/env python3
"""Adoption-only JSON/provenance checks; no scientific formula evaluation."""
from pathlib import Path
import argparse
import ast
import copy
import hashlib
import importlib.abc
import json
import re
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
BASE = '47da54279838a6b3e1497f36204ba7e53c4857f9'
SOURCE = '69de022dc8142f703423b1ce0bd42f05e123954e'
PROOF = 'b5a185cc0c3aae0e477bfc52b93061e86feb7dec'
ART = 'artifacts/track_b_m1s_gate0_v5_formal_adoption/2026-10-16/v1'


class BlockNumerics(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'scipy', 'openfermion', 'qiskit'} or fullname.endswith('.science'):
            raise RuntimeError('Scientific import forbidden: ' + fullname)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--write-report')
    args = p.parse_args()
    sys.meta_path.insert(0, BlockNumerics())
    formula_calls = []
    def deny_budget(frame, event, arg):
        if event == 'call' and frame.f_code.co_filename.endswith('/m1s_gate0_v5/budget.py') and frame.f_code.co_name not in {'<module>', '_Budget', 'formula_policy', 'is_formula'}:
            formula_calls.append(frame.f_code.co_name)
            raise RuntimeError('Formula evaluation forbidden')
    sys.setprofile(deny_budget)
    sys.path.insert(0, str(ROOT / 'src'))
    from trottertracks.algorithm_codesign.m1s_gate0_v5 import control as c
    checks = []
    def check(name, ok):
        checks.append({'name': name, 'status': 'PASS' if ok else 'FAIL'})
        if not ok:
            raise AssertionError(name)
    def git(*a):
        return subprocess.check_output(['git', '-C', str(ROOT), *a])
    def sha(raw):
        return hashlib.sha256(raw).hexdigest()
    def get(name):
        return c.read_json_bytes((ROOT / ART / name).read_bytes())
    record = get('user_final_contract_adoption_record.json')
    formal_raw = (ROOT / record['formal_contract_path']).read_bytes()
    formal = c.read_json_bytes(formal_raw)
    context = get('adoption_context.json')
    candidate_raw = git('show', BASE + ':' + record['candidate_contract_path'])
    candidate = c.read_json_bytes(candidate_raw)
    check('fixed_base_source_proof_commits', record['reviewed_candidate_commit'] == BASE and record['source_commit'] == SOURCE and record['proof_commit'] == PROOF)
    check('candidate_hash_bound', sha(candidate_raw) == record['candidate_contract_sha256'])
    check('formal_contract_hash_bound', sha(formal_raw) == record['formal_contract_sha256'])
    check('context_hash_bound', sha((ROOT / record['adoption_context_path']).read_bytes()) == record['adoption_context_sha256'])
    check('user_adoption_only', record['record_kind'] == 'FINAL_CONTRACT_ADOPTION_NOT_EXECUTION_AUTHORIZATION' and record['adopted_by'] == 'user')
    check('actual_authority_current_instruction', context['actual_authority'] == 'CURRENT_EXPLICIT_USER_INSTRUCTION')
    for key in ('user_instruction', 'independent_review', 'preceding_GPT_response'):
        row = context[key]
        raw = (ROOT / row['path']).read_bytes()
        check('original_context_bytes:' + key, len(raw) == row['bytes'] and sha(raw) == row['sha256'])
    request = (ROOT / context['user_instruction']['path']).read_text()
    review = (ROOT / context['independent_review']['path']).read_text()
    check('user_approval_and_science_withheld_preserved', '正式採択する方針を承認しました' in request and '科学計算の実行は承認しません' in request)
    check('earlier_phrase_source_distinguished', context['earlier_user_phrase']['quote'] == 'その方針で進めて' and 'QUOTED_BY_USER_IN_CURRENT_INSTRUCTION' in context['earlier_user_phrase']['provenance'])
    check('conditional_acceptance_quote_from_actual_review', bool(context['conditional_acceptance_quote']) and context['conditional_acceptance_quote'] in review)
    check('no_new_Codex_mathematical_review', context['new_GPT_review_performed_by_Codex'] is False and record['independent_review_reperformed_by_Codex'] is False)
    check('formal_adoption_flags', formal['status'] == 'ADOPTED' and formal['adopted'] is True and formal['formula_policy']['coverage_model_adopted'] is True)
    check('coverage_not_certified', record['coverage_model_conditions_certified'] is False and record['evidence_classification'] == formal['evidence_classification'] == c.EVIDENCE_CLASSIFICATION == 'CONDITIONAL_ENGINEERING_NOT_CERTIFIED')
    check('no_runtime_correctness_claim', record['runtime_or_numerical_correctness_verified'] is False)
    check('execution_approval_explicitly_withheld', record['scientific_execution_authorized'] is False and record['user_explicitly_withheld_scientific_execution'] is True)
    check('formal_adoption_string_binds_context', ART + '/adoption_context.json' in formal['adoption_record'] and record['adoption_context_path'] in formal['adoption_record'])
    normalized = copy.deepcopy(formal)
    for key in ('status', 'adopted', 'adoption_record'):
        normalized[key] = candidate[key]
    normalized['formula_policy']['coverage_model_adopted'] = candidate['formula_policy']['coverage_model_adopted']
    check('only_adoption_metadata_changed', normalized == candidate)
    diff = get('candidate_to_adopted_semantic_diff.json')
    check('exact_four_field_diff', {row['path'] for row in diff['changed_fields']} == {'.status', '.adopted', '.adoption_record', '.formula_policy.coverage_model_adopted'} and len(diff['changed_fields']) == 4)
    check('17_precision_items_unchanged', set(formal['precision']) == set(c.EPS_KEYS) | {'dtype'} and formal['precision'] == candidate['precision'])
    check('conditional_assumptions_unchanged', formal['formula_policy']['assumptions'] == candidate['formula_policy']['assumptions'])
    check('original_derivation_label_preserved_as_history', formal['formula_policy']['coefficient_bound_evidence'] == 'AUTHOR_CONDITIONAL_DERIVATION_NOT_INDEPENDENTLY_REVIEWED' and bool(record['frozen_author_not_reviewed_label_meaning']))
    check('global_and_local_formula_selectors_unchanged', formal['formula_policy'] == dict(candidate['formula_policy'], coverage_model_adopted=True))
    for key in ('input', 'split_ids', 'basis', 'law', 'state', 'runtime', 'limits', 'decisions', 'tail_bound_provenance', 'source_commit', 'source_sha256', 'scientific_scope_record'):
        check('frozen_condition:' + key, formal[key] == candidate[key])
    result = c.validate_config(formal, require_adopted=True)
    check('adopted_contract_JSON_complete', result['valid_static_shape'] and result['execution_ready'] and not result['errors'] and not result['pending'])
    check('configuration_completeness_not_permission', result['scientific_processing_performed'] is False and get('adoption_status.json')['actual_execution_permitted'] is False)
    for path, wanted in formal['source_sha256'].items():
        raw = (ROOT / path).read_bytes()
        check('frozen_source_bytes:' + path, sha(raw) == wanted and raw == git('show', SOURCE + ':' + path))
    for path, wanted in formal['tail_bound_provenance']['files_sha256'].items():
        raw = (ROOT / path).read_bytes()
        check('frozen_proof_bytes:' + path, sha(raw) == wanted and raw == git('show', PROOF + ':' + path))
    for earlier, later in ((PROOF, SOURCE), (SOURCE, BASE)):
        check('commit_ancestry:' + earlier + '<' + later, earlier != later and subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', earlier, later]).returncode == 0)
    input_raw = (ROOT / formal['input']['path']).read_bytes()
    check('input_raw_hash_only_no_arrays', len(input_raw) == 8652 and sha(input_raw) == formal['input']['sha256'])
    scope = formal['scientific_scope_record']
    scope_raw = (ROOT / scope['path']).read_bytes()
    check('original_scope_hash', sha(scope_raw) == scope['sha256'])
    scope_data = c.read_json_bytes(scope_raw)
    for key in ('input', 'split_ids', 'basis', 'law', 'state', 'decisions'):
        check('scope_consistency:' + key, scope_data[key] == formal[key])
    table = get('precision_specification_17_adopted.json')
    check('17_adopted_rows', table['precision_adopted'] is True and len(table['items']) == 17)
    for row in table['items']:
        check('adopted_precision:' + row['key'], row['formally_adopted'] is True and row['adopted_value_or_rule'] == formal['precision'][row['key']])
    rt = get('runtime_resource_adopted.json')
    check('runtime_caps_adopted_unchanged', rt['adopted'] is True and rt['runtime'] == formal['runtime'] and rt['limits'] == formal['limits'])
    check('no_execution_namespace', not (ROOT / c.AUTHORIZATION_PREFIX).exists() and not (ROOT / c.OUTPUT_PREFIX).exists())
    for auth_path, auth_commit in (('artifacts/track_b_m1s_gate0_execution_authorization/2026-10-16/v4/execution_authorization.json', BASE), (c.AUTHORIZATION_PREFIX + 'not_created.json', 'not-an-authorization-commit')):
        try:
            c.authorize(ROOT, record['formal_contract_path'], auth_commit, auth_path)
        except ValueError:
            rejected = True
        else:
            rejected = False
        check('missing_or_legacy_authority_rejected:' + auth_path, rejected)
    old_result = c.read_json_bytes((ROOT / 'artifacts/track_b_m1s_gate0_result/2026-10-16/v4_one_shot/result.json').read_bytes())
    check('v4_STOP_and_null_preserved', old_result['status'] == 'TECHNICAL_STOP' and all(all(row[k] is None for k in ('W', 'v', 'rho')) for row in old_result['rows']))
    marker = 'artifacts/track_b_m1s_gate0_execution_audit/2026-10-16/v4_one_shot/exclusive_launch_marker.json'
    check('v4_consumed_marker_unchanged', (ROOT / marker).read_bytes() == git('show', BASE + ':' + marker) == Path('/tmp/m1s_gate0_v4_once_b70abf111e6459c45c24331a2c7d99c05e7dc228.launch.json').read_bytes())
    auth_tree = ast.parse((ROOT / c.SOURCE_PATHS[1]).read_bytes())
    auth = next(n for n in auth_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'authorize')
    text = ast.unparse(auth)
    check('separate_execution_consent_required', 'explicit_user_scientific_execution_approval' in text and 'm1s_gate0_separate_execution_authorization_v5' in text)
    check('future_authorization_after_contract', 'cc == head' in text and 'sc in (head, cc)' in text)
    indexes = {'PROJECT_MAP.md', 'VALIDATION_STATUS.md', 'docs/research/研究概要・現状.md', 'docs/README.md', 'scripts/README.md', 'src/trotterlib/README.md'}
    allowed = indexes | {'artifacts/validation_manifest.json'}
    modified = {p for p in git('diff', '--name-only', '--diff-filter=MDR', '-z', BASE).decode().split('\0') if p}
    check('all_old_SOURCE_results_contracts_unchanged', modified == allowed)
    for path in sorted(indexes):
        check('index_body_preserved:' + path, (ROOT / path).read_bytes().endswith(git('show', BASE + ':' + path)))
        for target in re.findall(r'\]\(([^)]+)\)', (ROOT / path).read_text().split('\n\n', 1)[0]):
            destination = (ROOT / path).parent.joinpath(target.split('#')[0]).resolve()
            check('new_index_link:' + path, destination.is_relative_to(ROOT) and destination.is_file())
    before = c.read_json_bytes(git('show', BASE + ':artifacts/validation_manifest.json'))
    after = c.read_json_bytes((ROOT / 'artifacts/validation_manifest.json').read_bytes())
    added = set(after) - set(before)
    check('one_adoption_manifest_entry', added == {'track_b_m1s_gate0_v5_formal_adoption_20261016'} and all(after[k] == v for k, v in before.items()))
    docs = ['docs/tracks/algorithm_codesign/m1s_gate0_headroom_execution_contract_v5_formally_adopted_20261016.md', 'docs/research/研究ノート/2026-10-16_track_b_M1S_Gate0_v5_formal_adoption.md']
    for path in docs:
        body = (ROOT / path).read_text()
        check('mandatory_STOP:' + path, 'mandatory STOP' in body)
        for target in re.findall(r'\]\(([^)]+)\)', body):
            if target.startswith(('https://', 'http://', '#')):
                continue
            check('local_link:' + path + ':' + target, (ROOT / path).parent.joinpath(target.split('#')[0]).exists())
    check('no_formula_methods_called', not formula_calls)
    check('no_scientific_imports', not any(n in sys.modules for n in ('numpy', 'scipy', 'openfermion', 'qiskit')) and not any(n.endswith('.m1s_gate0_v5.science') for n in sys.modules))
    report = {'schema': 'm1s_gate0_v5_adoption_static_validation_v1', 'PASS': len(checks), 'FAIL': 0, 'checks': checks,
              'schema_validation': result, 'contract_configuration_complete': True, 'actual_execution_permitted': False,
              'scientific_execution_authorized': False, 'coverage_model_conditions_certified': False,
              'runtime_or_numerical_correctness_verified': False, 'new_mathematical_review_performed': False,
              'numerical_formula_evaluations': len(formula_calls), 'NPZ_arrays_decoded': 0,
              'Pauli_generation': 0, 'W_v_rho_evaluations': 0, 'scientific_runner_invocations': 0,
              'new_scientific_execution_authorizations': 0, 'new_grants': 0, 'new_markers': 0, 'mandatory_STOP': True}
    if args.write_report:
        Path(args.write_report).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'PASS': report['PASS'], 'FAIL': 0, 'contract_complete': True, 'actual_execution_permitted': False, 'mandatory_STOP': True}))


if __name__ == '__main__':
    main()
