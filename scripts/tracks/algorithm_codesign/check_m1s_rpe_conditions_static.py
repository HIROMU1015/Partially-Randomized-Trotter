"""Metadata-only audit. Never import scientific modules or evaluate formulas."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[3]
BASE = 'e5ed0db6b8f50ed02ee9bc17cc6d0e2d4ed0898b'
ART = 'artifacts/track_b_m1s_rpe_conditions/2026-10-17/v1'
DOC = 'docs/tracks/algorithm_codesign/m1s_rpe_applicability_and_total_cost_decision_packet_20261017.md'
CANDIDATE = 'docs/tracks/algorithm_codesign/m1s_rpe_next_verification_contract_candidate_20261017.md'
CHECKER = 'scripts/tracks/algorithm_codesign/check_m1s_rpe_conditions_static.py'
INDEX_PATHS = ['PROJECT_MAP.md', 'docs/README.md', 'docs/research/研究概要・現状.md', 'scripts/README.md']
NOTE = 'docs/research/研究ノート/2026-10-17_track_b_M1S_rpe_conditions_preparation.md'
MANIFEST = 'artifacts/validation_manifest.json'
KEY = 'track_b_m1s_rpe_conditions_prepared_20261017'


def git(*args):
    return subprocess.check_output(
        ['git', '-C', str(ROOT), *args],
        env=dict(os.environ, GIT_OPTIONAL_LOCKS='0'))


def main():
    checks = []

    def check(name, passed):
        checks.append({'name': name, 'status': 'PASS' if passed else 'FAIL'})

    def read_json(path):
        return json.loads((ROOT / path).read_text(encoding='utf-8'))

    contract = read_json(ART + '/next_verification_contract_candidate.json')
    identity = read_json(ART + '/input_identity.json')
    check('fixed_base_commit', identity['base_commit'] == BASE)
    check('candidate_not_execution_contract', contract['schema'] == 'm1s_rpe_conditions_draft_v1')
    for flag in ['adopted', 'coverage_model_adopted', 'contract_grants_execution',
                 'scientific_execution_authorized', 'run_ready']:
        check('closed:' + flag, contract[flag] is False)
    check('STOP_required', contract['mandatory_STOP'] is True)
    check('no_execution_entrypoint', contract['execution_entrypoint'] is None)
    check('no_selected_scientific_choices', all(v is None for v in contract['open_scientific_choices'].values()))
    check('all_D1_D11_present', set(contract['decision_items']) == {'D' + str(i) for i in range(1, 12)})
    check('no_numerical_outputs', all(v is None for v in contract['numerical_outputs'].values()))
    check('no_future_option_selected', all(v['selected'] is False for v in contract['future_options']))
    check('separate_future_execution_approval', all(
        v.get('requires_separate_scientific_execution_approval') is True
        for v in contract['future_options'] if v['id'] != 'V0'))
    for prohibition in ['shot_forecast', 'ranking', 'RPE_simulation',
                        'circuit_build_compile_synthesis', 'formal_scientific_adoption',
                        'execution_authorization_grant_marker_issuance']:
        check('explicit_prohibition:' + prohibition, prohibition in contract['prohibitions'])
    check('no_time_caps_by_default', contract['future_resource_policy']['wall_time_seconds'] is None
          and contract['future_resource_policy']['cpu_time_seconds'] is None)
    check('caps_not_adopted', contract['future_resource_policy']['adopted'] is False)
    check('formulas_unevaluated', contract['formulas_evaluated'] is False)
    check('no_future_scientific_source_prepared', contract['future_scientific_source_prepared'] is False)
    check('no_new_review', contract['new_detailed_scientific_review_started'] is False)
    check('no_source_or_criteria_adoption', contract['research_direction_adopted'] is False)
    expected_formula_ids = {'P1', 'P2', 'P3', 'R1', 'S1', 'C1', 'G1', 'G2'}
    check('formula_inventory', set(contract['symbolic_conditions']) == expected_formula_ids)
    for name, spec in contract['symbolic_conditions'].items():
        check('formula_not_evaluated:' + name, spec['evaluated'] is False)
        check('formula_scope_explicit:' + name, bool(spec['assumptions']) and bool(spec['classification']))
    for record in identity['files']:
        raw = (ROOT / record['path']).read_bytes()
        frozen = git('show', record['commit'] + ':' + record['path'])
        check('input_frozen_bytes:' + record['path'], raw == frozen)
        check('input_sha256:' + record['path'], hashlib.sha256(raw).hexdigest() == record['sha256'])
        check('input_bytes:' + record['path'], len(raw) == record['bytes'])

    for path in [DOC, CANDIDATE, NOTE]:
        text = (ROOT / path).read_text(encoding='utf-8')
        check('UTF8_LF:' + path, '\r' not in text and text.endswith('\n'))
        check('no_trailing_whitespace:' + path, all(line == line.rstrip() for line in text.splitlines()))
        check('math_delimiters:' + path, text.count(r'\[') == text.count(r'\]'))
        definitions = dict(re.findall(r'^\[([^\]]+)\]:\s*(\S+)', text, re.M))
        labels = re.findall(r'\]\[([^\]]+)\]', text)
        check('reference_labels:' + path, all(label in definitions for label in labels))
        for label, target in definitions.items():
            if not target.startswith(('http://', 'https://')):
                resolved = ((ROOT / path).parent / target).resolve()
                check('reference_exists:' + path + ':' + label,
                      resolved.is_file() or resolved == ROOT / ART / 'static_validation.json')
    packet = (ROOT / DOC).read_text(encoding='utf-8')
    for name in expected_formula_ids:
        check('formula_in_packet:' + name, '\\tag{' + name + '}' in packet)
    for token in ['二重計上', r'\pi/3', 'RMSE', 'IID', 'SPAM', 'SOURCE', 'proxy', 'D11', 'G9']:
        check('required_limitation:' + token, token in packet)

    for path in INDEX_PATHS:
        check('old_index_body_preserved:' + path, (ROOT / path).read_bytes().endswith(git('show', BASE + ':' + path)))
    old_manifest = json.loads(git('show', BASE + ':' + MANIFEST))
    new_manifest = read_json(MANIFEST)
    check('only_new_manifest_entry', set(new_manifest) - set(old_manifest) == {KEY})
    check('old_manifest_entries_preserved', all(new_manifest.get(k) == v for k, v in old_manifest.items()))
    check('manifest_not_numerical_evidence', new_manifest[KEY]['evidence_kind'] == 'NON_NUMERICAL_DESIGN_ONLY')
    check('manifest_not_adopted', new_manifest[KEY]['adopted'] is False)

    allowed = {DOC, CANDIDATE, CHECKER, NOTE, MANIFEST, *INDEX_PATHS}
    changed = set(filter(None, git('diff', '--name-only', '-z', BASE).decode().split('\0')))
    untracked = set(filter(None, git('ls-files', '--others', '--exclude-standard', '-z').decode().split('\0')))
    check('only_scoped_files_changed', all(p in allowed or p.startswith(ART + '/') for p in changed | untracked))
    check('no_old_numerical_source_changes', not any(p.startswith('src/') for p in changed | untracked))
    check('no_new_authorization_grant_marker_files', not any(
        any(t in Path(p).name.lower() for t in ('authorization', 'grant', 'marker'))
        for p in changed | untracked))

    # AST inspection only; no imports or execution of the inspected library.
    resource_source = (ROOT / 'src/trotterlib/rpe_resource_accounting.py').read_text()
    tree = ast.parse(resource_source)
    functions = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    check('existing_Hoeffding_helper_located', '_hoeffding_shots' in functions)
    check('strict_branch_constant_located', 'RPE_STRICT_BRANCH_CERTIFICATION_LIMIT = math.pi / 3.0' in resource_source)
    check('unit_radius_assumption_disclosed', 'unit_unattenuated_survival_signal_radius_assumed' in resource_source)
    check('reuse_limit_disclosed', 'rte_trajectory_reuse_not_covered_by_hoeffding_guarantee' in resource_source)
    own_tree = ast.parse((ROOT / CHECKER).read_text())
    imports = {n.names[0].name.split('.')[0] if isinstance(n, ast.Import) else n.module.split('.')[0]
               for n in ast.walk(own_tree) if isinstance(n, (ast.Import, ast.ImportFrom))}
    check('checker_standard_library_only', imports <= {'ast', 'hashlib', 'json', 'os', 'pathlib', 're', 'subprocess'})
    check('checker_no_eval_or_exec', not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
          and n.func.id in {'eval', 'exec', '__import__'} for n in ast.walk(own_tree)))
    report = {'schema': 'm1s_rpe_conditions_metadata_audit_v1',
              'base_commit': BASE,
              'PASS': sum(c['status'] == 'PASS' for c in checks),
              'FAIL': sum(c['status'] == 'FAIL' for c in checks),
              'scope': 'hash/schema/AST/document/preservation only; no scientific formula evaluation',
              'checks': checks, 'new_scientific_calculations': 0,
              'new_scientific_reanalysis': 0, 'mandatory_STOP': True}
    output = ROOT / ART / 'static_validation.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'checks'}, ensure_ascii=False))
    if report['FAIL']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
