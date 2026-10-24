#!/usr/bin/env python3
"""Document/schema and synthetic boundary checks; never import science code."""
import ast
import copy
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys

from information_boundary import validate_nonexecuting_contract, validate_synthetic_view

ROOT = Path(__file__).resolve().parents[1]
WORKTREE = ROOT.parents[2]
BASE = '8a2fbe9461d9cc262326fa30f9778b295adeb3f8'
PREFIX = ROOT.relative_to(WORKTREE).as_posix() + '/'
checks = []

# Reuse only these three stdlib helpers from the old fixed document checker.
# AST extraction avoids its imports, old campaign, and former branch checks.
old = subprocess.check_output(['git', '-C', str(WORKTREE), 'show',
    '501b4232b3e9ef9778e917f0a86f1d6c1d05f189:docs/research/prior_art_novelty_20261021/audit/check_documents.py'])
tree = ast.parse(old)
helpers = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                         and n.name in ('git', 'digest', 'check')], type_ignores=[])
exec(compile(helpers, '<reused-document-helpers>', 'exec'), globals())


def load(name):
    return json.loads((ROOT / name).read_text(), parse_float=str)


def reject(name, function, code):
    try:
        function()
    except ValueError as exc:
        check(name, str(exc) == code, str(exc))
    else:
        check(name, False, 'Expected rejection was absent')


def main():
    changes = [p for p in git('diff', BASE, '--name-only', '-z').decode().split('\0') if p]
    new = [p for p in git('ls-files', '--others', '--exclude-standard', '-z').decode().split('\0') if p]
    paths = changes + new
    check('separate_research_directory_only', bool(paths) and all(p.startswith(PREFIX) for p in paths),
          'First-study SOURCE/results/CEM/paper assets/grant/ledger/STOP unchanged by this branch')
    base_paths = set(git('ls-tree', '-r', '--name-only', BASE).decode().splitlines())
    check('no_existing_file_rewritten', not (set(changes) & base_paths), sorted(set(changes) & base_paths))

    catalog = load('source_catalog.json')
    issues = []
    for key, row in catalog.items():
        raw = git('show', row['commit'] + ':' + row['path'])
        if digest(raw) != row['sha256']:
            issues.append(key)
        if 'symbols' in row:
            nodes = {n.name: n for n in ast.walk(ast.parse(raw)) if isinstance(n, ast.FunctionDef)}
            if any(name not in nodes or nodes[name].lineno != spec['line'] for name, spec in row['symbols'].items()):
                issues.append(key + ':symbol')
    check('fixed_sources_and_real_symbols_resolve', not issues, {'sources': len(catalog), 'issues': issues})

    contract = load('pilot_contract_DRAFT.json')
    check('nonexecuting_contract', validate_nonexecuting_contract(contract), 'No runner SOURCE or valid grant')
    check('draft_event_budget_consistent', sum(contract['planned_outer_event_budget'].values()) == contract['planned_outer_total'] == 53,
          'Design only; internal implementation audit remains pending')
    pf = catalog['PF_INDEX']
    rows = list(csv.DictReader(io.StringIO(git('show', pf['commit'] + ':' + pf['path']).decode())))
    domain = contract['domain']
    selected = {r['id']: r for r in rows}
    check('pilot_16_ids_metadata_binding', len(domain) == 16 and len({r['id'] for r in domain}) == 16
          and all(all(str(row[k]) == selected[row['id']][k] for k in ('system', 'method', 'formula', 'prefix', 'q', 'dataset_pointer')) for row in domain),
          'No outcome-based selection or reranking')
    check('fair_full_domain_and_no_formula_alias', all(r['method'] == 'FULL' and r['formula'] in ('2nd', '4th') for r in domain)
          and {(r['system'], r['formula'], r['q']) for r in domain} == {(s, f, q) for s in ('H4', 'H6') for f in ('2nd', '4th') for q in (1, 2, 4, 8)},
          'Equal eight-candidate domain per system; no PR or m5 mixing')

    # Schema/read-only keys only. No arithmetic with physical endpoint values.
    ref = catalog['PF_DATA']
    data = json.loads(gzip.decompress(git('show', ref['commit'] + ':' + ref['path'])), parse_float=str)
    by_pointer = {'/candidates/' + str(i): r for i, r in enumerate(data['candidates'])}
    endpoint_ok = True
    for row in domain:
        candidate = by_pointer[row['dataset_pointer']]
        endpoint_ok &= candidate['candidate_status_saved']['id'] == row['id']
        endpoints = [r for r in candidate['endpoints_saved'] if r['epsilon'] == contract['physical']['epsilon']]
        endpoint_ok &= len(endpoints) == 1 and all(k in endpoints[0] for k in ('status', 'N_cos', 'N_sin', 'price', 'total_RZ_exact'))
    check('pilot_saved_endpoint_schema_exists', bool(endpoint_ok), 'Only metadata, field presence, epsilon identifier checked; outcome values not evaluated')

    with (ROOT / 'information_tiers.csv').open() as stream:
        fields = list(csv.DictReader(stream))
    names = {r['field'] for r in fields}
    header_names = set(rows[0])
    primary = catalog['PRIMARY']
    primary_header = set(next(csv.reader(io.StringIO(git('show', primary['commit'] + ':' + primary['path']).decode()))))
    check('all_primary_and_index_fields_classified', header_names | primary_header <= names, sorted((header_names | primary_header) - names))
    check('tier_sources_and_lineage_registered', len(names) == len(fields) and {r['tier'] for r in fields} == {'Tier0', 'Tier1', 'Tier2', 'Oracle'}
          and all(set(r['source_ref'].split(';')) <= set(catalog) and set(filter(None, r['ancestors'].split(';'))) <= names for r in fields),
          {'field_roles': len(fields)})

    def view(ancestors, **kw):
        return validate_synthetic_view('ARTIFICIAL', [{'name': 'synthetic_feature', 'ancestors': ancestors}], **kw)
    check('synthetic_tier0_and_structure_allowed', view(['q', 'structural_length']) == ['synthetic_feature'], 'No real feature scores computed')
    reject('oracle_label_refused', lambda: view(['G']), 'ORACLE_LEAKAGE')
    reject('renamed_oracle_feature_refused', lambda: view(['q', 'total_RZ_exact']), 'ORACLE_LEAKAGE')
    reject('unpaid_bias_refused', lambda: view(['bias_cos']), 'UNPAID_TIER2_INFORMATION')
    check('paid_query_channel_explicit', view(['eligibility_status'], paid_fields=['eligibility_status']) == ['synthetic_feature'], 'Only synthetic query declaration; no real query')
    reject('paid_status_does_not_release_cost', lambda: view(['G'], paid_fields=['eligibility_status']), 'ORACLE_LEAKAGE')
    reject('saved_normalization_not_free', lambda: view(['log_B_base']), 'UNPAID_TIER2_INFORMATION')
    reject('ground_state_information_refused', lambda: view(['ground_state_vector']), 'ORACLE_LEAKAGE')
    reject('winner_or_saved_eligibility_not_metadata', lambda: view(['eligible_epsilons']), 'ORACLE_LEAKAGE')
    reject('unregistered_proxy_refused', lambda: view(['perfect_proxy']), 'UNREGISTERED_SOURCE_FIELD')
    reject('identity_is_not_performance_feature', lambda: view(['candidate_id']), 'IDENTITY_IS_NOT_PERFORMANCE_FEATURE')
    reject('real_data_interface_refused', lambda: validate_synthetic_view('SAVED_REAL_VALUES', []), 'NO_REAL_DATA_EXECUTION_INTERFACE_IN_C0')
    reject('missing_lineage_refused', lambda: validate_synthetic_view('ARTIFICIAL', [{'name': 'x', 'ancestors': []}]), 'MISSING_SOURCE_LINEAGE')
    reject('duplicate_feature_names_refused', lambda: validate_synthetic_view('ARTIFICIAL', [{'name': 'x', 'ancestors': ['q']}] * 2), 'DUPLICATE_FEATURE_NAME')
    for key, value in [('execution_authorized', True), ('enabled', True), ('grant', {'valid': True}), ('runner_SOURCE', 'fake_source')]:
        altered = copy.deepcopy(contract); altered[key] = value
        reject('contract_' + key + '_activation_refused', lambda c=altered: validate_nonexecuting_contract(c), 'C0_CONTRACT_MUST_REMAIN_NONEXECUTABLE')

    stop = load('STOP.json')
    check('no_review_adoption_or_science', stop['new_scientific_calls'] == stop['new_saved_value_numerical_analysis'] == stop['new_grants_issued'] == 0
          and not stop['hypotheses_formally_adopted'] and not stop['detailed_scientific_review_started'], 'Mandatory STOP retained')
    check('policy_missing_original_recorded', not stop['policy_document']['formal_expected_original_obtained']
          and not stop['policy_document']['matches_expected'] and not stop['level1_review']['obtained'], 'Unavailable formal original/Level1 not claimed as read')
    docs = [p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    bad = []
    for p in docs:
        content = p.read_text()
        if re.search(r'/h[o]me/|/Us[e]rs/|/t[m]p/|gh[pousr]_[A-Za-z0-9]{20,}|BEGIN (?:RSA |OPENSSH )?PRIVATE KEY', content):
            bad.append(p.relative_to(ROOT).as_posix())
    check('new_public_text_contains_no_private_locators_or_credential_patterns', not bad, bad)
    all_text = '\n'.join(p.read_text() for p in ROOT.glob('*.md'))
    missing_links = []
    for path in re.findall(r'\]\(([^)]+)\)', all_text):
        if path == 'technical_checks.json':
            continue  # This check report is written after the checks below.
        if not path.startswith(('https://', 'http://')) and not (ROOT / path.split('#')[0]).exists():
            missing_links.append(path)
    check('local_document_links_resolve', not missing_links, missing_links)
    result = dict(schema='R2_CS_C0_TECHNICAL_CHECKS_V1',status='PASS' if all(r['pass'] for r in checks) else 'FAIL',
        checks=checks,check_count=len(checks),static_scientific_source_imports=0,new_scientific_calls=0,new_saved_value_numerical_analysis=0,
        field_classification_count=len(fields),fixed_source_count=len(catalog),metadata_preflight='PASS' if endpoint_ok else 'FAIL',
        executable_scientific_preflight='NOT_READY_NO_RUNNER_OR_APPROVAL',reviewer='PRIMARY_AGENT_AUTHOR_CHECK_NOT_INDEPENDENT_SCIENTIFIC_REVIEW')
    (ROOT / 'technical_checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k:result[k] for k in ('status', 'check_count', 'field_classification_count', 'fixed_source_count', 'new_scientific_calls', 'new_saved_value_numerical_analysis')}))
    for row in checks:
        if not row['pass']:
            print(json.dumps(row, ensure_ascii=False))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
