#!/usr/bin/env python3
"""Audit documentary metadata and literal transcription. Never evaluate scientific values.

Uses only stdlib and explicit Git blobs. No science runners, arrays, arithmetic of
costs/ratios, rankings, sampling, statistics, network or other worktree traversal.
Run from any directory; the default output is stdout. --output writes a new audit
receipt in the evidence bundle only. It does not alter any source input.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / 'artifacts/resource_applicability/track_a_h4_h8_cross_system_evidence/2026-10-17'
REQUIRED = 'system study_id study_scope geometry basis charge spin_policy electron_numbers spin_modes sector_dimension DF_policy DF_rank state_definition reference_definition T q_domain delta_domain prefix_domain PF_order PR_r_domain PR_K_domain control_policy gaussian_lowering compiler_profile cost_metric state_preparation_policy epsilon_domain allowance_policy alpha_axis candidate_domain deterministic_baselines PR_sampling_policy PR_sample_count selection_policy result_status SOURCE_commit result_commit evidence_paths limitations'.split()
COST = {'DIRECT_NATIVE_COMPILED', 'SAVED_COMPILED_MEASUREMENT', 'MODEL_ESTIMATED', 'FINITE_SAMPLE_MEAN', 'MIXED_MEASURED_AND_MODEL', 'NOT_AVAILABLE'}
COMPARABILITY = {'MATCHED_CONDITIONS', 'LIMITED_COMPARISON', 'DIFFERENT_SCIENTIFIC_CONTRACT', 'INSUFFICIENT_EVIDENCE'}

def git(*args):
    return subprocess.check_output(['git', '--no-optional-locks', '-c', 'core.fsmonitor=false', '-C', str(REPO), *args], stderr=subprocess.PIPE)

def load(name):
    return json.loads((BUNDLE / name).read_text(), parse_float=str)

def rows(name):
    with (BUNDLE / name).open(newline='') as f:
        return list(csv.DictReader(f))

def require(ok, message):
    if not ok:
        raise ValueError(message)

def audit():
    catalog = load('source_catalog_v1.json')['sources']
    require(len({x['evidence_id'] for x in catalog}) == len(catalog), 'duplicate evidence ID')
    texts = {}
    for x in catalog:
        require(re.fullmatch('[0-9a-f]{40}', x['commit']), 'non-exact commit')
        path = Path(x['path'])
        require(not path.is_absolute() and '..' not in path.parts, 'unsafe source path')
        git('cat-file', '-e', x['commit'] + '^{commit}')
        blob = git('rev-parse', x['commit'] + ':' + x['path']).decode().strip()
        raw = git('cat-file', 'blob', blob)
        require(blob == x['git_blob'], 'Git blob mismatch ' + x['evidence_id'])
        require(hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == blob, 'Git content hash')
        require(len(raw) == x['bytes'] and hashlib.sha256(raw).hexdigest() == x['sha256'], 'bytes/SHA mismatch')
        require(x['url'] == 'https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/' + x['commit'] + '/' + x['path'], 'URL/version mismatch')
        texts[x['evidence_id']] = raw.decode()
    lookup = {x['evidence_id']: x for x in catalog}
    conditions = rows('comparison_conditions_v1.csv')
    require(conditions == load('study_inventory_v1.json')['rows'], 'CSV/JSON study projection mismatch')
    studies = {x['study_id']: x for x in conditions}
    require(len(studies) == len(conditions), 'duplicate study ID')
    require(set(x['system'] for x in conditions) == {'H4', 'H5', 'H6', 'H7', 'H8'}, 'systems missing')
    commits = set()
    for x in conditions:
        require(all(k in x and x[k].strip() for k in REQUIRED), 'missing/blank required field')
        require(x['numerical_certificate'] == 'NOT_AVAILABLE' and x['population_mean_certificate'] == 'NOT_AVAILABLE', 'invented certificate')
        require(set(x['cost_evidence'].split(';')) <= COST, 'cost class')
        require(all(e in lookup for e in x['evidence_ids'].split(';')), 'unknown evidence ID')
        require(x['result_commit'] == lookup[x['evidence_ids'].split(';')[0]]['commit'], 'result reference mismatch')
        require(set(json.loads(x['source_roles']).values()) == set(x['SOURCE_commit'].split(';')), 'SOURCE roles mismatch')
        commits.update(x['SOURCE_commit'].split(';'))
        for p in x['parent_studies'].split(';'):
            require(p == 'NONE' or p in studies, 'unknown parent')
    for c in commits:
        require(re.fullmatch('[0-9a-f]{40}', c), 'invalid SOURCE commit')
        git('cat-file', '-e', c + '^{commit}')
    def visit(sid, stack):
        require(sid not in stack, 'cyclic parent study')
        for p in studies[sid]['parent_studies'].split(';'):
            if p != 'NONE':
                visit(p, stack | {sid})
    for sid in studies:
        visit(sid, set())

    ledger = rows('evidence_ledger_v1.csv')
    require(len({x['evidence_row_id'] for x in ledger}) == len(ledger), 'duplicate ledger ID')
    require(len({(x['study_id'], x['metric'], x['comparison_scope'], x['source_line'], x['source_field']) for x in ledger}) == len(ledger), 'duplicate transcription key')
    for x in ledger:
        require(x['study_id'] in studies and x['cost_evidence'] in COST, 'ledger study/class')
        e = lookup[x['evidence_id']]
        require(x['result_commit'] == e['commit'] and x['evidence_path'] == e['path'] and x['evidence_file_sha256'] == e['sha256'], 'ledger provenance mismatch')
        require(x['baseline_denominator'] and x['scientific_guarantee'], 'missing denominator/limit')
        source = texts[x['evidence_id']]
        if x['source_field'] == 'TEXT_LITERAL':
            line = source.splitlines()[int(x['source_line']) - 1]
            require(line == x['source_excerpt'] and x['value_literal'] in line, 'literal transcription mismatch ' + x['evidence_row_id'])
        else:
            source_row = list(csv.DictReader(io.StringIO(source)))[int(x['source_line']) - 2]
            require(source_row[x['source_field']] == x['value_literal'], 'saved CSV cell mismatch')
            for k in ['epsilon', 'factor', 'scope', 'bias_kind']:
                require(str(source_row[k]) in x['comparison_scope'], 'CSV selector mismatch')
            require(source_row['winner_after'] in x['baseline_denominator'], 'selected baseline mismatch')

    comps = rows('comparability_audit_v1.csv')
    require(len({x['comparison_id'] for x in comps}) == len(comps), 'duplicate comparison')
    require({x['status'] for x in comps} == COMPARABILITY, 'comparability states')
    issues = load('open_evidence_items_v1.json')['items']
    issue_fields = 'issue_id affected_studies missing_evidence existing_partial_evidence what_is_already_verified what_is_not_verified possible_codex_task new_scientific_computation_required user_authorization_required related_SOURCE related_result_commit'.split()
    require(len({x['issue_id'] for x in issues}) == len(issues), 'duplicate issue')
    for x in issues:
        require(all(k in x and x[k] for k in issue_fields), 'missing issue field')
    for x in comps + issues:
        key = 'study_ids' if 'study_ids' in x else 'affected_studies'
        require(all(s in studies for s in x[key].split(';')), 'unknown linked study')
        require(all(e in lookup for e in x['evidence_ids'].split(';')), 'unknown linked evidence')
    scope = load('audit_scope_v1.json')
    require(scope['study_count'] == len(studies) and scope['numerical_literal_rows'] == len(ledger) and scope['fixed_evidence_files'] == len(catalog), 'inventory counts')
    require(all(v == 0 for v in scope['scientific_operations'].values()), 'scope permits science')
    require(scope['next_scientific_stage_authorized'] is False, 'next stage authorized')

    # Only the new report/bundle links are inspected; no broad file traversal.
    report = REPO / 'docs/research/track_a_h4_h8_cross_system_evidence_inventory_20261017.md'
    docs = [report] + list(BUNDLE.glob('*.md'))
    checked_links = 0
    for doc in docs:
        for target in re.findall(r'\]\(([^)]+)\)', doc.read_text()):
            if target.startswith('https://'):
                require(target in {x['url'] for x in catalog}, 'unregistered external source URL')
                continue
            local = (doc.parent / target.split('#')[0]).resolve()
            require(local.is_relative_to(REPO) and local.exists(), 'missing/unsafe local link ' + target)
            checked_links += 1
    # New outputs contain no raw binaries, private grants or lock files.
    public_files = [p for p in BUNDLE.iterdir() if p.is_file()]
    require(all(p.suffix in {'.md', '.json', '.csv'} for p in public_files), 'unexpected artifact format')
    return dict(schema_version=1, status='PASS_DOCUMENTARY_METADATA_AUDIT', studies=len(studies), saved_literal_rows=len(ledger), fixed_evidence_files=len(catalog), SOURCE_commits_verified=len(commits), comparison_pairs=len(comps), open_items=len(issues), local_links_checked=checked_links, tests=['exact Git commit/blob and byte/SHA256', 'mandatory schema and CSV/JSON identity', 'SOURCE existence and role mapping (not full runtime closure)', 'parent lineage acyclic; no duplicate study/transcription keys', 'literal text lines / exact CSV cells; no evaluation of scientific values', 'explicit baseline denominator and evidence classifications', 'unknown/certificate status separation', 'local links and fixed external source URLs'], scientific_computation_executed=False, costs_ratios_statistics_recomputed=False, source_or_other_worktrees_modified=False, independent_review=False, next_stage_authorized=False)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', action='store_true', help='write metadata-only validation receipt in the new evidence bundle')
    args = parser.parse_args()
    result = audit()
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        (BUNDLE / 'technical_validation_v1.json').write_text(text)
    sys.stdout.write(text)

if __name__ == '__main__':
    main()
