"""Validate bounded O2 metadata and saved literals; no scientific arithmetic.

Reuses validate_synthesis.py/Phase2 patterns for Git, CSV and literal checks.
No scientific modules, report generators, simulators, compilers or RNG imports.
"""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import re
import subprocess

BASE = '3380f6de0d86c0ac1db9aeb6d97ab5d328dd43bd'
FILES = {'README.md', 'data_dictionary.md', 'study_catalog.csv', 'candidate_results.csv',
         'comparison_catalog.csv', 'df_pauli_matching.csv', 'SOURCE_INDEX.json',
         'VALIDATION.json', 'validate_catalog.py'}
ENV = dict(os.environ, GIT_OPTIONAL_LOCKS='0', GIT_NO_LAZY_FETCH='1')


def scalar(value):
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    assert isinstance(value, str), 'Numbers must be parsed as original literal strings'
    return value


def pointer(doc, path):
    assert path == '' or path.startswith('/')
    for token in path.split('/')[1:]:
        token = token.replace('~1', '/').replace('~0', '~')
        doc = doc[int(token)] if isinstance(doc, list) else doc[token]
    return doc


def read_csv(path):
    reader = csv.DictReader(io.StringIO(path.read_text(), newline=''))
    rows = list(reader)
    assert len(reader.fieldnames) == len(set(reader.fieldnames))
    assert rows and all(None not in row for row in rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args()
    out = Path(__file__).resolve().parent
    assert {p.name for p in out.iterdir() if p.is_file()} == FILES
    index = json.loads((out / 'SOURCE_INDEX.json').read_text())
    meta = json.loads((out / 'VALIDATION.json').read_text())
    assert index['base_integration'] == meta['base_integration'] == BASE
    assert all(value is False for value in meta['scope'].values())
    refs = index['references']

    def git(*arguments):
        return subprocess.check_output(['git', '-C', str(args.repo), *arguments], env=ENV)

    proc = subprocess.run(['git', '-C', str(args.repo), 'cat-file',
                           '--batch-check=%(objectname) %(objecttype) %(objectsize)'],
                          input=('\n'.join(e['commit'] + ':' + e['path'] for e in refs.values()) + '\n').encode(),
                          stdout=subprocess.PIPE, check=True, env=ENV)
    assert len(proc.stdout.splitlines()) == len(refs)
    for entry, line in zip(refs.values(), proc.stdout.decode().splitlines()):
        blob, kind, size = line.split()
        assert (blob, kind, size) == (entry['blob'], 'blob', str(entry['bytes']))
        assert re.fullmatch('[0-9a-f]{40}', entry['commit'])
        assert entry['url'] == ('https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/' +
                                entry['commit'] + '/' + entry['path'])

    cache = {}

    def raw(id):
        if id not in cache:
            entry = refs[id]
            cache[id] = git('show', entry['commit'] + ':' + entry['path'])
        return cache[id]

    def doc(id):
        return json.loads(raw(id), parse_float=str, parse_int=str)

    studies = read_csv(out / 'study_catalog.csv')
    candidates = read_csv(out / 'candidate_results.csv')
    comparisons = read_csv(out / 'comparison_catalog.csv')
    matching = read_csv(out / 'df_pauli_matching.csv')
    by_study = {row['study_id']: row for row in studies}
    by_comparison = {row['comparison_id']: row for row in comparisons}
    assert len(by_study) == len(studies)
    assert len(by_comparison) == len(comparisons)
    assert len({row['record_locator'] for row in candidates}) == len(candidates)
    assert len({row['matching_id'] for row in matching}) == len(matching)
    original_claims = list(csv.DictReader(io.StringIO(raw('BASE_CLAIMS').decode())))
    components = {row['component_id'] for row in original_claims}
    assert len(original_claims) == 35
    for row in studies:
        assert row['original_claim_component'] in components
        assert set(json.loads(row['result_refs'])) <= set(refs)
        assert set(json.loads(row['identity_refs'])) <= set(refs)
        assert row['representation'] in {'DF', 'Pauli-JW', 'DF/Pauli-JW'}
        assert json.loads(row['source_commit']) or row['source_resolution'].startswith('USE_FIXED_BINDING')
        recorded = {commit for role in json.loads(row['source_commit'])
                    for commit in re.findall(r'(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])', role)}
        availability = json.loads(row['source_availability'])
        assert {item['recorded_commit'] for item in availability} == recorded
        for item in availability:
            commit = item['recorded_commit']
            proc = subprocess.run(['git', '-C', str(args.repo), 'cat-file', '-t', commit],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=ENV)
            if item['local_state'] == 'LOCAL_COMMIT_PRESENT':
                assert proc.returncode == 0 and proc.stdout.strip() == b'commit'
            else:
                assert row['study_id'] == 'pr_boundary_execution_20261027'
                assert item['local_state'] == 'RECORDED_ID_ONLY;COMPLETE_SOURCE_PRIVATE'
                assert commit in raw('BOUNDARY_SUMMARY').decode()
    for study_id, ref_id in [
        ('pr2_matched_accuracy_m1_b1_execution', 'PRI_M1JSON'),
        ('pr2_matched_accuracy_m2_transfer_execution', 'PRI_M2JSON'),
        ('pr2_pm2_precision_analysis', 'PM2SUMMARY'),
        ('pr_boundary_execution_20261027', 'BOUNDARY_SUMMARY'),
        ('track_a_h8_long_time', 'H8_C_RAW'),
    ]:
        assert by_study[study_id]['research_status'] == doc(ref_id)['status']
    assert by_study['pr2_matched_accuracy_m1_b1_execution']['precision_domain'].startswith('epsilon_signal=.05（')
    assert by_study['pr2_matched_accuracy_m2_transfer_execution']['precision_domain'].startswith('epsilon_signal=.05（')
    assert by_study['pauli_pure_rte_mechanism_validation/bias']['cost_scope'].startswith('SIGNAL_DIAGNOSTIC_ONLY')
    for study_id, ref_id in [
        ('pauli_pure_rte_mechanism_validation/bias', 'H6_P1_BINDINGS'),
        ('pauli_pure_rte_mechanism_validation/cost', 'H6_P2_BINDINGS'),
        ('df_pauli_matched_partition_validation', 'P3_STAGE4_BINDINGS'),
        ('df_pauli_matched_partition_validation', 'P3_BIT_BINDINGS'),
    ]:
        assert doc(ref_id)['commit'] in by_study[study_id]['source_commit']
    projected = 0
    table_refs = 0
    for row in candidates:
        assert row['study_id'] in by_study
        entry = refs[row['ref_id']]
        assert (row['result_commit'], row['result_path']) == (entry['commit'], entry['path'])
        selector = json.loads(row['row_selector'])
        mapping = json.loads(row['projection_map'])
        if selector['type'] == 'csv_rows':
            # Table references with empty selector need only their original header.
            data = raw(row['ref_id']).decode('utf-8-sig')
            reader = csv.DictReader(io.StringIO(data, newline=''))
            assert set(selector['where']) <= set(reader.fieldnames)
            if row['record_kind'] == 'LITERAL_EXTRACT':
                selected = [record for record in reader if all(record[k] == v for k, v in selector['where'].items())]
                assert len(selected) == 1, row['record_locator']
                source = selected[0]
                for target, field in mapping.items():
                    assert row[target] == source[field], (row['record_locator'], target)
                    projected += 1
        elif selector['type'] == 'json_pointer':
            document = doc(row['ref_id'])
            source = pointer(document, selector['pointer'])
            if row['record_kind'] == 'LITERAL_EXTRACT':
                for target, field in mapping.items():
                    value = pointer(document, field['json_pointer']) if isinstance(field, dict) else source[field]
                    assert row[target] == scalar(value), (row['record_locator'], target)
                    projected += 1
        else:
            assert selector['type'] == 'text' and selector['anchor'] in raw(row['ref_id']).decode()
        if row['record_kind'] == 'TABLE_REFERENCE':
            table_refs += 1
            assert all(row[k] == 'NOT_PROJECTED' for k in ['N_cos', 'N_sin', 'C_cos', 'C_sin', 'G', 'n'])
        else:
            assert row['record_kind'] == 'LITERAL_EXTRACT'
            assert row['alias_of'].endswith('subset_view_not_another_experiment')
    for row in comparisons:
        assert row['study_id'] in by_study
        assert set(json.loads(row['related_study_ids'])) <= set(by_study)
        assert set(json.loads(row['data_refs'])) <= set(refs)
        assert row['new_comparison_created'] == row['new_numeric_evaluation'] == 'false'
        assert row['unseparated_factors']
    for row in matching:
        assert set(json.loads(row['comparison_ids'])) <= set(by_comparison)
        assert set(json.loads(row['study_ids'])) <= set(by_study)
        assert set(json.loads(row['evidence_refs'])) <= set(refs)
        assert row['causal_representation_only'] == 'NOT_ESTABLISHED'
        assert row['new_scientific_review'] == 'false'
        if row['comparison_class']:
            assert row['comparison_class'] in {'MATCHED', 'CONDITIONAL_MATCHED', 'CONTEXTUAL', 'NOT_COMPARABLE'}
        else:
            assert row['verification_state'] == 'UNCONFIRMED' and row['unconfirmed_items']
    by_match = {row['matching_id']: row for row in matching}
    assert by_match['H4_P3_common_hold']['comparison_class'] == 'CONDITIONAL_MATCHED'
    assert by_match['H4_DF12_vs_P3_DF10']['comparison_class'] == 'NOT_COMPARABLE'
    assert by_match['H4_control_only_support']['comparison_class'] == 'MATCHED'
    assert '同保存Pauli' in by_match['H4_control_only_support']['comparison_purpose']
    # These views share prices across precision and must not become new experiments.
    h6 = [r for r in candidates if r['record_kind'] == 'LITERAL_EXTRACT' and r['ref_id'] == 'PA_H6_CAND']
    assert len(h6) == 5 and len({r['shared_price_group'] for r in h6}) == 1
    h8 = [r for r in candidates if r['record_kind'] == 'LITERAL_EXTRACT' and r['ref_id'] == 'H8_C_RAW']
    assert {r['cohort_id'] for r in h8} == {'CONFIRMATION', 'EXPLORATION'}
    assert next(r for r in h8 if r['cohort_id'] == 'CONFIRMATION')['C_cos'] == 'NOT_RECORDED'
    assert next(r for r in h8 if r['cohort_id'] == 'EXPLORATION')['G_SE'] == 'null'
    assert any(r['original_decision'] == 'INCONCLUSIVE' for r in candidates)
    assert any(r['record_kind'] == 'LITERAL_EXTRACT' and r['original_decision'] == '' for r in candidates)
    # Existing synthesis, statuses and scientific paths remain untouched in output tree.
    protected = ['src', 'scripts', 'tests', 'artifacts', 'VALIDATION_STATUS.md', 'AGENTS.md',
                 'docs/research/pr_research_paper_evidence_synthesis_20261103']
    assert not git('diff', '--name-only', BASE, '--', *protected).strip()
    private = re.compile('/' + r'(?:home|tmp|mnt|scratch|Users)/|\b(?:\d{1,3}\.){3}\d{1,3}\b|'
                         r'gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----')
    for path in out.iterdir():
        if path.is_file():
            assert not private.search(path.read_text()), path.name
    for path in out.glob('*.md'):
        for target in re.findall(r'\[[^\]\n]+\]\(([^)\n]+)\)', path.read_text()):
            assert target in FILES and (out / target).is_file(), (path.name, target)
    expected = meta['counts']
    actual = dict(studies=len(studies), candidate_table_references=table_refs,
                  literal_records=len(candidates) - table_refs, saved_comparisons=len(comparisons),
                  matching_rows=len(matching), fixed_refs=len(refs))
    assert expected == actual
    print(json.dumps(dict(status='PASS_METADATA_AND_LITERAL_ONLY', **actual,
                          literal_fields_verified=projected, prior_claim_rows_unchanged=35,
                          new_scientific_execution=False, numeric_reanalysis=False,
                          new_scientific_PASS=False), indent=2))


if __name__ == '__main__':
    main()
