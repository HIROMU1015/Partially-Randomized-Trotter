"""Check figure planning metadata/literals; never generate figures or science.

Reuse the unchanged catalog's CSV, JSON-pointer and literal conventions and
both existing metadata validators. Only stdlib and read-only Git operations.
"""
import argparse
from collections import Counter
import importlib.util
import io
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys

BASE = '2779710a86d380bb484accc51d3dccba183370b8'
SYNTHESIS = '3380f6de0d86c0ac1db9aeb6d97ab5d328dd43bd'
CATALOG = 'docs/research/o2_data_catalog_20261105'
SYN_DIR = 'docs/research/pr_research_paper_evidence_synthesis_20261103'
OUTPUT = 'docs/research/o2_figure_planning_20261105'
FILES = {'README.md', 'figure_candidates.md', 'figure_data_plan.csv',
         'figure_readiness.csv', 'existing_figure_assets.csv', 'VALIDATION.json',
         'validate_figure_plan.py'}
READY = {'READY_FROM_SAVED_DATA', 'READY_WITH_CONDITIONS', 'NEEDS_NEW_ANALYSIS',
         'INSUFFICIENT_COMPARABILITY', 'UNCONFIRMED'}
ENV = dict(os.environ, GIT_OPTIONAL_LOCKS='0', GIT_NO_LAZY_FETCH='1',
           PYTHONDONTWRITEBYTECODE='1')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = Path(__file__).resolve().parent
    assert {p.name for p in out.iterdir() if p.is_file()} == FILES
    meta = json.loads((out / 'VALIDATION.json').read_text())
    assert meta['base_catalog'] == BASE and meta['base_synthesis'] == SYNTHESIS
    assert all(v is False for v in meta['scope'].values())
    assert meta['inherited_claim_scope'] == 'CURRENT_CLAIM_SCOPE_EVIDENCE_SUFFICIENT'
    # Import only unchanged stdlib metadata helpers; don't write into the catalog.
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('catalog_metadata', repo / CATALOG / 'validate_catalog.py')
    catalog = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(catalog)

    def git(*arguments):
        return subprocess.check_output(['git', '-C', str(repo), *arguments], env=ENV)

    # Output has no permission to alter the scientific tree or the base catalogs.
    changed = git('diff', '--name-only', BASE).decode().splitlines()
    allowed_pointers = {'PROJECT_MAP.md', 'docs/research/README.md'}
    assert all(p.startswith(OUTPUT + '/') or p in allowed_pointers for p in changed), changed
    assert not git('diff', '--name-only', BASE, '--', CATALOG, SYN_DIR).strip()
    for relative in [CATALOG, SYN_DIR]:
        for line in git('ls-tree', '-r', '--name-only', BASE, '--', relative).decode().splitlines():
            assert (repo / line).read_bytes() == git('show', BASE + ':' + line), line

    prior = {}
    for name, path in [('catalog', CATALOG + '/validate_catalog.py'),
                       ('synthesis', SYN_DIR + '/validate_synthesis.py')]:
        completed = subprocess.run([sys.executable, str(repo / path), '--repo', str(repo)],
                                   env=ENV, check=True, stdout=subprocess.PIPE, text=True)
        prior[name] = json.loads(completed.stdout)
        assert prior[name].get('new_scientific_PASS', False) is False

    index = json.loads((repo / CATALOG / 'SOURCE_INDEX.json').read_text())
    refs = dict(index['references'])
    assert not set(refs) & set(meta['supplemental_fixed_refs'])
    refs.update(meta['supplemental_fixed_refs'])
    for key, entry in meta['supplemental_fixed_refs'].items():
        assert re.fullmatch('[0-9a-f]{40}', entry['commit'])
        assert entry['url'] == ('https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/' +
                                entry['commit'] + '/' + entry['path'])
    batch = subprocess.run(['git', '-C', str(repo), 'cat-file',
                            '--batch-check=%(objectname) %(objecttype) %(objectsize)'],
                           input=('\n'.join(e['commit'] + ':' + e['path']
                                            for e in meta['supplemental_fixed_refs'].values()) + '\n').encode(),
                           stdout=subprocess.PIPE, check=True, env=ENV)
    lines = batch.stdout.decode().splitlines()
    assert len(lines) == len(meta['supplemental_fixed_refs'])
    for entry, line in zip(meta['supplemental_fixed_refs'].values(), lines):
        assert line.split() == [entry['blob'], 'blob', str(entry['bytes'])]

    cache = {}

    def raw(key):
        if key not in cache:
            e = refs[key]
            cache[key] = git('show', e['commit'] + ':' + e['path']).decode('utf-8-sig')
        return cache[key]

    for key, check in meta['static_code_checks'].items():
        assert check['executed'] is False
        assert all(anchor in raw(key) for anchor in check['anchors']), key

    plans = catalog.read_csv(out / 'figure_data_plan.csv')
    readiness = catalog.read_csv(out / 'figure_readiness.csv')
    assets = catalog.read_csv(out / 'existing_figure_assets.csv')
    studies = {r['study_id']: r for r in catalog.read_csv(repo / CATALOG / 'study_catalog.csv')}
    comparisons = {r['comparison_id']: r for r in catalog.read_csv(repo / CATALOG / 'comparison_catalog.csv')}
    candidates = {r['record_locator']: r for r in catalog.read_csv(repo / CATALOG / 'candidate_results.csv')}
    matches = {r['matching_id']: r for r in catalog.read_csv(repo / CATALOG / 'df_pauli_matching.csv')}
    by_id = {r['figure_candidate_id']: r for r in plans}
    by_ready = {r['figure_candidate_id']: r for r in readiness}
    by_asset = {r['asset_id']: r for r in assets}
    assert len(by_id) == len(plans) and set(by_ready) == set(by_id)
    assert len(by_ready) == len(readiness) and len(by_asset) == len(assets)
    bindings = literals = 0
    for p in plans:
        fid = p['figure_candidate_id']
        assert json.loads(p['study_ids']) and set(json.loads(p['study_ids'])) <= set(studies)
        assert set(json.loads(p['comparison_ids'])) <= set(comparisons)
        assert set(json.loads(p['source_refs'])) <= set(refs)
        assert set(json.loads(p['catalog_candidate_refs'])) <= set(candidates)
        assert set(json.loads(p['overlap_with'])) <= set(by_id)
        assert set(json.loads(p['existing_asset_ids'])) <= set(by_asset)
        assert json.loads(p['asset_reuse_policy'])
        assert p['plotting_readiness'] in READY
        assert p['plotting_readiness'] == by_ready[fid]['plotting_readiness']
        assert p['priority'] == by_ready[fid]['priority']
        assert 'analytic sufficient quantum shots' in p['unit_policy']
        assert 'classical price trajectories' in p['unit_policy']
        assert 'G and G_SE=RZ×shot' in p['unit_policy']
        assert p['cohort_policy'] and p['shared_price_groups'] and p['limitations']
        if p['matching_id']:
            saved = matches[p['matching_id']]
            assert json.loads(p['matching_record']) == saved
            assert p['comparison_class'] == (saved['comparison_class'] or 'UNCONFIRMED')
            assert saved['causal_representation_only'] == 'NOT_ESTABLISHED'
        if p['plotting_readiness'] == 'NEEDS_NEW_ANALYSIS':
            r = by_ready[fid]
            assert all(r[k] for k in ['needed_values_or_analysis', 'saved_data_insufficiency_reason',
                                     'new_scientific_work_needed', 'minimum_scope_if_later_requested'])
        if p['plotting_readiness'] == 'UNCONFIRMED':
            assert by_ready[fid]['unconfirmed_items']
        expected_groups = list(dict.fromkeys(c['shared_price_group'] for c in candidates.values()
                            if c['ref_id'] in {b['ref_id'] for b in json.loads(p['candidate_selectors'])}))
        assert json.loads(p['shared_price_groups']) == expected_groups
        for sid, identity in json.loads(p['identity_bindings']).items():
            s = studies[sid]
            assert identity == dict(H=s['Hamiltonian_identity'], state=s['state_identity'],
                                    reference=s['reference_identity'], identity_refs=json.loads(s['identity_refs']))
        for b in json.loads(p['candidate_selectors']):
            bindings += 1
            assert b['ref_id'] in json.loads(p['source_refs'])
            s = b['selector']
            if s['type'] == 'csv_rows':
                reader = csv.DictReader(io.StringIO(raw(b['ref_id']), newline=''))
                assert set(b['columns']) | set(s['where']) | set(s['where_in']) <= set(reader.fieldnames)
                selected = [(i, r) for i, r in enumerate(reader, 1)
                            if all(r[k] == v for k, v in s['where'].items()) and
                            all(r[k] in values for k, values in s['where_in'].items())]
                assert selected, (fid, b['ref_id'])
                for k, values in s['where_in'].items():
                    assert set(values) <= {r[k] for _, r in selected}, (fid, k)
                row_number, original = selected[0]
                preview = b['literal_preview']
                assert row_number == preview['csv_data_row']
                assert set(preview['values']) == set(b['columns'])
                for field, value in preview['values'].items():
                    assert value == original[field], (fid, field)
                    literals += 1
            elif s['type'] == 'json_pointers':
                document = json.loads(raw(b['ref_id']), parse_float=str, parse_int=str)
                assert set(b['literal_preview']) == set(s['pointers'])
                for ptr in s['pointers']:
                    assert catalog.pointer(document, ptr) == b['literal_preview'][ptr], (fid, ptr)
                    literals += 1
            else:
                assert s['type'] == 'text' and s['anchor'] in raw(b['ref_id'])
    for a in assets:
        assert set(json.loads(a['figure_candidate_ids'])) <= set(by_id)
        assert all(key in refs for col in ['asset_refs', 'data_refs', 'code_refs'] for key in json.loads(a[col]))
        assert a['code_execution'] == 'NOT_EXECUTED' and a['original_asset_modified'] == 'false'
        if json.loads(a['asset_refs']):
            assert a['existence_check'] == 'FIXED_BLOB_PRESENT'
        else:
            assert a['existence_check'].startswith('NO_STANDALONE_FIGURE_CONFIRMED')
    assert by_id['XP02']['comparison_class'] == 'CONDITIONAL_MATCHED'
    assert by_id['XP03']['comparison_class'] == 'MATCHED'
    assert '同一保存Pauli program' in by_id['XP03']['research_question']
    assert by_id['XP06']['plotting_readiness'] == by_id['XP07']['plotting_readiness'] == 'INSUFFICIENT_COMPARABILITY'
    assert by_id['ME10']['plotting_readiness'] == 'NEEDS_NEW_ANALYSIS'
    actual = dict(figure_candidates=len(plans), by_category=dict(Counter(r['category'] for r in plans)),
                  by_readiness=dict(Counter(r['plotting_readiness'] for r in plans)),
                  existing_asset_records=len(assets), supplemental_fixed_refs=len(meta['supplemental_fixed_refs']),
                  data_bindings=bindings, literal_preview_fields=literals)
    assert actual == meta['counts']
    # No private machine paths, credentials, binary figures, or broken local prose links.
    private = re.compile('/' + r'(?:home|tmp|mnt|scratch|Users)/|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----')
    for path in out.iterdir():
        assert path.is_file() and not private.search(path.read_text()), path.name
    for path in out.glob('*.md'):
        for target in re.findall(r'\[[^\]\n]+\]\(([^)\n]+)\)', path.read_text()):
            assert target in FILES and (out / target).is_file(), (path.name, target)
    result = dict(status='PASS_METADATA_AND_LITERAL_ONLY', **actual,
                  base_fixed_refs=len(index['references']), prior_literal_fields=prior['catalog']['literal_fields_verified'],
                  prior_claim_rows_unchanged=35, prior_validators=prior,
                  new_scientific_execution=False, numeric_reanalysis=False,
                  figure_generation=False, new_scientific_PASS=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
