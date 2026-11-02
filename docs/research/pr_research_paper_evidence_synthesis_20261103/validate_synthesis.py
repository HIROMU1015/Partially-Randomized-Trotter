"""Check document metadata only; no scientific runners or numerical evaluation.

Usage: python3 validate_synthesis.py --repo REPOSITORY
The repository must already contain the fixed objects. No fetch is performed.
Reuses the fixed coverage/binding checkers' CSV, Git and literal-check patterns.
"""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import re
import subprocess

FILES = {
    'RESEARCH_SYNTHESIS.md', 'CLAIM_EVIDENCE_MAP.csv',
    'PAPER_SCOPE_OPTIONS.md', 'PAPER_STORYLINE_OUTLINE.md',
    'STATUS_AND_OPEN_QUESTIONS.md', 'VALIDATION.json', 'validate_synthesis.py',
}
COLUMNS = ('claim_id component_id series claim_candidate_text phase3_original_claim_text '
           'phase3_original_claim_status phase3_original_assessment_status '
           'phase3_original_coverage_rows phase3_original_claimed_domain physical_domain '
           'comparators precision_and_metric evidence_kind original_required_evidence_refs '
           'reviewed_fixed_refs supporting_results negative_or_counter_results unassured_scope '
           'followup_evidence_separate_from_phase3 current_formal_adoption old_manuscript_treatment '
           'paper_placement_candidate not_required_for_scoped_claim proposal_status').split()
ENV = dict(os.environ, GIT_OPTIONAL_LOCKS='0', GIT_NO_LAZY_FETCH='1')
PREFIX = 'https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/'


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], env=ENV)


def table(raw):
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig'), newline=''))
    rows = list(reader)
    assert len(set(reader.fieldnames)) == len(reader.fieldnames)
    assert all(None not in row and all(value != '' for value in row.values()) for row in rows)
    return reader.fieldnames, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args()
    out = Path(__file__).resolve().parent
    meta = json.loads((out / 'VALIDATION.json').read_text())
    registry = meta['fixed_inputs']
    assert {p.name for p in out.iterdir() if p.is_file()} == FILES
    specs = [entry['commit'] + ':' + entry['path'] for entry in registry.values()]
    proc = subprocess.run(
        ['git', '-C', str(args.repo), 'cat-file', '--batch-check=%(objectname) %(objecttype) %(objectsize)'],
        input=('\n'.join(specs) + '\n').encode(), stdout=subprocess.PIPE, check=True, env=ENV)
    records = proc.stdout.decode().splitlines()
    assert len(records) == len(registry)
    for entry, line in zip(registry.values(), records):
        blob, kind, size = line.split()
        assert (blob, kind, size) == (entry['blob'], 'blob', str(entry['bytes']))
        assert entry['url'] == PREFIX + entry['commit'] + '/' + entry['path']
        assert re.fullmatch('[0-9a-f]{40}', entry['commit'])

    def fixed(id):
        entry = registry[id]
        return git(args.repo, 'show', entry['commit'] + ':' + entry['path'])

    _, oldclaims = table(fixed('P3CLAIMS'))
    _, oldcoverage = table(fixed('P3COV'))
    columns, rows = table((out / 'CLAIM_EVIDENCE_MAP.csv').read_bytes())
    assert columns == COLUMNS and len(rows) == 29
    assert len({row['component_id'] for row in rows}) == 29
    old_by_id = {row['claim_id']: row for row in oldclaims}
    bases = {row['claim_id'] for row in rows if row['component_id'].endswith('.BASE')}
    assert bases == set(old_by_id) | {'B04', 'B05', 'B06', 'B11'}
    assert len(oldclaims) == 21 and len(oldcoverage) == 44
    assert {r['claim_id'] for r in oldcoverage} - set(old_by_id) == {'HIST-UWC', 'HIST-DFSCREEN'}
    assert {row['component_id'] for row in rows if not row['component_id'].endswith('.BASE')} == {
        'R02.V1', 'R02.V2', 'R02.V3', 'R02.R1'}
    projected_rows = set()
    reference_links = 0
    for row in rows:
        id = row['claim_id']
        if id in old_by_id:
            old = old_by_id[id]
            for target, source in {
                'phase3_original_claim_text': 'claim_text_as_scoped',
                'phase3_original_claim_status': 'claim_status',
                'phase3_original_assessment_status': 'assessment_status',
                'phase3_original_claimed_domain': 'claimed_domain',
                'original_required_evidence_refs': 'required_evidence_refs',
                'not_required_for_scoped_claim': 'not_required_for_current_claim',
            }.items():
                assert row[target] == old[source], (row['component_id'], target)
            projection = [dict(csv_data_row=i + 1, coverage_state=r['coverage_state'],
                               claimed_domain=r['claimed_domain'], candidate_gap_id=r['candidate_gap_id'])
                          for i, r in enumerate(oldcoverage) if r['claim_id'] == id]
            assert json.loads(row['phase3_original_coverage_rows']) == projection
            projected_rows.update(item['csv_data_row'] for item in projection)
        else:
            assert row['phase3_original_claim_status'].startswith('NOT_IN_PHASE3_21_CLAIMS;')
            assert json.loads(row['phase3_original_coverage_rows']) == []
        assert row['current_formal_adoption'] == 'NO_NEW_ADOPTION;existing_source_status_retained'
        for ref in json.loads(row['reviewed_fixed_refs']):
            entry = registry[ref['evidence_id']]
            assert (ref['commit'], ref['path'], ref['git_blob'], ref['url']) == (
                entry['commit'], entry['path'], entry['blob'], entry['url'])
            assert ref['selector'] and ref['read_scope']
            reference_links += 1
    assert len(projected_rows) == 42
    _, bindings1 = table(fixed('G01'))
    _, bindings2 = table(fixed('G02'))
    assert len(bindings1) == 7 and all(r['match_status'] == 'MATCHED' for r in bindings1)
    assert len(bindings2) == 100  # Binding records; no experiment-count inference.

    # Check link definitions and known local output links; fixed sources above are immutable.
    document_links = 0
    for path in out.glob('*.md'):
        text = path.read_text()
        definitions = dict(re.findall(r'^\[([^\]]+)\]:\s*<([^>]+)>\s*$', text, re.M))
        uses = re.findall(r'\[[^\]\n]+\]\[([^\]\n]+)\]', text)
        assert set(uses) <= set(definitions), (path.name, set(uses) - set(definitions))
        for id in uses:
            assert definitions[id] == registry[id]['url']
        for target in re.findall(r'\[[^\]\n]+\]\(([^)\n]+)\)', text):
            if not target.startswith('https://'):
                assert target in FILES and (out / target).is_file(), (path.name, target)
            document_links += 1
        document_links += len(uses)
    # These checks protect this new public projection, not the full source repository.
    private = re.compile(
        '/' + r'(?:home|tmp|mnt|scratch|Users)/|\b(?:\d{1,3}\.){3}\d{1,3}\b|'
        r'gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----')
    for filename in FILES:
        assert not private.search((out / filename).read_text()), filename
    assert meta['scope']['new_scientific_execution'] is False
    assert meta['scope']['saved_numeric_reanalysis'] is False
    assert meta['scope']['independent_scientific_review'] is False
    assert meta['scope']['formal_claim_adoption'] is False
    print(json.dumps(dict(status='PASS_TECHNICAL_METADATA_ONLY', fixed_inputs=len(registry),
                          claim_rows=len(rows), original_claims=21, original_coverage_projection=42,
                          historical_lookup_rows_retained_in_original=2,
                          fixed_reference_links=reference_links, document_links=document_links,
                          new_scientific_execution=False, saved_numeric_reanalysis=False), indent=2))


if __name__ == '__main__':
    main()
