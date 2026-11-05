"""Check document metadata only; no scientific runners or numerical evaluation.

Usage: python3 validate_synthesis.py --repo REPOSITORY
The repository must already contain the fixed objects. No fetch is performed.
Reuses the fixed coverage/binding checkers' CSV, Git and literal-check patterns.
"""
import argparse
import csv
import hashlib
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
ORIGINAL_COLUMNS = ('claim_id component_id series claim_candidate_text phase3_original_claim_text '
           'phase3_original_claim_status phase3_original_assessment_status '
           'phase3_original_coverage_rows phase3_original_claimed_domain physical_domain '
           'comparators precision_and_metric evidence_kind original_required_evidence_refs '
           'reviewed_fixed_refs supporting_results negative_or_counter_results unassured_scope '
           'followup_evidence_separate_from_phase3 current_formal_adoption old_manuscript_treatment '
           'paper_placement_candidate not_required_for_scoped_claim proposal_status').split()
COLUMNS = ORIGINAL_COLUMNS + ('integration_as_of research_role candidate_acquisition_scope '
           'cost_and_compiler source_input_identity verification_level scoped_conclusion '
           'current_scope_gap_class necessary_remaining_work').split()
PREVIOUS = '2ce2b5f161e94567fec322d694fd8b01d03cd854'
LOCAL_BASE = '9de335a7c49367356fca43f077e8f5157b3a95f2'
FINAL_COMPONENTS = {'A11.BASE', 'P05.BASE', 'F01.BASE', 'A12.BASE', 'A08.BASE'}
APPEND_FIELDS = {'reviewed_fixed_refs', 'physical_domain', 'candidate_acquisition_scope',
                 'source_input_identity', 'supporting_results', 'verification_level',
                 'unassured_scope', 'followup_evidence_separate_from_phase3'}
NEW_COMPONENTS = {'A14.H8_AB', 'A14.H8_C', 'R02.POLICY_MEAN', 'R02.IID_ENERGY',
                  'R02.BACKEND', 'R02.FLOAT64'}
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
    assert columns == COLUMNS and len(rows) == 35
    assert len({row['component_id'] for row in rows}) == 35
    # Validate previous preservation, then today's explicitly permitted append-only edits.
    _, previous_rows = table(git(args.repo, 'show', PREVIOUS + ':' +
        'docs/research/pr_research_paper_evidence_synthesis_20261103/CLAIM_EVIDENCE_MAP.csv'))
    by_component = {row['component_id']: row for row in rows}
    _, local_rows = table(git(args.repo, 'show', LOCAL_BASE + ':' +
        'docs/research/pr_research_paper_evidence_synthesis_20261103/CLAIM_EVIDENCE_MAP.csv'))
    local_by_component = {row['component_id']: row for row in local_rows}
    assert [row['component_id'] for row in rows] == [row['component_id'] for row in local_rows]
    updated = set()
    for previous in local_rows:
        current = by_component[previous['component_id']]
        changed = {key for key in COLUMNS if current[key] != previous[key]}
        if not changed:
            continue
        updated.add(previous['component_id'])
        assert previous['component_id'] in FINAL_COMPONENTS and changed <= APPEND_FIELDS
        for key in changed:
            if key == 'reviewed_fixed_refs':
                before, after = json.loads(previous[key]), json.loads(current[key])
                assert after[:len(before)] == before
                assert len({ref['evidence_id'] for ref in after}) == len(after)
            else:
                assert current[key].startswith(previous[key] + ' | 3環境追補2026-11-05: ')
    assert updated == FINAL_COMPONENTS
    assert len(previous_rows) == 29
    for previous in previous_rows:
        assert {key: local_by_component[previous['component_id']][key] for key in ORIGINAL_COLUMNS} == previous
    assert set(by_component) - {row['component_id'] for row in previous_rows} == NEW_COMPONENTS
    old_by_id = {row['claim_id']: row for row in oldclaims}
    bases = {row['claim_id'] for row in rows if row['component_id'].endswith('.BASE')}
    assert bases == set(old_by_id) | {'B04', 'B05', 'B06', 'B11'}
    assert len(oldclaims) == 21 and len(oldcoverage) == 44
    assert {r['claim_id'] for r in oldcoverage} - set(old_by_id) == {'HIST-UWC', 'HIST-DFSCREEN'}
    assert {row['component_id'] for row in rows if not row['component_id'].endswith('.BASE')} == {
        'R02.V1', 'R02.V2', 'R02.V3', 'R02.R1'} | NEW_COMPONENTS
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
        assert row['scoped_conclusion'] in {'支持', '限定支持', '証拠不足'}
        assert row['current_scope_gap_class'] in {'A', 'B', 'C', 'D'}
        assert row['integration_as_of'] == '2026-11-05'
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

    # Inspect saved strings/identities only. No numerical/statistical evaluation.
    h8 = json.loads(fixed('H8_C_RAW'))
    assert h8['scope']['SOURCE'] == registry['H8_C_SOURCE']['commit']
    assert h8['PR_summary']['mean_G_exact'] == '67335317264200'
    assert h8['PR_summary']['fixed_N_cos'] == 84916951
    assert h8['PR_summary']['fixed_N_sin'] == 88339169
    assert h8['PR_summary']['cohort'] == 'CONFIRMATION'
    assert h8['PR_summary']['n'] == 9 and h8['PR_summary']['pooled'] is False
    assert h8['PR_summary']['population_tail_bound'] is None
    assert h8['fixed_PF']['mean_G_exact'] == '89291219516568'
    assert h8['fixed_PF']['cohort'] == 'EXPLORATION'
    assert h8['representative_checks'] == {'count': 7, 'index': 0, 'status': 'PASS'}
    for path, sha256 in h8['scope']['source_files'].items():
        raw = git(args.repo, 'show', h8['scope']['SOURCE'] + ':' + path)
        assert hashlib.sha256(raw).hexdigest() == sha256
    plan = json.loads(fixed('H8_C_PLAN'))
    assert h8['scope']['binding'] == plan['binding']
    assert h8['scope']['comparison_ids'] == plan['comparison_ids']
    mean = json.loads(fixed('MEAN_RAW'))
    assert mean['decision'] == 'CONDITIONAL_IDEAL_CONTROLLED_IID_MEAN_LOCAL_CONDITIONS_CONFIRMED'
    energy = json.loads(fixed('ENERGY_RAW'))
    assert energy['SOURCE']['commit'] == registry['ENERGY_SOURCE']['commit']
    assert energy['decision'] == 'CONDITIONAL_IDEAL_CONTROLLED_IID_MATHEMATICAL_ENERGY_95_SUFFICIENT'
    assert energy['aggregate']['failure_upper_exact'] == (
        '1218468535910657108111929276348813106077652184592843003/'
        '24519928653854221733733552434404946937899825954937634816')
    assert energy['aggregate']['success_lower_exact'] == (
        '23301460117943564625621623158056133831822173770344791813/'
        '24519928653854221733733552434404946937899825954937634816')
    assert energy['RPE_tolerance_certified'] is False
    assert energy['scientific_status'] == 'UNDETERMINED'
    assert energy['U_virtual_IID_upper'] is None and energy['U_exec_upper'] is None
    assert energy['proof']['Float64_decoder'] == 'UNKNOWN/STOP'
    # Use the existing execution binding map rather than rebuilding inputs.
    for entry in json.loads(fixed('ENERGY_BINDINGS'))['files'].values():
        raw = git(args.repo, 'show', entry['commit'] + ':' + entry['path'])
        assert len(raw) == entry['bytes']
        assert hashlib.sha256(raw).hexdigest() == entry['sha256']
        assert git(args.repo, 'rev-parse', entry['commit'] + ':' + entry['path']).decode().strip() == entry['blob']
    for id, literals in {
        'H8_C': ['67335317264200', '0.754109056061', '24.58909439%',
                 '103180211675.02368', '121655623062.28098'],
        'ENERGY_REPORT': ['0.0496929886343339467', '0.950307011365666053',
                          '0.000138689200360119854'],
        'MEAN_REPORT': ['7.55735552e-9'],
    }.items():
        source_text = fixed(id).decode()
        synthesis = (out / 'RESEARCH_SYNTHESIS.md').read_text()
        for literal in literals:
            assert literal in source_text and literal in synthesis, (id, literal)
    quoted_rows = [
        ('CNG', {'system': 'H4', 'T': '0.8', 'epsilon': '0.05', 'family': 'PF',
                 'higher_formula': '4th(m5_best)'},
         ['higher_over_S2_N', 'higher_over_S2_C_eff', 'higher_over_S2_G']),
        ('CNG', {'system': 'H4', 'T': '1.6', 'epsilon': '0.0005', 'family': 'PF',
                 'higher_formula': '4th(m5_best)'},
         ['higher_over_S2_N', 'higher_over_S2_C_eff', 'higher_over_S2_G']),
        ('CNG', {'system': 'H4', 'T': '0.8', 'epsilon': '0.0005', 'family': 'PR',
                 'higher_formula': '4th(m5_best)'},
         ['higher_over_S2_N', 'higher_over_S2_C_eff', 'higher_over_S2_G']),
        ('CNG', {'system': 'H4', 'T': '1.6', 'epsilon': '0.0005', 'family': 'PR',
                 'higher_formula': '4th(m5_best)'},
         ['higher_over_S2_N', 'higher_over_S2_C_eff', 'higher_over_S2_G']),
    ]
    for formula in ['2nd', '4th(m5_best)']:
        quoted_rows.append(('MATCHED', {'system': 'H4', 'T': '0.8', 'epsilon': '0.05',
            'q': '1', 'prefix': '3', 'formula': formula}, ['log_B', 'N_cos', 'N_sin', 'G']))
    for cutoff in ['2', '4', '6']:
        quoted_rows.append(('RK', {'system': 'H4', 'T': '0.8', 'epsilon': '0.05',
            'q': '1', 'prefix': '3', 'formula': '4th(m5_best)', 'R_outer': '16', 'K': cutoff},
            ['N_cos', 'N_sin', 'C_cos', 'C_sin']))
    for id, selector, fields in quoted_rows:
        matched = [row for row in csv.DictReader(io.StringIO(fixed(id).decode()))
                   if all(row[key] == value for key, value in selector.items())]
        assert len(matched) == 1, (id, selector)
        for field in fields:
            assert matched[0][field] in synthesis, (id, selector, field)

    # Reuse server projections and primary saved fields, without re-accounting results.
    s1 = json.loads(fixed('S1_CHECKS'))
    s2 = json.loads(fixed('S2_EVIDENCE'))
    assert s1['verdict'] == 'NO_MATERIAL_DIFFERENCE'
    assert s2['verdict'] == 'MATERIAL_NEW_EVIDENCE'
    assert s1['previous_synthesis_commit'] == PREVIOUS
    assert s2['previous_synthesis'] == PREVIOUS
    # Original boundary rows intentionally have missing signal/diagnostic cells.
    boundary = list(csv.DictReader(io.StringIO(fixed('BOUNDARY_RESULTS').decode())))
    boundary_by = {(row['id'], row['cohort'], row['epsilon']): row for row in boundary}
    for id, cohort, reference, diagnostic, n in [
        ('f85eb7b56ba5131d1eb1', 'BOUNDARY_20261027_CONFIRMATION',
         '45f9fc0a53c1a37c7345', 'HIGHER', '745'),
        ('3a55c1dac3947ae23c76', 'BOUNDARY_20261027_CONFIRMATION',
         '45f9fc0a53c1a37c7345', 'HIGHER', '1185'),
        ('0e0e9fb6da33cb6ab2ab', 'BOUNDARY_20261027_CONFIRMATION',
         'efbb07c848b303a6abab', 'INCONCLUSIVE', '229'),
    ]:
        row = boundary_by[id, cohort, '0.0005']
        assert row['reference_id'] == reference and row['diagnostic'] == diagnostic and row['n'] == n
        for field in ['N_cos', 'N_sin', 'C_cos', 'C_sin', 'G']:
            assert row[field] in synthesis, (id, field)
        assert row['T'] == '2.0' and row['q'] == '1' and row['formula'] == '4th(m5_best)'
    assert boundary_by['0e0e9fb6da33cb6ab2ab', 'BOUNDARY_20261027_CONFIRMATION', '0.0005']['point_lower'] == 'True'
    assert boundary_by['d9cb4e923530349e940a', 'EXPLORATION', '0.0005']['n'] == '9'
    summary = json.loads(fixed('BOUNDARY_SUMMARY'))
    assert summary['execution_SOURCE'] == 'ad1286021bdc0e5a0b1760064cefb3c1168b24eb'
    assert summary['accounting_SOURCE'] == '0f2d167f9c928ba94bfad1a167bf4d3f5964ad1a'
    assert summary['mandatory_STOP'] is True and summary['original_tables_overwritten'] is False
    for id, source in [
        ('H6_P1_BINDINGS', 'a0a3aff915f98456733d6bf09ed641a1be423628'),
        ('H6_P2_BINDINGS', '11e50e262f95630b14fa4070ad30894ba3f64b74'),
        ('P3_STAGE4_BINDINGS', '2f0d31f75ab1ecacc73d9ee5c34ddfc997396102'),
        ('P3_BIT_BINDINGS', 'd22ca967955a664260a3a9857df243b3f151610f'),
    ]:
        binding = json.loads(fixed(id))
        assert binding['commit'] == source
        for path, expected in binding['code_sha256'].items():
            assert hashlib.sha256(git(args.repo, 'show', source + ':' + path)).hexdigest() == expected
        # Hash bound public inputs without decoding scientific arrays or archives.
        if id in {'P3_STAGE4_BINDINGS', 'P3_BIT_BINDINGS'}:
            for path, entry in binding.get('input_files', binding.get('inputs', {})).items():
                assert hashlib.sha256(git(args.repo, 'show', entry['ref'] + ':' + path)).hexdigest() == entry['sha256']
    p1_binding, p2_binding = json.loads(fixed('H6_P1_BINDINGS')), json.loads(fixed('H6_P2_BINDINGS'))
    for key in ['commit', 'path', 'sha256']:
        assert p1_binding['inputs']['model'][key] == p2_binding['inputs']['model'][key]
    for binding in [p1_binding, p2_binding]:
        model = binding['inputs']['model']
        assert hashlib.sha256(git(args.repo, 'show', model['commit'] + ':' + model['path'])).hexdigest() == model['sha256']
    assert p2_binding['cohort'] == 'confirmation' and p2_binding['index'] == 0
    assert all(p1_binding['candidates'][key] == value for key, value in p2_binding['candidates'].items())
    primary = s2['b3_primary_table_identity']
    assert len(primary) == 8
    for item in primary:
        assert item['same_blob'] is True
        assert git(args.repo, 'rev-parse', item['public_commit'] + ':' + item['path']).decode().strip() == item['public_blob']
    assert s2['legacy_h8_custody']['present_in_known_server2_object_store'] is True
    assert s2['legacy_h8_custody']['complete_raw_audit_performed'] is False
    assert 'EXPLAINED_NO_SCIENTIFIC_IMPACT' in fixed('P8_CAUSE').decode()
    assert "saved_exact_export_match" in fixed('P8_LOADER').decode()
    assert "('faithful:p' + str(p)) in export" in fixed('P8_LOADER').decode()
    for id, literals in {
        'H6_P1': ['STOP_UNRESOLVED_COMPONENTS', 'H32/H64', 'UNRESOLVED'],
        'H6_P2': ['confirmation/0000', '4732', '6735', '3081', '2500'],
        'P3_STAGE4': ['ELIGIBLE_CONDITIONAL', 'compiled ρ=0'],
        'P3_BIT': ['certified_gap_upper=null', 'implemented_accuracy_guaranteed=false',
                   'actual_implemented_headroom=null'],
    }.items():
        for literal in literals:
            assert literal in fixed(id).decode(), (id, literal)

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
                if target.startswith('#'):
                    assert 'id="' + target[1:] + '"' in text, (path.name, target)
                else:
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
                          previous_LOCAL_rows_retained=35, components_appended=len(updated),
                          other_components_unchanged=30, H8_stage_AB_C_unchanged=True,
                          descriptive_current_scope='CURRENT_CLAIM_SCOPE_EVIDENCE_SUFFICIENT',
                          new_scientific_execution=False, saved_numeric_reanalysis=False), indent=2))


if __name__ == '__main__':
    main()
