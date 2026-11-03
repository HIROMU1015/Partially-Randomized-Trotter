"""Metadata/literal checks only, adapted from the fixed B/C document validators.

Usage: python3 validate_addendum.py --repo REPOSITORY
Requires existing fixed Git objects. No fetch, scientific import, numeric
aggregation, RNG, array decode, circuit construction, or compile is performed.
"""
import argparse
import ast
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
FILES = {'REPORT.md', 'H468_COVERAGE_ADDENDUM.csv', 'O1_H8_CLAIM_EVIDENCE_UPDATE.md',
         'STATUS_AND_REMAINING_GAPS.md', 'VALIDATION.json', 'validate_addendum.py'}
STATUS = 'ANONYMIZED_COMPLETION_METADATA_AVAILABLE_WITH_REMAINING_LIMITS'
ENV = dict(os.environ, GIT_OPTIONAL_LOCKS='0', GIT_NO_LAZY_FETCH='1')
UPDATES = {'H8_DF28_P_1', 'H8_DF28_P_2', 'INPUT_H8', 'SIGNAL_H8', 'NATIVE_H8',
           'COHORT_H8', 'H8_G04', 'H8_PROFILE_UNRESOLVED'}


def table(raw):
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig'), newline=''))
    rows = list(reader)
    assert len(set(reader.fieldnames)) == len(reader.fieldnames)
    assert all(None not in row for row in rows)
    return reader.fieldnames, rows


def obj(raw):
    # Preserve saved numeric tokens as strings; do not evaluate scientific values.
    return json.loads(raw, parse_int=str, parse_float=str)


def check(repo):
    metadata = json.loads((HERE/'VALIDATION.json').read_text())
    registry = metadata['fixed_inputs']
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args], env=ENV)
    cache = {}
    def raw(id):
        if id not in cache:
            item = registry[id]
            cache[id] = git('show', item['commit']+':'+item['path'])
        return cache[id]
    def objects(id):
        return obj(raw(id))
    assert {p.name for p in HERE.iterdir() if p.is_file()} == FILES
    specs = [v['commit']+':'+v['path'] for v in registry.values()]
    result = subprocess.run(['git', '-C', str(repo), 'cat-file',
                             '--batch-check=%(objectname) %(objecttype) %(objectsize)'],
                            input=('\n'.join(specs)+'\n').encode(), stdout=subprocess.PIPE,
                            check=True, env=ENV)
    lines = result.stdout.decode().splitlines()
    assert len(lines) == len(registry)
    for entry, line in zip(registry.values(), lines):
        blob, kind, size = line.split()
        assert (blob, kind, size) == (entry['blob'], 'blob', str(entry['bytes']))
        assert entry['url'] == 'https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/'+entry['commit']+'/'+entry['path']

    old_columns, old_rows = table(raw('B_COVERAGE'))
    columns, rows = table((HERE/'H468_COVERAGE_ADDENDUM.csv').read_bytes())
    assert columns[:len(old_columns)] == old_columns and len(rows) == len(old_rows) == 36
    assert len({r['coverage_id'] for r in rows}) == 36
    assert [{k:r[k] for k in old_columns} for r in rows] == old_rows
    assert all(all(v != '' for v in r.values()) for r in rows)
    by_id = {r['coverage_id']:r for r in rows}
    assert {r['coverage_id'] for r in rows if r['addendum_status'] != 'UNCHANGED_PRIOR_RECORDED_SCOPE'} == UPDATES
    assert by_id['H8_G04']['addendum_status'] == STATUS
    assert by_id['H8_DF28_P_1']['saved_exact_export_match'] == 'false'
    assert by_id['H8_DF28_P_2']['saved_exact_export_match'] == 'true'
    assert all(r['scientific_status_changed'] == 'NO_NEW_SCIENTIFIC_PASS_OR_ADOPTION' for r in rows)
    assert all(r['current_O1_new_science_required'] == 'NO' for r in rows)
    historical = [{k:r[k] for k in ['claim_id','claimed_domain','coverage_state','candidate_gap_id']}
                  for r in table(raw('PHASE3_COVERAGE'))[1] if 'G04' in r['candidate_gap_id']]
    assert {r['coverage_state'] for r in historical} == {'NOT_ACCESSIBLE', 'SUPPORTED_WITHIN_RECORDED_SCOPE'}
    prior_claims = {r['claim_id']:r for r in table(raw('C_CLAIMS'))[1] if r['component_id'].endswith('.BASE')}
    for r in rows:
        if r['coverage_id'] in UPDATES:
            assert json.loads(r['phase3_historical_G04_coverage']) == historical
        if r['synthesis_claim_ids'] in {'A07','A08','A13'}:
            old = prior_claims[r['synthesis_claim_ids']]
            assert json.loads(r['synthesis_historical_status']) == {k:old[k] for k in
                ['phase3_original_claim_status','phase3_original_assessment_status','current_formal_adoption']}
        for ref in json.loads(r['addendum_sources']):
            assert ref['url'] == registry[ref['evidence_id']]['url'] and ref['selector']

    audit = objects('A_AUDIT')
    evidence = table(raw('A_EVIDENCE'))[1]
    originals = table(raw('ORIGINAL_COMPARISONS'))[1]
    main = {r['comparison_id']:r for r in objects('MAIN')['rows']}
    mechanism = {r['contrast_id']:r for r in table(raw('MECHANISM'))[1]}
    design = objects('DESIGN')
    targets = {r['scope']:r for r in design['targets']}
    scopes = {r['scope']:r for r in audit['scopes']}
    figure = table(raw('FIGURE2B'))[1]
    assert len(evidence) == 2 and len(scopes) == 2
    pair_ids = {'H8_DF28_P_1':'H8_CONFIRM_1','H8_DF28_P_2':'H8_CONFIRM_3'}
    for row in evidence:
        sc = scopes[row['scope']]
        orig = next(r for r in originals if r['scope']==row['scope'] and r['mode']=='primary')
        target = targets[row['scope']]
        item = main[row['comparison_id']]
        alias = pair_ids[row['comparison_id']]
        assert row['mechanism_id'] == item['mechanism_core_alias'] == alias
        assert row['candidate'] == target['cell']['id']
        assert row['L_D'] == target['cell']['prefix'] == item['L_D']
        for field in ['q','r','K','T','delta']:
            assert row[field] == item[field]
        assert row['epsilon'] == target['primary_epsilon']
        for new, old in {'candidate':'candidate','epsilon':'epsilon','original_trajectories':'n',
                         'PR_N_cosine':'shots_cosine','PR_N_sine':'shots_sine','PF_id':'fixed_PF_id',
                         'PR_G_mean_exact':'mean_exact','PR_paired_SE':'paired_SE','PF_G_exact':'fixed_PF_G_exact',
                         'point_decision':'point_decision'}.items():
            assert row[new] == orig[old], (row['comparison_id'], new)
        for key in ['n','mean_exact','sample_SD','paired_SE','fixed_PF_G_exact','PR_reduction_percent_exact','point_decision']:
            assert sc['saved_primary'][key] == orig[key], key
        secondary = next(r for r in originals if r['scope']==row['scope'] and r['mode']=='secondary')
        for key in ['fixed_PF_G_exact','PR_reduction_percent_exact']:
            assert sc['saved_secondary'][key] == (secondary[key] or None)
        assert sc['saved_primary']['confidence_interval'] is None and sc['saved_primary']['p_value'] is None
        assert sc['saved_primary']['old_samples_pooled'] is False
        for key in ['hamiltonian','state','snapshot','compiler','profile']:
            assert sc['input_binding'][key] == design['binding'][key]
        assert sc['preparation']['preparation_hash'] == target['preparation_sha256']
        assert sc['preparation']['matches_fixed_design_hash'] is True
        assert row['execution_SOURCE'] == sc['execution_SOURCE'] == item['SOURCE']
        assert row['dataset_seed_SOURCE'] == sc['dataset_SOURCE']
        for key, original in {'PR_id':'candidate','PF_id':'fixed_PF_id','epsilon':'epsilon','n':'n'}.items():
            assert item[key] == mechanism[alias][key] == orig[original]
        assert item['G_PR_exact'] == orig['mean_exact'] and item['G_PF_exact'] == orig['fixed_PF_G_exact']
        assert item['paired_SE'] == orig['paired_SE']
        assert mechanism[alias]['cohorts_pooled'] == 'False'
        for metric in ['R_N','R_Ceff','R_G']:
            found = [r for r in figure if r['panel']=='2B' and r['contrast_id']==alias and r['metric']==metric]
            assert len(found)==1 and found[0]['saved_value_text']==item['saved_core_ratios'][metric]
        assert sc['driver'] == {'exit_code':'0','reaped':True}
        assert sc['n']=='48' and sc['complete_axes']=='96'
        assert sc['original_STOP_preserved'] is True and sc['old_reservations_refunded'] is False
    p8,p7 = scopes['H8_CONFIRM_P8_Q2'],scopes['H8_CONFIRM_P7_Q16']
    assert p8['preparation']['saved_exact_export_match'] is False
    assert p8['preparation']['status'] == 'UNRESOLVED_SAVED_EXPORT_COMPARISON_FLAG'
    assert p7['preparation']['saved_exact_export_match'] is True
    assert (p8['new_trajectory_parents'],p8['reused_trajectories'],p8['new_nested_children'])==('47','1','94')
    assert (p7['new_trajectory_parents'],p7['reused_trajectories'],p7['new_nested_children'])==('48','0','101')
    assert audit['compiler']['original_profile_bytes_verified'] is False
    assert audit['compiler']['original_compiler_output_identity_verified'] is False
    assert audit['seed_audit']['full_historical_coverage_certified'] is False
    assert audit['seed_audit']['prior_derived_seed_audit_rerun'] is False

    axes = table(raw('A_AXES'))[1]
    old_axes = table(raw('ORIGINAL_AXES'))[1]
    key = lambda r:(r['scope'],r['replica'],r['axis'])
    old_index = {key(r):r for r in old_axes}
    assert len(axes)==len(old_axes)==len(old_index)==192
    assert len({key(r) for r in axes})==192
    expected_keys = {(scope,str(replica),axis) for scope in scopes for replica in range(48) for axis in ['cosine','sine']}
    assert {key(r) for r in axes}==set(old_index)==expected_keys
    for row in axes:
        original = old_index[key(row)]
        for col in ['candidate','cohort','reused_complete_trajectory','RZ']:
            assert row[col]==original[col], (key(row),col)
        reused = row['scope']=='H8_CONFIRM_P8_Q2' and row['replica']=='0'
        assert row['reused_complete_trajectory']==str(reused)
        assert row['measurement_SOURCE']==(p8['dataset_SOURCE'] if reused else p8['execution_SOURCE'])
        assert row['compiler_hash']==design['binding']['compiler']
        assert re.fullmatch('[0-9a-f]{64}',row['fingerprint'])
        assert re.fullmatch('[0-9a-f]{64}',row['task_result_sha256'])
        assert row['worker_exit_code']=='0' and row['worker_reaped']=='True'
        assert row['pair_event_and_seed_index_binding']=='MATCHED_ORIGINAL_METADATA'
        partner = next(r for r in axes if r['scope']==row['scope'] and r['replica']==row['replica'] and r['axis']!=row['axis'])
        assert row['task_result_sha256']==partner['task_result_sha256']

    closure = audit['SOURCE_closure']
    assert len(closure)==11
    for i,item in enumerate(closure,1):
        source = registry['SOURCE_'+str(i).zfill(2)]
        assert (source['commit'],source['path'],source['blob'])==(item['commit'],item['path'],item['git_blob'])
        data=raw('SOURCE_'+str(i).zfill(2))
        assert str(len(data))==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
        if item['path'].endswith('.py'):
            ast.parse(data)  # Parse only; never import/execute the scientific SOURCE.
    h4=objects('H4_COMPLETION')
    assert h4['native_exit_code']=='0' and h4['all_workers_reaped'] is True
    for item in objects('H6_COMPLETION'):
        assert item['driver']['exit_code']=='0' and item['driver']['reaped'] is True
    assert objects('H4_SEEDS')['future_unregistered_campaigns_covered'] is False
    # Preserve legacy documents/statuses; no broad historical inference is introduced.
    assert metadata['scope']['new_scientific_execution'] is False
    assert metadata['scope']['saved_numeric_reanalysis'] is False
    assert metadata['scope']['mandatory_stop_after_publication'] is True
    for path in HERE.glob('*.md'):
        text=path.read_text()
        defs=dict(re.findall(r'^\[([^\]]+)\]: <([^>]+)>$',text,re.M))
        for id in re.findall(r'\[[^\]\n]+\]\[([^\]\n]+)\]',text):
            assert defs[id]==registry[id]['url']
        for target in re.findall(r'\[[^\]\n]+\]\(([^)\n]+)\)',text):
            if not target.startswith('https://'):
                assert target in FILES and (HERE/target).is_file()
        assert 'mandatory STOP' in text
    private=re.compile('/'+r'(?:home|tmp|mnt|srv|scratch|Users)/|\b(?:\d{1,3}\.){3}\d{1,3}\b|'
                       r'gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----')
    for file in FILES:
        assert not private.search((HERE/file).read_text()),file
    return dict(status='PASS_METADATA_LITERAL_ONLY',fixed_objects=len(registry),old_coverage_rows_preserved=36,
                old_columns_preserved=32,H8_followup_rows=8,fixed_H8_comparisons=2,axis_literal_links=192,
                source_closure_files=11,p8_saved_exact_export_match=False,
                new_scientific_execution=False,saved_numeric_reanalysis=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    print(json.dumps(check(parser.parse_args().repo),ensure_ascii=False,indent=2))
