"""Read-only documentary checks, except an optional local report write.

No scientific module/test imports, execution, numerical reanalysis or network.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = 'd4163acd7b577afaa7d6db8c29db64f76ec03ae7'
PREFIX = HERE.relative_to(ROOT).as_posix() + '/'
DOCS = ['README.md', 'generation_pipeline_alignment.md',
        'known_method_composition_audit.md', 'access_precision_complexity_comparison.md',
        'remaining_independent_claim_candidates.md', 'handoff.md']


def verify():
    checks = []

    def check(name, result):
        checks.append({'check': name, 'pass': bool(result)})

    def read(name):
        return json.loads((HERE / name).read_text())

    source, literature, evidence = map(read, ['source_inventory.json', 'literature_scope.json', 'evidence_index.json'])
    source_ids = {r['id'] for r in source['entries']}
    papers = {r['id']: r for r in literature['papers']}
    check('fixed_worktree_base', source['worktree_base_commit'] == evidence['worktree_base_commit'] == BASE)
    check('source_inventory_unique', len(source_ids) == len(source['entries']) == 22)
    check('source_packet_fixed_refs', all(r['source_commit'] == BASE for r in source['entries'] if r['id'].startswith('C') and r['id'] != 'CP'))
    check('nonpauli_read_only_ref', all(r['source_commit'] == '128a05765985c1c6aec3a1d0ce35e19ffbb3b3f6' for r in source['entries'] if r['id'].startswith('N')))
    for r in source['entries']:
        check(r['id'] + '_fixed_commit', bool(re.fullmatch('[0-9a-f]{40}', r['source_commit'])))
        b = subprocess.check_output(['git', '-C', str(ROOT), 'show', r['source_commit'] + ':' + r['path']])
        check(r['id'] + '_git_hash', len(b) == r['bytes'] and hashlib.sha256(b).hexdigest() == r['sha256'])
        check(r['id'] + '_remote_hash', r['remote_bytes_equal_git_blob'] and r['remote_sha256'] == r['sha256'])
        check(r['id'] + '_reading_scope', bool(r['reading_scope']))
    check('seven_primary_bodies', set(papers) == {'WAN', 'AK', 'MT', 'PTSC', 'SCU', 'SPS', 'IS'})
    for ident, p in papers.items():
        check(ident + '_retrieved_versioned_body', p['full_body_retrieved'] and p['retrieval_status'] == 'RETRIEVED')
        check(ident + '_prior_body_identity', p['prior_bytes_equal'] and p['sha256'] == p['retrieved_sha256'] and p['bytes'] == p['retrieved_bytes'])
        check(ident + '_scope_and_no_proof_claim', bool(p['read_scope_this_task']) and not p['whole_paper_proofs_independently_reproved'])
    records = evidence['alignment']
    expected = {(m, 'P' + str(i)) for m in papers for i in range(1, 13)}
    check('all_84_cells_exactly_once', len(records) == 84 and {(r['method'], r['stage']) for r in records} == expected)
    fields = ['g6_operation', 'g6_matrix_references', 'g6_source_locator', 'prior_documented_operation',
              'target_and_output_agreement_condition', 'same_information_oracle_comparison',
              'preprocessing_normalizer_coefficient_sampling_difference', 'finite_bit_bias_range_confidence_comparison',
              'controlled_phase_and_circuit_comparison', 'complexity_and_additional_acquisition_scope',
              'primary_reference', 'remaining_information_or_proof']
    pipeline = (HERE / 'generation_pipeline_alignment.md').read_text()
    for r in records:
        check(r['id'] + '_complete_comparison_fields', all(r.get(k) for k in fields))
        check(r['id'] + '_valid_status', r['status'] in evidence['allowed_statuses'])
        check(r['id'] + '_primary_ref', r['primary_reference']['id'] in papers and bool(r['primary_reference']['locator']))
        check(r['id'] + '_no_new_proof', r['new_proof_performed'] is False)
        check(r['id'] + '_rendered_row', f"| {r['stage']} | `{r['status']}` | {r['prior_documented_operation']} |" in pipeline)
    check('direct_components_explicit', len(evidence['directly_documented_components']) == 8 and all(r['status'] == 'DIRECTLY_DOCUMENTED' for r in evidence['directly_documented_components']))
    check('proof_boundary_conditional', evidence['unresolved_scientific_proof_scope']['conditional'] and evidence['unresolved_scientific_proof_scope']['status'] == 'REQUIRES_NEW_SCIENTIFIC_PROOF')
    check('zero_science_source_design_execution', all(v == 0 for v in evidence['execution_accounting'].values()))
    check('no_novelty_adjudication', not evidence['scientific_novelty_or_paper_go_adjudicated'])
    check('mandatory_stop', evidence['mandatory_stop_after_publication'])
    permitted_urls = {r['url'] for r in source['entries']} | {p['url'] for p in papers.values()}
    for name in DOCS:
        path = HERE / name
        check(name + '_exists', path.is_file() and path.stat().st_size > 0)
        text = path.read_text()
        for target in re.findall(r'\[[^\]]*\]\(([^)\s]+)\)', text):
            if target.startswith(('https://', 'http://', '#')):
                if target.startswith('https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/'):
                    check(name + '_fixed_source_' + target, target.split('#', 1)[0] in permitted_urls)
                continue
            check(name + '_relative_link_' + target, (path.parent / target.split('#', 1)[0]).resolve().is_file())
        definitions = dict(re.findall(r'^\[([^\]]+)\]:\s+(https?://\S+)', text, re.M))
        for ident, url in definitions.items():
            check(name + '_reference_definition_' + ident, url in permitted_urls)
    changed = subprocess.check_output(['git', '-C', str(ROOT), 'diff', '--name-only', BASE]).decode().splitlines()
    extra = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '--others', '--exclude-standard']).decode().splitlines()
    check('only_new_dedicated_directory_changed', all(p.startswith(PREFIX) for p in changed + extra))
    protected = subprocess.check_output(['git', '-C', str(ROOT), 'diff', '--name-only', BASE, '--', 'src', 'scripts', 'tests', 'artifacts', 'AGENTS.md', 'PROJECT_MAP.md', 'VALIDATION_STATUS.md', 'docs/research/track_b_g6_construction_paper_evidence_20261018']).decode()
    check('old_packet_science_and_existing_indexes_unchanged', protected == '')
    payload = []
    for p in sorted(HERE.iterdir()):
        if p.is_file() and p.name not in ('verification.json', 'publication_receipt.json'):
            b = p.read_bytes()
            payload.append({'path': p.name, 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()})
    return {'schema_version': 1, 'kind': 'NON_SCIENTIFIC_DOCUMENTARY_METADATA_CHECK',
            'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'passed': all(r['pass'] for r in checks), 'checks': checks, 'payload_manifest': payload,
            'self_hash_exclusions': ['verification.json', 'publication_receipt.json'],
            'scientific_source_or_test_executed': False, 'mathematical_or_novelty_certificate': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--write-report', action='store_true')
    args = parser.parse_args()
    report = verify()
    if args.write_report:
        (HERE / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'passed': report['passed'], 'checks': len(report['checks']),
                      'failed': [r['check'] for r in report['checks'] if not r['pass']]}))
    raise SystemExit(0 if report['passed'] else 1)
