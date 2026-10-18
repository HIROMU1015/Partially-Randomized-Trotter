"""Document/JSON/link/hash checks only. Never imports scientific source."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = '101bf3426fcd04cc0c01c0d202d0916268765d97'
PREFIX = HERE.relative_to(ROOT).as_posix() + '/'
DOCS = ['README.md', 'candidate_operator_inventory.md',
        'pr_rte_connection_and_access.md', 'controlled_native_feasibility.md',
        'pauli_and_known_method_baselines.md', 'application_feasibility_matrix.md',
        'handoff.md']


def verify():
    checks = []

    def check(name, condition):
        checks.append({'check': name, 'pass': bool(condition)})

    def read(name):
        return json.loads((HERE / name).read_text())

    source = read('source_inventory.json')
    evidence = read('evidence_index.json')
    literature = read('literature_scope.json')
    source_ids = {r['id'] for r in source['entries']}
    lit_ids = {r['id'] for r in literature['entries']}
    fixed_urls = {r['url'] for r in source['entries']}
    check('fixed_worktree_base', source['worktree_base_commit'] == BASE == evidence['worktree_base_commit'])
    check('unique_source_ids', len(source_ids) == len(source['entries']) == 28)
    for r in source['entries']:
        check(r['id'] + '_fixed_commit', bool(re.fullmatch('[0-9a-f]{40}', r['source_commit'])))
        check(r['id'] + '_remote_match', r['retrieval_status'] == 'REMOTE_BYTES_EQUAL_GIT_BLOB' and r['sha256'] == r['remote_sha256'])
        check(r['id'] + '_explicit_reading_scope', bool(r['reading_scope']))
        blob = subprocess.check_output(['git', '-C', str(ROOT), 'show', r['source_commit'] + ':' + r['path']])
        check(r['id'] + '_git_blob_hash', len(blob) == r['bytes'] and hashlib.sha256(blob).hexdigest() == r['sha256'])
    check('seven_candidate_ids', {r['id'] for r in evidence['candidates']} == {'C0' + str(i) for i in range(1, 8)})
    axes = set(evidence['evaluation_axes'])
    check('seven_independent_axes', len(axes) == 7)
    for r in evidence['candidates']:
        check(r['id'] + '_axes', set(r['axes']) == axes)
        check(r['id'] + '_valid_evidence', all(x in source_ids | lit_ids for x in r['evidence']))
        check(r['id'] + '_explicit_auxiliary_status', bool(r['auxiliary_status']))
    for claim in evidence['claims']:
        check(claim['id'] + '_references', all(x in source_ids for x in claim.get('sources', [])) and all(x in lit_ids for x in claim.get('literature', [])))
    check('no_unsupported_direct_classification', not any(r['overall_classification'] == 'DIRECTLY_SUPPORTED_BY_EXISTING_SOURCE' for r in evidence['candidates']))
    check('no_scientific_execution_reported', all(x == 0 for x in evidence['execution_accounting'].values()))
    check('mandatory_stop', evidence['mandatory_stop_after_publication'] and not evidence['automatic_merge_or_integration_authorized'])
    check('publication_supplement_failure_retained', next(r for r in literature['entries'] if r['id'] == 'L05')['status'] == 'NOT_RETRIEVED_NOT_READ_THIS_RUN')
    for name in DOCS:
        p = HERE / name
        check(name + '_exists', p.is_file() and p.stat().st_size > 0)
        for target in re.findall(r'\[[^\]]*\]\(([^)\s]+)\)', p.read_text()):
            if target.startswith('https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/'):
                check(name + '_fixed_source_' + target, target.split('#', 1)[0] in fixed_urls)
                if '#L' in target:
                    ref = next(r for r in source['entries'] if r['url'] == target.split('#', 1)[0])
                    raw = subprocess.check_output(['git', '-C', str(ROOT), 'show', ref['source_commit'] + ':' + ref['path']])
                    check(name + '_line_anchor_' + target, max(map(int, re.findall(r'L(\d+)', target.split('#', 1)[1]))) <= len(raw.splitlines()))
            if target.startswith(('https://', 'http://', '#')):
                continue
            local = target.split('#', 1)[0]
            check(name + '_link_' + target, (p.parent / local).resolve().is_file())
    changed = subprocess.check_output(['git', '-C', str(ROOT), 'diff', '--name-only', BASE]).decode().splitlines()
    untracked = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '--others', '--exclude-standard']).decode().splitlines()
    check('only_dedicated_directory_changed', all(p.startswith(PREFIX) for p in changed + untracked))
    protected = subprocess.check_output(['git', '-C', str(ROOT), 'diff', '--name-only', BASE, '--', 'src', 'scripts', 'tests', 'artifacts', 'AGENTS.md', 'PROJECT_MAP.md', 'VALIDATION_STATUS.md']).decode()
    check('scientific_source_results_and_existing_indexes_unchanged', protected == '')
    # No self-hash claim: report and publication receipt are deliberately excluded.
    manifest = []
    for p in sorted(HERE.iterdir()):
        if not p.is_file() or p.name in ('verification.json', 'publication_receipt.json'):
            continue
        data = p.read_bytes()
        manifest.append({'path': p.name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    return {'schema_version': 1, 'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'kind': 'NON_SCIENTIFIC_DOCUMENT_METADATA_VERIFICATION',
            'passed': all(c['pass'] for c in checks), 'checks': checks,
            'payload_manifest': manifest,
            'self_hash_exclusions': ['verification.json', 'publication_receipt.json'],
            'scientific_tests_run': False, 'scientific_source_imported_or_executed': False}


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
