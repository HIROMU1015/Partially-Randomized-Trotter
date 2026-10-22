"""Verify Phase 2A documents and fixed Git bytes; never import science SOURCE.

Run with the Python standard library. --repository OWNER/NAME=CHECKOUT may be
repeated for external repository object caches. This does not fetch or execute
the files it inspects. No saved scientific estimate is recomputed.
"""
import argparse
import ast
import hashlib
import json
import pathlib
import re
import subprocess

BASE = 'a9bdcf1d44061daceffe896637e2c92fcb3353b4'
REPO = 'HIROMU1015/Partially-Randomized-Trotter'
NAMESPACE = 'docs/research/pr_algorithm_improvement_phase2a_20261023'
REQUIRED = {
    'README.md', 'BOTTLENECK_AND_LEVER_MAP.md',
    'PAST_IMPROVEMENTS_AND_FAILURE_MODES.md', 'OPEN_RESEARCH_PROBLEMS.md',
    'STRONG_BASELINE_AND_OVERLAP_MATRIX.md', 'EVIDENCE_LIMITATIONS.md',
    'PHASE2B_REVIEW_PREPARATION.md', 'SOURCE_EVIDENCE_INDEX.json', 'handoff.md',
    'PRIMARY_SOURCE_GUIDE.md', 'RESEARCH_PROBLEMS.json',
    'FAILURE_MODE_CATALOG.json', 'STUDY_SCOPE_AND_TRIAGE.md', 'STUDY_SCOPE.json',
    'verify_documents.py', 'VERIFICATION.json', 'PUBLICATION.json',
}
FIELDS = {
    'problem_statement', 'existing_bottleneck', 'evidence_text',
    'proposed_research_freedom', 'strong_baseline',
    'improvement_mechanism_hypothesis', 'counterevidence', 'scientific_value',
    'smallest_distinguishing_evidence', 'failure_stop_criterion',
    'current_readiness', 'evidence',
}
ASSESSMENT_FIELDS = {
    'repeat_or_overlap', 'overcome_prior_failure', 'strong_baseline',
    'important_cost_share', 'information_and_preprocessing',
    'same_task_accuracy', 'unresolved_layer', 'existing_evidence_defines_question',
}


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args])


def blob_oid(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def verify(directory, repositories, allow_pending=False):
    root = pathlib.Path(git(directory, 'rev-parse', '--show-toplevel').decode().strip())
    files = {p.name for p in directory.iterdir() if p.is_file()}
    assert REQUIRED <= files, sorted(REQUIRED - files)
    assert not any(p.is_dir() for p in directory.iterdir()), 'unexpected subdirectory'
    documents = {p.name: p.read_text() for p in directory.iterdir() if p.is_file()}
    for name, value in documents.items():
        assert value.endswith('\n'), name
        if name.endswith('.json'):
            json.loads(value)
        if name.endswith('.py'):
            tree = ast.parse(value)
            imported = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.extend(n.name for n in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imported.append(node.module or '')
            assert all(x.split('.')[0] in {
                'argparse', 'ast', 'hashlib', 'json', 'pathlib', 're', 'subprocess'
            } for x in imported), imported
        # Public output must not contain local absolute paths or credentials.
        if name.endswith(('.md', '.json')):
            assert not re.search(r'/(?:home|tmp|Users)/|ssh://|github_pat_|ghp_[A-Za-z0-9]{20}|BEGIN [A-Z ]*PRIVATE KEY', value), name

    catalog = json.loads(documents['RESEARCH_PROBLEMS.json'])
    problems = catalog['problems']
    assert [p['id'] for p in problems] == [f'RQ-{i:02d}' for i in range(1, 8)]
    assert catalog['scientific_priority_assigned'] is False
    assert catalog['adopted'] is False and catalog['new_algorithm_designed'] is False
    sources = json.loads(documents['SOURCE_EVIDENCE_INDEX.json'])['sources']
    lookup = {x['evidence_id']: x for x in sources}
    assert len(lookup) == len(sources)
    for p in problems:
        assert FIELDS <= p.keys(), p['id']
        assert all(p[k] for k in FIELDS), p['id']
        assert set(p['evidence']) <= lookup.keys(), p['id']
        assert p['id'] in documents['PHASE2B_REVIEW_PREPARATION.md']
        section = documents['OPEN_RESEARCH_PROBLEMS.md'].split('## ' + p['id'], 1)[1]
        section = section.split('\n## RQ-', 1)[0]
        assert len(re.findall(r'^### \d+\.', section, re.M)) == 12, p['id']
        assert len(p['review_questions']) >= 2
    expected_op = {f'OP-I{i}' for i in range(1, 5)} | {f'OP-II{i}' for i in range(1, 5)} | {'OP-III1', 'OP-III2'}
    assessments = catalog['opportunity_assessments']
    assert {x['id'] for x in assessments} == expected_op and len(assessments) == 10
    assert {x for p in problems for x in p['ops']} == expected_op
    for item in assessments:
        assert ASSESSMENT_FIELDS <= item.keys() and all(item[k] for k in ASSESSMENT_FIELDS)
    scope = json.loads(documents['STUDY_SCOPE.json'])
    assert scope['catalog_units'] == len(scope['studies']) == 70
    assert len({x['study_id'] for x in scope['studies']}) == 70
    assert scope['targeted_primary_record_recheck_units'] == sum(x['targeted_primary_record_recheck'] for x in scope['studies']) == 30
    failures = json.loads(documents['FAILURE_MODE_CATALOG.json'])['entries']
    assert len(failures) == 23
    for x in failures:
        assert set(x['evidence_ids']) <= lookup.keys()

    link_count = 0
    for name, value in documents.items():
        if not name.endswith('.md'):
            continue
        for link in re.findall(r'\]\(([^)]+)\)', value):
            if link.startswith(('https://', 'http://', '#')):
                continue
            assert (directory / link.split('#')[0]).is_file(), (name, link)
            link_count += 1
    references = []
    skipped = []
    for x in sources:
        assert re.fullmatch('[0-9a-f]{40}', x['commit'])
        assert x['url'] == f"https://github.com/{x['repository']}/blob/{x['commit']}/{x['path']}"
        assert x['science_import_or_execution_this_phase'] is False
        cache = repositories.get(x['repository'])
        if cache is None:
            skipped.append(x['evidence_id'])
            continue
        data = git(cache, 'show', x['commit'] + ':' + x['path'])
        assert len(data) == x['bytes'], x['evidence_id']
        assert hashlib.sha256(data).hexdigest() == x['sha256'], x['evidence_id']
        assert blob_oid(data) == x['blob_sha1'], x['evidence_id']
        if x['path'].endswith('.py'):
            ast.parse(data)  # Parse text, never import or execute the module.
        if x['path'].endswith('.json'):
            json.loads(data)  # Container only, no scientific fields recalculated.
        references.append({'evidence_id': x['evidence_id'], 'bytes': len(data),
                           'sha256': x['sha256'], 'blob_sha1': x['blob_sha1']})

    # No modification/deletion of any base file is allowed, including new docs
    # accidentally staged outside this namespace.
    diff = git(root, 'diff', '--name-status', BASE).decode().splitlines()
    for row in diff:
        action, path = row.split('\t', 1)
        assert action == 'A' and path.startswith(NAMESPACE + '/'), row
    untracked = git(root, 'ls-files', '--others', '--exclude-standard').decode().splitlines()
    assert all(x.startswith(NAMESPACE + '/') for x in untracked), 'untracked path outside namespace'
    publication = json.loads(documents['PUBLICATION.json'])
    assert allow_pending or publication['state'] == 'PAYLOAD_VERIFIED'
    inventory = [{'path': NAMESPACE + '/' + n, 'bytes': len(v.encode()),
                  'sha256': hashlib.sha256(v.encode()).hexdigest(),
                  'blob_sha1': blob_oid(v.encode())}
                 for n, v in sorted(documents.items()) if n != 'VERIFICATION.json']
    return {
        'schema_version': 1, 'status': 'PASS', 'scope': 'document/identity verification only',
        'base_commit': BASE, 'problem_count': 7, 'old_opportunity_count': 10,
        'required_fields_per_problem': 12, 'assessment_questions_per_opportunity': 8,
        'study_catalog_units': 70, 'targeted_primary_record_units': 30,
        'failure_mode_entries': 23, 'relative_links_checked': link_count,
        'reference_files_checked': len(references), 'reference_files_skipped': skipped,
        'http_exact_byte_checks_recorded': sum(bool(x['retrieval'].get('github_http_bytes_checked')) for x in sources),
        'fixed_reference_checks': references, 'new_document_inventory': inventory,
        'inventory_self_exclusion': 'VERIFICATION.json excluded from its own digest inventory',
        'all_base_paths_unmodified': True, 'namespace_only': True,
        'public_path_and_secret_pattern_scan': 'PASS for Markdown/JSON; verifier source inspected separately',
        'publication_state_at_check': publication['state'],
        'science_execution_or_import': False, 'scientific_reanalysis_or_reranking': False,
        'scientific_review_or_adoption': False,
        'scientific_correctness_certified': False,
        'final_head_and_self_bytes_verification': 'Performed after publication; reported in final chat. Not recursively claimed by this file.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', action='append', default=[])
    parser.add_argument('--allow-pending-publication', action='store_true')
    parser.add_argument('--preservation-report', help='Sanitized read-only workspace observation JSON')
    parser.add_argument('--output')
    args = parser.parse_args()
    directory = pathlib.Path(__file__).resolve().parent
    root = pathlib.Path(git(directory, 'rev-parse', '--show-toplevel').decode().strip())
    repositories = {REPO: root}
    for item in args.repository:
        name, path = item.split('=', 1)
        repositories[name] = pathlib.Path(path)
    result = verify(directory, repositories, args.allow_pending_publication)
    if args.preservation_report:
        result['original_workspace_observation'] = json.loads(pathlib.Path(args.preservation_report).read_text())
    data = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        pathlib.Path(args.output).write_text(data)
    print(json.dumps({k: result[k] for k in [
        'status', 'problem_count', 'old_opportunity_count', 'reference_files_checked',
        'reference_files_skipped', 'http_exact_byte_checks_recorded', 'namespace_only'
    ]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
