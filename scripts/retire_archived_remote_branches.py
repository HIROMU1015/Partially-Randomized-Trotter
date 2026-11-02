#!/usr/bin/env python3
"""Execute the fixed, user-authorized 2026-11-02 branch retirement plan.

Three explicit phases: prepare (local backup), tags (publish archive refs),
delete (guarded retirement). Never changes worktree files or scientific data.
"""
import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVER = Path(__file__).resolve()
DATA = ROOT / 'docs/research/branch_archive_20261102'
REPOSITORY = 'HIROMU1015/Partially-Randomized-Trotter'
PREFIX = 'archive/research-map-20261102/'


def run(repo, *args, input_text=None, check=True):
    result = subprocess.run(['git', '-C', str(repo), *args], input=input_text,
                            text=True, capture_output=True)
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def remote_refs(repo):
    text = run(repo, 'ls-remote', '--heads', '--tags', 'origin').stdout
    return {line.split('\t')[1]: line.split('\t')[0] for line in text.splitlines()}


def verify_origin(repo):
    allowed = {'https://github.com/' + REPOSITORY, 'git@github.com:' + REPOSITORY}
    for options in [('--all',), ('--push', '--all')]:
        urls = run(repo, 'remote', 'get-url', *options, 'origin').stdout.splitlines()
        if not urls or any(url.removesuffix('.git') not in allowed for url in urls):
            raise ValueError('Unexpected origin fetch/push repository.')


def load_plan():
    plan = json.loads((DATA / 'manifest.json').read_text())
    assert plan['repository'] == 'https://github.com/' + REPOSITORY
    if digest(DRIVER) != plan['driver_sha256']:
        raise ValueError('Execution driver differs from the plan.')
    for name, expected in plan['files_sha256'].items():
        assert digest(DATA / name) == expected, ('Plan file changed', name)
    with (DATA / 'branches.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    for row in plan['archive_tags']:
        assert row['tag'].startswith(PREFIX)
        assert re.fullmatch('[0-9a-f]{40}', row['commit'])
    for row in rows:
        assert re.fullmatch('[0-9a-f]{40}', row['fixed_commit'])
        if row['action'].startswith('DELETE_'):
            assert row['initial_remote_sha'] == row['fixed_commit']
            assert row['old_branch'] not in plan['keep_branches']
            assert row['old_branch'] != 'main'
    return plan, rows


def verify_archive_refs(plan, refs):
    for row in plan['archive_tags']:
        assert refs.get('refs/tags/' + row['tag']) == row['commit'], ('Archive absent or moved', row['tag'])


def deletion_arguments(rows, keep):
    for row in rows:
        assert row['old_branch'] not in keep and row['old_branch'] != 'main'
        assert re.fullmatch('[0-9a-f]{40}', row['initial_remote_sha'])
    return ['push', '--atomic', '--porcelain', *[
        '--force-with-lease=refs/heads/' + r['old_branch'] + ':' + r['initial_remote_sha'] for r in rows
    ], 'origin', *[':refs/heads/' + r['old_branch'] for r in rows]]


def prepare(repo, state, plan, rows, bundles):
    assert not (state / 'prepared.json').exists(), 'Existing prepared state: reuse it for tags/delete instead of overwriting.'
    refs = remote_refs(repo)
    save(state / 'remote_refs_before.json', refs)
    # Optional bundles are imported as objects only; existing branches are not moved.
    for bundle in bundles:
        run(repo, 'bundle', 'unbundle', str(bundle.resolve()))
    for entry in plan['archive_tags']:
        sha = entry['commit']
        if run(repo, 'cat-file', '-e', sha + '^{commit}', check=False).returncode:
            run(repo, 'fetch', '--no-tags', 'origin', sha)
        ref = 'refs/tags/' + entry['tag']
        current = run(repo, 'show-ref', '--verify', '--hash', ref, check=False)
        if current.returncode == 0:
            assert current.stdout.strip() == sha, ('Local archive tag collision', ref)
        else:
            run(repo, 'update-ref', ref, sha, '0' * 40)
    main = refs['refs/heads/main']
    run(repo, 'fetch', '--no-tags', 'origin', 'main')
    tips = sorted({s for row in rows for s in json.loads(row['known_commit_versions'])})
    exclusions = ['^' + r['commit'] for r in plan['archive_tags']] + ['^' + main]
    uncovered = int(run(repo, 'rev-list', '--count', '--stdin', input_text='\n'.join(tips + exclusions) + '\n').stdout)
    assert uncovered == 0, ('Known history not preserved', uncovered)
    backup = state / 'before-delete.bundle'
    assert not backup.exists(), 'Use a fresh state directory after an interrupted preparation.'
    run(repo, 'bundle', 'create', str(backup), '--all')
    run(repo, 'bundle', 'verify', '-q', str(backup))
    save(state / 'prepared.json', dict(status='PREPARED', plan_sha256=digest(DATA / 'manifest.json'),
         backup_file=backup.name, backup_sha256=digest(backup), backup_bytes=backup.stat().st_size,
         known_history_uncovered_commits=uncovered, archive_tags_prepared=len(plan['archive_tags'])))
    print('PREPARED: complete local backup and', len(plan['archive_tags']), 'archive tags; no remote changes.')


def require_prepared(repo, state):
    receipt = json.loads((state / 'prepared.json').read_text())
    assert receipt['plan_sha256'] == digest(DATA / 'manifest.json'), 'Prepared plan differs from current plan.'
    backup = state / receipt['backup_file']
    assert digest(backup) == receipt['backup_sha256'], 'Backup bytes changed.'
    run(repo, 'bundle', 'verify', '-q', str(backup))


def publish_tags(repo, state, plan):
    require_prepared(repo, state)
    refs = remote_refs(repo)
    pending = []
    for entry in plan['archive_tags']:
        ref = 'refs/tags/' + entry['tag']
        if ref in refs:
            assert refs[ref] == entry['commit'], ('Remote tag collision', ref)
        else:
            pending.append(entry)
    for i in range(0, len(pending), 20):
        batch = pending[i:i + 20]
        run(repo, 'push', '--atomic', 'origin', *[
            'refs/tags/' + r['tag'] + ':refs/tags/' + r['tag'] for r in batch])
        print('Archive tags published:', min(i + 20, len(pending)), '/', len(pending), flush=True)
    verify_archive_refs(plan, remote_refs(repo))
    save(state / 'tags_verified.json', dict(status='PASS', archive_tags=len(plan['archive_tags'])))
    print('TAGS_VERIFIED; no branches deleted.')


def retire(repo, state, plan, rows, extra_keep):
    require_prepared(repo, state)
    assert json.loads((state / 'tags_verified.json').read_text())['status'] == 'PASS'
    refs = remote_refs(repo)
    verify_archive_refs(plan, refs)
    main = refs['refs/heads/main']
    run(repo, 'fetch', '--no-tags', 'origin', 'main')
    # Require exact public ledger, rather than trusting uncommitted local files.
    for name, expected in plan['files_sha256'].items():
        raw = subprocess.check_output(['git', '-C', str(repo), 'show', main + ':docs/research/branch_archive_20261102/' + name])
        assert hashlib.sha256(raw).hexdigest() == expected, ('Public ledger mismatch', name)
    raw = subprocess.check_output(['git', '-C', str(repo), 'show', main + ':scripts/retire_archived_remote_branches.py'])
    if hashlib.sha256(raw).hexdigest() != plan['driver_sha256']:
        raise ValueError('Public execution driver differs from the plan.')
    index = run(repo, 'show', main + ':docs/research/RESEARCH_RESULTS_INDEX.md').stdout
    assert 'branch_archive_20261102/README.md' in index
    keep = set(plan['keep_branches']) | set(extra_keep) | {'main'}
    progress_file = state / 'deletion_progress.json'
    results = json.loads(progress_file.read_text()) if progress_file.exists() else []
    # A previous interrupted/readback-failed push is reconciled, never retried.
    for entry in results:
        if entry['status'] == 'UNCONFIRMED_AFTER_PUSH':
            current = refs.get('refs/heads/' + entry['branch'])
            entry.update(actual_sha=current, status=(
                'ABSENT_CONFIRMED_AFTER_UNCONFIRMED_PUSH' if current is None else
                'KEPT_AFTER_UNCONFIRMED_PUSH' if current == entry['expected_sha'] else
                'KEPT_REMOTE_CHANGED_DURING_PUSH'))
    save(progress_file, results)
    completed = {r['branch'] for r in results}
    pending = [r for r in rows if r['action'].startswith('DELETE_') and r['old_branch'] not in completed]
    anchors = {r['commit'] for r in plan['archive_tags']}
    pending.sort(key=lambda r: (r['fixed_commit'] in anchors, r['old_branch']))
    while pending:
        refs = remote_refs(repo)
        verify_archive_refs(plan, refs)
        batch = []
        while pending and len(batch) < 20:
            row = pending.pop(0)
            current = refs.get('refs/heads/' + row['old_branch'])
            status = ('KEPT_EXPLICITLY' if row['old_branch'] in keep else
                      'ALREADY_ABSENT' if current is None else
                      'KEPT_REMOTE_ADVANCED' if current != row['initial_remote_sha'] else None)
            if status:
                results.append(dict(branch=row['old_branch'], status=status, expected_sha=row['initial_remote_sha'], actual_sha=current))
            else:
                batch.append(row)
        if batch:
            entries = [dict(branch=r['old_branch'], expected_sha=r['initial_remote_sha'],
                            status='UNCONFIRMED_AFTER_PUSH') for r in batch]
            results.extend(entries)
            # Persist before contacting the server, including crash/transport ambiguity.
            save(progress_file, results)
            result = run(repo, *deletion_arguments(batch, keep), check=False)
            for entry in entries:
                entry['push_returncode'] = result.returncode
            save(progress_file, results)
            if result.returncode:
                with (state / 'push_rejections.log').open('a') as f:
                    f.write(result.stderr + '\n')
            # The return code alone cannot tell whether the server deleted refs.
            after = remote_refs(repo)  # Failure leaves UNCONFIRMED records for explicit resume.
            for entry in entries:
                current = after.get('refs/heads/' + entry['branch'])
                entry.update(actual_sha=current, status=(
                    ('DELETED_WITH_SHA_LEASE' if result.returncode == 0 else
                     'ABSENT_CONFIRMED_AFTER_PUSH_ERROR') if current is None else
                    'KEPT_REMOTE_CHANGED_DURING_PUSH' if current != entry['expected_sha'] else
                    'KEPT_ATOMIC_PUSH_REJECTED' if result.returncode else
                    'KEPT_PRESENT_AFTER_PUSH_SUCCESS'))
            save(progress_file, results)
            verify_archive_refs(plan, after)
        save(progress_file, results)
        print('Candidates processed:', len(results), flush=True)
    final = remote_refs(repo)
    verify_archive_refs(plan, final)
    for entry in results:
        current = final.get('refs/heads/' + entry['branch'])
        entry['final_sha'] = current
        if current is not None and (entry['status'].startswith('ABSENT_') or
                                    entry['status'] in {'DELETED_WITH_SHA_LEASE', 'ALREADY_ABSENT'}):
            entry['status'] = 'KEPT_REAPPEARED_AT_FINAL_CHECK'
    save(progress_file, results)
    receipt = dict(status='COMPLETED_WITH_GUARDED_RETENTIONS' if any(r['status'].startswith('KEPT_') for r in results) else 'COMPLETED',
                   deleted_count=sum(r['status'] == 'DELETED_WITH_SHA_LEASE' for r in results),
                   already_absent=sum(r['status'] == 'ALREADY_ABSENT' for r in results),
                   absent_after_uncertain_push=sum(r['status'].startswith('ABSENT_CONFIRMED_') for r in results),
                   retained_candidates=sum(r['status'].startswith('KEPT_') for r in results),
                   final_heads={k.removeprefix('refs/heads/'): v for k, v in final.items() if k.startswith('refs/heads/')},
                   archive_tags_verified=len(plan['archive_tags']), results=results,
                   worktree_files_changed=False, scientific_execution=False)
    save(state / 'execution.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != 'results'}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'tags', 'delete'])
    parser.add_argument('--state-dir', type=Path, default=ROOT / '.branch-cleanup-20261102')
    parser.add_argument('--bundle', type=Path, action='append', default=[], help='Optional local backup parts in restore order (prepare only).')
    parser.add_argument('--keep', action='append', default=[], help='Additional branch to retain (delete only).')
    args = parser.parse_args()
    state = args.state_dir.resolve()
    assert not state.is_relative_to(DATA), 'Execution state/backups must not be inside public ledger directory.'
    state.mkdir(parents=True, exist_ok=True)
    verify_origin(ROOT)
    plan, rows = load_plan()
    if args.phase == 'prepare':
        prepare(ROOT, state, plan, rows, args.bundle)
    elif args.phase == 'tags':
        publish_tags(ROOT, state, plan)
    else:
        retire(ROOT, state, plan, rows, args.keep)


if __name__ == '__main__':
    main()
