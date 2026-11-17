#!/usr/bin/env python3
"""Place saved PR-DF inspection dependencies; never execute a scientific leaf.

The input package contains only files already bound by the saved inspection
REQUESTs. Keep this package and the resulting runtime directory private.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare(tasks, profile, package, repository, directory, bundle=None):
    tasks = tasks.get('tasks', tasks) if isinstance(tasks, dict) else tasks
    if not tasks or any(task.get('method') != 'PR_DF' or
                        task['preparation'].get('dry_check_only') is not True for task in tasks):
        raise ValueError('Only labelled saved PR-DF dry-check-only tasks are accepted')
    requirements = {}
    for task in tasks:
        for alias, required in task['preparation'].get('required_paths', {}).items():
            if alias in requirements and requirements[alias] != required:
                raise ValueError('Conflicting saved runtime requirement: ' + alias)
            requirements[alias] = required
    expected = {}
    for alias, required in requirements.items():
        if '/' in alias or alias in ('.', '..'):
            raise ValueError('Unsafe runtime alias')
        if required['kind'] == 'file':
            expected[alias + '/config.json'] = required['sha256']
        elif required['kind'] == 'directory':
            expected.update({alias + '/' + name: sha for name, sha in required['files'].items()})
        else:
            raise ValueError('Unsupported saved path kind')
    directory = directory.resolve()
    runtime = directory / 'runtime'
    with tarfile.open(package) as archive:
        members = archive.getmembers()
        if len({member.name for member in members}) != len(members) or {member.name for member in members} != set(expected):
            raise ValueError('Input package does not match the saved required file set')
        for member in members:
            path = Path(member.name)
            if not member.isfile() or path.is_absolute() or '..' in path.parts:
                raise ValueError('Unsafe input member')
            raw = archive.extractfile(member).read()
            if sha256(raw) != expected[member.name]:
                raise ValueError('Saved runtime hash differs: ' + member.name)
            target = runtime / path
            if target.exists() and target.read_bytes() != raw:
                raise ValueError('Existing owned runtime differs: ' + str(target))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
    if bundle:
        subprocess.run(['git', 'bundle', 'verify', str(bundle.resolve())], cwd=repository,
                       check=True, capture_output=True)
        for index, line in enumerate(subprocess.check_output(
                ['git', 'bundle', 'list-heads', str(bundle.resolve())], text=True).splitlines()):
            commit, reference = line.split(maxsplit=1)
            target = 'refs/rpe-preparation-inputs/' + sha256(bundle.read_bytes())[:16] + '/' + str(index)
            subprocess.run(['git', 'fetch', str(bundle.resolve()), reference + ':' + target],
                           cwd=repository, check=True, capture_output=True)
    for task in tasks:
        for locator in task['preparation'].get('input_hashes', []):
            raw = subprocess.check_output(['git', 'show', locator['commit'] + ':' + locator['path']], cwd=repository)
            if sha256(raw) != locator['sha256']:
                raise ValueError('Saved Git input hash differs: ' + locator['path'])
    result = deepcopy(profile)
    paths = {alias: {'path': str(runtime / alias / ('config.json' if required['kind'] == 'file' else ''))}
             for alias, required in requirements.items()}
    result.setdefault('preparation_paths', {}).update(paths)
    for entry in result.get('preparation_profiles', []):
        entry['profile'].setdefault('preparation_paths', {}).update(paths)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'PROFILE.json').write_text(json.dumps(result, indent=2) + '\n')
    record = dict(scope='SAVED_PR_DF_INPUT_PLACEMENT_ONLY', host_id=result['host_id'],
                  package_sha256=sha256(package.read_bytes()), files_verified=len(expected),
                  git_inputs_verified=sum(len(task['preparation'].get('input_hashes', [])) for task in tasks),
                  runtime_paths=paths, original_profile_modified=False,
                  scientific_calls=0, compile_calls=0, queue_operations=0)
    (directory / 'PLACEMENT.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--git-bundle', type=Path)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    value = prepare(json.loads(args.tasks.read_text()), json.loads(args.profile.read_text()),
                    args.package, args.repository, args.directory, args.git_bundle)
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
