"""Additive recovery of the derived preparation Git object cache."""
import hashlib
import os
from pathlib import Path
import re
import subprocess
import tempfile


def git(repo, *args, env=None):
    return subprocess.check_output(['git', '-C', str(repo), *map(str, args)],
                                   env=env, stderr=subprocess.PIPE, text=True)


def flattened_object_directories(directories):
    """Return direct object-store references, deepest first, without copying Git objects."""
    seen, visiting, ordered = set(), set(), []
    def visit(directory):
        directory = Path(directory).resolve()
        # An old alternate may have been detached. Keep Git's normal
        # tolerance for missing optional stores; exact input checks still hold.
        if not directory.is_dir():
            return
        if directory in seen:
            return
        if directory in visiting:
            return
        visiting.add(directory)
        alternate = directory/'info/alternates'
        if alternate.is_file():
            for line in alternate.read_text().splitlines():
                if line:
                    path = Path(line)
                    visit(path if path.is_absolute() else directory/path)
        visiting.remove(directory)
        seen.add(directory)
        ordered.append(str(directory))
    for directory in directories:
        visit(directory)
    return ordered


def flatten_new_clone_alternates(repo):
    """Only called immediately after creating a private clone."""
    objects = Path(git(repo,'rev-parse','--path-format=absolute','--git-path','objects').strip())
    alternate = objects/'info/alternates'
    if not alternate.is_file():
        return
    source = [objects/line for line in alternate.read_text().splitlines() if line]
    entries = flattened_object_directories(source)
    # Do not rewrite an inherited object database, only this new clone.
    alternate.write_text(''.join(path+'\n' for path in entries))


def bundle_heads(path, repo):
    heads = [line.split(maxsplit=1) for line in
             git(repo, 'bundle', 'list-heads', path).splitlines()]
    if not heads or any(len(item) != 2 for item in heads):
        raise ValueError('Invalid metadata bundle heads')
    return heads


def metadata_ref_rows(repo):
    rows = git(repo, 'for-each-ref', '--format=%(refname) %(objectname)',
               'refs/rpe-metadata/').splitlines()
    return [row.split() for row in rows]


def missing_refs(repo):
    rows = metadata_ref_rows(repo)
    if not rows:
        return []
    result = subprocess.check_output(['git', '-C', str(repo), 'cat-file', '--batch-check'],
        input=''.join(oid+'\n' for _, oid in rows), text=True).splitlines()
    if len(result) != len(rows):
        raise ValueError('Incomplete Git object inventory')
    return [(ref, oid) for (ref, oid), line in zip(rows, result)
            if line.endswith(' missing')]


def restore_bundle(repo, path, sha, *, required_ref=None, expected=None):
    """Caller holds IMPORT.lock. Never remove refs or change scientific SOURCE."""
    repo, path = Path(repo).resolve(), Path(path).resolve()
    with path.open('rb') as stream:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    if checksum.hexdigest() != sha:
        raise ValueError('Metadata bundle content hash mismatch')
    heads = bundle_heads(path, repo)
    if required_ref is not None and required_ref not in [ref for _, ref in heads]:
        raise ValueError('Metadata bundle lacks its declared ref')
    desired = {'refs/rpe-metadata/'+sha+'/'+str(i): oid
               for i, (oid, _) in enumerate(heads)}
    existing = dict(metadata_ref_rows(repo))
    if expected is not None and any(desired.get(ref) != oid for ref, oid in expected):
        raise ValueError('Metadata recovery does not match the recorded refs')
    if any(ref in existing and existing[ref] != oid for ref, oid in desired.items()):
        raise ValueError('Existing metadata ref conflicts with its bundle')
    # Git's usual verify traverses all refs, so an unrelated missing object
    # poisons every import. Verify in an empty ref namespace sharing precisely
    # the original object database (including its alternates).
    env = dict(os.environ, GIT_OBJECT_DIRECTORY=str(repo/'objects'))
    with tempfile.TemporaryDirectory(prefix='rpe-metadata-verify-') as scratch:
        subprocess.run(['git', 'init', '--bare', '-q', scratch], check=True)
        git(scratch, 'bundle', 'verify', path, env=env)
        git(scratch, 'bundle', 'unbundle', path, env=env)
    for ref, oid in desired.items():
        git(repo, 'cat-file', '-t', oid)
        if ref not in existing:
            git(repo, 'update-ref', ref, oid)
    return len(desired)


def missing_bundles(repo):
    """Only return refs with an exact content-addressed bundle owner."""
    out = {}
    for ref, oid in missing_refs(repo):
        match = re.fullmatch(r'refs/rpe-metadata/([0-9a-f]{64})/[0-9]+', ref)
        if match:
            out.setdefault(match.group(1), []).append((ref, oid))
    return out
