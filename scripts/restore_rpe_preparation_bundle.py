#!/usr/bin/env python3
"""Verify or restore a preparation input bundle from its frozen REQUEST."""
import argparse
from contextlib import contextmanager
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from trottertracks.rpe_preparation_artifacts import ArtifactCache, artifact_hash
from trottertracks.rpe_preparation_bridge import build_preparation_git_bundle


def file_sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@contextmanager
def verified_bundle(sha, cache_directory):
    """Keep reconstructed bytes private until they match the frozen digest."""
    cache = ArtifactCache(cache_directory) if cache_directory is not None else None
    target = cache.path(sha) if cache is not None else None
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.rpe-rebuild-', dir=target.parent if target else None) as temporary:
        packed = Path(temporary) / 'input.bundle'
        yield packed, target


def publish_bundle(packed, target, sha):
    if file_sha256(packed) != sha:
        raise ValueError('Rebuilt bundle SHA-256 differs from the frozen REQUEST')
    status = 'VERIFIED'
    if target is not None:
        with packed.open('rb') as stream:
            os.fsync(stream.fileno())
        try:
            os.link(packed, target)
        except FileExistsError:
            if target.is_symlink() or not target.is_file() or file_sha256(target) != sha:
                raise ValueError('Existing cache entry differs from the frozen REQUEST')
            status = 'ALREADY_PRESENT'
        else:
            status = 'RESTORED'
            fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    return status


def assemble_archive_bundle(archive, recipe, destination):
    entries = archive/'pack-entries.bin'
    size = entries.stat().st_size
    with entries.open('rb') as source, destination.open('wb') as target:
        target.write(bytes.fromhex(recipe['header']))
        for offset, length in recipe['entries']:
            if offset < 0 or length <= 0 or offset + length > size:
                raise ValueError('Archive Git pack entry has an invalid range')
            source.seek(offset)
            while length:
                chunk = source.read(min(length, 1024**2))
                if not chunk:
                    raise ValueError('Archive Git pack entry is truncated')
                target.write(chunk)
                length -= len(chunk)
        target.write(bytes.fromhex(recipe['trailer']))


def restore(request, sha, repository=None, cache_directory=None, *, archive=None):
    """Rebuild one delivery bundle, optionally publishing it to a cache."""
    sha = artifact_hash(sha)
    spec = request['preparation']
    rows = [row for row in spec['input_git_bundles'] if row['sha256'] == sha]
    if len(rows) != 1 or rows[0].get('workspace_objects'):
        raise ValueError('SHA does not identify a reconstructible input bundle in REQUEST')
    ref = rows[0]['ref']
    match = re.fullmatch(r'refs/rpe-preparation/input-bundles/inputs-([0-9a-f]{40})', ref)
    if match is None:
        raise ValueError('Only declared inputs-* bundles are supported')
    source = spec['source']['commit']
    if re.fullmatch(r'[0-9a-f]{40}', source) is None:
        raise ValueError('REQUEST must declare a fixed preparation SOURCE')
    tip = match[1]
    locators = sorted({(entry['commit'], entry['path']) for entry in spec['input_hashes']})
    with verified_bundle(sha, cache_directory) as (packed, target):
        if archive is not None:
            archive = Path(archive)
            with gzip.open(archive / (sha + '.recipe.json.gz'), 'rt') as stream:
                recipe = json.load(stream)
            if (recipe['sha256'], recipe['source'], recipe['ref']) != (sha, source, ref):
                raise ValueError('Archive recipe differs from the frozen REQUEST')
            assemble_archive_bundle(archive, recipe, packed)
        else:
            repository = Path(repository or spec['controller_root']).resolve(strict=True)
            actual = subprocess.check_output(
                ['git', 'rev-parse', '--verify', ref + '^{commit}'], cwd=repository,
                text=True, stderr=subprocess.PIPE,
            ).strip()
            if actual != tip:
                raise ValueError('Input ref no longer points to the commit recorded in REQUEST')
            build_preparation_git_bundle(repository, ref, [source], locators, packed)
        status = publish_bundle(packed, target, sha)
        return dict(status=status, source=source, ref=ref, input_head=tip, sha256=sha,
                    bytes=packed.stat().st_size, input_locators=len(locators))


def restore_run(run, archive, cache_directory=None):
    """Restore a run's original bundle paths before using its fixed Resume."""
    run = run.resolve(strict=True)
    receipt = json.loads((archive/'ARCHIVE.json').read_bytes())
    results = {}
    for row in receipt['requests']:
        original = Path(row['request'])
        owner = original.parents[3]
        if owner.name != run.name:
            continue
        request_path = run/original.relative_to(owner)
        if file_sha256(request_path) != row['request_sha256']:
            raise ValueError('Archived run REQUEST changed: ' + str(request_path))
        sha = row['artifact_sha256']
        if sha not in results:
            results[sha] = restore(json.loads(request_path.read_bytes()), sha, archive=archive,
                                   cache_directory=cache_directory or run/'distributed/artifacts')
    if not results:
        raise ValueError('Archive does not contain the specified run')
    return dict(run_id=run.name, bundles=list(results.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--request', type=Path)
    target.add_argument('--archived-run', type=Path, help='restore all delivery bundles for a frozen run')
    parser.add_argument('--sha256')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--repository', type=Path, help='checkout retaining the fixed input ref and objects')
    source.add_argument('--archive', type=Path, help='standalone preparation input archive')
    parser.add_argument('--restore-into', type=Path, help='publish matching bytes to this artifact cache')
    args = parser.parse_args()
    try:
        if args.archived_run:
            if args.archive is None or args.sha256 is not None:
                parser.error('--archived-run needs --archive and no --sha256')
            result = restore_run(args.archived_run, args.archive, args.restore_into)
        else:
            if args.sha256 is None:
                parser.error('--request needs --sha256')
            result = restore(json.loads(args.request.read_bytes()), args.sha256, args.repository,
                             args.restore_into, archive=args.archive)
    except subprocess.CalledProcessError as error:
        parser.exit(1, f'Input bundle reconstruction failed: {error}\n')
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f'Input bundle reconstruction failed: {error}\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
