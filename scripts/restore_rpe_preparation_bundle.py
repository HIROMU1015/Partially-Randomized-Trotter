#!/usr/bin/env python3
"""Verify or restore a preparation input bundle from its frozen REQUEST."""
import argparse
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


def restore(request, sha, repository=None, cache_directory=None):
    """Rebuild one source-only inputs-* bundle, optionally publishing it to a cache."""
    sha = artifact_hash(sha)
    spec = request['preparation']
    rows = [row for row in spec['input_git_bundles'] if row['sha256'] == sha]
    if len(rows) != 1 or rows[0].get('workspace_objects'):
        raise ValueError('SHA does not identify a reconstructible input bundle in REQUEST')
    row = rows[0]
    ref = row['ref']
    match = re.fullmatch(r'refs/rpe-preparation/input-bundles/inputs-([0-9a-f]{40})', ref)
    if match is None:
        raise ValueError('Only declared inputs-* bundles are supported')
    source = spec['source']['commit']
    if re.fullmatch(r'[0-9a-f]{40}', source) is None:
        raise ValueError('REQUEST must declare a fixed preparation SOURCE')
    repository = Path(repository or spec['controller_root']).resolve(strict=True)
    tip = subprocess.check_output(
        ['git', 'rev-parse', '--verify', ref + '^{commit}'], cwd=repository,
        text=True, stderr=subprocess.PIPE,
    ).strip()
    if tip != match[1]:
        raise ValueError('Input ref no longer points to the commit recorded in REQUEST')
    locators = sorted({(entry['commit'], entry['path']) for entry in spec['input_hashes']})

    cache = ArtifactCache(cache_directory) if cache_directory is not None else None
    target = cache.path(sha) if cache is not None else None
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.rpe-rebuild-', dir=target.parent if target else None) as temporary:
        packed = Path(temporary) / 'input.bundle'
        build_preparation_git_bundle(repository, ref, [source], locators, packed)
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
        return dict(status=status, source=source, ref=ref, input_head=tip, sha256=sha,
                    bytes=packed.stat().st_size, input_locators=len(locators))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--repository', type=Path, help='checkout retaining the fixed input ref and objects')
    parser.add_argument('--restore-into', type=Path, help='publish matching bytes to this artifact cache')
    args = parser.parse_args()
    try:
        result = restore(json.loads(args.request.read_bytes()), args.sha256, args.repository, args.restore_into)
    except subprocess.CalledProcessError as error:
        parser.exit(1, f'Input bundle reconstruction failed: {error}\n')
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f'Input bundle reconstruction failed: {error}\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
