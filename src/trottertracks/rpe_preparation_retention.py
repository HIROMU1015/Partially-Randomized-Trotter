"""Restore archived preparation inputs for a pinned run."""
from contextlib import contextmanager
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


from .rpe_preparation_artifacts import ArtifactCache, artifact_hash


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
            from .rpe_preparation_bridge import build_preparation_git_bundle
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


def restore_resume_archives(state, run):
    """Rehydrate only cache entries explicitly reclaimed from the selected run."""
    run = Path(run).resolve(strict=True)
    archives = Path(state).resolve().parent / 'archives/preparation-inputs'
    cache = ArtifactCache(run / 'distributed/artifacts')
    result = {'restored': [], 'already_present': []}
    found = set()
    for marker in sorted(archives.glob('*.reclaimed.json')):
        record = json.loads(marker.read_bytes())
        directory = archives / marker.name.removesuffix('.reclaimed.json')
        selected = {sha: row for sha, row in record['bundles'].items() if row['run_id'] == run.name}
        if not selected:
            continue
        if Path(record['archive']).resolve() != directory.resolve() or directory.is_symlink():
            raise ValueError('Reclaim receipt points outside its archive')
        receipt = json.loads((directory / 'ARCHIVE.json').read_bytes())
        requests = {row['artifact_sha256']: row for row in receipt['requests']
                    if Path(row['request']).parents[3].resolve() == run}
        for sha, row in selected.items():
            path = cache.path(sha)
            request = Path(row['request'])
            if (row['source_path'] != str(path) or not request.resolve().is_relative_to(run / 'jobs')
                    or requests.get(sha, {}).get('request') != str(request)
                    or file_sha256(request) != requests[sha]['request_sha256']):
                raise ValueError('Reclaim receipt does not match the frozen request: ' + sha)
            if sha in found:
                continue
            found.add(sha)
            if path.is_file():
                if path.is_symlink() or file_sha256(path) != sha:
                    raise ValueError('Existing preparation cache entry differs: ' + sha)
                result['already_present'].append(sha)
                continue
            restored = restore(json.loads(request.read_bytes()), sha, archive=directory, cache_directory=cache.root)
            name = 'restored' if restored['status'] == 'RESTORED' else 'already_present'
            result[name].append(sha)
    return result
