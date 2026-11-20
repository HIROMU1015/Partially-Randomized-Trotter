#!/usr/bin/env python3
"""Deduplicate the encoded Git objects of frozen preparation delivery bundles."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from trottertracks.rpe_preparation_retention import ArtifactCache, file_sha256, restore


def git(repository, *command, **kwargs):
    return subprocess.check_output(['git', *command], cwd=repository, stderr=subprocess.PIPE, **kwargs)


def bundle_ranges(stream, scratch):
    # index-pack must read from precisely the first PACK byte.
    header = bytearray()
    while line := stream.readline():
        header.extend(line)
        if line == b'\n':
            break
    pack_start = stream.tell()
    pack_header = stream.read(12)
    if not header.startswith((b'# v2 git bundle\n', b'# v3 git bundle\n')) or not header.endswith(b'\n\n'):
        raise ValueError('Incomplete Git bundle header')
    if (not pack_header.startswith(b'PACK')
            or (b'@object-format=' in header and b'@object-format=sha1\n' not in header)):
        raise ValueError('Only SHA-1 Git packs are supported')
    stream.seek(pack_start)
    pack_id = git(scratch, 'index-pack', '--stdin', stdin=stream).split()[-1].decode()
    pack = scratch/'objects/pack'/f'pack-{pack_id}'
    with pack.with_suffix('.idx').open('rb') as index:
        offsets = sorted(int(line.split()[0]) for line in git(scratch, 'show-index', stdin=index).splitlines())
    pack.with_suffix('.idx').unlink()
    pack.with_suffix('.pack').unlink()
    pack_end = stream.seek(0, os.SEEK_END) - pack_start - len(bytes.fromhex(pack_id))
    if (not offsets or offsets[0] != len(pack_header) or offsets[-1] >= pack_end
            or len(offsets) != struct.unpack('>I', pack_header[8:12])[0]):
        raise ValueError('Git pack offsets disagree with the received bundle')
    stream.seek(pack_start + pack_end)
    trailer = stream.read()
    if trailer.hex() != pack_id:
        raise ValueError('Original pack trailer differs from its index')
    ranges = [(pack_start + start, end - start)
              for start, end in zip(offsets, offsets[1:] + [pack_end], strict=True)]
    return bytes(header) + pack_header, trailer, ranges


def sync_path(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def archive_inputs(requests, output):
    if output.exists():
        raise FileExistsError(output)
    if any(os.environ.get(key) for key in ('GIT_DIR', 'GIT_COMMON_DIR', 'GIT_OBJECT_DIRECTORY',
                                          'GIT_ALTERNATE_OBJECT_DIRECTORIES')):
        raise ValueError('Clear the Git repository and alternate object overrides before archiving')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.rpe-input-archive-', dir=output.parent) as temporary:
        temporary = Path(temporary)
        scratch, archive = temporary/'index.git', temporary/'archive'
        scratch.mkdir()
        git(scratch, 'init', '-q', '--bare')
        archive.mkdir()
        shared_entries, selected, locations = {}, {}, []
        with (archive/'pack-entries.bin').open('wb') as shared:
            for path in requests:
                path = path.resolve(strict=True)
                cache = ArtifactCache(path.parents[3]/'distributed/artifacts')
                request = json.loads(path.read_bytes())
                spec = request['preparation']
                for row in spec['input_git_bundles']:
                    ref, sha = row['ref'], row['sha256']
                    if row.get('workspace_objects') or not ref.startswith('refs/rpe-preparation/input-bundles/inputs-'):
                        continue
                    original = cache.path(sha)
                    if original.is_symlink() or file_sha256(original) != sha:
                        raise ValueError(f'Original delivery bundle differs from REQUEST: {sha}')
                    locations.append(dict(request=str(path), request_sha256=file_sha256(path),
                                          artifact_sha256=sha, bundle_bytes=original.stat().st_size))
                    if sha in selected:
                        continue
                    selected[sha] = request
                    entries = []
                    with original.open('rb', buffering=0) as stream:
                        header, trailer, ranges = bundle_ranges(stream, scratch)
                        for offset, size in ranges:
                            stream.seek(offset)
                            raw = stream.read(size)
                            if len(raw) != size:
                                raise ValueError(f'Truncated Git pack entry: {sha}')
                            key = hashlib.sha256(raw).digest()
                            if key not in shared_entries:
                                shared_entries[key] = (shared.tell(), size)
                                shared.write(raw)
                            entries.append(shared_entries[key])
                    recipe = dict(sha256=sha, source=spec['source']['commit'], ref=ref, header=header.hex(),
                                  trailer=trailer.hex(), entries=entries)
                    with gzip.open(archive/(sha+'.recipe.json.gz'), 'wt') as stream:
                        json.dump(recipe, stream, separators=(',', ':'))
        if not selected:
            raise ValueError('No inputs-* bundles were selected')
        # Every SHA must rebuild without the original packs, checkout or Git.
        for sha, request in selected.items():
            restore(request, sha, archive=archive)
        receipt = dict(schema='RPE_PREPARATION_PACK_ENTRY_ARCHIVE_V1', bundle_count=len(selected),
                       unique_entries=len(shared_entries), byte_count=(archive/'pack-entries.bin').stat().st_size,
                       requests=locations)
        (archive/'ARCHIVE.json').write_text(json.dumps(receipt, indent=2)+'\n')
        for path in archive.iterdir():
            sync_path(path)
        sync_path(archive)
        archive.rename(output)
        sync_path(output.parent)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', action='append', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        receipt = archive_inputs(args.request, args.output.absolute())
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Archive failed: {error}\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
