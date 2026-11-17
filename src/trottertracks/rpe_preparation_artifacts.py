"""Immutable, content-addressed preparation artifacts shared through the relay.

The controller and each worker have their own cache. Only missing hashes cross
SSH/HTTP; no filesystem or direct host-to-host connection is assumed.
"""
import base64
import hashlib
import os
from pathlib import Path, PurePosixPath
import uuid


def artifact_hash(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Invalid artifact SHA256')
    return value


def artifact_path(value):
    path = PurePosixPath(value)
    if not isinstance(value, str) or path.is_absolute() or not path.parts or any(p in ('', '.', '..') for p in path.parts):
        raise ValueError('Unsafe artifact path')
    if path.parts[0] == '.git' or '.git' in path.parts:
        raise ValueError('Git metadata is transferred as a bundle, not paths')
    return path


class ArtifactCache:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def path(self, sha):
        sha = artifact_hash(sha)
        return self.root / sha[:2] / sha

    def missing(self, hashes):
        result = []
        for sha in dict.fromkeys(hashes):
            path = self.path(sha)
            if not path.is_file():
                result.append(sha)
            elif path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                raise ValueError('Corrupt cached artifact: ' + sha)
        return result

    def hashes(self):
        if not self.root.exists():return []
        return sorted(p.name for p in self.root.glob('*/*') if p.is_file() and len(p.name) == 64)

    def put(self, blobs):
        count = size = 0
        for sha, encoded in blobs.items():
            path = self.path(sha)
            raw = base64.b64decode(encoded, validate=True)
            if hashlib.sha256(raw).hexdigest() != sha:
                raise ValueError('Transferred artifact hash differs: ' + sha)
            if not self.missing([sha]):continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + '.transfer-' + uuid.uuid4().hex)
            with temporary.open('xb') as stream:
                stream.write(raw);stream.flush();os.fsync(stream.fileno())
            try:
                # Concurrent uploads of the same hash may finish in either order.
                os.link(temporary, path)
                count += 1;size += len(raw)
            except FileExistsError:
                if self.missing([sha]):raise ValueError('Concurrent artifact install failed')
            finally:temporary.unlink()
        return dict(files=count, bytes=size)

    def get(self, hashes):
        missing = self.missing(hashes)
        if missing:raise ValueError('Missing cached artifacts: ' + ','.join(missing))
        return {sha:base64.b64encode(self.path(sha).read_bytes()).decode('ascii') for sha in dict.fromkeys(hashes)}

    def materialize(self, manifest, destination):
        destination = Path(destination).resolve()
        if self.missing(manifest.values()):raise ValueError('Required input artifacts are not cached')
        for name, sha in manifest.items():
            relative = artifact_path(name)
            target = destination.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(destination):raise ValueError('Artifact escapes owned workspace')
            if target.is_symlink():raise ValueError('Symlink artifact target')
            target.parent.mkdir(parents=True, exist_ok=True)
            raw = self.path(sha).read_bytes()
            if target.exists():
                if not target.is_file() or target.read_bytes() != raw:raise ValueError('Changed immutable input artifact: ' + name)
                continue
            temporary=target.with_name(target.name+'.materialize-'+uuid.uuid4().hex)
            with temporary.open('xb') as stream:
                stream.write(raw);stream.flush();os.fsync(stream.fileno())
            try:
                # Publish only complete bytes. Two candidates may adopt the
                # same accepted calibration concurrently; neither sees a
                # partial write or needs to run that scientific task again.
                os.link(temporary,target)
            except FileExistsError:
                if target.is_symlink()or not target.is_file()or target.read_bytes()!=raw:
                    raise ValueError('Changed immutable input artifact: '+name)
            finally:temporary.unlink()

    def capture(self, directory, outputs):
        directory = Path(directory).resolve();manifest = {}
        for name in outputs:
            relative = artifact_path(name);path = directory.joinpath(*relative.parts)
            if not path.resolve().is_relative_to(directory):raise ValueError('Output escapes owned workspace')
            if not path.exists():raise ValueError('Missing declared preparation output: ' + name)
            paths = sorted(path.rglob('*')) if path.is_dir() else [path]
            for item in paths:
                if item.is_symlink():raise ValueError('Symlink preparation artifact')
                if not item.is_file():continue
                rel = item.relative_to(directory).as_posix();artifact_path(rel)
                raw = item.read_bytes();sha = hashlib.sha256(raw).hexdigest()
                self.put({sha:base64.b64encode(raw).decode('ascii')});manifest[rel] = sha
        return manifest


def input_manifest(request):
    inputs = request.get('preparation', {}).get('inputs', {})
    if isinstance(inputs, list):inputs = {row['path']:row['sha256'] for row in inputs}
    if not isinstance(inputs, dict):raise ValueError('Preparation inputs must be an artifact manifest')
    return {str(artifact_path(name)):artifact_hash(sha) for name, sha in inputs.items()}


def worker_cache(profile):
    return ArtifactCache(profile.get('artifact_cache_directory') or Path(profile['work_dir']) / 'artifacts')
