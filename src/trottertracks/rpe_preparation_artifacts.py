"""Immutable, content-addressed preparation artifacts shared through the relay.

The controller and each worker have their own cache. Only missing hashes cross
SSH/HTTP; no filesystem or direct host-to-host connection is assumed.
"""
from contextlib import closing, suppress
from functools import lru_cache
import base64
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import stat
import threading
import time
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


def df_calibration_receipt(name, output):
    """Identify a path receipt or axis receipt inside one declared calibration."""
    base = artifact_path(output)/'df_prices/calibration/acquisition'
    path = artifact_path(name)
    if path.is_relative_to(base):
        return re.fullmatch(r'm(\d+)/p(\d+)/(?:axes_(?:XY|X|Y)/)?(PATH_RECEIPT|cosine|sine)\.json',
                            path.relative_to(base).as_posix())
    return None


def file_digest(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):digest.update(block)
        return digest.hexdigest()


def copy_file(source, target):
    with Path(source).open('rb') as reader, Path(target).open('xb') as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())


class ArtifactCache:
    def __init__(self, root, *, prior_roots=()):
        self.root = Path(root).resolve()
        self.prior_roots = tuple(dict.fromkeys(Path(path).resolve() for path in prior_roots
                                               if Path(path).resolve() != self.root))
        self._verified = {}
        self._verification_locks = {}
        self._verification_lock = threading.Lock()
        self._inventory_lock = threading.Lock()
        self._inventory = {}

    @staticmethod
    def _identity(path):
        st=path.lstat()
        if not stat.S_ISREG(st.st_mode):
            raise ValueError('Cached artifact is not an ordinary file: '+str(path))
        return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)

    def path(self, sha):
        sha = artifact_hash(sha)
        return self.root / sha[:2] / sha

    def available(self, hashes):
        """Cheap advisory presence check; never authorizes content adoption.

        The dispatch/transfer/materialization paths still hash exact bytes.
        A scheduler predicate must not read a multi-GiB bundle under its lock.
        """
        for sha in dict.fromkeys(hashes):
            path = self.path(sha)
            candidates = (path, *(root/sha[:2]/sha for root in self.prior_roots))
            for candidate in candidates:
                try:
                    self._identity(candidate)
                except (OSError, ValueError):
                    continue
                break
            else:
                return False
        return True

    def missing(self, hashes):
        result = []
        for sha in dict.fromkeys(hashes):
            path = self.path(sha)
            # Immutable CAS files may be shared by dozens of queued requests.
            # Rehash at first use or after any inode/content metadata change,
            # rather than reading multi-GiB inputs on every scheduler poll.
            with self._verification_lock:
                lock=self._verification_locks.setdefault(sha,threading.Lock())
            with lock:
                try:
                    identity=self._identity(path)
                except FileNotFoundError:
                    self._verified.pop(sha,None)
                    if not self._link_prior(sha, path):
                        result.append(sha);continue
                    identity=self._identity(path)
                # Another consumer may materialize a hard link while we read
                # the cache: this changes ctime, not the verified bytes. Use
                # the new identity on a second read rather than fail adoption.
                for attempt in range(4):
                    stable=time.time_ns()-max(identity[-2:])>=1_000_000_000
                    if stable and self._verified.get(sha)==identity:break
                    if file_digest(path)!=sha:
                        self._verified.pop(sha,None)
                        raise ValueError('Corrupt cached artifact: '+sha)
                    current=self._identity(path)
                    if current==identity:
                        if stable:self._verified[sha]=identity
                        break
                    self._verified.pop(sha,None)
                    if attempt==3:
                        # Identical inode, size and mtime + a valid SHA is
                        # sufficient for this read, but never cache it.
                        if current[:4]!=identity[:4]:
                            raise ValueError('Cached artifact changed during verification: '+sha)
                        break
                    identity=current
        return result

    def _link_prior(self, sha, path):
        # The CAS is immutable; sharing a verified inode avoids both upload and
        # a second physical copy. Cross-device archives use the normal upload.
        for root in self.prior_roots:
            source = root/sha[:2]/sha
            try:
                identity = self._identity(source)
                if file_digest(source) != sha or self._identity(source)[:4] != identity[:4]:
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, path)
                except FileExistsError:
                    pass
                return True
            except (OSError, ValueError):
                continue
        return False

    def hashes(self):
        """Advisory inventory: only rescan shards whose directory entries changed."""
        with self._inventory_lock:
            updated = {}
            for shard in self.root.glob('*'):
                try:
                    info = shard.stat()
                    if not stat.S_ISDIR(info.st_mode):continue
                    revision = (info.st_dev,info.st_ino,info.st_mtime_ns,info.st_ctime_ns)
                    cached = self._inventory.get(shard)
                    # Coarse/shared filesystem timestamp ticks can hide
                    # rapid add/remove operations. Reuse only a stable shard.
                    stable = time.time_ns()-max(revision[-2:])>=1_000_000_000
                    if cached is None or cached[0] != revision or not stable:
                        names = tuple(p.name for p in shard.iterdir() if p.is_file() and len(p.name)==64)
                        cached = revision,names
                    updated[shard]=cached
                except FileNotFoundError:
                    continue  # concurrent safe reclamation; inventory is advisory
            self._inventory=updated
            return sorted(name for _,names in updated.values() for name in names)

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

    def materialize(self, manifest, destination, *, immutable=False):
        """Publish verified blobs at their named paths. Workspaces get private copies."""
        destination = Path(destination).resolve()
        if self.missing(manifest.values()):raise ValueError('Required input artifacts are not cached')
        for name, sha in manifest.items():
            relative = artifact_path(name)
            target = destination.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(destination):raise ValueError('Artifact escapes owned workspace')
            if target.is_symlink():raise ValueError('Symlink artifact target')
            target.parent.mkdir(parents=True, exist_ok=True)
            source = self.path(sha)
            if immutable and not target.exists():
                with self._verification_lock:
                    lock=self._verification_locks.setdefault(sha,threading.Lock())
                with lock:
                    before=self._identity(source)
                    links=source.stat().st_nlink
                    verified=self._verified.get(sha)==before
                    try:
                        os.link(source,target)
                    except FileExistsError:
                        pass  # Another reader materialized the result first.
                    except OSError as error:
                        if error.errno not in (errno.EXDEV, errno.EOPNOTSUPP, errno.ENOSYS, errno.EMLINK):
                            raise
                    else:
                        after=self._identity(source)
                        # Only our own successful link can carry a previously
                        # verified identity over the ctime change. The ordinary
                        # one-second stability rule in missing() still applies.
                        if (verified and before[:4]==after[:4]
                                and source.stat().st_nlink==links+1
                                and self._identity(target)==after):
                            self._verified[sha]=after
                        else:
                            self._verified.pop(sha,None)
                        continue
            if immutable and target.is_file() and not target.is_symlink() and source.samefile(target):
                continue
            if target.exists():
                if not target.is_file() or file_digest(target) != sha:raise ValueError('Changed immutable input artifact: ' + name)
                continue
            temporary=target.with_name(target.name+'.materialize-'+uuid.uuid4().hex)
            try:
                copy_file(source, temporary)
                if file_digest(temporary) != sha:raise ValueError('Cached artifact changed during materialization: '+name)
                # Publish only complete bytes. Two candidates may adopt the
                # same accepted calibration concurrently; neither sees a
                # partial write or needs to run that scientific task again.
                os.link(temporary,target)
            except FileExistsError:
                if target.is_symlink()or not target.is_file()or file_digest(target)!=sha:
                    raise ValueError('Changed immutable input artifact: '+name)
            finally:temporary.unlink(missing_ok=True)

    def put_file(self, source, sha):
        """Install an immutable file without building a whole-file byte/base64 copy."""
        path = self.path(sha)
        if not self.missing([sha]):return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.transfer-' + uuid.uuid4().hex)
        try:
            copy_file(source, temporary)
            if file_digest(temporary) != sha:raise ValueError('Captured artifact changed: '+sha)
            try:os.link(temporary, path)
            except FileExistsError:
                if self.missing([sha]):raise ValueError('Concurrent artifact install failed')
        finally:temporary.unlink(missing_ok=True)

    def capture(self, directory, outputs):
        directory = Path(directory).resolve();manifest = {}; files = {}
        for name in outputs:
            relative = artifact_path(name);path = directory.joinpath(*relative.parts)
            if not path.resolve().is_relative_to(directory):raise ValueError('Output escapes owned workspace')
            if not path.exists():raise ValueError('Missing declared preparation output: ' + name)
            paths = sorted(path.rglob('*')) if path.is_dir() else [path]
            for item in paths:
                if item.is_symlink():raise ValueError('Symlink preparation artifact')
                if not item.is_file():continue
                rel = item.relative_to(directory).as_posix();artifact_path(rel)
                stat = item.stat()
                inode = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                sha = files.get(inode)
                if sha is None:
                    sha = file_digest(item)
                    self.put_file(item, sha);files[inode] = sha
                manifest[rel] = sha
        return manifest

    def share_completed_inputs(self, directory, outputs, inputs):
        """Keep unchanged archived workspace copies backed by the same cached bytes."""
        for name, sha in outputs.items():
            if inputs.get(name) != sha or not name.startswith(('workspace/', 'workspaces/')):
                continue
            target = directory.joinpath(*artifact_path(name).parts)
            source = self.path(sha)
            temporary = target.with_name('.share-input-' + uuid.uuid4().hex)
            try:
                local = target.stat()
                cached = source.stat()
                if (target.is_symlink() or source.is_symlink() or not target.resolve().is_relative_to(directory)
                        or local.st_dev != cached.st_dev or local.st_ino == cached.st_ino
                        or local.st_size != cached.st_size or local.st_mode != cached.st_mode):
                    continue
                os.link(source, temporary)
                os.replace(temporary, target)
            except OSError:
                # The receipt and complete private copy are already durable.
                continue
            finally:
                with suppress(OSError):
                    temporary.unlink(missing_ok=True)


def input_manifest(request):
    inputs = request.get('preparation', {}).get('inputs', {})
    if isinstance(inputs, list):inputs = {row['path']:row['sha256'] for row in inputs}
    if not isinstance(inputs, dict):raise ValueError('Preparation inputs must be an artifact manifest')
    return {str(artifact_path(name)):artifact_hash(sha) for name, sha in inputs.items()}


def preparation_dependency_manifest(request, outcome_for):
    """Resolve named inputs from accepted predecessor outcomes."""
    preparation=request.get('preparation') or {}
    manifest=input_manifest(request)
    for predecessor,paths in preparation.get('dependency_inputs',{}).items():
        outcome=outcome_for(predecessor)
        for source,target in paths.items():
            target=str(artifact_path(target))
            sha=artifact_hash(outcome['artifacts'][source])
            if target in manifest and manifest[target]!=sha:
                raise ValueError('Dependency artifact binding conflicts: '+target)
            manifest[target]=sha
    return manifest


def assigned_artifacts(assignment):
    manifest=input_manifest(assignment['request'])
    for name,sha in assignment.get('artifact_manifest',{}).items():
        name=str(artifact_path(name));artifact_hash(sha)
        if name in manifest and manifest[name]!=sha:raise ValueError('Assignment changed an explicit input artifact')
        manifest[name]=sha
    return manifest


def preparation_bundle_hashes(assignment):
    spec = assignment.get('request', assignment).get('preparation', {})
    bundles = list(spec.get('input_git_bundles', []))
    for workspace in spec.get('workspaces', {}).values():
        bundles.extend(workspace.get('input_git_bundles', []))
    return list(dict.fromkeys(bundle['sha256'] for bundle in bundles))


def readiness_artifact_hashes(assignment):
    hashes = preparation_bundle_hashes(assignment)
    spec = assignment.get('request', assignment).get('preparation', {})
    if spec.get('pf_compile_context') is True:
        hashes.extend(assigned_artifacts(assignment).values())
    return list(dict.fromkeys(hashes))


def read_preparation_readiness(directory, database=None):
    """Read the old checkpoint plus per-request updates in the queue database."""
    directory = Path(directory)
    if database is None:
        path = (directory/'QUEUE.sqlite3').resolve()
        if path.is_file():
            with closing(sqlite3.connect(path.as_uri()+'?mode=ro', uri=True)) as db:
                return read_preparation_readiness(directory, db)
    checkpoint = directory/'PREPARATION_PENDING.json'
    saved = json.loads(checkpoint.read_text()) if checkpoint.is_file() else {}
    if saved and saved.get('schema') != 'RPE_PENDING_PREPARATION_READINESS_V1':
        raise ValueError('Unknown saved preparation readiness schema')
    tasks = dict(saved.get('tasks', {}))
    if database is not None and database.execute(
            "SELECT 1 FROM sqlite_master WHERE name='preparation_readiness'").fetchone():
        for job, record in database.execute('SELECT job_id, record FROM preparation_readiness'):
            if record is None:
                tasks.pop(job, None)
            else:
                tasks[job] = json.loads(record)
    return tasks


@lru_cache(maxsize=32)
def _worker_cache(root):
    return ArtifactCache(root)


def worker_cache(profile):
    root = Path(profile.get('artifact_cache_directory') or Path(profile['work_dir']) / 'artifacts').resolve()
    return _worker_cache(root)
