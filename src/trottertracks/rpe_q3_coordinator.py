"""Loopback RPC and immutable artifact transfer for the single Q3 controller.

SSH supplies the connection boundary. This module never chooses scientific
candidates, creates approvals, expires leases, or starts a numerical worker.
"""
import base64
import errno
import hashlib
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import stat
import threading
import uuid


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as output:
        json.dump(value, output, indent=2, default=str)
        output.write('\n')
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    if os.name != 'nt':  # Windows cannot open directory handles with os.open.
        _sync_directory(path.parent)


def source_manifest(root, commit):
    paths = ('scripts/run_rpe_optimizer_q3_job.py', 'scripts/run_rpe_q3_worker.py',
             'scripts/run_rpe_method_optimizer.py', 'src/trottertracks/rpe_optimizer_concurrent.py',
             'src/trottertracks/rpe_q3_coordinator.py', 'src/trottertracks/rpe_q3_store.py',
             'src/trottertracks/rpe_q3_worker.py', 'src/trottertracks/rpe_dp_prefix_reuse.py')
    # A separately pinned old computation root keeps its old byte manifest.
    # New roots which import timing must bind that additional dependency.
    for name in ('rpe_q3_evidence', 'rpe_q3_timing', 'rpe_large_shot_bounds', 'rpe_q3_round_tables',
                 'rpe_dry_check_recovery', 'rpe_pbs_attempt_control'):
        path = f'src/trottertracks/{name}.py'
        if (Path(root)/path).exists():
            paths += (path,)
    if (Path(root)/'src/trottertracks/rpe_q3_resources.py').exists():
        paths += ('src/trottertracks/rpe_q3_resources.py',)
    if (Path(root)/'src/trottertracks/rpe_worker_storage.py').exists():
        paths += ('src/trottertracks/rpe_worker_storage.py',)
    for path in ('scripts/rpe_hchain_q3.py', 'scripts/run_rpe_hchain_preparation.py',
                 'scripts/run_rpe_preparation_job.py', 'src/trottertracks/rpe_preparation_worker.py',
                 'src/trottertracks/rpe_preparation_artifacts.py',
                 'src/trottertracks/rpe_metadata_bundle.py'):
        if (Path(root)/path).exists():paths += (path,)
    for path in ('src/trottertracks/rpe_pbs.py', 'src/trottertracks/rpe_pbs_state.py', 'src/trottertracks/rpe_pbs_worker.py',
                 'scripts/run_rpe_pbs_worker.py', 'scripts/rpe_pbs_gateway.py'):
        if (Path(root)/path).exists():paths += (path,)
    from .rpe_preparation_worker import fixed_file_sha
    return dict(commit=commit, files={p: fixed_file_sha(Path(root)/p) for p in paths})


@lru_cache(maxsize=16384)
def _immutable_git_blob_sha(root, object_id):
    # Git objects are content-addressed. Keep the scientific SHA-256 of
    # the complete blob, not just the Git (possibly SHA-1) object name.
    blob = subprocess.check_output(['git','--no-replace-objects','cat-file','blob',object_id],cwd=root)
    return hashlib.sha256(blob).hexdigest()


def fixed_git_blob_sha(root, commit, path):
    root = str(Path(root).resolve())
    locator = commit+':'+path
    if len(commit) in (40,64) and all(c in '0123456789abcdef' for c in commit) and '\n' not in path:
        # One batch-check resolves the locator and verifies that the blob
        # exists even on a cache hit (sparse/promisor roots).
        result = subprocess.check_output(['git','--no-replace-objects','cat-file','--batch-check=%(objectname) %(objecttype)'],
                                         cwd=root,text=True,input=locator+'\n').strip().split()
        if len(result)!=2 or result[1]!='blob':
            raise ValueError('Fixed Git input blob is missing: '+path)
        return _immutable_git_blob_sha(root,result[0])
    return hashlib.sha256(subprocess.check_output(['git','show',locator],cwd=root)).hexdigest()


def input_hashes(root, request, cache=None):
    """Hash fixed locator blobs; selectors retain the original full input blob."""
    found = {}
    cache = {} if cache is None else cache
    def visit(value):
        if isinstance(value, list):
            for item in value: visit(item)
        elif isinstance(value, dict):
            if isinstance(value.get('commit'), str) and isinstance(value.get('path'), str):
                key = (value['commit'], value['path'])
                if key not in found:
                    if key not in cache:
                        cache[key] = dict(commit=key[0], path=key[1], sha256=fixed_git_blob_sha(root,*key))
                    found[key] = cache[key]
            for item in value.values(): visit(item)
    if request['route']['kind'] != 'artificial':
        effective = dict(request['route'])
        if 'configurations' in effective:
            effective['configurations'] = {request['configuration_id']:effective['configurations'][request['configuration_id']]}
        visit(effective)
        if effective['kind'] == 'pauli_b_common_entry':
            entry = (effective.get('configurations') or {}).get(request['configuration_id'], {})
            locator = entry.get('configuration') or {}
            if locator.get('path', '').endswith('.json') and locator.get('commit'):
                dependency_key = ('pauli_dictionary', digest(locator))
                if dependency_key not in cache:
                    selected = json.loads(subprocess.check_output(
                        ['git', 'show', locator['commit']+':'+locator['path']], cwd=root))
                    for part in locator.get('selector', []):
                        selected = selected[part]
                    # Sparse worker views also need the fixed residual receipt
                    # referenced inside the canonical Pauli dictionary identity.
                    cache[dependency_key] = selected['spec']['split'].get('tail_dictionary')
                visit(cache[dependency_key])
    return list(found.values())


def prefix_key(request):
    # Allocation is deliberately omitted: the actual prefix helper separately
    # checks prefix shots and numerical contracts on the host. This is affinity,
    # never permission to accept a cache hit or reuse a scientific evaluation.
    rule = request['route']
    effective = {k: v for k, v in rule.items() if k not in ('view', 'interpreter', 'configurations', 'configuration_ids')}
    effective['configuration'] = (rule.get('configurations') or {}).get(request['configuration_id'])
    return digest(dict(method=request['method'], configuration=request['configuration_id'],
                       expected=request['expected'], route=effective))


def _artifact_destination(job, name):
    forbidden = {'REQUEST.json', 'LAUNCH.json', 'STARTING.json'}
    part = PurePosixPath(name)
    if part.is_absolute() or '..' in part.parts or not part.parts or part.as_posix() in forbidden:
        raise ValueError('Unsafe or controller-owned artifact path: '+name)
    destination = job.joinpath(*part.parts)
    if not destination.resolve().is_relative_to(job.resolve()):
        raise ValueError('Artifact leaves job directory')
    return destination


def _sync_directory(path):
    directory = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def install_files(job, files):
    """Idempotent import; controller-owned request/launch records stay original."""
    job = Path(job)
    shared_events = {}
    for name, encoded in files.items():
        destination = _artifact_destination(job, name)
        value = base64.b64decode(encoded, validate=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Preserve common/X/Y event originals through transport. They are
        # immutable and already shared by the native producer.
        key = hashlib.sha256(value).hexdigest() if destination.name == 'events_PRIVATE.json' else None
        if destination.exists():
            if destination.read_bytes() != value: raise ValueError('Changed immutable artifact: '+name)
        else:
            source = shared_events.get(key) if key else None
            if source is not None and not source.is_symlink() and source.is_file():
                try:
                    os.link(source, destination)
                except OSError:
                    pass
                else:
                    _sync_directory(destination.parent)
                    continue
            temporary = destination.with_name(destination.name+'.transfer')
            with temporary.open('wb') as output:
                output.write(value); output.flush(); os.fsync(output.fileno())
            temporary.replace(destination)
            _sync_directory(destination.parent)
        if key is not None and not destination.is_symlink():
            shared_events.setdefault(key, destination)


def local_result_digest(files, manifest):
    """Verify local source bytes and reproduce the *legacy* base64 completion digest.

    Stream groups divisible by three, never build a base64 document in memory.
    Keeping the legacy digest preserves stopped-run recovery/publication.
    """
    if set(files) != set(manifest):
        raise ValueError('Local result manifest does not match completed files')
    digest_state=hashlib.sha256();digest_state.update(b'{')
    for index,name in enumerate(sorted(files)):
        path=Path(files[name]);expected=manifest[name]
        before=path.lstat()
        if (not stat.S_ISREG(before.st_mode) or before.st_mode & 0o222
                or type(expected.get('size')) is not int or expected['size']!=before.st_size):
            raise ValueError('Local result is not finalized: '+name)
        if index:digest_state.update(b',')
        digest_state.update((json.dumps(name)+':"').encode())
        checksum=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(3*256*1024), b''):
                checksum.update(block);digest_state.update(base64.b64encode(block))
        digest_state.update(b'"')
        after=path.lstat()
        if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns) or checksum.hexdigest()!=expected.get('sha256'):
            raise ValueError('Local result content differs: '+name)
    digest_state.update(b'}')
    return digest_state.hexdigest()


def install_local_result(job, files, manifest):
    """Install finalized files at trusted owned paths, retaining exact content."""
    import shutil
    job=Path(job)
    for name,source in files.items():
        source=Path(source)
        target=_artifact_destination(job,name)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() or target.is_symlink():
            if not _same_artifact(source,target):raise ValueError('Changed immutable artifact: '+name)
            continue
        temporary=target.with_name(target.name+'.local-transfer-'+uuid.uuid4().hex)
        try:
            try:
                os.link(source,temporary)
            except OSError as error:
                if error.errno != errno.EXDEV:raise
                with source.open('rb') as original, temporary.open('xb') as copy:
                    shutil.copyfileobj(original,copy,1<<20)
                    copy.flush();os.fsync(copy.fileno())
                temporary.chmod(temporary.stat().st_mode & ~0o222)
            with temporary.open('rb') as stream:
                checksum=hashlib.file_digest(stream,'sha256').hexdigest()
                os.fsync(stream.fileno())
            if checksum!=manifest[name]['sha256']:
                raise ValueError('Changed local result during transfer: '+name)
            temporary.replace(target)
            _sync_directory(target.parent)
        finally:
            temporary.unlink(missing_ok=True)


def _same_artifact(source, destination):
    if destination.is_symlink() or not destination.is_file():
        return False
    if source.samefile(destination):
        return True
    if source.stat().st_size != destination.stat().st_size:
        return False
    with source.open('rb') as original, destination.open('rb') as saved:
        return all(block == saved.read(len(block)) for block in iter(lambda: original.read(1 << 20), b''))


def publication_directory_stamp(path):
    info = Path(path).lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError('Publication directory is not an ordinary directory')
    return [info.st_dev,info.st_ino,info.st_mtime_ns,info.st_ctime_ns]


def publish_staged_files(stage, job, *, capture_signature=False):
    """Expose accepted immutable bytes at the job path, with OUTCOME last.

    Optionally capture the directory identities during the existing traversal,
    so a clean first Resume need not walk every staged file again.
    """
    stage, job = Path(stage), Path(job)
    stage_dirs, job_dirs = {}, {'.'}
    linked = capture_signature
    if linked:
        try:stage_dirs['.']=publication_directory_stamp(stage)
        except (OSError,ValueError):linked=False
    sources=[]
    for path in stage.rglob('*'):
        if linked and path.is_dir() and not path.is_symlink():
            try:stage_dirs[path.relative_to(stage).as_posix()]=publication_directory_stamp(path)
            except (OSError,ValueError):linked=False
        if path != stage/'REQUEST.json' and (path.is_file() or path.is_symlink()):
            sources.append(path)
    for source in sorted(sources, key=lambda path: (path == stage/'OUTCOME.json', path)):
        name = source.relative_to(stage).as_posix()
        if source.is_symlink() or not source.resolve().is_relative_to(stage.resolve()):
            raise ValueError('Unsafe staged artifact path: '+name)
        destination = _artifact_destination(job, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
        except FileExistsError:
            if not _same_artifact(source, destination):
                raise ValueError('Changed immutable artifact: '+name)
            if linked and not source.samefile(destination):linked=False
        except OSError as error:
            if error.errno not in (errno.EXDEV, errno.EOPNOTSUPP, errno.ENOSYS, errno.EMLINK):
                raise
            # Some deployments place jobs on another filesystem. The original
            # atomic installer remains the fallback for that file only.
            install_files(job, {name: base64.b64encode(source.read_bytes())})
            linked=False
        if linked:job_dirs.update(parent.as_posix() for parent in PurePosixPath(name).parents)
        _sync_directory(destination.parent)
    if linked:
        try:
            if all(publication_directory_stamp(stage/name)==stamp for name,stamp in stage_dirs.items()):
                return dict(stage=stage_dirs,job={name:publication_directory_stamp(job/name) for name in job_dirs})
        except (OSError,ValueError):pass
    return None


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(executor, bind, port):
    if bind not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('Distributed Q3 RPC must bind loopback; use SSH transport')
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if size <= 0: raise ValueError('JSON body required')
                value = json.loads(self.rfile.read(size))
                result = executor.rpc(self.path, value)
                code = 200
            except Exception as exc:
                code, result = 409, dict(error=type(exc).__name__+': '+str(exc))
            payload = json.dumps(result, default=str).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            try: self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError): pass
    server = Server((bind, port), Handler)
    thread = threading.Thread(target=server.serve_forever, name='rpe-q3-rpc', daemon=True)
    thread.start()
    return server, thread
