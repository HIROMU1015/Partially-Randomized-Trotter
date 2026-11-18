"""Loopback RPC and immutable artifact transfer for the single Q3 controller.

SSH supplies the connection boundary. This module never chooses scientific
candidates, creates approvals, expires leases, or starts a numerical worker.
"""
import base64
import errno
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import threading


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
    if (Path(root)/'src/trottertracks/rpe_q3_evidence.py').exists():
        paths += ('src/trottertracks/rpe_q3_evidence.py',)
    if (Path(root)/'src/trottertracks/rpe_q3_timing.py').exists():
        paths += ('src/trottertracks/rpe_q3_timing.py',)
    if (Path(root)/'src/trottertracks/rpe_q3_resources.py').exists():
        paths += ('src/trottertracks/rpe_q3_resources.py',)
    if (Path(root)/'src/trottertracks/rpe_worker_storage.py').exists():
        paths += ('src/trottertracks/rpe_worker_storage.py',)
    for path in ('scripts/run_rpe_preparation_job.py', 'src/trottertracks/rpe_preparation_worker.py',
                 'src/trottertracks/rpe_preparation_artifacts.py'):
        if (Path(root)/path).exists():paths += (path,)
    for path in ('src/trottertracks/rpe_pbs.py', 'src/trottertracks/rpe_pbs_worker.py',
                 'scripts/run_rpe_pbs_worker.py', 'scripts/rpe_pbs_gateway.py'):
        if (Path(root)/path).exists():paths += (path,)
    return dict(commit=commit, files={p: hashlib.sha256((Path(root)/p).read_bytes()).hexdigest() for p in paths})


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
                        blob = subprocess.check_output(['git', 'show', key[0]+':'+key[1]], cwd=root)
                        cache[key] = dict(commit=key[0], path=key[1], sha256=hashlib.sha256(blob).hexdigest())
                    found[key] = cache[key]
            for item in value.values(): visit(item)
    if request['route']['kind'] != 'artificial':
        effective = dict(request['route'])
        if 'configurations' in effective:
            effective['configurations'] = {request['configuration_id']:effective['configurations'][request['configuration_id']]}
        visit(effective)
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
    for name, encoded in files.items():
        destination = _artifact_destination(job, name)
        value = base64.b64decode(encoded, validate=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.read_bytes() != value: raise ValueError('Changed immutable artifact: '+name)
        else:
            temporary = destination.with_name(destination.name+'.transfer')
            with temporary.open('wb') as output:
                output.write(value); output.flush(); os.fsync(output.fileno())
            temporary.replace(destination)
            _sync_directory(destination.parent)


def _same_artifact(source, destination):
    if destination.is_symlink() or not destination.is_file():
        return False
    if source.samefile(destination):
        return True
    if source.stat().st_size != destination.stat().st_size:
        return False
    with source.open('rb') as original, destination.open('rb') as saved:
        return all(block == saved.read(len(block)) for block in iter(lambda: original.read(1 << 20), b''))


def publish_staged_files(stage, job):
    """Expose accepted immutable bytes at the job path, with OUTCOME last."""
    stage, job = Path(stage), Path(job)
    sources = (path for path in stage.rglob('*')
               if path != stage/'REQUEST.json' and (path.is_file() or path.is_symlink()))
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
        except OSError as error:
            if error.errno not in (errno.EXDEV, errno.EOPNOTSUPP, errno.ENOSYS, errno.EMLINK):
                raise
            # Some deployments place jobs on another filesystem. The original
            # atomic installer remains the fallback for that file only.
            install_files(job, {name: base64.b64encode(source.read_bytes())})
        _sync_directory(destination.parent)


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
