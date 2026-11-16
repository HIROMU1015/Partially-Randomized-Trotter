"""Loopback RPC and immutable artifact transfer for the single Q3 controller.

SSH supplies the connection boundary. This module never chooses scientific
candidates, creates approvals, expires leases, or starts a numerical worker.
"""
import base64
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
        import os
        os.fsync(output.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)


def source_manifest(root, commit):
    paths = ('scripts/run_rpe_optimizer_q3_job.py', 'scripts/run_rpe_q3_worker.py',
             'scripts/run_rpe_method_optimizer.py', 'src/trottertracks/rpe_optimizer_concurrent.py',
             'src/trottertracks/rpe_q3_coordinator.py', 'src/trottertracks/rpe_q3_store.py',
             'src/trottertracks/rpe_q3_worker.py', 'src/trottertracks/rpe_dp_prefix_reuse.py')
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


def install_files(job, files):
    """Idempotent import; controller-owned request/launch records stay original."""
    job = Path(job)
    forbidden = {'REQUEST.json', 'LAUNCH.json', 'STARTING.json'}
    for name, encoded in files.items():
        part = PurePosixPath(name)
        if part.is_absolute() or '..' in part.parts or not part.parts or name in forbidden:
            raise ValueError('Unsafe or controller-owned artifact path: '+name)
        destination = job.joinpath(*part.parts)
        if not destination.resolve().is_relative_to(job.resolve()):
            raise ValueError('Artifact leaves job directory')
        value = base64.b64decode(encoded, validate=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.read_bytes() != value: raise ValueError('Changed immutable artifact: '+name)
        else:
            temporary = destination.with_name(destination.name+'.transfer')
            with temporary.open('wb') as output:
                output.write(value); output.flush(); os.fsync(output.fileno())
            temporary.replace(destination)
            directory = os.open(destination.parent, os.O_RDONLY)
            try: os.fsync(directory)
            finally: os.close(directory)


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
