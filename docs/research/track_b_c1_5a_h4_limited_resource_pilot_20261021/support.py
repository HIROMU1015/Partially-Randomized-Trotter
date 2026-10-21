"""C1-5A task-local I/O and frozen dependency connection. No molecular generation."""
from contextlib import contextmanager
from fractions import Fraction as F
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = 'af75af03c0b334c29b5a4311bd2b4578a999460c'
FIXED = '243eea8e840f3c472753f7935e510391626fd325'
R1 = ROOT / 'docs/research/track_b_c1_1_r1_new_partition_exact_model_20261021'
PRIOR = ROOT / 'docs/research/track_b_c1_2_c1_4_integrated_adapter_validation_20261021'
MODEL = 'e96ea0faea31bc6a784dfb8dc6f7cf4856789b0256e4a1c53b63ce087c86dc49'

def require(test, message):
    if not test:
        raise ValueError(message)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def digest(path):
    return sha(Path(path).read_bytes())

def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()

def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=True) + '\n')
    os.replace(tmp, path)

def gzip_bytes(raw):
    return gzip.compress(raw, compresslevel=6, mtime=0)

def save_blob(directory, record):
    # Atomic hard-link from a fully written private temp file; shared equal blobs deduplicate.
    raw = record.document.encode()
    path = Path(directory) / (record.digest + '.json.gz')
    path.parent.mkdir(parents=True, exist_ok=True)
    compressed = gzip_bytes(raw)
    tmp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    tmp.write_bytes(compressed)
    try:
        os.link(tmp, path)
        created = True
    except FileExistsError:
        require(path.read_bytes() == compressed, 'CONTENT_ADDRESS_COLLISION')
        created = False
    finally:
        tmp.unlink()
    return record.digest, created

def verify_rows(rows):
    for row in rows:
        path = Path(row['path'])
        if not path.is_absolute():
            path = ROOT / path
        require(digest(path) == row['sha256'], 'IDENTITY_MISMATCH:' + str(path))

def isolate():
    for name in ('trotterlib', 'trotterlib.df_trotter', 'trottertracks', 'trottertracks.algorithm_codesign'):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(ROOT / 'src' / name.replace('.', '/'))]
            sys.modules[name] = module

isolate()

def prior_adapter():
    spec = importlib.util.spec_from_file_location('c1_fixed_adapter', PRIOR / 'adapter.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class Timers:
    def __init__(self):
        self.rows = {}

    @contextmanager
    def stage(self, name):
        start = time.perf_counter(), time.process_time()
        try:
            yield
        finally:
            row = self.rows.setdefault(name, dict(calls=0, wall_seconds=0., CPU_seconds=0.))
            row['calls'] += 1
            row['wall_seconds'] += time.perf_counter() - start[0]
            row['CPU_seconds'] += time.process_time() - start[1]

def resource_snapshot():
    return dict(pid=os.getpid(), monotonic=time.monotonic(),
        self_user_CPU=resource.getrusage(resource.RUSAGE_SELF).ru_utime,
        self_system_CPU=resource.getrusage(resource.RUSAGE_SELF).ru_stime,
        max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        meminfo=Path('/proc/meminfo').read_text(),
        pressure={k:Path('/proc/pressure/'+k).read_text() for k in ('cpu','memory','io')})
