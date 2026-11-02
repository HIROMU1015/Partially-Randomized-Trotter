"""Task-local runtime isolation and provenance; no package installation."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = '2937e6b655ba3cd77f982d6901d766355a8b92c1'
FIXED = '243eea8e840f3c472753f7935e510391626fd325'
R1 = ROOT / 'docs/research/track_b_c1_1_r1_new_partition_exact_model_20261021'
MODEL = 'e96ea0faea31bc6a784dfb8dc6f7cf4856789b0256e4a1c53b63ce087c86dc49'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(',', ':')).encode()

def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()

def isolate():
    # Skip broad __init__ exports; imported science submodules remain unmodified.
    for name in ('trotterlib', 'trotterlib.df_trotter', 'trottertracks',
                 'trottertracks.algorithm_codesign'):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(ROOT / 'src' / name.replace('.', '/'))]
            sys.modules[name] = module

def verify_rows(rows):
    for row in rows:
        path = Path(row['path'])
        if not path.is_absolute():
            path = ROOT / path
        if digest(path) != row['sha256']:
            raise RuntimeError('PROVENANCE_MISMATCH: ' + str(path))

def environment():
    packages = {}
    for name in ('numpy', 'scipy', 'qiskit', 'qiskit-nature', 'qiskit-algorithms',
                 'openfermion', 'pyscf', 'numba', 'h5py'):
        distribution = importlib.metadata.distribution(name)
        packages[name] = distribution.version
    return {'executable': sys.executable, 'realpath': str(Path(sys.executable).resolve()),
            'executable_sha256': digest(Path(sys.executable).resolve()),
            'python_version': sys.version, 'packages': packages,
            'isolation': 'namespace paths; unchanged selected submodules; no broad package initializers'}
