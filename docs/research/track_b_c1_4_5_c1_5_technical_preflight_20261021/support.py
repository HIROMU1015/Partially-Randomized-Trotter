"""C1-4.5 synthetic-only harness; no H4 input loader or scientific runner."""
import hashlib
from functools import lru_cache
import json
from pathlib import Path
import subprocess
import sys
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FIXED = '243eea8e840f3c472753f7935e510391626fd325'
BASE = '068f0ae8771fccf28c529bda12a84abe8b7e7648'
PRIOR = ROOT / 'docs/research/track_b_c1_2_c1_4_integrated_adapter_validation_20261021'
R1 = ROOT / 'docs/research/track_b_c1_1_r1_new_partition_exact_model_20261021'
MODEL = 'e96ea0faea31bc6a784dfb8dc6f7cf4856789b0256e4a1c53b63ce087c86dc49'

def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(',', ':')).encode()

def digest(path):
    return sha(Path(path).read_bytes())

def sha(data):
    return hashlib.sha256(data).hexdigest()

def save(name, data):
    path = HERE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')

def isolate():
    for name in ('trotterlib', 'trotterlib.df_trotter', 'trottertracks',
                 'trottertracks.algorithm_codesign'):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(ROOT / 'src' / name.replace('.', '/'))]
            sys.modules[name] = module

@lru_cache(None)
def pinned_rows():
    paths = sorted((ROOT / 'src/trottertracks/algorithm_codesign').glob('g[6789]*.py'))
    paths += [ROOT / 'src/trottertracks/algorithm_codesign/return_aggregation.py',
              ROOT / 'src/trotterlib/df_rte_tail.py', ROOT / 'src/trotterlib/df_rte_qiskit.py',
              ROOT / 'src/trotterlib/rpe_hadamard_interrogation.py', PRIOR / 'adapter.py']
    out = []
    for path in paths:
        rel = str(path.relative_to(ROOT))
        pin = BASE if path.parent == PRIOR else FIXED
        expected = subprocess.check_output(['git', 'show', pin + ':' + rel], cwd=ROOT)
        if path.read_bytes() != expected:
            raise ValueError('FIXED_SOURCE_MISMATCH:' + rel)
        out.append(dict(path=rel, bytes=len(expected), sha256=sha(expected)))
    return out

def source_rows():
    rows = pinned_rows()
    for row in rows:
        if sha((ROOT / row['path']).read_bytes()) != row['sha256']:
            raise ValueError('FIXED_SOURCE_MISMATCH:' + row['path'])
    return rows

isolate()
