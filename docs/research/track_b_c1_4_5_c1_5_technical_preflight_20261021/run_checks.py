"""Capture deterministic technical tests; no scientific execution authorization."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
tag = sys.argv[1]
assert tag.isdigit()
out = HERE / 'evidence'
out.mkdir(exist_ok=True)
stem = 'test_run_' + tag
assert not (out / (stem + '.json')).exists(), 'preserve earlier test records'
start = time.monotonic()
r = subprocess.run([sys.executable, '-B', str(HERE / 'test_synthetic.py')],
                   capture_output=True, env=os.environ)
for suffix, data in (('.stdout.log', r.stdout), ('.stderr.log', r.stderr)):
    (out / (stem + suffix)).write_bytes(data)
sources = [dict(path=p.name, bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(HERE.glob('*.py'))]
doc = dict(scope='SYNTHETIC_ONLY', exit_code=r.returncode, source=sources,
    python=sys.version, executable=sys.executable,
    executable_sha256=hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
    packages={k:importlib.metadata.version(k) for k in ('numpy','scipy','qiskit')},
    elapsed_seconds=time.monotonic()-start, duration_is_not_benchmark=True,
    H4_inputs_opened_for_numerical_work=False, H4_scientific_calls=0,
    native_backend_compile_transpile_synthesis=0, sufficient_N_computed=0, shots=0,
    statistical_Monte_Carlo_trajectories=0,
    note='Qiskit logical Gate.control may use internal definition passes; no native target or price acquisition')
(out / (stem + '.json')).write_text(json.dumps(doc, indent=2)+'\n')
print(r.stderr.decode())
sys.exit(r.returncode)
