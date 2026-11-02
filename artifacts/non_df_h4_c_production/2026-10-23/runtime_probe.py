"""Import/runtime inventory only; no data loader or scientific evaluation."""
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
import ctypes,fcntl,signal,subprocess,datetime,resource
legacy='--legacy' in sys.argv
if legacy:
    old=Path('/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/non-df-h4-execution-foundation-20261023')
    sys.path.insert(0,str(old/'src'))
    import numpy,scipy,qiskit
    import trotterlib.h4_pilot_science
    distributions=['numpy','scipy','qiskit','qiskit-nature','openfermion','openfermionpyscf','pyscf']
else:
    import flint,numpy
    s=importlib.util.spec_from_file_location('C_probe_engine',ROOT/'src/trotterlib/h4_ball_evidence.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    distributions=['python-flint','numpy']
paths={Path(sys.executable).resolve()}
for distribution in distributions:
    d=importlib.metadata.distribution(distribution)
    for file in d.files or []:
        path=Path(d.locate_file(file)).resolve()
        if path.is_file() and path.suffix!='.pyc':paths.add(path)
for module in list(sys.modules.values()):
    name=getattr(module,'__file__',None)
    if name:
        p=Path(name).resolve()
        if p.is_file() and p.suffix!='.pyc' and not p.is_relative_to(ROOT) and (not legacy or not p.is_relative_to(old)):paths.add(p)
for line in Path('/proc/self/maps').read_text().splitlines():
    name=line.split()[-1]
    if name.startswith('/') and Path(name).is_file():paths.add(Path(name).resolve())
record=dict(scope='Runtime imports and file SHA only; no scientific input read',legacy=legacy,python=sys.version,executable=str(Path(sys.executable).resolve()),versions={name:importlib.metadata.version(name) for name in distributions},files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)})
if not legacy:record['FLINT']=flint.__FLINT_VERSION__
print(json.dumps(record,sort_keys=True))
