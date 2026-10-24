import hashlib, importlib.metadata, json, os, platform, sys
from pathlib import Path
root=Path.cwd();sys.path.insert(0,str(root/'src'))
from trottertracks import c1_stage1_execution as run
role=sys.argv[1]
record=run.runtime_identity(root,role)
record.update(platform=platform.platform(),machine=platform.machine(),role=role,
              scipy_source_files={},scientific_calls=0)
import scipy.linalg._matfuncs as matfuncs
record['scipy_source_files']['linalg/_matfuncs.py']=run.sha(matfuncs.__file__)
if role=='fit':
    import ffsim,jax
    source=run.read(root/run.ENV_DIR/'ffsim-python-source.json')
    mismatches=[name for name,want in source['files'].items()
                if run.sha(Path(ffsim.__file__).parent/name)!=want]
    assert not mismatches and jax.config.jax_enable_x64
    record.update(ffsim_commit=source['commit'],ffsim_python_files_verified=len(source['files']),
                  mismatches=mismatches,jax_x64=True,devices=list(map(str,jax.devices())),
                  compatibility_version=__import__('trottertracks.c1_ffsim_compat',fromlist=['VERSION']).VERSION)
run.save(root/run.ENV_DIR/f'{role}-runtime.json',record)
print(json.dumps(dict(role=role,identity='PASS',packages=len(record['packages']),scientific_calls=0)))
