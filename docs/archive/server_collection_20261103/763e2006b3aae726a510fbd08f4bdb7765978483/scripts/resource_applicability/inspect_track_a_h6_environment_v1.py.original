#!/usr/bin/env python3
"""Freeze dedicated interpreter/dependency bytes and compiler metadata only."""
import argparse
import ast
import base64
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import platform
import importlib.util
import py_compile
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.h6_matched_contract_v1 import environment,file_hash
from trottertracks.resource_applicability.h6_matched_reuse_v3 import COMPILER_HASH
from trottertracks.resource_applicability.ax2a_preparation import digest
from trottertracks.resource_applicability.h6_server_native_dependencies_v2 import native_dependencies


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--build-dependencies',type=Path,required=True);a=p.parse_args()
    if not Path(sys.prefix).is_relative_to('/home/AbeHiromu/projects/h6-environments'):
        raise ValueError('H6_DEDICATED_ENVIRONMENT_REQUIRED')
    native=native_dependencies()
    files={};dependencies=[];regenerated=[]
    def capture(path):
        path=Path(path).resolve()
        if not path.is_relative_to('/home/AbeHiromu'):raise ValueError('DEPENDENCY_OUTSIDE_PRIVATE_ENVIRONMENT')
        if path.is_file():files[str(path)]=file_hash(path)
    capture(sys.executable)
    for d in sorted(md.distributions(),key=lambda x:x.metadata['Name'].lower()):
        record=d.read_text('RECORD')
        if record is None:raise ValueError('DEPENDENCY_RECORD_MISSING')
        checked=0
        for f in d.files or []:
            path=Path(d.locate_file(f));capture(path)
            if f.hash:
                if f.hash.mode!='sha256':raise ValueError('DEPENDENCY_HASH_ALGORITHM')
                expected=base64.urlsafe_b64decode(f.hash.value+'='*(-len(f.hash.value)%4)).hex()
                actual_sha=file_hash(path)
                if actual_sha!=expected:
                    # pip regenerates bytecode under the selected interpreter;
                    # a wheel's bytecode RECORD hash can describe its build host.
                    # Prove regeneration and bind the actual bytes; never exempt source/native files.
                    if path.suffix!='.pyc':raise ValueError('DEPENDENCY_RECORD_BYTES:'+str(path))
                    source=importlib.util.source_from_cache(str(path))
                    with tempfile.TemporaryDirectory(dir=__import__('os').environ['TMPDIR']) as temp:
                        target=Path(temp)/'regenerated.pyc'
                        optimization=int(path.name.split('.opt-')[1].split('.')[0]) if '.opt-' in path.name else 0
                        py_compile.compile(source,cfile=str(target),dfile=source,doraise=True,optimize=optimization,
                            invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
                        if file_hash(target)!=actual_sha:raise ValueError('UNEXPLAINED_BYTECODE:'+str(path))
                    regenerated.append(dict(path=str(path),wheel_RECORD_sha256=expected,actual_sha256=actual_sha,
                        source_sha256=file_hash(Path(source)),independent_regeneration_verified=True))
                checked+=1
        dependencies.append(dict(name=d.metadata['Name'],version=d.version,
            RECORD_sha256=hashlib.sha256(record.encode()).hexdigest(),hashed_records_verified=checked))
    for p in (Path(sys.base_prefix)/'lib').rglob('*'):
        if p.is_file():capture(p)
    for p in (a.build_dependencies/'usr/lib/x86_64-linux-gnu').rglob('*'):
        if p.is_file():capture(p)
    import qiskit
    from trotterlib.rte import CompilerSettings
    from trotterlib.rte_compiled_cost import compiler_settings_hash
    settings=CompilerSettings(('rz','sx','x','cx'),None,None,1,None,None,17,qiskit.__version__)
    if compiler_settings_hash(settings)!=COMPILER_HASH:raise ValueError('COMPILER_HASH_CHANGED')
    entry=Path(md.distribution('qiskit').locate_file('qiskit/compiler/transpiler.py'))
    tree=ast.parse(entry.read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='transpile')
    defaults=[None]*(len(function.args.args)-len(function.args.defaults))+function.args.defaults
    actual={arg.arg:('<required>' if d is None else ast.unparse(d)) for arg,d in zip(function.args.args,defaults)}
    eps=md.entry_points()
    entries=[x for group in eps.keys() for x in eps[group]] if hasattr(eps,'keys') else list(eps)
    plugins=sorted([dict(group=x.group,name=x.name,value=x.value) for x in entries
        if x.group.startswith('qiskit.')],key=lambda x:(x['group'],x['name'],x['value']))
    profile=dict(schema='h6_server_environment_compiler_v1',python=sys.executable,venv=sys.prefix,
        python_full_version=sys.version,platform=platform.platform(),environment=environment(),
        native_lowering_dependencies=native,
        dependencies=dependencies,critical_file_hashes=[dict(path=p,sha256=sha) for p,sha in sorted(files.items())],
        install_generated_bytecode_differences=regenerated,
        compiler=dict(settings_hash=COMPILER_HASH,qiskit_version=qiskit.__version__,
            basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=17,
            backend=None,coupling_map=None,layout_method=None,routing_method=None,num_processes=1,
            transpiler_entry_sha256=file_hash(entry),inherited_defaults=actual,plugins=plugins,
            process_limit_source='QISKIT_NUM_PROCS=1; QISKIT_PARALLEL=FALSE; frozen transpile call unchanged'),
        old_native_binary_inventory_available=False,old_native_binary_bit_identity_verified=False,
        real_H6_output_equivalence_verified=False,science_run_performed=False)
    profile['fingerprint']=digest(profile)
    with a.output.open('x') as f:f.write(json.dumps(profile,sort_keys=True,indent=2)+'\n')
    print('FROZEN_PRIVATE_ENVIRONMENT',len(dependencies),'distributions',len(files),'files',profile['fingerprint'])


if __name__=='__main__':raise SystemExit(main())
