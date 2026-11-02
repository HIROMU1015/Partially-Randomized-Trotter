"""B-ENV identity collection/verification only; never executes B or issues grants.

Run using the dedicated Python with -I -B. NumPy import is the sole third-party
smoke check. file_identity is reused verbatim from the prior P1* management
helper; provenance is recorded in EXISTING_ENVIRONMENT_AND_CODE_REUSE.md.
"""
import argparse
import base64
import concurrent.futures
import csv
import datetime
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import queue
import re
import site
import subprocess
import sys
import sysconfig
import zipfile


def file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        while chunk := stream.read(1024**2):
            size += len(chunk)
            digest.update(chunk)
    return {'path':str(path),'bytes':size,'sha256':digest.hexdigest()}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def sha(value):
    return hashlib.sha256(value).hexdigest()


def save_new(path, value):
    with Path(path).open('xb') as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())


def guard_imports():
    banned = {'trottertracks', 'trotterlib', 'qiskit', 'scipy', 'mpmath',
              'openfermion', 'pyscf', 'pytest', 'sympy', 'z3'}
    if any(name.split('.')[0] in banned for name in sys.modules):
        raise RuntimeError('research/extra package already imported')
    def audit(event, args):
        if event == 'import' and args[0].split('.')[0] in banned:
            raise RuntimeError('B-ENV import boundary')
    sys.addaudithook(audit)


def isolation():
    prefix = Path(sys.prefix).resolve()
    base = Path(sys.base_prefix).resolve()
    if prefix == base or not sys.flags.isolated or not sys.dont_write_bytecode:
        raise RuntimeError('dedicated venv, -I and -B required')
    if platform.python_version() != '3.11.9':
        raise RuntimeError('Python patch mismatch')
    cfg = (prefix/'pyvenv.cfg').read_text()
    if 'include-system-site-packages=false' not in cfg.lower().replace(' ', ''):
        raise RuntimeError('system site enabled')
    if site.ENABLE_USER_SITE is not False or not sys.flags.ignore_environment:
        raise RuntimeError('user site/environment isolation failed')
    roots = (prefix, base)
    for value in sys.path:
        p = Path(value).resolve()
        if not value or not any(p.is_relative_to(root) for root in roots):
            raise RuntimeError('external sys.path')
    site_root = prefix/'lib/python3.11/site-packages'
    if list(site_root.glob('*.pth')) or (site_root/'sitecustomize.py').exists():
        raise RuntimeError('site injection file present')
    packages = sorted((d.metadata['Name'].lower(), d.version)
                      for d in importlib.metadata.distributions())
    if packages != [('numpy', '1.26.4')]:
        raise RuntimeError('runtime package closure mismatch')
    for root in (prefix, base):
        if (root/'lib/python3.11/sitecustomize.py').exists():
            raise RuntimeError('base site injection')
    return {'prefix':str(prefix), 'base_prefix':str(base), 'sys_path':sys.path,
            'isolated':True, 'ignore_environment':True, 'no_user_site':True,
            'dont_write_bytecode':True, 'system_site_packages':False,
            'packages':[{'name':n, 'version':v} for n,v in packages],
            'pyvenv_cfg':file_identity(prefix/'pyvenv.cfg'),
            'environment_variable_names_present':sorted(k for k in os.environ
                if k.startswith(('PYTHON', 'LD_', 'OPENBLAS', 'OMP_', 'MKL_', 'BLIS_')))}


def tree_records(root):
    root = Path(root).resolve()
    paths = sorted(p for p in root.rglob('*')
                   if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    workers = len(os.sched_getaffinity(0))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(file_identity, paths))
    return {str(p.relative_to(root)):{'bytes':r['bytes'], 'sha256':r['sha256']}
            for p,r in zip(paths, rows)}


def numpy_wheel_and_record(wheel):
    dist = importlib.metadata.distribution('numpy')
    prefix = Path(sys.prefix).resolve()
    record = Path(dist._path)/'RECORD'
    rows = list(csv.reader(io.StringIO(record.read_text())))
    verified = 0
    for name, encoded, size in rows:
        path = Path(dist.locate_file(name)).resolve()
        if not path.is_relative_to(prefix) or not path.is_file():
            raise RuntimeError('RECORD file missing or escapes venv')
        if not encoded:
            if path != record: raise RuntimeError('unexpected unhashed RECORD file')
            continue
        algorithm, digest = encoded.split('=', 1)
        if algorithm != 'sha256': raise RuntimeError('unexpected RECORD hash')
        raw = path.read_bytes()
        if base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b'=').decode() != digest:
            raise RuntimeError('installed RECORD hash mismatch')
        if len(raw) != int(size): raise RuntimeError('RECORD size mismatch')
        verified += 1
    ident = file_identity(wheel)
    expected = '666dbfb6ec68962c033a450943ded891bed2d54e6755e35e5835d63f4f6931d5'
    if ident['sha256'] != expected: raise RuntimeError('wheel identity mismatch')
    wheel_rows = 0
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.endswith('/') or name.endswith('.dist-info/RECORD'):continue
            # pip generates the f2py entry point; it is covered by installed RECORD.
            if '.data/' in name: raise RuntimeError('unhandled wheel data relocation')
            actual = Path(dist.locate_file(name))
            if actual.read_bytes() != archive.read(name):
                raise RuntimeError('wheel/installed bytes differ')
            wheel_rows += 1
        metadata = archive.read('numpy-1.26.4.dist-info/METADATA').decode()
        if any(line.startswith('Requires-Dist:') for line in metadata.splitlines()):
            raise RuntimeError('unexpected NumPy transitive package dependency')
    return {'version':dist.version, 'wheel':ident,
            'METADATA':file_identity(Path(dist._path)/'METADATA'),
            'WHEEL':file_identity(Path(dist._path)/'WHEEL'),
            'RECORD':file_identity(record), 'record_files_verified':verified,
            'wheel_files_compared':wheel_rows,
            'record_unhashed_files':['numpy-1.26.4.dist-info/RECORD']}


def smoke_and_native():
    # Stdlib imports mirror B's declared module requirements, without B SOURCE.
    import dataclasses, decimal, fractions, itertools, math, operator, typing, warnings
    import numpy
    if numpy.__version__ != '1.26.4':raise RuntimeError('NumPy import version')
    if not Path(numpy.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError('NumPy imported outside dedicated venv')
    mapped = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith('/'):continue
        p = Path(fields[5])
        if not p.is_file():raise RuntimeError('mapped file missing')
        with p.open('rb') as stream:
            if stream.read(4) == b'\x7fELF':mapped.add(p.resolve())
    # Resolve DT_NEEDED against actual process mappings and SONAMEs. Calling ldd
    # on a transitive wheel library alone loses its parent's RPATH context.
    def elf_info(p):
        result = subprocess.run(['/usr/bin/readelf','-d',str(p)],
                                capture_output=True,text=True,check=True)
        return {'object':str(p), 'needed':re.findall(r'\(NEEDED\).*?\[(.*?)\]',result.stdout),
                'soname':re.findall(r'\(SONAME\).*?\[(.*?)\]',result.stdout),
                'dynamic_table':result.stdout}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
        resolutions = list(pool.map(elf_info, sorted(mapped)))
    index = {}
    for row in resolutions:
        for name in [Path(row['object']).name]+row['soname']:
            index.setdefault(name,set()).add(row['object'])
    for row in resolutions:
        row['resolved'] = {}
        for name in row['needed']:
            choices = index.get(name,set())
            if len(choices) != 1:raise RuntimeError('DT_NEEDED not uniquely mapped: '+name)
            row['resolved'][name] = next(iter(choices))
    identities = [file_identity(p) for p in sorted(mapped)]
    openblas = [r for r in identities if 'openblas' in Path(r['path']).name]
    if len(openblas) != 1:raise RuntimeError('one mapped OpenBLAS required')
    config = numpy.__config__.CONFIG
    return {'numpy_import':'PASS_VERSION_AND_ABI_IMPORT_ONLY',
            'numpy_module_file':str(Path(numpy.__file__).resolve()),
            'numpy_extension_module_files':sorted(str(Path(m.__file__).resolve())
                for name,m in tuple(sys.modules.items()) if name.startswith('numpy')
                and getattr(m,'__file__',None) and str(m.__file__).endswith('.so')),
            'numpy_configuration':config, 'mapped_elf_files':[str(p) for p in sorted(mapped)],
            'native_identities':identities, 'loader_resolutions':resolutions,
            'scientific_source_imports':0, 'user_research_numeric_calls':0}


def source_binding(root):
    root = Path(root).resolve()
    base = '3de42d3a3efe9ebc74dde8a1d0e1f1bdbadd607c'
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    if head != base:raise RuntimeError('SOURCE root must be frozen B-PRE commit')
    folder = root/'docs/research/pr_algorithm_improvement_phase2d_binding_preparation_20261023'
    closure = json.loads((folder/'SOURCE_CLOSURE.json').read_bytes())
    files = {}
    for row in closure['source_records']:
        if file_identity(root/row['path'])['sha256'] != row['sha256']:
            raise RuntimeError('SOURCE bytes mismatch')
        files[row['path']] = row['sha256']
    if len(files) != 10:raise RuntimeError('SOURCE closure must have ten files')
    docs = {str((folder/name).relative_to(root)):file_identity(folder/name)['sha256']
            for name in ('B_INPUT_SPECIFICATION.md','B_CALL_INVENTORY.json','B_SCHEMAS.json')}
    return {'source_commit':base, 'source_files':files,
            'source_closure_sha256':sha(canonical(files)), 'specification_files':docs,
            'source_closure_document_sha256':file_identity(folder/'SOURCE_CLOSURE.json')['sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['collect','verify'])
    parser.add_argument('--source-root',required=True)
    parser.add_argument('--wheel',required=True)
    parser.add_argument('--private-dir',required=True)
    args = parser.parse_args()
    guard_imports()
    out = Path(args.private_dir).resolve()
    if out.stat().st_uid != os.getuid() or out.stat().st_mode & 0o077:
        raise RuntimeError('owned mode-0700 private directory required')
    binding = source_binding(args.source_root)
    iso = isolation()
    prefix = Path(sys.prefix).resolve()
    base = Path(sys.base_prefix).resolve()
    if args.mode == 'verify':
        receipt = json.loads((out/'environment_receipt_private.json').read_bytes())
        for root,expected in ((prefix,receipt['venv_files']),(base,receipt['base_files'])):
            if tree_records(root) != expected:raise RuntimeError('runtime file tree changed')
        for row in receipt['smoke']['native_identities']:
            if file_identity(row['path']) != row:raise RuntimeError('native bytes changed')
        if iso != receipt['isolation'] or binding != receipt['source_binding']:
            raise RuntimeError('source/isolation binding changed')
        if file_identity(__file__)['sha256'] != receipt['management_helper']['sha256']:
            raise RuntimeError('management helper changed')
        package = numpy_wheel_and_record(args.wheel)
        if package != receipt['numpy']:raise RuntimeError('package receipt changed')
        smoke = smoke_and_native()
        if smoke != receipt['smoke']:raise RuntimeError('ABI/native smoke identity changed')
        result = {'status':'PASS_ENVIRONMENT_MANAGEMENT_ONLY',
                  'source_files':10, 'scientific_calls':0,
                  'private_receipt_sha256':sha((out/'environment_receipt_private.json').read_bytes())}
        save_new(out/'environment_verification_private.json',result)
        print(json.dumps(result))
        return
    package = numpy_wheel_and_record(args.wheel)
    smoke = smoke_and_native()
    base_files = tree_records(base)
    venv_files = tree_records(prefix)
    environment = {'status':'DEDICATED_FROZEN_ENVIRONMENT', 'python_patch':'3.11.9',
        'numpy':'1.26.4', 'dedicated_prefix_identity':sha(str(prefix).encode()),
        'wheel_sha256':{Path(args.wheel).name:package['wheel']['sha256']},
        'package_file_hashes':{p:r['sha256'] for p,r in venv_files.items()
                               if (prefix/p).resolve().is_relative_to(prefix)},
        'native_runtime_hashes':{**{str((base/p).resolve()):r['sha256']
            for p,r in base_files.items() if 'site-packages' not in Path(p).parts},
            **{r['path']:r['sha256'] for r in smoke['native_identities']}}}
    # The fixed seven-field B schema permits this record; no authorization is made.
    receipt = {'schema_version':'phase2d-b-env-private-v1',
        'created_at_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'python':{'version':platform.python_version(),'implementation':platform.python_implementation(),
            'version_string':sys.version,'executable':sys.executable,'binary':file_identity(sys.executable),
            'platform':platform.platform(),'machine':platform.machine(),
            'SOABI':sysconfig.get_config_var('SOABI'),'config_vars':sysconfig.get_config_vars()},
        'management_helper':file_identity(__file__), 'source_binding':binding,
        'isolation':iso,'numpy':package,'smoke':smoke,'base_files':base_files,'venv_files':venv_files,
        'environment_record_sha256':sha(canonical(environment)),
        'scientific_execution_authorized':False,'scientific_input_sha256':None,
        'hash_workers':len(os.sched_getaffinity(0)),
        'scope':'environment identity; no research SOURCE import or scientific calls'}
    save_new(out/'environment.json',environment)
    save_new(out/'environment_receipt_private.json',receipt)
    print(json.dumps({'status':'COLLECTED_ENVIRONMENT_IDENTITY_ONLY',
        'python':'3.11.9','numpy':'1.26.4','native_files':len(smoke['native_identities']),
        'base_files':len(base_files),'venv_files':len(venv_files),'scientific_calls':0}))


if __name__ == '__main__':
    main()
