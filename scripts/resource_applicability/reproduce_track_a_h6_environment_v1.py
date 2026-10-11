#!/usr/bin/env python3
"""Reproduce the dedicated rc1 environment in a new home directory only.

No shared package installation, scientific runner, or execution grant.
Input archives and every wheel are SHA pinned in the prepared bundle.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import urllib.request


def download(url,path,expected):
    with urllib.request.urlopen(url,timeout=60) as source,path.open('xb') as target:
        while data:=source.read(1024*1024):target.write(data)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('DOWNLOAD_SHA:'+path.name)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
    p.add_argument('--destination',type=Path,required=True);p.add_argument('--build-inputs',type=Path,required=True)
    p.add_argument('--requirements',type=Path,required=True);a=p.parse_args()
    dest=a.destination
    if (not dest.is_absolute() or not dest.is_relative_to('/home/AbeHiromu') or '..' in dest.parts
        or any(x.is_symlink() for x in (dest,*dest.parents))):raise ValueError('EXCLUSIVE_HOME_DESTINATION')
    if not a.execute:print('PREPARATION_RECIPE_ONLY; no environment created');return 0
    dest.mkdir(parents=True,exist_ok=False);inputs=json.loads(a.build_inputs.read_text())
    for name in ('tmp','downloads','build-dependencies','logs'): (dest/name).mkdir()
    env=dict(os.environ,TMPDIR=str(dest/'tmp'),TEMP=str(dest/'tmp'),TMP=str(dest/'tmp'),
        XDG_CACHE_HOME=str(dest/'cache'),PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
        MAKEFLAGS='-j1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',RAYON_NUM_THREADS='1')
    def run(command,cwd):
        with (dest/'logs'/'reproduction.log').open('a') as log:
            subprocess.run(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
    for item in inputs['debs']:
        path=dest/'downloads'/item['file'];download(item['url'],path,item['sha256'])
        run(['dpkg-deb','--extract',str(path),str(dest/'build-dependencies')],dest)
    archive=dest/'downloads'/'Python-3.11.0rc1.tar.xz'
    download(inputs['python_url'],archive,inputs['python_sha256'])
    # Python 3.12 orchestrator uses the data filter; the rc1 interpreter is the output.
    with tarfile.open(archive) as t:t.extractall(dest,filter='data')
    dep=dest/'build-dependencies';lib=dep/'usr/lib/x86_64-linux-gnu';prefix=dest/'python-3.11.0rc1'
    env.update(CPPFLAGS=f'-I{dep}/usr/include -I{dep}/usr/include/x86_64-linux-gnu',
        LDFLAGS=f'-L{lib} -Wl,-rpath,{lib}',PKG_CONFIG_LIBDIR=str(lib/'pkgconfig'),LD_LIBRARY_PATH=str(lib))
    src=dest/'Python-3.11.0rc1'
    run(['nice','-n','10','./configure','--prefix='+str(prefix),'--with-ensurepip=install',
        '--with-openssl='+str(dep/'usr'),'--with-openssl-rpath='+str(lib)],src)
    run(['nice','-n','10','make','-j1'],src);run(['nice','-n','10','make','install'],src)
    py=prefix/'bin/python3.11'
    run([str(py),'-I','-P','-B','-m','venv',str(dest/'venv')],dest)
    py=dest/'venv/bin/python'
    run([str(py),'-I','-P','-B','-m','pip','--isolated','install','--cache-dir',str(dest/'pip-cache'),
         '--index-url','https://pypi.org/simple','--only-binary=:all:','--require-hashes','-r',str(a.requirements.resolve())],dest)
    run([str(py),'-I','-P','-B','-m','pip','check'],dest)
    print('DEDICATED_ENVIRONMENT_REPRODUCED; a new profile/seal is required before any science')
    return 0


if __name__=='__main__':raise SystemExit(main())
