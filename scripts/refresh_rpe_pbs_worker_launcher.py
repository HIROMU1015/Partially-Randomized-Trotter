#!/usr/bin/env python3
"""Refresh operational worker modules in a PBS launcher without losing its pilot logic.

Use together with the same release's fixed preparation/transport checkout.
The input launcher remains unchanged. No PBS operation is performed here.
"""
import argparse
import hashlib
from pathlib import Path
import os
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODULES = {
    'trottertracks/_collection_pbs_worker.py': 'src/trottertracks/_pipeline_pbs_worker.py',
    'trottertracks/_collection_q3_worker.py': 'src/trottertracks/_pipeline_pbs_q3_worker.py',
    'trottertracks/_pipeline_preparation_artifacts.py': 'src/trottertracks/rpe_preparation_artifacts.py',
    'trottertracks/_pbs_storage_report.py': 'src/trottertracks/rpe_worker_storage.py',
    'trottertracks/rpe_pbs_offline.py': 'src/trottertracks/rpe_pbs_offline.py',
}


def refresh(base, output, root=ROOT):
    base, output, root = map(Path, (base, output, root))
    if base.resolve() == output.resolve() or output.exists():
        raise ValueError('Use a new launcher filename; never replace the active one')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        with base.open('rb') as raw:
            prefix = raw.readline(4096)
        if prefix.startswith(b'#!'):
            temporary.write_bytes(prefix)
        with zipfile.ZipFile(base) as old, zipfile.ZipFile(temporary, 'a') as new:
            missing = (set(MODULES) - {'trottertracks/rpe_pbs_offline.py'}) - set(old.namelist())
            if missing:
                raise ValueError('Base launcher does not contain the required worker modules: '+str(sorted(missing)))
            for entry in old.infolist():
                data = ((root/MODULES[entry.filename]).read_bytes()
                        if entry.filename in MODULES else old.read(entry))
                if entry.filename.endswith('.py'):
                    compile(data,entry.filename,'exec')
                new.writestr(entry, data)
            if 'trottertracks/rpe_pbs_offline.py' not in old.namelist():
                new.write(root/MODULES['trottertracks/rpe_pbs_offline.py'],'trottertracks/rpe_pbs_offline.py')
        temporary.chmod(base.stat().st_mode & 0o777)
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    with output.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument('base', type=Path)
    args.add_argument('output', type=Path)
    ns = args.parse_args()
    print(refresh(ns.base, ns.output))


if __name__ == '__main__':
    main()
