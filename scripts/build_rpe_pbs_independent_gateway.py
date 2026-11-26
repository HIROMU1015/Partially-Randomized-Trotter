#!/usr/bin/env python3
"""Build a new PBS gateway zipapp for an independent-search release; no submission."""
import argparse
import hashlib
from pathlib import Path
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def build(output):
    output=Path(output)
    if output.exists():
        raise ValueError('Use a new gateway filename')
    modules={'__main__.py':ROOT/'scripts/rpe_pbs_pilot.py',
             'rpe_pbs_gateway.py':ROOT/'scripts/rpe_pbs_gateway.py',
             'rpe_dry_check_recovery.py':ROOT/'src/trottertracks/rpe_dry_check_recovery.py'}
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent,delete=False) as stream:
        temporary=Path(stream.name)
    try:
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            for name,source in modules.items():
                data=source.read_bytes()
                compile(data,name,'exec')
                archive.writestr(name,data)
        import os
        os.link(temporary,output)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(output.read_bytes()).hexdigest()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    print(build(parser.parse_args().output))
