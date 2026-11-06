#!/usr/bin/env python3
"""Prepare/passively verify the server SOURCE; --execute needs user permission."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--execute',action='store_true')
    a = p.parse_args()
    if a.prepare and a.execute: raise ValueError('PREPARE_THEN_REVIEW_SOURCE_BEFORE_EXECUTION')
    root = a.root.resolve();sys.path.insert(0,str(root/'src'))
    from trottertracks import c_integrated_server as server
    result = server.prepare(root) if a.prepare else server.preflight(root)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
    if a.prepare:server.write(root/server.TECHNICAL/'preflight.json',result)
    if a.execute:
        from trottertracks.c_integrated_execution import Run, environment
        os.environ.update(environment(root,'server_parent'))
        Run(root,root/server.SOURCE,contract_path=server.CONTRACT,family=server.FAMILY).execute()


if __name__ == '__main__':main()
