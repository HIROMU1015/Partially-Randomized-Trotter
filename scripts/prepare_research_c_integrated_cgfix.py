#!/usr/bin/env python3
"""Prepare/check new saved17+new1 scope. --execute requires explicit permission."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode=True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--prepare',action='store_true');p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if a.prepare and a.execute:raise ValueError('PREPARE_AND_REVIEW_NEW_SOURCE_BEFORE_EXECUTION')
    root=a.root.resolve();sys.path.insert(0,str(root/'src'))
    from trottertracks import c_integrated_cgfix as job
    result=job.prepare(root) if a.prepare else job.preflight(root)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
    if a.prepare:job.server.write(root/job.TECHNICAL/'preflight.json',result)
    if a.execute:
        os.environ.update(job.execution.environment(root,'cgfix_parent'))
        job.Run(root,root/job.SOURCE).execute()


if __name__=='__main__':main()
