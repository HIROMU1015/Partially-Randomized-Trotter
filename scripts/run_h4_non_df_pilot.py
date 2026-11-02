#!/usr/bin/env python3
"""H4 execution entrypoint. Defaults remain inactive until explicit batch seal."""
import argparse
import json
from pathlib import Path
import sys

from trotterlib.h4_execution_contract import completed_contract,activate,checked_pack


def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--contract',action='store_true')
    g.add_argument('--activate',action='store_true')
    g.add_argument('--execute',action='store_true')
    g.add_argument('--synthetic',action='store_true')
    g.add_argument('--worker')
    p.add_argument('--pack',type=Path)
    p.add_argument('--contract-file',type=Path)
    p.add_argument('--approval-file',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--release',type=Path)
    p.add_argument('--artificial-worker',action='store_true')
    a=p.parse_args()
    if a.contract:
        print(json.dumps(completed_contract(),ensure_ascii=False,indent=2));return
    if a.pack is None:p.error('--pack required')
    if a.activate:
        if a.contract_file is None or a.approval_file is None:p.error('explicit contract and approval required')
        print(json.dumps(activate(a.contract_file,a.approval_file,a.pack)));return
    if a.execute:checked_pack(a.pack)  # Reject before science/dispatcher import.
    from trotterlib.h4_pilot_execution import run_pipeline,worker
    if a.worker:
        if a.output is None or a.release is None:p.error('worker output/release required')
        worker(a.pack,a.worker,a.output,a.release,artificial=a.artificial_worker)
    else:
        print(json.dumps(run_pipeline(a.pack,artificial=a.synthetic)))


if __name__=='__main__':main()
