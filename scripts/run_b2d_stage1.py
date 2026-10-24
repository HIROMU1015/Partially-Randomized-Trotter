#!/usr/bin/env python3
"""Metadata inspection or separately authorized B2-D Stage-1 execution."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect",action="store_true")
    mode.add_argument("--execute",action="store_true")
    parser.add_argument("--packet")
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/"src"))
    from trottertracks.b2d_evaluator.scope import candidate_graph, registry
    from trottertracks.b2d_evaluator.launcher import launch
    if args.inspect:
        head = subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
        print(json.dumps(dict(status="PROPOSAL_NOT_EXECUTION_APPROVAL",source_commit=head,
            production_authority="UNCONNECTED",scope=registry(candidate_graph(head))),indent=2))
        return 0
    if not args.packet or not args.output:
        parser.error("--execute requires --packet and --output")
    result = launch(args.packet,root,args.output)
    print(json.dumps(result,indent=2))
    return 0 if result["status"] == "COMPLETED_DESCRIPTIVE_MANDATORY_STOP" else 2


if __name__ == "__main__":
    raise SystemExit(main())
