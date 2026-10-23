#!/usr/bin/env python3
"""L one-shot runner; grant, launch and numerical worker are separate actions.

No default execution. This implementation-stage SOURCE ships only an inactive
contract, never an issued scientific grant or approval. Artificial tests call
the small numerical kernel directly with manually supplied fixtures.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("local_L_controller", ROOT / "src/trotterlib/h4_local_control.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check-contract", type=Path)
    group.add_argument("--issue-grant", type=Path)
    group.add_argument("--launch", type=Path)
    group.add_argument("--worker", type=Path)
    group.add_argument("--witness", type=Path)
    parser.add_argument("--seal", type=Path)
    parser.add_argument("--namespace", type=Path)
    parser.add_argument("--command-json")
    args = parser.parse_args()
    try:
        if args.check_contract:
            c.validate_contract(c.read(args.check_contract))
            print(json.dumps(dict(status="VALID_INACTIVE_OR_ACTIVE_CONTRACT_NOT_EXECUTION", science_calls=0)))
        elif args.issue_grant:
            c.require(args.seal is not None and args.namespace is not None, "SEAL_AND_NAMESPACE_REQUIRED")
            c.issue_grant(args.issue_grant, args.seal, args.namespace)
        elif args.launch:
            c.launch(args.launch)
        elif args.worker:
            c.worker(args.worker)
        else:
            command = json.loads(args.command_json)
            c.require(type(command) is list and command and all(type(x) is str for x in command), "WITNESS_COMMAND")
            return c.witness(args.witness, command)
    except (ValueError, KeyError, OSError) as error:
        print("L_STOP: " + str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
