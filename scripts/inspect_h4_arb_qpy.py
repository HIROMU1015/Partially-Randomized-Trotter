#!/usr/bin/env python3
"""Decode a hash-bound saved QPY; no evolution, compiler or grant interface."""
import argparse
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType('arb_bridge_private')
pkg.__path__ = [str(ROOT / 'src/trotterlib')]
sys.modules['arb_bridge_private'] = pkg
from arb_bridge_private import h4_arb_bridge as B


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qpy', type=Path, required=True)
    parser.add_argument('--axis', choices=('cosine', 'sine'), required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--fingerprint', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    program = B.inspect_qpy(args.qpy, axis=args.axis, expected_sha256=args.sha256, expected_fingerprint=args.fingerprint)
    # Exclusive creation prevents accidental replacement of saved evidence.
    with args.output.open('x') as stream:
        json.dump(program, stream, sort_keys=True, separators=(',', ':'), allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status='STRUCTURAL_DECODE_ONLY', instructions=len(program['instructions']), state_actions=0)))


if __name__ == '__main__': main()
