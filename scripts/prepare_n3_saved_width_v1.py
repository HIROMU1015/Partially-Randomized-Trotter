#!/usr/bin/env python3
"""Prepare or verify small saved-evidence views. Never evaluates new widths."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks.representation_exploration.n3_saved_evidence_v1 import extract, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--extract', action='store_true')
    mode.add_argument('--verify', action='store_true')
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    report = extract(ROOT, args.directory) if args.extract else verify(ROOT, args.directory)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
