#!/usr/bin/env python3
"""Execute only the separately approved four-case C4 packet."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', required=True, type=Path)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    from trottertracks.pr_dictionary_validation.rustiq_execution import verify_packet, execute
    verified = verify_packet(ROOT, args.packet, args.inputs, args.output)
    print(json.dumps(execute(verified), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
