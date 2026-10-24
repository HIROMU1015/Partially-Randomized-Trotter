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
    parser.add_argument('--packet', type=Path)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--scope', choices=('C4_Q13_LIMITED_V1','C4_B2_SAVED_COMPARISON_V1'),
                        default='C4_Q13_LIMITED_V1')
    parser.add_argument('--preflight-only', action='store_true',
                        help='B2 technical bytes/schema/environment checks; no science or output')
    args = parser.parse_args()
    from trottertracks.pr_dictionary_validation.rustiq_execution import (
        verify_packet, execute, verify_b2_preflight, B2_SCOPE)
    if args.preflight_only:
        if args.scope != B2_SCOPE or args.packet is not None or args.output is not None:
            parser.error('--preflight-only requires B2 scope and no packet/output')
        verified = verify_b2_preflight(ROOT, args.inputs)
        print(json.dumps(dict(status='B2_TECHNICAL_PREFLIGHT_PASS_NO_SCIENCE',
            source_identity=verified['source_identity'], input_identity=verified['input_identity'],
            scientific_execution_authorized=False), indent=2))
        return
    if args.packet is None or args.output is None:
        parser.error('--packet and --output are required for an approved execution')
    verified = verify_packet(ROOT, args.packet, args.inputs, args.output, scope=args.scope)
    print(json.dumps(execute(verified), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
