#!/usr/bin/env python3
"""Scope display and built-in synthetic fixture only. Never opens science inputs."""
import argparse
import json
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'/'trotterlib'))
from h8_rare_tail_selection import conditional_counts, draw_orders


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--synthetic', action='store_true')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--config')
    p.add_argument('--grant')
    args = p.parse_args(argv)
    # Reject before reading ANY externally supplied path, even purported approval.
    if args.execute or args.config or args.grant:
        print('EXPLICIT_SCIENTIFIC_EXECUTION_APPROVAL_REQUIRED; '
              'PRODUCTION_DISPATCH_UNAVAILABLE_IN_READINESS_SOURCE', file=sys.stderr)
        return 2
    result = dict(authorization_status='UNAPPROVED', scientific_calls=0,
                  native_compiler_connected=False, mandatory_STOP=True)
    if args.synthetic:
        law = conditional_counts(3, (.5, .3, .2), 's4')
        orders = draw_orders(law, random.Random(17))
        result.update(input_kind='SYNTHETIC_BUILTIN', orders=orders,
                      conditional_support=len(law.counts))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
