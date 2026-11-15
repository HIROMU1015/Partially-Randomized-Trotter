#!/usr/bin/env python3
"""Passive B2 JSON boundary. Never invokes scientific producers."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks.pr_price_rules import PriceRules  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=ROOT)
    action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument('--input', type=Path)
    action.add_argument('--template', help='e.g. DF_LD7 or PAULI_LD5')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    try:
        rules = PriceRules.from_repo(args.root)
        if args.template:
            e = next(e for e in rules.entries if e['name'] == args.template)
            out = dict(configuration_source=e['configuration_source'],
                       implementation=e['implementation'], policy='MODEL')
        else:
            req = json.loads(args.input.read_text())
            if 'configuration' not in req:
                req['configuration'] = rules.reader.read(req['configuration_source'])
            out = rules.quote(req)
    except (KeyError, ValueError, StopIteration) as err:
        print(json.dumps(dict(status='INVALID_PRICE_INPUT', reason=str(err))), file=sys.stderr)
        return 2
    text = json.dumps(out, indent=2, ensure_ascii=False) + '\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
