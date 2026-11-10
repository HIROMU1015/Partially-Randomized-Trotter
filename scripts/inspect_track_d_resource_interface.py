#!/usr/bin/env python3
"""Inspect saved Track D method results; no numerical evaluation is invoked."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks.track_d_resource_interface import SavedCDF8Study, METHODS, SCENARIOS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--scenario', choices=SCENARIOS, default='A')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    study = SavedCDF8Study(ROOT)
    result = study.select(study.inputs, args.method, scenario=args.scenario)
    # dataclasses contain saved records and are JSON metadata, not circuits.
    result['prices'] = asdict(result['prices'])
    result['read_only_inspection'] = True
    result['new_H4_shot_calculations'] = 0
    result['new_H4_total_aggregations'] = 0
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as f:
            f.write(payload)
        print('Saved-reference inspection written; no N solver or resource aggregation')
    else:
        print(payload, end='')


if __name__ == '__main__':
    main()
