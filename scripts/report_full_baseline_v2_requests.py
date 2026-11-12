#!/usr/bin/env python3
"""Write the Full PF / DF Full-H RTE SOURCE catalog and an artificial request example.

Metadata only: pinned Git text and the published light-study summary are read.
The example uses a synthetic RMSE policy and a synthetic kappa; it is a schema
illustration, not a menu, adopted condition or scientific request.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
from trottertracks import full_baseline_v2_requests as D  # noqa: E402


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def artificial_example():
    from check_full_baseline_v2_requests import full_pf, full_h
    configs = dict(FULL_PF=full_pf(), FULL_H_RTE=full_h())
    return dict(status='ARTIFICIAL_SCHEMA_EXAMPLE_NOT_A_MENU_OR_ADOPTED_CONDITION',
        note='Synthetic RMSE policy, kappa, h=0.1, M=3, delta=0.1 and layout q16/R16 at m3 only; '
             'binding and Full-H preparation are the published H4 metadata.',
        shown_rounds=[0, 3],
        configurations={k: dict(configuration_id=v['id'], spec=v['spec']) for k, v in configs.items()},
        requests={k: dict(D.requests(v), rounds=[r for r in D.requests(v)['rounds'] if r['m'] in (0, 3)])
                  for k, v in configs.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    catalog = D.catalog()
    catalog['saved_full_h_preparation'] = {s: D.saved_full_h_preparation(s) for s in ('H4', 'H6')}
    write(args.output_dir/'SOURCE_CATALOG.json', catalog)
    write(args.output_dir/'REQUEST_EXAMPLE_ARTIFICIAL.json', artificial_example())


if __name__ == '__main__':
    main()
