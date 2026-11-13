#!/usr/bin/env python3
"""Write the Full PF / DF Full-H RTE SOURCE catalog and an artificial request example.

Metadata only: pinned Git text and the published light-study summary are read.
The example uses a synthetic RMSE policy (Full-H RTE without kappa); it is a schema
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
    packets = {'FULL_PF_2nd': full_pf('2nd'), 'FULL_PF_4th': full_pf('4th'), 'FULL_H_RTE_explicit_layout': full_h()}
    def show(packet):
        out = D.requests(packet)
        return dict(v2_status=packet['v2_status'], identity_map=packet['identity_map'], b_interface=packet['b_interface'],
                    resolved_pending=packet['resolved_pending'],
                    spec=packet['physical']['spec'], requests=dict(out, rounds=[r for r in out['rounds'] if r['m'] in (0, 3)]))
    return dict(status='ARTIFICIAL_SCHEMA_EXAMPLE_NOT_A_MENU_OR_ADOPTED_CONDITION',
        note='Synthetic RMSE policy, h=0.1, M=3, delta=0.1, K=4 and layout q16/R16 at m3 only; no kappa. '
             'Binding and Full-H preparation are the published H4 metadata.',
        shown_rounds=[0, 3], packets={k: show(v) for k, v in packets.items()})


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
