"""Reconcile saved C4 costs/timings using stdlib only; never build or simulate."""
from pathlib import Path
import argparse
import json


def summarize(root):
    root = Path(root)
    rows = json.loads((root/'metrics.json').read_bytes())
    if len(rows) != 24:
        raise ValueError('Expected four cases, three routes and both axes')
    keys = ('rz_count', 'cx_count', 'rz_depth', 'cx_depth', 'total_depth',
            'quantum_only_depth', 'circuit_size', 'measurement_count')
    indexed = {(r['case'], r['route'], r['axis']): r for r in rows}
    if len(indexed) != 24:
        raise ValueError('Duplicate resource records')
    for case in ('SS_motif_2', 'SS_motif_3', 'SS_motif_4', 'PP_motif_0'):
        for axis in ('X', 'Y'):
            if any(indexed[case, 'B0', axis][k] != indexed[case, 'B1', axis][k] for k in keys):
                raise ValueError('B0/B1 partition control differs')
    totals = {route: {k: sum(r[k] for r in rows if r['route'] == route) for k in keys}
              for route in ('B0', 'B1', 'R1')}
    changes = {k: dict(B0=totals['B0'][k], B1=totals['B1'][k], R1=totals['R1'][k],
                      delta=totals['R1'][k]-totals['B0'][k],
                      relative_change=totals['R1'][k]/totals['B0'][k]-1 if totals['B0'][k] else None)
               for k in keys}
    timing = {}
    for path in sorted((root/'journals').glob('*.jsonl')):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['status'] != 'COMPLETED':
                continue
            route = ('B0_B1_literal' if '_B0_B1_literal' in row['id'] else
                     next((r for r in ('B0', 'B1', 'R1') if '_'+r+'_' in row['id']+'_'), 'shared'))
            key = route+'/'+row['group']
            value = timing.setdefault(key, dict(calls=0, wall_seconds_sum=0, cpu_seconds_sum=0))
            value['calls'] += 1
            value['wall_seconds_sum'] += row['wall_seconds']
            value['cpu_seconds_sum'] += row['cpu_seconds']
    for path in sorted((root/'reports').glob('*_build.json')):
        route = path.stem.rsplit('_', 2)[-2]
        for record in json.loads(path.read_bytes())['trace']:
            key = route+'/'+record['kind']
            value = timing.setdefault(key, dict(calls=0, wall_seconds_sum=0, cpu_seconds_sum=0))
            value['calls'] += 1
            value['wall_seconds_sum'] += record['seconds']
            value['cpu_seconds_sum'] += record['cpu_seconds']
    return dict(status='SAVED_COST_RECONCILIATION_PASS', B0_B1_all_metrics_equal=True,
                scope='Unweighted sum of eight designed case/axis wrappers, not a law expectation',
                changes=changes, classical_timings_by_route=timing,
                timing_scope='Sums of per-task intervals; nested adapter/synthesis are subsets of builder time, not additive. Process import/startup is outside kernel intervals.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    report = summarize(args.root)
    if args.check:
        saved = json.loads((args.root/'comparison.json').read_bytes())
        if saved['changes'] != report['changes'] or saved['B0_B1_all_metrics_equal'] is not True:
            raise ValueError('Saved cost comparison does not reconcile')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
