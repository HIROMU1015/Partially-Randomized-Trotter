"""Publish tables/curves from an already generated technical ordering replay.

Reads no scientific archive. CSV/plots retain the replay's censor boundaries;
recorded service-time simulation is never described as measured elapsed time.
"""
import argparse
import csv
import gzip
import json
from pathlib import Path


def number(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return value['numerator'] / value['denominator']
    return float(value)


def csv_file(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def publish(data, output, plot=False):
    output.mkdir(parents=True, exist_ok=True)
    budget_rows, elapsed_rows, curve_rows = [], [], []
    summary = {k: v for k, v in data.items() if k not in ('results', 'budget_scan')}
    summary['results'] = {}
    keep = ('conditions', 'paired_observable_horizon', 'qualification')
    result_keep = ('stop', 'censor', 'seconds', 'initial_best', 'best_adopted', 'best_all_completed',
                   'Q3_launched', 'unknown_Q3_requests', 'completed', 'revealed_records', 'adopted',
                   'completed_not_adopted', 'target_reach', 'timing_qualification', 'physical_scientific_calls')
    for name, case in data['results'].items():
        result = {k: case[k] for k in keep}
        result['policies'] = {p: {k: r[k] for k in result_keep} for p, r in case['policies'].items()}
        divergence = case['first_divergence']
        result['first_divergence'] = None if divergence is None else dict(
            index=divergence['index'], same_pool=divergence['same_pool'],
            **{p: {k: divergence[p][k] for k in ('seconds', 'selected', 'kind', 'selection')} for p in ('A', 'B')})
        summary['results'][name] = result
        for row in case['equal_elapsed']:
            for policy, point in row['policies'].items():
                elapsed_rows.append(dict(scenario=name, simulated_seconds=row['seconds'], policy=policy,
                    paired_observable=row['paired_observable'], status=point['status'],
                    best_all_RZ=number(point.get('best_all')), Q3_launched=point.get('launched')))
        for policy, r in case['policies'].items():
            for point in r['curve']:
                curve_rows.append(dict(scenario=name, policy=policy, simulated_seconds=point['seconds'],
                    Q3_launched=point['launched'], best_all_RZ=number(point['best_all']),
                    adopted=point['adopted'], job=point['job'], censor_seconds=r['seconds'] if r['censor'] else None))
    for cap, cases in data.get('budget_scan', {}).items():
        for name, case in cases.items():
            for policy, r in case['policies'].items():
                budget_rows.append(dict(scenario=name, Q3_budget=int(cap), policy=policy,
                    stop=r['stop'], censored=r['censor'] is not None, simulated_seconds=r['seconds'],
                    Q3_launched=r['Q3_launched'], completed=r['completed'], adopted=r['adopted'],
                    best_all_RZ=number(r['best_all_completed'])))
    summary['budget_scan'] = budget_rows
    (output / 'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    csv_file(output / 'BUDGETS.csv', budget_rows)
    csv_file(output / 'EQUAL_ELAPSED.csv', elapsed_rows)
    csv_file(output / 'CURVES.csv', curve_rows)
    with (output / 'COMPARISON.json.gz').open('wb') as stream:
        with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as zipped:
            zipped.write((json.dumps(data, ensure_ascii=False, separators=(',', ':'))+'\n').encode())
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        case = data['results']['LD5_KAPPA_LE64_NO_SPECULATION']
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
        for policy, r in case['policies'].items():
            label = 'A: round05 rule' if policy == 'ROUND05' else 'B: improvement / time'
            rows = r['curve']
            axes[0].step([x['seconds'] / 60 for x in rows], [number(x['best_all']) / 1e6 for x in rows], where='post', label=label)
        axes[0].axvline(case['paired_observable_horizon'] / 60, color='black', linestyle=':', label='Paired evidence ends')
        axes[0].set(xlabel='Simulated elapsed time (minutes)', ylabel='Best feasible cost (million RZ)')
        axes[0].legend(fontsize=8)
        for policy, label in [('ROUND05', 'A'), ('IMPROVEMENT_PER_TIME_V1', 'B')]:
            rows = [r for r in budget_rows if r['policy'] == policy]
            axes[1].plot([r['Q3_budget'] for r in rows], [r['best_all_RZ'] / 1e6 for r in rows], marker='o', label=label)
        axes[1].set(xlabel='Equal Q3 launch budget', ylabel='Best feasible cost (million RZ)')
        axes[1].legend()
        for ax in axes:
            ax.grid(alpha=.25)
        fig.suptitle('Saved-log replay: PR-Pauli, requested width h, LD5, kappa <= 64, no speculation')
        fig.savefig(output / 'CURVES.svg')
        fig.savefig(output / 'CURVES.png', dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    opener = gzip.open if args.comparison.suffix == '.gz' else open
    with opener(args.comparison, 'rt') as stream:
        data = json.load(stream)
    publish(data, args.output, args.plot)


if __name__ == '__main__':
    main()
