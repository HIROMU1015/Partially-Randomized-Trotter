"""Report only saved B2 evidence, using stdlib. Never load QPY/NPY or run science.

The earlier C4 summarizer supplied the saved-JSON/journal approach. Its across-
case cost sums are deliberately not used for this paired comparison contract.
"""
from pathlib import Path
import argparse
import csv
import json


CASES = ('SS_motif_2', 'SS_motif_3', 'SS_motif_4', 'PP_motif_0')
FIELDS = ('rz_count', 'cx_count', 'rz_depth', 'cx_depth', 'total_depth',
          'quantum_only_depth', 'circuit_size', 'measurement_count')


def read(root, name):
    return json.loads((root/name).read_bytes())


def summarize(root):
    root = Path(root)
    summary = read(root, 'summary.json')
    contract = read(root, 'preparation_contract.json')
    if (summary['status'] != 'C4_B2_CORRECTNESS_PASS_RESOURCES_RECORDED_MANDATORY_STOP'
            or read(root, 'STOP.json')['status'] != 'COMPLETED_MANDATORY_STOP'):
        raise ValueError('A complete passing B2 run is required')
    rows = read(root, 'paired_compiled_metrics.json')
    if len(rows) != 8 or {(r['case'], r['axis']) for r in rows} != {
            (c, a) for c in CASES for a in ('X', 'Y')}:
        raise ValueError('All four cases and both axes required')
    comparisons = []
    for row in rows:
        values = row['metrics']
        if set(values) != {'B0', 'B1', 'B2', 'R1'}:
            raise ValueError('Route inventory drift')
        for metric in values.values():
            if set(metric) != set(FIELDS) or metric['measurement_count'] != 1:
                raise ValueError('Metric/wrapper inventory drift')
        comparisons.append(dict(case=row['case'], axis=row['axis'], metrics=values,
            B0_equals_B1=values['B0'] == values['B1'],
            B2_equals_R1=values['B2'] == values['R1'],
            B2_minus_B1={k: values['B2'][k]-values['B1'][k] for k in FIELDS},
            B2_minus_R1={k: values['B2'][k]-values['R1'][k] for k in FIELDS}))
    comparison = dict(status='SAVED_PAIRED_COMPARISON_COMPLETE',
        scope='Four preselected synthetic cases, both axes, same full-wrapper/compiler; no weighted or across-case cost totals.',
        B0_B1_all_metrics_equal=all(r['B0_equals_B1'] for r in comparisons),
        B2_R1_all_eight_wrappers_all_eight_metrics_equal=all(r['B2_equals_R1'] for r in comparisons),
        rows=comparisons,
        interpretation='Simple adjacent controlled-Pauli angle fusion reproduces the observed compiled metric gains on this fixture. No residual compiled metric difference from saved R1 is observed. This does not uniquely identify Rustiq internals or establish general superiority/newness.',
        next_decision='User/research C/coordination decides whether to retain this known implementation improvement and close C4 as an independent research direction. No R2 or further experiment is authorized.')
    # Count saved reports and summarize errors only; never rematerialize operators.
    local, before, full, adjoint, compiled, unitarity = [], [], [], [], [], []
    for path in sorted((root/'reports').glob('*_macro*_B2.json')):
        report = json.loads(path.read_bytes())
        local.append(report)
        before.append(report['before_fusion_literal'])
        unitarity.append(report['unitarity'])
    for cid in CASES:
        build = read(root, 'reports/'+cid+'_B2_build.json')
        full.append(build['operator'])
        unitarity.append(build['operator']['unitarity'])
        adjoint.append(read(root, 'reports/'+cid+'_B2_adjoint.json'))
        for axis in ('X', 'Y'):
            item = read(root, 'reports/'+cid+'_B2_'+axis+'_compiled.json')
            compiled.append(item['operator'])
            if (item['cold_misses'], item['warm_hits'], item['warm_extra_transpile']) != (1, 1, 0):
                raise ValueError('Saved cache accounting drift')
    groups = dict(local_independent=local, local_before_fusion=before, full=full,
                  actual_adjoint=adjoint, compiled_wrapper=compiled, unitarity=unitarity)
    expected = dict(local_independent=10, local_before_fusion=10, full=4,
                    actual_adjoint=4, compiled_wrapper=8, unitarity=14)
    gate_summary = {}
    for name, reports in groups.items():
        if len(reports) != expected[name]:
            raise ValueError('Saved gate count drift: '+name)
        for item in reports:
            if item['tolerance'] != 1e-10 or item['rtol'] != 0 or item['max_entry'] > 1e-10:
                raise ValueError('Saved matrix gate FAIL or tolerance drift')
        gate_summary[name] = dict(count=len(reports), tolerance=1e-10, rtol=0,
            max_entry=max(r['max_entry'] for r in reports),
            max_frobenius=max(r['frobenius'] for r in reports),
            max_spectral=max(r['spectral'] for r in reports))
    for kind, reports in read(root, 'signal_reports.json').items():
        if len(reports) != 4 or any(r['tolerance'] != 2e-10 or r['error'] > 2e-10
                                     for r in reports.values()):
            raise ValueError('Saved signal count/gate FAIL or tolerance drift')
        gate_summary[kind+'_complex_XY'] = dict(count=4, tolerance=2e-10,
            max_error=max(r['error'] for r in reports.values()))
    counts, intervals, line_count = {}, [], 0
    for path in sorted((root/'journals').glob('*.jsonl')):
        for line in path.read_text().splitlines():
            item = json.loads(line)
            line_count += 1
            if item['status'] == 'FAILED':
                raise ValueError('Failed saved request: '+item['id'])
            if item['status'] == 'COMPLETED':
                counts[item['group']] = counts.get(item['group'], 0)+1
                intervals.append(dict(journal=path.name, **item))
    if counts != contract['journal_counts'] or counts != summary['counts'] or line_count != 356:
        raise ValueError('Saved call accounting drift')
    trace = []
    for cid in CASES:
        for item in read(root, 'reports/'+cid+'_B2_build.json')['trace']:
            if item['kind'] == 'synthesis':
                raise ValueError('B2 unexpectedly contains Rustiq synthesis')
            trace.append(dict(case=cid, **{k:v for k,v in item.items()
                if k in ('kind','region','seconds','cpu_seconds','emitter_seconds','emitter_cpu_seconds')}))
    timing = dict(scope='Single local B2 execution; per-request intervals and nested plan/fold/emitter measurements. No matched R1 timing and no speedup estimate.',
        wall_seconds=summary['wall_seconds'], parent_cpu_seconds=summary['parent_cpu_seconds'],
        children_cpu_seconds=summary['children_cpu_seconds'],
        max_recorded_process_peak_RSS_KiB=max(r['peak_RSS_KiB'] for r in intervals),
        memory_scope='Maximum per-process high-water RSS in completed journal records, not concurrent aggregate memory.',
        request_intervals=intervals, builder_subintervals=trace,
        Rustiq_synthesis='NOT_CALLED', nested_times_must_not_be_added_to_builder=True)
    return comparison, gate_summary, timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    comparison, gates, timing = summarize(args.root)
    if args.write:
        for name, value in [('comparison_summary.json', comparison), ('gate_summary.json', gates),
                            ('classical_costs.json', timing)]:
            with (args.root/name).open('x') as stream:
                json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
                stream.write('\n')
        for filename, rows, fields in [
            ('metrics.csv', read(args.root, 'metrics.json'), ('case', 'route', 'axis', *FIELDS)),
            ('paired_metrics.csv', [dict(case=r['case'], axis=r['axis'], route=route, **metric)
                for r in comparison['rows'] for route,metric in r['metrics'].items()],
             ('case', 'axis', 'route', *FIELDS))]:
            with (args.root/filename).open('x', newline='') as stream:
                writer = csv.DictWriter(stream, fields)
                writer.writeheader()
                writer.writerows(rows)
    else:
        for name, value in [('comparison_summary.json', comparison), ('gate_summary.json', gates),
                            ('classical_costs.json', timing)]:
            if read(args.root, name) != value:
                raise ValueError('Derived saved report mismatch: '+name)
    print(json.dumps(dict(B2_R1_all_metrics_equal=comparison[
        'B2_R1_all_eight_wrappers_all_eight_metrics_equal'], gate_summary=gates,
        max_recorded_process_peak_RSS_KiB=timing['max_recorded_process_peak_RSS_KiB']), indent=2))


if __name__ == '__main__':
    main()
