"""Export completed H4/1.00-Angstrom JSON summaries; no molecular evaluation.

Run from the repository root. Reads saved summaries only, never NPZ/QPY, and
preserves the original execution contract, raw results and mandatory STOP.
"""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'artifacts/research_c_integrated/2026-11-06-picklefix'
OUT = Path(__file__).resolve().parent
DOWN = BASE / 'development/run01/downstream'
METRICS = ('rz_count', 'rz_depth', 'cx_count', 'cx_depth', 'total_depth', 'circuit_size')


def read(path):
    return json.loads(path.read_text())


def tsv(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t',
                                lineterminator='\n', restval='NA')
        writer.writeheader()
        writer.writerows({key: 'NA' if value is None else value
                          for key, value in row.items()} for row in rows)


def main():
    source = read(BASE / 'technical/SOURCE.json')
    bad = [p for p, digest in source['files'].items()
           if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != digest]
    assert not bad, bad
    selected = read(DOWN / 'selected.json')
    comparisons, costs, deterministic, checkpoints = [], [], [], []
    for path in sorted(DOWN.glob('comparison_*.json')):
        value = read(path)
        _, start, budget, *regime_parts = path.stem.split('_')
        regime = '_'.join(regime_parts)
        row = dict(start=start, budget=int(budget[1:]), regime=regime,
                   status=value['status'], report=str(path.relative_to(ROOT)))
        result = value['comparison']
        if result is not None:
            row.update(PR_total_RZ=result['total_points']['PR'],
                       ENTRY_total_RZ=result['total_points']['ENTRY'],
                       point_ratio_PR_ENTRY=result['point_ratio_PR_ENTRY'],
                       PR_reduction_percent=100 * (1 - result['point_ratio_PR_ENTRY']),
                       confidence=result['confidence'],
                       difference_low=result['difference_interval'][0],
                       difference_high=result['difference_interval'][1],
                       precision_met=result['precision_met'], decision=result['decision'])
        for arm in ('PR', 'ENTRY'):
            chosen = selected[f"{start}/{row['budget']}/{arm}"][regime]
            row[arm + '_id'] = None if chosen is None else chosen['id']
            row[arm + '_strength'] = None if chosen is None else chosen['strength']
            row[arm + '_r'] = None if chosen is None else chosen['r']
        comparisons.append(row)
    for path in sorted((DOWN / 'final').glob('*/r*/result.json')):
        value = read(path)
        assert value['samples'] == 64 and value['cohort_role'] == 'final'
        assert read(path.parent / 'call_summary.json')['failed'] is None
        bind = value['bindings']['ideal_exact_native']
        stats = value['statistics']
        lane = stats['lanes']['ideal_exact_native']
        nominal = lane['RZ_PR_D_nominal']
        row = dict(value['task'], r=value['r'], candidate_sha256=value['candidate_sha256'],
                   samples=64, cohort_role='final', shots=bind['shots'], W=bind['W'],
                   report=str(path.relative_to(ROOT)),
                   same_JU_D_reference=value['same_JU_D_reference'])
        for metric in METRICS:
            kernel = stats['calibration']['sample_mean'][metric]
            shell = bind['shell'][metric]
            assert lane['totals'][metric] == bind['W'] * kernel + shell
            row.update({metric + '_mean_kernel': kernel, metric + '_shell': shell,
                        metric + '_total': lane['totals'][metric],
                        metric + '_same_JU_D': lane['same_JU_D'][metric],
                        metric + '_ratio_to_D': lane['point_ratios'][metric]})
        row.update(D_ratio_low=nominal['ratio_interval'][0],
                   D_ratio_high=nominal['ratio_interval'][1],
                   D_nominal_confidence=nominal['confidence'],
                   D_nominal_decision=nominal['decision'], D_precision_met=nominal['precision_met'])
        costs.append(row)
    for path in sorted((DOWN / 'deterministic').glob('*/D/result.json')):
        value = read(path)
        assert read(path.parent.parent / 'call_summary.json')['failed'] is None
        lane = value['ideal_exact_native']
        w = sum(2 * x['shots'] * (1 << x['m']) for x in lane['rounds'])
        row = dict(id=path.parent.parent.name, candidate_sha256=value['candidate'],
                   shots=lane['cost']['shots'], W=w, report=str(path.relative_to(ROOT)))
        for metric in METRICS:
            kernel = value['metrics'][metric]
            total = lane['cost']['totals'][metric]
            row.update({metric + '_kernel': kernel, metric + '_total': total,
                        metric + '_shell': total - w * kernel})
        deterministic.append(row)
    handoff = read(BASE / 'technical/saved_input_handoff.json')
    jobs = [(x['job_id'], ROOT / x['result_path']) for x in handoff['completed']]
    jobs.append(('H4_100_C0_ENTRY_g03',
                 BASE / 'development/run01/optimization/H4_100_C0_ENTRY_g03/result.json'))
    for job, path in jobs:
        value = read(path)
        summary = value['summary']
        for budget, checkpoint in value['checkpoints'].items():
            evidence = checkpoint['evidence']
            physical, state = evidence['physical'], evidence['state']
            checkpoints.append(dict(id=job, start=value['start'], arm=value['arm'],
                strength=value['strength'], budget=int(budget),
                trajectory_status=summary['status'],
                accepted_updates=summary['trajectory']['accepted_U_updates'],
                DF_budget_PASS=physical['budget_PASS'], sector_error_Ha=physical['sector_error'],
                numerical_guard_Ha=physical['numerical_guard'],
                exact_tail_lambda_Ha=physical['exact_tail_lambda_Ha'],
                J_stationarity=state['J_stationarity']['status'],
                U_stationarity=state['U_stationarity']['status'],
                candidate_sha256=physical['candidate_sha256'], report=str(path.relative_to(ROOT))))
    selection = read(DOWN / 'selection_rows.json')
    for path in (DOWN / 'selection').glob('*/call_summary.json'):
        assert read(path)['failed'] is None
    counts = dict(optimization_trajectories=len(jobs), endpoint_checkpoints=len(checkpoints),
                  selection_candidates=len(list((DOWN / 'selection').glob('*/result.json'))),
                  selection_policies=len(selection),
                  eligible_selection_policies=sum(x['eligible'] for x in selection),
                  selection_native_samples=sum(x['actual_samples'] for x in selection),
                  final_candidate_r_pairs=len(costs), final_native_samples=sum(x['samples'] for x in costs),
                  deterministic_candidates=len(deterministic),
                  completed_comparisons=sum(x['status'] == 'REGISTERED_COMPARISON_COMPLETE' for x in comparisons),
                  unresolved_comparisons=sum(x['status'] != 'REGISTERED_COMPARISON_COMPLETE' for x in comparisons))
    assert counts['selection_native_samples'] == 2880 and counts['final_native_samples'] == 1408
    progress = [json.loads(line) for line in (BASE / 'execution_progress.jsonl').read_text().splitlines()]
    stop = read(BASE / 'INTEGRATED_STOP.json')
    primary = [x for x in comparisons if x['start'] == 'C0' and x['budget'] == 64]
    output = dict(status='H4_100_COMPLETED_SAVED_RESULT_SUMMARY',
        generated_JST=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
        scientific_SOURCE_sha256=hashlib.sha256((BASE / 'technical/SOURCE.json').read_bytes()).hexdigest(),
        contract_sha256=hashlib.sha256((BASE / 'technical/contract.json').read_bytes()).hexdigest(),
        source_files_verified=len(source['files']), source_files_mismatched=bad,
        scope='Linear H4/1.00 Angstrom/STO-3G;number CDF8;sector36;LD2;partial S2/s1/h=.1/K0;m0..12;P0/fenced serial;own-DF ground;state preparation excluded',
        counts=counts, primary_comparisons=primary, comparisons=comparisons, final_costs=costs,
        deterministic_costs=deterministic, optimization_checkpoints=checkpoints,
        development_complete_elapsed_seconds=next(x['elapsed_seconds'] for x in progress
            if x.get('stage') == 'development_downstream' and x['phase'] == 'stage_complete'),
        historical_execution_stop=stop,
        current_transfer_scope='User cancelled further H4/1.15-Angstrom work on 2026-11-06;no restart,repair,or transfer performance claim',
        actual_CA_achieved=None, guaranteed_resource_superiority=None,
        statistical_qualification='Stored approximate fixed-N IID-model Welch differences;individual nominal Student-t comparisons to same-JU D;coverage and all-law price transfer unvalidated',
        new_scientific_computations=0, NPZ_QPY_decoded=False,
        important_scientific_review=False, publication_GO=None)
    (OUT / 'summary.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    tsv('comparisons.tsv', comparisons)
    tsv('final_costs.tsv', costs)
    tsv('deterministic_costs.tsv', deterministic)
    tsv('optimization_checkpoints.tsv', checkpoints)
    tsv('selection_rows.tsv', selection)
    print(json.dumps(dict(counts=counts, SOURCE_files_verified=len(source['files']),
        development_minutes=output['development_complete_elapsed_seconds']/60), ensure_ascii=False))


if __name__ == '__main__':
    main()
