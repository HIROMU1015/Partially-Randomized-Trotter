"""Guard saved-data execution, then independently compare public CSV results."""
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def guard(event, args):
    if event in ('subprocess.Popen', 'os.system', 'socket.connect'):
        raise RuntimeError('No process launch or network in saved-data analysis')
    if event == 'import' and str(args[0]).split('.')[0] in ('qiskit', 'scipy', 'pyscf'):
        raise RuntimeError('Scientific runtime import forbidden')
    if event == 'open' and isinstance(args[0], (str, bytes)):
        path = str(args[0])
        if path.endswith(('.npz', '.qpy', '.qpy.gz', '.npz.gz')):
            raise RuntimeError('Scientific array/circuit access forbidden')


def rows(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        return list(csv.DictReader(stream))


def main():
    sys.addaudithook(guard)
    # Execute existing tests directly: this host does not have pytest installed.
    tests = runpy.run_path(str(ROOT / 'tests/test_h46_long_time_accounting.py'))
    tests['test_paired_cost_uncertainty_includes_covariance']()
    tests['test_deterministic_integer_accounting_has_no_sampling_error']()
    runpy.run_path(str(HERE / 'analyze.py'), run_name='__main__')
    public = ROOT / 'artifacts/resource_applicability/track_a_h46_long_time/2026-10-25'
    original = rows(public / 'candidate_precision_accounting.csv')
    mapped = rows(HERE / 'candidate_endpoint_coverage.csv.gz')
    screened = rows(HERE / 'all_unpriced_screening.csv.gz')
    assert len(original) == len(mapped) == 58160
    indexed = {(x['id'], x['epsilon']): x for x in original}
    for x in mapped:
        old = indexed[(x['id'], x['epsilon'])]
        assert x['signal_status'] == old['status']
        assert x['N_cos'] == old['N_cos'] and x['N_sin'] == old['N_sin']
        assert (x['price_status'] == 'COMPATIBLE_ACQUIRED') == (old['native_cost_status'] == 'ACQUIRED')
    minima = defaultdict(list)
    for x in original:
        if x['G_exact']:
            cell = (x['system'], x['T'], x['epsilon'], 'PR' if x['method'] == 'PR' else 'PF')
            minima[cell].append((Fraction(x['G_exact']), x['id']))
    minima = {k: min(v) for k, v in minima.items()}
    for x in screened:
        cell = (x['system'], x['T'], x['epsilon'])
        g, identity = min(minima[(*cell, 'PF')], minima[(*cell, 'PR')])
        assert x['current_best_id'] == identity and Fraction(x['current_best_G_exact']) == g
        assert Fraction(x['equal_axis_break_even_exact']) == g / (int(x['N_cos']) + int(x['N_sin']))
        assert x['excluded_by_rigorous_bound'] == 'False'
    selected = rows(HERE / 'promising_candidates.csv')
    assert len({x['id'] for x in selected}) == len(selected) == 16
    assert sum(x['recommendation'] == 'INITIAL' for x in selected) == 7
    assert all(not indexed[(x['id'], x['epsilon'])]['G_exact'] for x in selected)
    # Check the extra saved measured m5 catalogue independently of saved_assets.
    catalogue = ROOT / 'artifacts/resource_applicability/track_a_h46_higher_pf_dataset_v1/2026-10-20/measured_gate_costs_96.csv'
    by_condition = {}
    fields = ('system', 'method', 'formula', 'prefix', 'q')
    for x in original:
        if float(x['T']) == .8:
            by_condition[tuple(x[k] for k in fields)] = x
    catalogue_matches = 0
    for x in rows(catalogue):
        if tuple(x[k] for k in fields) not in by_condition:
            continue
        old = by_condition[tuple(x[k] for k in fields)]
        assert old['native_cost_status'] == 'ACQUIRED'
        # The catalogue includes ineligible endpoints; use any priced epsilon.
        priced = next(v for v in original if v['id'] == old['id'] and v['C_cos'])
        axis = 'C_cos' if x['axis'] == 'cosine' else 'C_sin'
        assert float(priced[axis]) == int(x['rz_count'])
        catalogue_matches += 1
    summary = json.loads((HERE / 'summary.json').read_text())
    for name, digest in summary['source_hashes'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    result = dict(status='PASS', existing_accounting_tests=2,
        public_CSV_endpoint_rows_compared=len(mapped), exact_break_even_rows_checked=len(screened),
        baseline_family_minima_reproduced=len(minima), extra_m5_catalogue_matching_axes=catalogue_matches,
        protected_source_hashes_checked=len(summary['source_hashes']),
        guarded_analysis_completed=True, new_scientific_calls=0,
        validation_scope='LOCAL_SAVED_METADATA_AND_ARITHMETIC; not CI or external reproduction')
    (HERE / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
