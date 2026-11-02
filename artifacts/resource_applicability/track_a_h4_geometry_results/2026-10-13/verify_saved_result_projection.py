"""Read-only, stdlib-only reproduction from the committed saved-value projection."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import resource
import signal


def read_json(path):
    def invalid(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(path.read_text(), parse_constant=invalid)


def near(a, b):
    if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-6):
        raise ValueError(f'numerical mismatch: {a!r} / {b!r}')


def evaluate(row, epsilon):
    counts = []
    for axis in ('cosine', 'sine'):
        allowance = epsilon / math.sqrt(2) - row['saved']['bias'][axis]
        if allowance <= 0:
            return None
        counts.append(math.ceil(2 * row['saved']['normalization']**2 / allowance**2 * math.log(2 / .025)))
    stats = row['_statistics']
    point = counts[0]*stats[0] + counts[1]*stats[1]
    variance = (counts[0]**2*stats[2] + 2*counts[0]*counts[1]*stats[3] + counts[1]**2*stats[4])/stats[5]
    se = math.sqrt(max(0., variance))
    return {'work':point, 'SE':se, 'shots':counts, 'engineering_interval':[point-2*se, point+2*se]}


def statistics(pairs):
    n = len(pairs)
    if n not in (1, 32) or any(len(pair) != 2 for pair in pairs):
        raise ValueError('paired cost shape')
    if any(not isinstance(v, int) or v < 0 for pair in pairs for v in pair):
        raise ValueError('invalid cost')
    a, b = (math.fsum(p[i] for p in pairs)/n for i in (0, 1))
    if n == 1:return (a,b,0.,0.,0.,n)
    return (a,b,math.fsum((p[0]-a)**2 for p in pairs)/(n-1),
            math.fsum((p[0]-a)*(p[1]-b) for p in pairs)/(n-1),
            math.fsum((p[1]-b)**2 for p in pairs)/(n-1),n)


def verify(bundle):
    manifest = read_json(bundle/'artifact_manifest_v1.json')
    for name, expected in manifest['files'].items():
        raw = (bundle/name).read_bytes()
        if len(raw) != expected['bytes'] or hashlib.sha256(raw).hexdigest() != expected['sha256']:
            raise ValueError('artifact hash mismatch: '+name)
    values = read_json(bundle/'saved_candidates_v1.json')
    summary = read_json(bundle/'summary.json')
    groups = defaultdict(list)
    fingerprints = set()
    for row in values['candidates']:
        fp = row['candidate_fingerprint']
        if fp in fingerprints:raise ValueError('duplicate candidate')
        fingerprints.add(fp)
        ident = values['geometry_identities'][row['geometry']]
        if ident['source'] != summary['dataset_identity_source']:
            raise ValueError('dataset source mismatch')
        row['_statistics'] = statistics(row['saved']['paired_costs'])
        for epsilon, expected in row['selected_display_points'].items():
            actual = evaluate(row, float(epsilon))
            if (actual is not None) != expected['eligible']:raise ValueError('eligibility mismatch')
            if actual:
                if actual['shots'] != expected['shots']:raise ValueError('shot mismatch')
                for key in ('work','SE'):near(actual[key],expected[key])
        groups[row['geometry']].append(row)
    if len(fingerprints) != 1308 or len(groups) != 6:raise ValueError('campaign coverage')
    template_sets = []
    for geometry, rows in groups.items():
        tids = {row['template_id'] for row in rows}
        if len(rows) != 218 or len(tids) != 218:raise ValueError('template coverage: '+geometry)
        template_sets.append(tids)
    if any(t != template_sets[0] for t in template_sets):raise ValueError('geometry template mismatch')
    grid = [.005*(.1/.005)**(i/300) for i in range(301)]
    grid[0], grid[-1] = .005, .1
    grid = sorted(set(grid+[.05]))
    winner_methods = {}
    for geometry, rows in groups.items():
        counts = Counter()
        for epsilon in grid:
            eligible = [(row,evaluate(row,epsilon)) for row in rows]
            eligible = [(row,p) for row,p in eligible if p is not None]
            row,p = min(eligible,key=lambda x:x[1]['work'])
            counts[row['template_id'].split('-rank')[0]] += 1
        winner_methods[geometry] = dict(counts)
    if winner_methods != summary['point_min_method_counts_across_302_saved_precision_points']:
        raise ValueError('precision-grid point minimum mismatch')
    for expected in summary['selected_precision_comparisons']:
        rows = groups[expected['geometry_A']]
        eligible = [(row,evaluate(row,expected['epsilon'])) for row in rows]
        eligible = [(row,p) for row,p in eligible if p is not None]
        row,p = min(eligible,key=lambda x:x[1]['work'])
        det, dp = min(((row,p) for row,p in eligible if row['template_id'].startswith(('B0-','B1-','PM1-B0-'))),key=lambda x:x[1]['work'])
        if len(eligible) != expected['eligible_candidates']:raise ValueError('eligible count')
        if row['template_id'] != expected['point_min_template']:raise ValueError('point minimum template')
        if det['template_id'] != expected['best_deterministic_template']:raise ValueError('deterministic comparator')
        near(p['work'],expected['RZ_work']);near(p['SE'],expected['SE'])
        near(dp['work'],expected['best_deterministic_RZ_work'])
        near(p['work']/dp['work'],expected['ratio_to_best_deterministic'])
    with (bundle/'precision_comparison.csv').open(newline='') as f:
        csv_rows = list(csv.DictReader(f))
    if len(csv_rows) != 18:raise ValueError('CSV coverage')
    for row, expected in zip(csv_rows,summary['selected_precision_comparisons']):
        for key in ('geometry_A','point_min_template','best_deterministic_template'):
            if row[key] != expected[key]:raise ValueError('CSV label mismatch')
        for key in ('epsilon','RZ_work','SE','ratio_to_best_deterministic','reduction_percent'):
            near(float(row[key]),expected[key])
    return {'status':'PASS_COMMITTED_SAVED_VALUE_PROJECTION', 'candidates':1308,
            'candidate_precision_points':1308*302, 'geometry_precision_minima':6*302,
            'selected_comparisons':18, 'new_signal_build_compile_transpile':0,
            'private_runtime_and_NPZ_access':0,
            'scope':'reproduces saved-value resource arithmetic; not a new physics run or external reproduction'}


def main():
    resource.setrlimit(resource.RLIMIT_AS,(256*2**20,256*2**20))
    resource.setrlimit(resource.RLIMIT_CPU,(45,45))
    signal.alarm(60)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(verify(args.bundle),ensure_ascii=False))


if __name__ == '__main__':
    main()
