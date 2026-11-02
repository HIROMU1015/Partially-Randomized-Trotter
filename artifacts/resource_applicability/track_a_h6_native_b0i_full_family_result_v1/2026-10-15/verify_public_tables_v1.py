"""Check saved scalar tables and exact rational envelopes; no scientific APIs.

Run with Python -P -B in a clone containing the referenced D0 Git object.
Only a read-only `git show` child is used. No network or result generation.
"""
import csv
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess

resource.setrlimit(resource.RLIMIT_AS, (512*2**20, 512*2**20))
resource.setrlimit(resource.RLIMIT_CPU, (60, 62))
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
METRICS = ('RZ', 'CX', 'RZ_depth', 'CX_depth', 'depth', 'size')
checks = 0


def require(value):
    global checks
    checks += 1
    if not value:
        raise ValueError('PUBLISHED_TABLE_INCONSISTENCY_CHECK_' + str(checks))


def read(name):
    return json.loads((HERE/name).read_bytes())


def rational(value):
    return Fraction(int(value['numerator']), int(value['denominator']))


summary = read('result_summary_v1.json')
rows = read('b0i_accounting_v1.json')
saved = read('comparison_and_P_sensitivity_v1.json')
with (HERE/'axis_compiled_metrics_v1.csv').open(newline='') as f:
    axes = {(int(r['prefix']), int(r['q']), r['axis']): r for r in csv.DictReader(f)}
require(len(axes) == 18 and len(rows) == 120)
require(len({(r['cell_id'], r['epsilon'], r['factor']) for r in rows}) == 120)
for row in rows:
    if row['status'] == 'INELIGIBLE':
        require(row['N_total'] is None and row['metrics'] == {})
        continue
    require(row['status'] == 'DERIVED_SAVED_SCALAR')
    a = row['accounting']['axes']
    require(row['N_total'] == a['real']['shots'] + a['imag']['shots'])
    p, q = row['cell']['prefix'], row['cell']['q']
    for metric in METRICS:
        expected = sum(a[channel]['shots']*int(axes[(p, q, axis)][metric])
                       for channel, axis in (('real', 'cosine'), ('imag', 'sine')))
        require(expected == row['metrics'][metric]['G_point'])

ref = summary['D0_baseline']
raw = subprocess.check_output(['git', '-C', str(ROOT), '-c', 'maintenance.auto=false',
                               '-c', 'gc.auto=0', 'show', ref['commit']+':'+ref['path']],
                              env=dict(os.environ, GIT_OPTIONAL_LOCKS='0'))
require(len(raw) == ref['bytes'] and hashlib.sha256(raw).hexdigest() == ref['sha256'])
baseline = json.loads(raw)['H6_CONFIRMATION_UNION8']
require(len(baseline) == 280)
require(all(r['sample_phase'] == 'confirmation' and r['sample_count'] == 32
            for r in baseline if r['method'] == 'B2'))
all_rows = baseline + rows
for cmp in saved['comparisons']:
    good = [r for r in all_rows if r['epsilon'] == cmp['epsilon']
            and float(r['factor']) == cmp['factor'] and r['status'] != 'INELIGIBLE']
    metric = cmp['metric']
    def best(selected):
        if not selected:
            return None
        row = min(selected, key=lambda r: (r['metrics'][metric]['G_point'], r['cell_id']))
        return dict(cell_id=row['cell_id'], G_point=row['metrics'][metric]['G_point'])
    methods = {'B0': lambda r: r['method'] == 'B0',
               'B0I': lambda r: r['method'] == 'B0I',
               'full_S2': lambda r: r['method'] == 'B1' and r['cell']['formula'] == '2nd',
               'full_S4': lambda r: r['method'] == 'B1' and r['cell']['formula'] == '4th'}
    for name, predicate in methods.items():
        require(cmp['family_minima'][name] == best([r for r in good if predicate(r)]))
    det = best([r for r in good if r['method'] in ('B0', 'B0I', 'B1')])
    pr = best([r for r in good if r['method'] == 'B2'])
    require(det == cmp['best_registered_deterministic'] and pr == cmp['confirmed_PR_union8'])
    require(cmp['PR_over_deterministic'] == pr['G_point']/det['G_point'])
    require(not cmp['population_winner_certified'] and cmp['matched_B0_delta'] is None)

segments_checked = 0
for item in saved['P_sensitivity']:
    good = [r for r in all_rows if r['epsilon'] == item['epsilon']
            and float(r['factor']) == item['factor'] and r['status'] != 'INELIGIBLE']
    lines = {r['cell_id']: (Fraction(r['N_total']), Fraction(r['metrics'][item['metric']]['G_point'])) for r in good}
    def winners(at):
        values = {cid: slope*at+intercept for cid, (slope, intercept) in lines.items()}
        value = min(values.values())
        return {cid for cid, cost in values.items() if cost == value}
    segments = item['lower_envelope']['segments']
    previous_end = Fraction(0)
    for index, segment in enumerate(segments):
        lo = rational(segment['P_start'])
        hi = rational(segment['P_end']) if segment['P_end'] is not None else None
        require(lo == previous_end and (hi is None or hi > lo))
        slope, intercept = Fraction(segment['slope']), rational(segment['intercept'])
        ids = set(segment['cell_ids'])
        require(all(lines[cid] == (slope, intercept) for cid in ids))
        require(ids <= winners(lo))
        if hi is None:
            require(index == len(segments)-1 and slope == min(x[0] for x in lines.values()))
        else:
            require(ids <= winners(hi))
        require(ids == winners((lo+hi)/2 if hi is not None else lo+1))
        previous_end = hi
        segments_checked += 1
    require(segments[-1]['P_end'] is None)
    for vertex in item['lower_envelope']['vertices']:
        require(set(vertex['tied_cells']) == winners(rational(vertex['P'])))

print(json.dumps(dict(status='PASS',checks=checks,accounting_rows=120,compiled_axes=18,
                      comparisons=len(saved['comparisons']),P_envelopes=len(saved['P_sensitivity']),
                      P_segments=segments_checked,scientific_imports=0,new_compile_calls=0)))
