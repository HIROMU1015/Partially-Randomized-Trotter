#!/usr/bin/env python3
"""Path-level observed variation of saved LD2 prices (descriptive; S0 item 4 fallback).

run01 could not segment any path (P2 FAIL), so per-step variation inside multi-step paths is not
identified. For single-step cells (q=1) a path is one step, so the variance of path totals is the
variance of the step contribution in the first-step context, without any decomposition. Multi-step
path differences are listed for reference only. Inputs: receipt rz_count values bound by run01 P1
(INPUTS.json) and C2's published m9 receipts (a7fdd200d). X axis only (Y is the same path; X=Y in
every receipt). P2R66 is excluded (derived from S66). No fit, no CI, no prediction.
"""
from fractions import Fraction
import csv
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
M9 = ('a7fdd200da8d499e93965d63b5b2e5b01e0b8940',
      'docs/research/d_pauli_ld2_m9_longpoint_results_20261115/run01/COMPILE_RECEIPTS.json')


def main():
    data = json.loads((HERE / 'INPUTS.json').read_text())
    cells = {}
    for p in data['paths']:
        if p['composed_of']:
            continue
        x, y = p['axes']['X']['metrics']['rz_count'], p['axes']['Y']['metrics']['rz_count']
        assert x == y
        cells.setdefault((p['cell'], p['q'], p['R']), []).append(x)
    m9 = json.loads(subprocess.run(['git', 'show', f'{M9[0]}:{M9[1]}'], cwd=ROOT, check=True,
                                   capture_output=True).stdout)
    assert m9['p0X']['record']['metrics']['rz_count'] == m9['p0Y']['record']['metrics']['rz_count']
    cells[('M9_ROUND (C2, reference)', 512, 526)] = [m9['p0X']['record']['metrics']['rz_count'],
                                                    m9['p1X']['record']['metrics']['rz_count']]
    rows = []
    for (cell, q, R), vals in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][2])):
        n = len(vals)
        row = dict(cell=cell, q=q, R=R, n_paths=n, rz_values='|'.join(map(str, vals)))
        if n >= 2:
            mean = Fraction(sum(vals), n)
            var = sum((Fraction(v) - mean) ** 2 for v in vals) / (n - 1)
            row.update(path_sample_variance=float(var), dof=n - 1,
                       per_step_variance_if_iid_steps=float(var / q),
                       note=('single-step cell: equals the step-contribution variance (first-step context)'
                             if q == 1 else 'multi-step path: per-step value assumes independent, identically '
                                            'distributed, additive steps (H_loc), which S0 did not establish'))
        else:
            row.update(path_sample_variance=None, dof=0, per_step_variance_if_iid_steps=None,
                       note='not identifiable (n=1)')
        rows.append(row)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(HERE / 'PATH_LEVEL_VARIATION.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == '__main__':
    main()
