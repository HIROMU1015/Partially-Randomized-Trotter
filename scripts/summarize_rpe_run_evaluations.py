"""Compact evaluation table of a round directory (every completed job): configuration, allocation, status, outward 30-digit
MSE bounds, exact cost, MSE upper / tau, origin, OUTCOME sha256.  The exact bounds stay in each job's OUTCOME.json.

    python summarize_rpe_run_evaluations.py --problem P --run-dir DIR [--labels LABELS.json]
"""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from trottertracks import rpe_method_optimizer as O  # noqa: E402
import run_rpe_method_optimizer as CLI  # noqa: E402

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--problem', type=Path, required=True);ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--labels', type=Path, help='{configuration_id: label} (e.g. L_D/kappa names)')
    a = ap.parse_args();tau = O.tau(O.load_json(a.problem));labels = json.loads(a.labels.read_text()) if a.labels else {}
    rows = []
    for req in sorted((a.run_dir / 'jobs' / 'PR_PAULI').glob('*/REQUEST.json')):
        job = req.parent;q = json.loads(req.read_text());o = job / 'OUTCOME.json'
        if not o.exists():
            rows.append(dict(job=job.name, configuration_id=q['configuration_id'], status='NO_OUTCOME'));continue
        out = json.loads(o.read_text());lo = None if out.get('MSE_lower') is None else O.frac(out['MSE_lower'])
        up = None if out.get('MSE_upper') is None else O.frac(out['MSE_upper'])
        rows.append(dict(job=job.name, configuration_id=q['configuration_id'], label=labels.get(q['configuration_id']), N_X=q['N_X'], N_Y=q['N_Y'],
                         status=out['status'], cost=O.frac(q['cost']), MSE_lower=lo and CLI.outward(lo), MSE_upper=up and CLI.outward(up, up=True),
                         MSE_upper_over_tau=None if up is None else float(up / tau), origin=q.get('origin', {}).get('kind'),
                         outcome_sha256=hashlib.sha256(o.read_bytes()).hexdigest()))
    (a.run_dir / 'EVALUATIONS.json').write_text(json.dumps(O.jsonable(dict(schema='RPE_RUN_EVALUATIONS_V1', tau=tau, rows=rows,
        exact_bounds='MSE bounds as outward 30-digit decimals; exact values in jobs/PR_PAULI/<job>/OUTCOME.json (server)')), indent=1) + '\n')
    keys = ['label', 'configuration_id', 'N_X', 'status', 'MSE_upper_over_tau', 'cost', 'origin', 'job', 'outcome_sha256']
    with (a.run_dir / 'EVALUATIONS.csv').open('w', newline='') as f:
        w = csv.writer(f, lineterminator='\n');w.writerow(keys)
        for r in sorted(rows, key=lambda r: (str(r.get('label')), r.get('cost') or 0)):
            w.writerow([' '.join(map(str, r[k])) if isinstance(r.get(k), list) else (O.show(r[k]) if isinstance(r.get(k), Fraction) else r.get(k)) for k in keys])
    print(len(rows), sum(1 for r in rows if r['status'] == O.PASS))


if __name__ == '__main__':
    main()
