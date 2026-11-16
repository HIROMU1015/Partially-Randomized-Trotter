"""Ledger of every completed PR Q3 job of the given runs, each checked before it may be reused (no Q3, no price fit).

Per job
  original result   OUTCOME.json read through the executor (``O.job_outcome``: a separately collected outcome only with its
                    original sha and a clean process-exit gate); status PASS / EXCEEDS / INCONCLUSIVE
  evaluation        same job key and route as the round would use (``Executor.prior_records``: configuration, allocation,
  contract          total view / binding / signal and contract identity); status consistent with the bounds under the
                    route's rule (PASS iff MSE_upper <= tau; EXCEEDS iff MSE_lower > tau)
  price             the request's cost equals the allocation priced with the configuration's current D2 MODEL prices
Verified jobs are the reusable knowledge.  Two minima are reported separately per configuration and per method:
  rule_adopted      the cheapest feasible point the search rule took (ROUND_RESULT of the latest run: knowledge it loaded + its
                    own adopted picks)
  verified_all      the cheapest feasible point among all verified jobs (lookahead jobs the rule never consumed included; a
                    small-gain stop is not a reason to leave a found feasible point out of the cost comparison)

    python build_rpe_q3_ledger.py --problem P --round R --runs RUN1 RUN2 ... --output DIR
"""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from trottertracks import rpe_method_optimizer as O  # noqa: E402
import rpe_round_tools as T  # noqa: E402

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)


def adopted_keys(run):
    """(configuration prefix, cost display) of picks the run's rule consumed: ADOPT events, or the interrupted trace."""
    keys = set()
    prog = run / 'PROGRESS.jsonl'
    if prog.exists():
        for line in prog.read_text().splitlines():
            e = json.loads(line)
            if e.get('event') == 'ADOPT':
                keys.add((e['configuration'][:8], e['cost']))
    for name in ('ROUND_RESULT_INTERRUPTED.json',):
        f = run / name
        if f.exists():
            for m in O.load_json(f)['methods'].values():
                for t in m.get('trace') or []:
                    if t.get('answer_source') != 'SAVED':
                        keys.add((t['configuration_id'][:8], O.show(O.frac(t['cost']))))
    return keys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--problem', type=Path, required=True);ap.add_argument('--round', type=Path, required=True)
    ap.add_argument('--runs', nargs='+', required=True);ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--method', default='PR_PAULI')
    a = ap.parse_args();t0 = time.time();out = a.output.resolve();out.mkdir(parents=True, exist_ok=True)
    spec, problem, context, routes = T.round_context(a.problem, a.round, out / 'preparation')
    tau = context['tau'];name = a.method
    cands = [c for c in context['candidates'] if c['method'] == name];ready = {c['configuration_id']: c for c in cands if c['ready']}
    rows, refused, known_all = [], [], {}
    for run in a.runs:
        run = (ROOT / run).resolve()
        ex = T.executor(problem, context, routes, out / 'executor_scratch', [str(run)]).for_method(name)
        known, ref = ex.prior_records(cands)
        adopted = adopted_keys(run)
        refused += [dict(r, run=str(run)) for r in ref]
        for (cid, v), r in known.items():
            job = Path(r['job']) if r.get('job') else None
            q = json.loads((job / 'REQUEST.json').read_text()) if job else {}
            c = ready[cid];cost = O.cost(v, c['prices']);req_cost = O.frac(q['cost']) if q.get('cost') is not None else None
            lo, up = r.get('MSE_lower'), r.get('MSE_upper')
            consistent = (r['status'] == O.PASS and up is not None and up <= tau) or \
                         (r['status'] == O.FAIL and lo is not None and lo > tau) or \
                         (r['status'] == 'INCONCLUSIVE' and lo is not None and up is not None and lo <= tau < up)
            outcome, opath = O.job_outcome(job)
            row = dict(run=run.name if run.name != 'run01' else run.parent.name + '/run01', job=job.name if job else None,
                       configuration_id=cid, candidate=c['name'], L_D=c['key'].get('L_D'), kappa=c.get('kappa_value'),
                       N_X=O.split(v)[0], N_Y=O.split(v)[1], status=r['status'], MSE_lower_over_tau=None if lo is None else lo / tau,
                       MSE_upper_over_tau=None if up is None else up / tau, cost=cost, request_cost=req_cost,
                       price_match=req_cost == cost, status_consistent=consistent, identity='EXECUTOR_PRIOR_RECORD_CHECK_PASSED',
                       outcome_file=opath.name, outcome_sha256=hashlib.sha256(opath.read_bytes()).hexdigest(),
                       adopted_by_rule=(cid[:8], O.show(cost)) in adopted)
            row['verified'] = row['price_match'] and row['status_consistent']
            rows.append(row)
            if row['verified']:
                known_all[(cid, v)] = row
    # minima
    last = O.load_json(Path(ROOT / a.runs[-1]).resolve() / 'ROUND_RESULT.json')['methods'][name]
    by_cfg = {}
    for b in last['by_configuration']:
        by_cfg[b['configuration_id']] = dict(candidate=b['candidate'], rule_adopted=None if not b.get('best') else O.frac(b['best']['cost']))
    for (cid, v), row in known_all.items():
        if row['status'] == O.PASS:
            d = by_cfg.setdefault(cid, dict(candidate=row['candidate'], rule_adopted=None))
            if d.get('verified_all') is None or row['cost'] < d['verified_all']:
                d.update(verified_all=row['cost'], verified_all_job=row['job'], verified_all_run=row['run'],
                         verified_all_adopted=row['adopted_by_rule'], verified_all_N=row['N_X'])
    for cid, d in by_cfg.items():
        c = ready.get(cid);d.update(L_D=c and c['key'].get('L_D'), kappa=c and c.get('kappa_value'))
        ra, va = d.get('rule_adopted'), d.get('verified_all')
        d['verified_below_rule_by'] = None if ra is None or va is None else ra - va
    feas = [d for d in by_cfg.values() if d.get('rule_adopted') is not None]
    method = dict(rule_adopted_min=min((d['rule_adopted'] for d in feas), default=None),
                  verified_all_min=min((d['verified_all'] for d in by_cfg.values() if d.get('verified_all') is not None), default=None))
    summary = dict(schema='RPE_Q3_LEDGER_V1', method=name, tau=tau, runs=a.runs, round_context=str(a.round),
                   jobs_verified=sum(1 for r in rows if r['verified']), jobs_checked=len(rows), jobs_refused=len(refused),
                   not_verified=[r for r in rows if not r['verified']], refused=refused, method_minima=method,
                   by_configuration=by_cfg, source=T.O.__file__ and __import__('subprocess').check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   wall_seconds=round(time.time() - t0, 1), science='none: saved outcomes, identity checks and price arithmetic only')
    (out / 'LEDGER_SUMMARY.json').write_text(json.dumps(O.jsonable(summary), indent=1, ensure_ascii=False, default=str) + '\n')
    cols = ['run', 'job', 'candidate', 'L_D', 'kappa', 'N_X', 'status', 'MSE_lower_over_tau', 'MSE_upper_over_tau', 'cost', 'price_match',
            'status_consistent', 'adopted_by_rule', 'verified', 'outcome_file', 'outcome_sha256']
    with (out / 'LEDGER.csv').open('w', newline='') as f:
        w = csv.writer(f, lineterminator='\n');w.writerow(cols)
        for r in sorted(rows, key=lambda r: (r['run'], r['candidate'], r['cost'])):
            w.writerow([' '.join(map(str, r[c])) if isinstance(r[c], (list, tuple)) else (O.show(r[c]) if c == 'cost' else
                        ('%.9g' % float(r[c]) if isinstance(r[c], Fraction) else r[c])) for c in cols])
    print(json.dumps(dict(checked=len(rows), verified=summary['jobs_verified'], refused=len(refused),
                          rule_adopted_min=method['rule_adopted_min'] and O.show(method['rule_adopted_min']),
                          verified_all_min=method['verified_all_min'] and O.show(method['verified_all_min']),
                          seconds=summary['wall_seconds'])))


if __name__ == '__main__':
    main()
