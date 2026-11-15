"""One command: per-method RPE cost optimization, report mode or an approved search round.

Approved round (``--round ROUND.json --run-dir DIR``): methods and new-Q3 budget come from the
round; each request is executed on the method's existing Q3 route, validated, ingested, and the
next candidate is chosen, until a stop rule fires.  Re-running the same command resumes: saved
and completed results are reused, running jobs are never launched twice.

Report mode (no ``--round``; new-Q3 budget 0, nothing is launched):
  FULL     all saved Q3 known; per-method best, pending next Q3, not-ready candidates
  INGEST   one externally supplied Q3 result (artificial here) is added; the next
           request is recomputed from the updated knowledge
  REPLAY   start from listed saved seeds only; saved results answer the search
  FIXTURE  artificial Q3 backend (labelled ARTIFICIAL) to exercise several steps
  OWN_PRICE same integer vectors priced by each PR configuration's own prices
  ROUTE_DRY_CHECK (``--dry-check-routes``) builds each next request on its real route and runs
           route metadata/count/runtime checks only
and writes RESULT.json, PER_METHOD.csv, NEEDS.json, INGEST_DEMO.json,
REPLAY.json, FIXTURE.json, OWN_PRICE.json and CHECKS.json.
"""
import argparse
import csv
from fractions import Fraction
import inspect
import json
from pathlib import Path
import subprocess
import time

from trottertracks import rpe_method_optimizer as O

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(O.jsonable(value), ensure_ascii=False, indent=2) + '\n')


def candidate_rows(context):
    return [dict(name=c['name'], method=c['method'], configuration=c['key'], configuration_id=c['configuration_id'],
                 ready=c['ready'], needs=c['needs'], price_status=c['price_status'], signal_source=c['signal_source'],
                 labels=c['labels'], saved_points=c['saved_points'], outside_index=c.get('outside_index', False))
            for c in context['candidates']]


def table(result, path):
    with path.open('w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['method', 'best_candidate', 'configuration', 'N_X', 'N_Y', 'status', 'MSE_lower', 'MSE_upper',
                    'cost_RZ', 'stop_reason', 'next_Q3_candidate', 'next_Q3_N_X', 'next_Q3_cost_RZ', 'ready_candidates',
                    'not_ready_candidates', 'evaluated'])
        for name, r in result.items():
            b, p = r['best'], r['pending_request']
            w.writerow([name, b and b['candidate'], b and json.dumps(b['configuration']), b and ' '.join(map(str, b['N_X'])),
                        b and ' '.join(map(str, b['N_Y'])), b and b['status'],
                        b and '%.16e' % float(b['MSE_lower']), b and '%.16e' % float(b['MSE_upper']),
                        b and b['cost_display'], r['stop_reason'], p and p['candidate'],
                        p and ' '.join(map(str, O.split(p['vector'])[0])), p and O.show(p['cost']),
                        len(r['ready_candidates']), len(r['not_ready_candidates']), r['evaluated_count']])


def fixture_function(context, method):
    """Artificial MSE (labelled): sum_m w_m (1/N_mX + 1/N_mY)/2 with late rounds weighted more."""
    t = context['tau']
    weights = [Fraction((m + 1) ** 2, 1) for m in range(13)]
    scale = t * Fraction(3, 5) / sum(w / n for w, n in zip(weights, [59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 13, 4]))

    def mse(config_id, vector):
        nx, ny = O.split(vector)
        value = scale * sum(w * (Fraction(1, x) + Fraction(1, y)) / 2 for w, x, y in zip(weights, nx, ny))
        return value, value
    return mse


def head():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def resume_input(paths):
    """Resume records exactly as written by a previous run (ROUND_RESULT/RESULT methods.*.resume_records, or a list)."""
    records = []
    for path in paths or []:
        value = O.load_json(path)
        if isinstance(value, list):
            records += value
        else:
            for r in value.get('methods', {}).values():
                records += r.get('resume_records', [])
    return records


def scoped(problem, overrides):
    """Apply an approved search scope (formulas, width range, L_D values, configuration allow-list) to methods."""
    p = json.loads(json.dumps(problem))
    for name, change in (overrides or {}).items():
        p['methods'][name].update(change)
    return p


def round_mode(a, problem, extra):
    """Approved search round: execute requests on existing routes within the round budget; resumable."""
    spec = O.load_json(a.round);run = a.run_dir;run.mkdir(parents=True, exist_ok=True);start = time.perf_counter()
    problem = scoped(problem, spec.get('method_overrides'));context = O.prepare(ROOT, problem)
    extra = extra + resume_input([Path(x) for x in spec.get('resume_from', [])])
    stored = run / 'ROUND.json'
    if stored.exists():
        old = O.load_json(stored)
        for key in ('round_id', 'methods', 'initial', 'routes_override', 'method_overrides'):
            if old.get(key) != spec.get(key):
                raise SystemExit('Run directory belongs to a different approved round (' + key + ')')
    else:
        write(stored, spec)
    routes = dict(problem['execution_routes']);routes.update(spec.get('routes_override', {}))
    search = dict(problem['search'], max_steps_per_method=spec.get('max_steps_per_method', problem['search']['max_steps_per_method']))
    executor = O.Executor(ROOT, run, routes, context['expected'], problem, optimizer_source=head())
    result = O.optimize(ROOT, dict(problem, search=search), context=context, methods=spec['methods'],
                        initial=spec.get('initial', 'all'), max_new=spec['max_new_q3_per_method'], executor=executor,
                        extra_results=extra)
    write(run / 'ROUND_RESULT.json', dict(schema='RPE_OPTIMIZER_ROUND_RESULT_V1', round=spec, methods=result,
                                          optimizer_source=head(), wall_seconds=time.perf_counter() - start))
    table(result, run / 'PER_METHOD.csv')
    write(run / 'NEEDS.json', dict(candidates=[r for r in candidate_rows(context) if r['method'] in spec['methods']],
                                   pending_Q3={m: r['pending_request'] for m, r in result.items()}))
    with (run / 'INVOCATIONS.jsonl').open('a') as f:
        f.write(json.dumps(O.jsonable(dict(time=time.time(), budget=spec['max_new_q3_per_method'],
            launched={m: executor.launched(m) for m in spec['methods']},
            stop={m: r['stop_reason'] for m, r in result.items()}, new_calls={m: r['new_calls'] for m, r in result.items()})),
            ensure_ascii=False) + '\n')
    print(json.dumps(O.jsonable({m: dict(stop=r['stop_reason'], best=r['best'] and r['best']['cost_display'],
        new_calls=r['new_calls'], spent_before=r['budget']['spent_before'],
        next=r['pending_request'] and (r['pending_request']['candidate'], O.show(r['pending_request']['cost'])))
        for m, r in result.items()}), ensure_ascii=False, indent=1))


def dry_check_routes(problem, context, full, out):
    """Build the next request(s) of each method on its real route and run metadata/count/runtime checks only."""
    rows = []
    for name, r in full.items():
        pend = r['pending_request']
        if pend is None:
            continue
        seen, picks = set(), []
        for q in [pend] + pend['queue']:
            cid = next(c['configuration_id'] for c in context['candidates'] if c['name'] == q['candidate'])
            if cid not in seen:
                seen.add(cid);picks.append(dict(q, configuration_id=cid, vector=tuple(q['vector'])))
        ex = O.Executor(ROOT, out / 'ROUTE_DRY_CHECK', problem['execution_routes'], context['expected'], problem,
                        optimizer_source=head()).for_method(name)
        for q in picks:
            rule = ex.route(q['configuration_id']);job = ex.job_dir(q)
            if rule is None:
                rows.append(dict(method=name, candidate=q['candidate'], status='NO_ROUTE_ADAPTER'));continue
            job.mkdir(parents=True, exist_ok=True)
            (job / 'REQUEST.json').write_text(json.dumps(O.jsonable(ex.request(q, job, rule)), ensure_ascii=False, indent=2) + '\n')
            for stale in ('DRY_CHECK.json', 'JOB_STAGES.jsonl'):
                (job / stale).unlink(missing_ok=True)
            code = ex.run_adapter(job, rule, 'dry-check').wait()
            check = O.load_json(job / 'DRY_CHECK.json') if (job / 'DRY_CHECK.json').exists() else dict(status='NO_OUTPUT')
            rows.append(dict(method=name, candidate=q['candidate'], route=rule['kind'], view_commit=rule['view_commit'],
                             status=check['status'], returncode=code, bounds=check.get('bounds'), template=check.get('template'),
                             science_calls=0, job=str(job.relative_to(ROOT)) if job.is_relative_to(ROOT) else str(job)))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--problem', type=Path, required=True)
    p.add_argument('--output', type=Path, help='report-mode output directory')
    p.add_argument('--round', type=Path, help='approved search round JSON (methods, new-Q3 budget, optional routes override)')
    p.add_argument('--run-dir', type=Path, help='round directory (jobs, results); re-run the same command to resume')
    p.add_argument('--dry-check-routes', action='store_true')
    p.add_argument('--ingest', type=Path, help='JSON list of Q3 results {configuration_id,N_X,N_Y,status,MSE_lower,MSE_upper,source}')
    p.add_argument('--resume-from', type=Path, action='append', help='previous ROUND_RESULT.json/RESULT.json (resume_records) to reuse')
    a = p.parse_args();start = time.perf_counter()
    problem = O.load_json(a.problem)
    extra = (O.load_json(a.ingest) if a.ingest else []) + resume_input(a.resume_from)
    if a.round:
        if a.run_dir is None:
            raise SystemExit('--run-dir is required with --round')
        return round_mode(a, problem, extra)
    context = O.prepare(ROOT, problem)
    out = a.output;out.mkdir(parents=True, exist_ok=True)
    full = O.optimize(ROOT, problem, context=context, extra_results=extra)
    checks = {}
    minima = context['catalog']['method_minima']
    checks['C_catalog_minima_reproduced'] = {m: (full[m]['best'] is not None and minima.get(m) is not None
                                              and full[m]['best']['cost'] == O.frac(minima[m]['conditional_cost'])) for m in full}
    priced = [c for c in context['candidates'] if c['prices'] is not None]
    checks['own_prices_reproduce_every_saved_total'] = all(c['price_status'] == 'SAVED_UNIT_PRICES_REPRODUCE_ALL_SAVED_TOTALS' for c in priced)
    checks['priced_configurations'] = len(priced)
    for name, spec in problem['methods'].items():
        if spec['family'] == 'PR':
            lds = sorted({c['key']['L_D'] for c in context['candidates'] if c['method'] == name})
            checks['L_D_enumerated_' + name] = dict(values=lds, fragment_count=context['facts']['fragment_count'],
                                                     from_rule=lds == list(range(spec['L_D_range']['min'],
                                                         context['facts']['fragment_count'] + spec['L_D_range']['max_offset_from_fragment_count'] + 1)))
        else:
            checks['PF_candidates_' + name] = len([c for c in context['candidates'] if c['method'] == name])
    write(out / 'RESULT.json', dict(schema='RPE_METHOD_OPTIMIZER_RESULT_V1', rule=O.RULE, problem=str(a.problem),
                                     facts=context['facts'], tau=context['tau'], methods=full, ingested=extra,
                                     saved_catalog=problem['evidence']['saved_catalog_inputs'],
                                     scope='MINIMUM_WITHIN_EVALUATED_SET_AND_FINITE_SEARCH_RULE; no new science'))
    table(full, out / 'PER_METHOD.csv')
    write(out / 'NEEDS.json', dict(candidates=candidate_rows(context),
                                   pending_Q3={m: r['pending_request'] for m, r in full.items()}))

    # INGEST: an artificial result for the TRUNCATED_PF pending proposal changes the next request.
    pend = full['TRUNCATED_PF']['pending_request']
    fake = dict(configuration_id=pend['configuration_id'], N_X=O.split(pend['vector'])[0], N_Y=O.split(pend['vector'])[1],
                status=O.PASS, MSE_lower=O.enc(context['tau'] * Fraction(19, 20)), MSE_upper=O.enc(context['tau'] * Fraction(19, 20)),
                source='ARTIFICIAL_INGEST_DEMO')
    after = O.optimize(ROOT, problem, context=context, methods=['TRUNCATED_PF'], extra_results=extra + [fake])['TRUNCATED_PF']
    write(out / 'INGEST_DEMO.json', dict(ingested=fake, before_pending=pend, after_best=after['best'],
                                         after_pending=after['pending_request'], label='ARTIFICIAL_RESULT_NOT_SCIENCE'))
    checks['ingest_changes_next_request'] = (after['best']['cost'] == pend['cost'] and after['pending_request'] is not None
                                             and after['pending_request']['cost'] < pend['cost'])

    # REPLAY from listed saved seeds; saved results answer each step.
    replay = O.optimize(ROOT, problem, context=context, methods=list(problem['verification']['replay_seeds']),
                        initial=problem['verification']['replay_seeds'])
    write(out / 'REPLAY.json', replay)
    for name, r in replay.items():
        steps = r['trace']
        seed_cost = steps[0]['incumbent_before'] if steps else None
        checks['replay_' + name] = dict(seed_cost=seed_cost, best_cost=r['best'] and r['best']['cost'],
                                         reached_cheaper_feasible=bool(steps) and r['best']['cost'] < seed_cost,
                                         steps=len(steps), new_calls=r['new_calls'], stop_reason=r['stop_reason'],
                                         next_from_updated_state=all(s2['origin'].get('base_vector') == s1['vector']
                                                                     for s1, s2 in zip(steps, steps[1:])
                                                                     if s1['improved'] and s2['candidate'] == s1['candidate']
                                                                     and s2['origin']['kind'].startswith('DECREMENT')))

    # FIXTURE: artificial backend only, from no knowledge.
    fx = O.FixtureBackend(fixture_function(context, 'TRUNCATED_PF'), context['tau'])
    fixture = O.optimize(ROOT, problem, context=context, methods=['TRUNCATED_PF'], initial='none', backend=fx, max_new=12,
                          saved_is_free=False)
    write(out / 'FIXTURE.json', dict(fixture, label='ARTIFICIAL_FIXTURE_NOT_SCIENCE'))
    f = fixture['TRUNCATED_PF']
    checks['fixture'] = dict(steps=len(f['trace']), improvements=sum(t['improved'] for t in f['trace']),
                             continued_after_first_improvement=len([t for t in f['trace'] if t['improved']]) > 1
                             or (any(t['improved'] for t in f['trace']) and f['trace'][-1]['improved'] is False),
                             stop_reason=f['stop_reason'], new_calls=f['new_calls'])

    # OWN_PRICE: same vectors under each ready PR configuration's own prices.
    saved = {r['label']: (cid, v, r) for (cid, v), r in context['saved'].table.items()}
    vectors = {lab: saved[lab][1] for lab in problem['verification']['own_price_vectors_from_labels'] if lab in saved}
    pr = [c for c in context['candidates'] if c['family'] == 'PR']
    rows = O.own_price_comparison(pr, vectors)
    for row in rows:
        cid = next(c['configuration_id'] for c in pr if c['name'] == row['candidate'])
        hit = context['saved'].table.get((cid, vectors[row['vector_label']]))
        row.update(saved_status=hit and hit['status'], saved_cost_matches=hit is None or hit['cost'] == row['cost'])
    write(out / 'OWN_PRICE.json', rows)
    checks['own_price_rows'] = len(rows)
    checks['own_price_saved_costs_match'] = all(r['saved_cost_matches'] for r in rows)
    source = inspect.getsource(O)
    checks['no_known_answer_constants_in_optimizer'] = not any(s in source for s in ('395962678', '559457386', '435481090', '[59, 55'))
    if a.dry_check_routes:
        rows = dry_check_routes(problem, context, full, out)
        write(out / 'ROUTE_DRY_CHECK.json', rows)
        checks['route_dry_check'] = {r['method'] + ':' + r['candidate'].split(':')[-1]: r['status'] for r in rows}
    checks['wall_seconds'] = time.perf_counter() - start
    checks['execution_source'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    write(out / 'CHECKS.json', checks)
    print(json.dumps(O.jsonable({k: v for k, v in checks.items() if k != 'execution_source'}), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
