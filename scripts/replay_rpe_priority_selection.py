"""Saved-log replay: how many Q3 the new candidate selection (METHOD_MINIMUM_PRIORITY_V2) needs to reach known
low-cost feasible points, compared with the stopped round's rule (FINITE_LOCAL_DESCENT_V1, configuration scope).

Arithmetic on saved results only: the answers are C's saved Q3 and the stopped round's completed jobs (validated with
the executor's identity checks).  A proposal of the new rule without an answer in the log is recorded as UNEVALUATED
(not counted, not assumed feasible or infeasible) and the replay continues with the next candidate.

Q3 counts
  old_actual     jobs of the stopped round finished (all configurations, lookahead included) when the rule adopted the
                 point -- what the round really spent (18 slots, 9 configurations in parallel)
  old_rule       rule-consumed jobs (all configurations) by that time, lookahead excluded
  new            answered round jobs consumed by the new rule (one at a time); saved C points cost 0 in both

    python replay_rpe_priority_selection.py --problem P --round R --run-dir RUN --output DIR [--variant NAME=JSON ...]
"""
import argparse
import bisect
import csv
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from trottertracks import rpe_method_optimizer as O  # noqa: E402
from trottertracks import rpe_priority_search as P  # noqa: E402
import run_rpe_method_optimizer as CLI  # noqa: E402

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)


class LogBackend:
    def __init__(self, table):
        self.table = table

    def answer(self, proposal, allow_new=True):
        hit = self.table.get((proposal['configuration_id'], tuple(proposal['vector'])))
        return None if hit is None else dict(hit)


def first(rows, test):
    return next((r for r in rows if test(r)), None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--problem', type=Path, required=True);ap.add_argument('--round', type=Path, required=True)
    ap.add_argument('--run-dir', type=Path, required=True);ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--variant', action='append', default=[], help='NAME=JSON search overrides for an extra replay of the new rule')
    a = ap.parse_args();t0 = time.perf_counter();run = a.run_dir.resolve()
    spec = O.load_json(a.round);problem = CLI.scoped(O.load_json(a.problem), spec.get('method_overrides'))
    name = spec['methods'][0];context = O.prepare(ROOT, problem, methods=[name]);tau = context['tau']
    routes = dict(problem['execution_routes']);routes.update(spec.get('routes_override', {}))
    O.apply_published(context['candidates'], routes)
    cands = [c for c in context['candidates'] if c['method'] == name];ready = {c['configuration_id']: c for c in cands if c['ready']}
    stopped = O.load_json(run / 'ROUND_RESULT_INTERRUPTED.json')['methods'][name]

    # Answers: completed jobs of the stopped round (validated) + C's saved points.
    ex = O.Executor(ROOT, run, routes, context['expected'], problem, launch_disabled=True).for_method(name)
    table, finished = {}, {}
    for req in sorted((run / 'jobs' / name).glob('*/REQUEST.json')):
        job = req.parent;q = json.loads(req.read_text())
        if not (job / 'OUTCOME.json').exists() or q['configuration_id'] not in ready:
            continue
        v = O.interleave(q['N_X'], q['N_Y']);r = ex.ingest(job, dict(configuration_id=q['configuration_id'], vector=v), calls=1)
        if r['status'] in (O.PASS, O.FAIL, 'INCONCLUSIVE'):
            table[(q['configuration_id'], v)] = dict(r, calls=1, source='LOGGED_ROUND_JOB')
            finished[job.name] = json.loads((job / 'OUTCOME.json').read_text()).get('finished_UTC')
    for (cid, v), r in context['saved'].table.items():
        if cid in ready:
            table[(cid, v)] = dict(r, calls=0)
    search = dict(problem['search'], max_steps_per_method=100000);search.update(spec.get('search_overrides', {}))

    # Old rule: trajectories of the stopped round, adoption time = max finish time of the consumed jobs so far.
    trace = stopped['trace'];jobs_by_finish = sorted(finished.items(), key=lambda kv: kv[1])
    saved_best = {}
    for (cid, v), r in context['saved'].table.items():
        if cid in ready and r['status'] == O.PASS:
            t = O.cost(v, ready[cid]['prices']);saved_best[cid] = min(saved_best.get(cid, t), t)
    events, clock = [], {}
    for t in trace:
        cid = t['configuration_id'];jn = Path(t['job']).name if t.get('job') else None
        clock[cid] = max(clock.get(cid, ''), finished.get(jn, '')) if jn else clock.get(cid, '')
        events.append(dict(cid=cid, time=clock[cid], status=t['status'], cost=O.frac(t['cost']), new=t['answer_source'] != 'SAVED'))
    events.sort(key=lambda e: e['time'])
    cfg_best = dict(saved_best);method_curve = [];rule_count = 0;finish_times = [f for _, f in jobs_by_finish]
    for e in events:
        rule_count += e['new']
        if e['status'] == O.PASS and e['cost'] < cfg_best.get(e['cid'], e['cost'] + 1):
            cfg_best[e['cid']] = e['cost']
        method_curve.append(dict(time=e['time'], old_rule=rule_count, old_actual=bisect.bisect_right(finish_times, e['time']),
                                 method_min=min(cfg_best.values()) if cfg_best else None))
    cfg_stop = {b['configuration_id']: O.frac(b['cheapest_feasible_evaluated']['cost']) for b in stopped['by_configuration']
                if b.get('cheapest_feasible_evaluated')}
    saved_min = min(O.frac(p['conditional_cost']) for p in context['catalog']['candidates']
                    if p['method'] == name and p['q3_status'] == O.PASS and p['conditional_cost'] is not None)
    best_all = min(cfg_stop.values())
    targets = sorted({saved_min, best_all * Fraction(101, 100), best_all * Fraction(1001, 1000), best_all}
                     | {v for v in cfg_stop.values() if v <= saved_min}, reverse=True)

    def run_new(overrides):
        s = dict(search, rule=P.RULE_V2, on_missing='SKIP', **overrides)
        knowledge = O.initial_knowledge(context['saved'], cands, 'all')
        res = P.run_priority(name, cands, knowledge, LogBackend(table), s, tau, context['generate'], None,
                             baseline=(problem['methods'][name].get('generation') or {}).get('baseline'),
                             free=lambda p: (p['configuration_id'], tuple(p['vector'])) in table, science_approved=True,
                             on_missing='SKIP', facts=context['facts'])
        curve, used = [], 0
        for t in res['trace']:
            used += t['calls']
            curve.append(dict(q3=used, method_min=t['method_min_after'], cid=t['configuration_id'], cls=t['cls'],
                              cfg_best=t['configuration_best_after']))
        return res, curve

    variants = [('NEW_DEFAULT', {})] + [(v.split('=', 1)[0], json.loads(v.split('=', 1)[1])) for v in a.variant]
    results = {}
    for vname, ov in variants:
        ts = time.perf_counter();res, curve = run_new(ov)
        reach = []
        start_min = min(saved_best.values()) if saved_best else None
        for T in targets:
            if start_min is not None and start_min <= T:  # already reached by C's saved points before any new Q3
                reach.append(dict(target=T, target_display=O.show(T), old_actual=0, old_rule=0, new=0, new_reached=True,
                                  new_unevaluated_before=0, note='SAVED_POINT'))
                continue
            o = first(method_curve, lambda r: r['method_min'] is not None and r['method_min'] <= T)
            n = first(curve, lambda r: r['method_min'] is not None and r['method_min'] <= T)
            reach.append(dict(target=T, target_display=O.show(T), old_actual=o and o['old_actual'], old_rule=o and o['old_rule'],
                              new=n and n['q3'], new_reached=n is not None,
                              new_unevaluated_before=None if n is None else sum(1 for u in res['unevaluated'] if u['step_index'] < curve.index(n) + 1)))
        per_cfg = []
        for cid, c in ready.items():
            stop_best = cfg_stop.get(cid)
            if stop_best is None:
                continue
            row = dict(candidate=c['name'], configuration_id=cid, L_D=c['key'].get('L_D'), kappa=c['key'].get('kappa_multiplier', '1'),
                       stop_best=stop_best, new_picks=sum(1 for t in res['trace'] if t['configuration_id'] == cid),
                       new_best=next((b['best']['cost'] for b in res['by_configuration'] if b['configuration_id'] == cid and b['best']), None))
            ctrace = [t for t in trace if t['configuration_id'] == cid]
            for lab, f in (('1pct', Fraction(101, 100)), ('0.1pct', Fraction(1001, 1000)), ('exact', Fraction(1))):
                T = stop_best * f;best = saved_best.get(cid);k_old = None;n_new_steps = 0
                for i, t in enumerate(ctrace):
                    if t['status'] == O.PASS:
                        best = min(best, O.frac(t['cost'])) if best is not None else O.frac(t['cost'])
                    if best is not None and best <= T:
                        k_old = sum(1 for x in ctrace[:i + 1] if x['answer_source'] != 'SAVED');break
                if saved_best.get(cid) is not None and saved_best[cid] <= T:
                    k_old = 0
                used = 0;k_new = None
                if saved_best.get(cid) is not None and saved_best[cid] <= T:
                    k_new = (0, 0)
                else:
                    for t in res['trace']:
                        used += t['calls']
                        if t['configuration_id'] == cid:
                            n_new_steps += t['calls']
                            if t['configuration_best_after'] is not None and t['configuration_best_after'] <= T:
                                k_new = (n_new_steps, used);break
                row['old_rule_steps_' + lab] = k_old
                row['new_configuration_q3_' + lab] = None if k_new is None else k_new[0]
                row['new_total_q3_' + lab] = None if k_new is None else k_new[1]
            per_cfg.append(row)
        results[vname] = dict(overrides=ov, stop_reason=res['stop_reason'], q3_used=sum(t['calls'] for t in res['trace']),
                              picks=len(res['trace']), unevaluated=len(res['unevaluated']), left=res['left'],
                              classes={k: sum(1 for t in res['trace'] if t['cls'] == k) for k in P.CLASS_ORDER},
                              final_method_min=res['trace'][-1]['method_min_after'] if res['trace'] else None,
                              reach=reach, per_configuration=per_cfg, input_requests=res['input_requests'],
                              settings=res['settings'], wall_seconds=time.perf_counter() - ts,
                              unevaluated_rows=res['unevaluated'])
        print(vname, 'picks', len(res['trace']), 'q3', results[vname]['q3_used'], 'unevaluated', len(res['unevaluated']),
              'final', results[vname]['final_method_min'] and O.show(results[vname]['final_method_min']), res['stop_reason'],
              '%.0fs' % results[vname]['wall_seconds'])
        for r in reach:
            print('   target', r['target_display'], 'old_actual', r['old_actual'], 'old_rule', r['old_rule'], 'new', r['new'],
                  'uneval_before', r['new_unevaluated_before'])
    old_total = dict(jobs=len(table) - len([1 for k in table if table[k]['calls'] == 0]),
                     rule_consumed=sum(1 for t in trace if t['answer_source'] != 'SAVED'),
                     lookahead_not_adopted=len(stopped['interrupted']['computed_not_adopted']))
    # A logged FAIL dominating a logged PASS (same configuration): the bound is not exactly monotone in shots.
    viol = {}
    for cid in ready:
        rows = [(v, r) for (c, v), r in table.items() if c == cid and r.get('MSE_upper') is not None]
        fails = [(v, r['MSE_upper'] / tau) for v, r in rows if r['status'] == O.FAIL]
        passes = [v for v, r in rows if r['status'] == O.PASS]
        for lab, m in (('any', Fraction(1)), ('margin_1e-4', Fraction(10001, 10000))):
            viol.setdefault(lab, 0)
            viol[lab] += sum(1 for p in passes for f, h in fails if h >= m and all(x <= y for x, y in zip(p, f)))
    out = dict(schema='RPE_PRIORITY_SELECTION_REPLAY_V1', rule_new=P.RULE_V2, rule_old=O.RULE + ' (incumbent_scope CONFIGURATION, lookahead 3)',
               round_id=spec['round_id'], tau=tau, answers_in_log=len(table), saved_PR_Pauli_minimum=saved_min,
               stop_time_cheapest_by_configuration={ready[c]['name']: v for c, v in cfg_stop.items()},
               old=old_total, monotonicity_violations=viol, variants=results, source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               wall_seconds=time.perf_counter() - t0,
               science='no Q3, price, signal or compile: answers are saved results only; UNEVALUATED rows are proposals with no logged answer')
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'REPLAY_COMPARISON.json').write_text(json.dumps(O.jsonable(out), indent=1, ensure_ascii=False) + '\n')
    with (a.output / 'REPLAY_REACH.csv').open('w', newline='') as f:
        w = csv.writer(f, lineterminator='\n');w.writerow(['variant', 'target_RZ', 'old_actual_Q3', 'old_rule_Q3', 'new_Q3', 'new_unevaluated_before'])
        for vname, r in results.items():
            for x in r['reach']:
                w.writerow([vname, x['target_display'], x['old_actual'], x['old_rule'], x['new'], x['new_unevaluated_before']])
    with (a.output / 'REPLAY_UNEVALUATED.csv').open('w', newline='') as f:
        w = csv.writer(f, lineterminator='\n');w.writerow(['variant', 'class', 'candidate', 'N_X', 'cost_RZ', 'origin', 'status'])
        for vname, r in results.items():
            for u in r['unevaluated_rows']:
                w.writerow([vname, u['cls'], u['candidate'], ' '.join(map(str, u['N_X'])), O.show(u['cost']), u['origin'].get('kind'), u['status']])
    print('violations', viol, 'answers', len(table), 'old', old_total, '%.0fs' % out['wall_seconds'])


if __name__ == '__main__':
    main()
