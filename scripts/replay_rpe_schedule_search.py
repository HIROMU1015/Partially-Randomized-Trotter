"""Saved-result replay of kappa / L_D / shot search rules with reveal-after-selection (no Q3, no signal, no price fit).

Every rule selects a schedule (outer rule) and an allocation (V2 inner rule) BEFORE its saved result is looked up; the answer
is a verified completed Q3 of rounds 01-04 (executor identity checks, as in build_rpe_q3_ledger.py) or C's saved points.
A selected schedule with no saved configuration, and a selected allocation with no saved answer, are counted as
UNDETERMINED (never assumed feasible or infeasible); the rule continues with its next choice.

Rules
  OLD   round04: V2 (lookahead 16, chain speculation) + schedule-aware kappa neighbourhood (centre 2, bounds [1/2, 8])
  NEW   SCHEDULE_WIDE_V1 (rpe_schedule_search) + V2 with coarse_gain_fraction (FINE class) + independent-slot lookahead
  NEW_CHAIN  the NEW rule with round04's chain speculation instead (slot-policy comparison)
Clock: a discrete-event model of the executor (``slots`` concurrent jobs, each job's recorded wall time, mean for jobs
without a record); speculative jobs occupy slots even when never adopted or not answerable.

    python replay_rpe_schedule_search.py --problem P --round R --runs RUN... --output DIR
"""
import argparse
import csv
from fractions import Fraction
import heapq
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from trottertracks import rpe_method_optimizer as O  # noqa: E402
from trottertracks import rpe_priority_search as P  # noqa: E402
from trottertracks import rpe_kappa_schedule as KS  # noqa: E402
from trottertracks import rpe_schedule_search as W  # noqa: E402
import rpe_round_tools as T  # noqa: E402

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)
NAME = 'PR_PAULI'


class Sim:
    """Executor clock: launch / prefetch / answer with slot limits and recorded durations."""

    def __init__(self, table, durations, slots, mean):
        self.table, self.dur, self.slots, self.mean = table, durations, slots, mean
        self.t, self.finish, self.busy, self.launched = 0.0, {}, [], 0
        self.unevaluated_selected = 0

    def _free(self):
        while self.busy and self.busy[0] <= self.t:
            heapq.heappop(self.busy)
        return self.slots - len(self.busy)

    def _launch(self, key):
        d = self.dur.get(key, self.mean)
        self.finish[key] = self.t + d;heapq.heappush(self.busy, self.t + d);self.launched += 1

    def prefetch(self, picks):
        for p in picks:
            key = (p['configuration_id'], tuple(p['vector']))
            if key in self.finish or (key in self.table and self.table[key]['calls'] == 0):
                continue
            if self._free() <= 0:
                break
            self._launch(key)

    def answer(self, choice):
        key = (choice['configuration_id'], tuple(choice['vector']));hit = self.table.get(key)
        if hit is None:
            self.unevaluated_selected += 1
            return None
        if hit['calls'] == 0:
            return dict(hit)
        if key not in self.finish:
            while self._free() <= 0:
                self.t = self.busy[0]
            self._launch(key)
        self.t = max(self.t, self.finish[key])
        return dict(hit)


def build_table(problem, context, routes, runs, out):
    cands = [c for c in context['candidates'] if c['method'] == NAME]
    table, durations = {}, {}
    for run in runs:
        run = (ROOT / run).resolve()
        ex = T.executor(problem, context, routes, out / 'executor_scratch', [str(run)]).for_method(NAME)
        known, _ = ex.prior_records(cands)
        walls = {}
        prog = run / 'PROGRESS.jsonl'
        if prog.exists():
            for line in prog.read_text().splitlines():
                e = json.loads(line)
                if e.get('event') == 'EXIT' and e.get('wall_seconds') is not None:
                    walls[e['job']] = float(e['wall_seconds'])
        for key, r in known.items():
            table[key] = dict(r, calls=1, source='SAVED_ROUND_JOB')
            if r.get('job'):
                fin = O.load_json(Path(r['job']) / 'OUTCOME.json').get('finished_UTC')
                table[key]['finished_UTC'] = fin;table[key]['run'] = run.parent.name
            j = Path(r['job']).name if r.get('job') else None
            if j in walls:
                durations[key] = walls[j]
    for key, r in context['saved'].table.items():
        if key[0] in {c['configuration_id'] for c in cands}:
            table[key] = dict(r, calls=0)
    return table, durations


def families_of(cands):
    """(L_D) -> Family from any generated delta=h configuration; every candidate's schedule R by its exact kappa."""
    fams, sched = {}, {}
    for c in cands:
        g = c.get('generated_configuration')
        if g and KS._frac(g['spec']['delta_pf']) == KS._frac(g['spec']['h']):
            fams.setdefault(c['key']['L_D'], W.Family(c['key']['L_D'], KS.constants(g)))
    for c in cands:
        ld = c['key'].get('L_D')
        if ld in fams and c.get('kappa_value') is not None:
            sched[c['configuration_id']] = (ld, fams[ld].R(Fraction(str(c['kappa_value']))), Fraction(str(c['kappa_value'])))
    return fams, sched


def run_rule(rule, cands, context, problem, spec, table, durations, slots, mean, opts):
    tau = context['tau'];fams, sched = families_of(cands);by_R = {}
    for cid, (ld, R, k) in sched.items():
        by_R.setdefault((ld, R), cid)
    ready = {c['configuration_id']: c for c in cands if c['ready']}
    sim = Sim(table, durations, slots, mean);knowledge = O.initial_knowledge(context['saved'], cands, 'all')
    search = dict(problem['search'], **spec.get('search_overrides', {}));search.pop('kappa_neighbourhood', None)
    search.update(opts.get('search', {}));search.update(max_steps_per_method=opts['batch'], on_missing='SKIP')
    nb = dict(spec['search_overrides']['kappa_neighbourhood'])
    saved_cids = {cid for (cid, v), r in context['saved'].table.items() if cid in ready}
    active = set(saved_cids)  # both rules start from C's saved configurations (round01 start)
    points, undetermined, curve, used, steps = {}, [], [], 0, 0
    if rule == 'OLD':
        for ld in range(1, 10):
            points[(ld, Fraction(2))] = dict(L_D=ld, kappa=Fraction(2), status='PENDING')
    trace_all, cycles = [], 0
    while cycles < opts['max_cycles']:
        cycles += 1
        for (ld, k), p in points.items():  # map selected schedules to saved configurations
            if p['status'] == 'PENDING':
                cid = by_R.get((ld, fams[ld].R(k))) if ld in fams else None
                if cid is None:
                    p['status'] = 'UNAVAILABLE';undetermined.append(dict(kind='SCHEDULE_WITHOUT_SAVED_CONFIGURATION', L_D=ld, kappa=str(k),
                                                                       cycle=cycles))
                else:
                    p['cid'] = cid;active.add(cid);p['status'] = 'IN_PROGRESS'
        cand_now = [c for c in cands if c['configuration_id'] in active]
        res = P.run_priority(NAME, cand_now, knowledge, sim, search, tau, context['generate'], None,
                             baseline=(problem['methods'][NAME].get('generation') or {}).get('baseline'), science_approved=True,
                             free=lambda p: table.get((p['configuration_id'], tuple(p['vector'])), {}).get('calls') == 0,
                             on_missing='SKIP', facts=context.get('facts'), prefetch=sim.prefetch)
        for t in res['trace']:
            used += t['calls'];steps += 1
            curve.append(dict(q3=used, launched=sim.launched, seconds=sim.t, method_min=t['method_min_after']))
        trace_all += res['trace'];undetermined += [dict(kind='ALLOCATION_WITHOUT_SAVED_ANSWER', candidate=u['candidate'], cycle=cycles)
                                                   for u in res['unevaluated']]
        by = {b['configuration_id']: b for b in res['by_configuration']}
        # outer plan
        if rule == 'OLD':
            pts = []
            for (ld, k), p in points.items():
                b = by.get(p.get('cid'));best = None if not b or not b['best'] else b['best']['cost']
                st = {'UNAVAILABLE': 'HELD', 'IN_PROGRESS': 'EVALUATED'}.get(p['status'], p['status'])
                pts.append(dict(L_D=ld, kappa=k, status=st, best=best, promising=b and b['promising'], picks=b and b['picks'], reason=None))
            sch = KS.Sched({str(ld): KS.encode_rows(f.rows) for ld, f in fams.items()}, nb['bounds'])
            proposals, decisions, _ = P.kappa_neighbourhood_plan(pts, nb, sch)
            new = [(q['L_D'], Fraction(q['kappa'])) for q in proposals]
        else:
            pts = []
            for cid in active:
                if cid not in sched:
                    continue
                ld, R, k = sched[cid];b = by.get(cid);best = None if not b or not b['best'] else b['best']['cost']
                lows = [r['MSE_lower'] / tau for (c, v), r in knowledge.items() if c == cid and r.get('MSE_lower') is not None]
                status = 'FEASIBLE' if best is not None else ('NO_FEASIBLE_FOUND' if b and not b['promising'] else 'IN_PROGRESS')
                pts.append(dict(family=ld, kappa=k, status=status, best=best, min_lower_over_tau=min(lows) if lows else None))
            for (ld, k), p in points.items():
                if p['status'] == 'UNAVAILABLE':
                    pts.append(dict(family=ld, kappa=k, status='UNAVAILABLE', best=None))
            proposals, decisions, report = W.plan_wide(fams, pts, opts.get('wide'))
            new = [(q['family'], q['kappa']) for q in proposals]
        added = 0
        for ld, k in new:
            if (ld, k) not in points:
                points[(ld, k)] = dict(L_D=ld, kappa=k, status='PENDING', reason=next(q['reason'] for q in proposals
                                                                                       if (q.get('L_D', q.get('family')), Fraction(q['kappa'])) == (ld, k)));added += 1
        if not added and res['stop_reason'] in ('ONLY_SMALL_OR_DEFERRED_LEFT', 'NO_CANDIDATE_LEFT_IN_RANGE', 'NO_FEASIBLE_POINT_IN_RANGE'):
            break
    # outcomes
    finished_pass = []
    for key, fin in sim.finish.items():
        r = table.get(key)
        if r is not None and r['status'] == O.PASS and key[0] in ready:
            finished_pass.append((O.cost(key[1], ready[key[0]]['prices']), key[0], fin))
    adopted_min = min((b['best']['cost'] for b in res['by_configuration'] if b['best']), default=None)
    examined = {t['configuration_id'] for t in trace_all}
    classes = {}
    for t in trace_all:
        classes[t['cls']] = classes.get(t['cls'], 0) + 1
    return dict(rule=rule, cycles=cycles, classes=classes, q3_answered=used, picks=steps, jobs_launched=sim.launched, seconds=sim.t,
                adopted_min=adopted_min, all_finished_min=min((x[0] for x in finished_pass), default=None),
                saved_points_min=min((O.cost(v, ready[c]['prices']) for (c, v), r in context['saved'].table.items()
                                      if r['status'] == O.PASS and c in ready), default=None),
                schedules_examined=len(examined), schedules_selected=len(points), undetermined_schedules=sum(
                    1 for u in undetermined if u['kind'] == 'SCHEDULE_WITHOUT_SAVED_CONFIGURATION'),
                undetermined_allocations=sum(1 for u in undetermined if u['kind'] == 'ALLOCATION_WITHOUT_SAVED_ANSWER'),
                stop=res['stop_reason'], curve=curve, points=[dict(L_D=ld, kappa=str(k), status=p['status'], reason=p.get('reason'),
                                                                  cid=p.get('cid')) for (ld, k), p in sorted(points.items())],
                undetermined=undetermined, report=None if rule == 'OLD' else report)


def reach(curve, target):
    for c in curve:
        if c['method_min'] is not None and c['method_min'] <= target:
            return dict(q3=c['q3'], launched=c['launched'], seconds=round(c['seconds']))
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--problem', type=Path, required=True);ap.add_argument('--round', type=Path, required=True)
    ap.add_argument('--runs', nargs='+', required=True);ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--slots', type=int, default=32);ap.add_argument('--batch', type=int, default=64)
    a = ap.parse_args();t0 = time.time();out = a.output.resolve();out.mkdir(parents=True, exist_ok=True)
    spec, problem, context, routes = T.round_context(a.problem, a.round, out / 'preparation')
    cands = [c for c in context['candidates'] if c['method'] == NAME]
    table, durations = build_table(problem, context, routes, a.runs, out)
    mean = sum(durations.values()) / len(durations)
    rules = dict(
        OLD=dict(rule='OLD', batch=128, max_cycles=60, search=dict(lookahead=16, adaptive_lookahead=True, lookahead_mode='CHAIN')),
        NEW=dict(rule='NEW', batch=a.batch, max_cycles=200, search=dict(lookahead=16, coarse_gain_fraction='1/100', lookahead_mode='INDEPENDENT',
                                                                       adaptive_lookahead=False)),
        NEW_CHAIN=dict(rule='NEW', batch=a.batch, max_cycles=200, search=dict(lookahead=16, coarse_gain_fraction='1/100',
                                                                             lookahead_mode='CHAIN', adaptive_lookahead=True)),
        NEW_HYBRID=dict(rule='NEW', batch=a.batch, max_cycles=200, search=dict(lookahead=16, coarse_gain_fraction='1/100',
                                                                              lookahead_mode='HYBRID', adaptive_lookahead=False)),
        OLD_HYBRID=dict(rule='OLD', batch=128, max_cycles=60, search=dict(lookahead=16, lookahead_mode='HYBRID', adaptive_lookahead=False)))
    results = {}
    for name, o in rules.items():
        ts = time.time();r = run_rule(o['rule'], cands, context, problem, spec, table, durations, a.slots, mean, o)
        r['wall_seconds'] = round(time.time() - ts, 1);results[name] = r
        print(name, 'q3', r['q3_answered'], 'launched', r['jobs_launched'], 'sim_min', round(r['seconds'] / 60), 'adopted_min',
              r['adopted_min'] and O.show(r['adopted_min']), 'all_finished_min', r['all_finished_min'] and O.show(r['all_finished_min']),
              'schedules', r['schedules_examined'], 'undet', r['undetermined_schedules'], r['undetermined_allocations'], r['stop'], flush=True)
    # what the rounds actually spent (all completed jobs, lookahead included, in finishing order)
    ready = {c['configuration_id']: c for c in cands if c['ready']};hist, best = [], min(
        (O.cost(v, ready[c]['prices']) for (c, v), r in context['saved'].table.items() if r['status'] == O.PASS and c in ready), default=None)
    for n, ((c, v), r) in enumerate(sorted(((k, r) for k, r in table.items() if r['calls']), key=lambda kv: kv[1].get('finished_UTC') or ''), 1):
        if r['status'] == O.PASS and c in ready:
            x = O.cost(v, ready[c]['prices']);best = x if best is None else min(best, x)
        hist.append(dict(q3=n, launched=n, seconds=0, method_min=best, run=r.get('run')))
    levels = sorted({Fraction(3508632333, 7), Fraction(3065260145, 7) * 1, Fraction(410199237), Fraction(410199237) * Fraction(101, 100)},
                    reverse=True)
    for r in results.values():
        r['reach'] = [dict(target=O.show(L), **(reach(r['curve'], L) or dict(q3=None))) for L in levels]
    history = [dict(target=O.show(L), **(dict((lambda h: dict(jobs=h['q3'], run=h['run']))(next(h for h in hist if h['method_min'] is not None
                                                                                                and h['method_min'] <= L)))
                                         if any(h['method_min'] is not None and h['method_min'] <= L for h in hist) else dict(jobs=None)))
               for L in levels]
    print('HISTORY', history)
    summary = dict(schema='RPE_SCHEDULE_SEARCH_REPLAY_V1', answers=len(table), answers_new_q3=sum(1 for v in table.values() if v['calls']),
                   answers_saved_points=sum(1 for v in table.values() if not v['calls']), slots=a.slots, mean_job_seconds=mean,
                   round04_measured=dict(jobs=284, adopted=82, v2_minutes=72, mean_in_flight=10.1),
                   history_rounds01_04=history,
                   rules={k: {x: y for x, y in v.items() if x != 'curve'} for k, v in results.items()},
                   source=__import__('subprocess').check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   wall_seconds=round(time.time() - t0, 1),
                   science='none: verified saved Q3 and C saved points; UNDETERMINED rows were selected but have no saved result')
    (out / 'REPLAY_SUMMARY.json').write_text(json.dumps(O.jsonable(summary), indent=1, ensure_ascii=False, default=str) + '\n')
    with (out / 'REPLAY_REACH.csv').open('w', newline='') as f:
        w = csv.writer(f, lineterminator='\n');w.writerow(['rule', 'target_RZ', 'q3_answered', 'jobs_launched', 'simulated_seconds'])
        for k, r in results.items():
            for x in r['reach']:
                w.writerow([k, x['target'], x.get('q3'), x.get('launched'), x.get('seconds')])
    for k, r in results.items():
        print(k, [(x['target'], x.get('q3'), x.get('launched'), x.get('seconds')) for x in r['reach']])


if __name__ == '__main__':
    main()
