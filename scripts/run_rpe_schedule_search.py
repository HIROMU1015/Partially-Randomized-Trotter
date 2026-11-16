"""Wide kappa / L_D / shot search for PR-Pauli (SCHEDULE_WIDE_V1), resumable; the user names no kappa, L_D or shot count.

One cycle
  1. plan: rpe_schedule_search.plan_wide over every (L_D, width) family -- schedules (exact kappa intervals), not kappa values;
     saved / evaluated schedules are reused, so no schedule is evaluated twice;
  2. inputs: a family with a prepared base goes through A's preparation entry (configuration), D2 PriceRules MODEL and the
     C/A common calibration-request entry (round04 driver); a schedule whose proven price floor (one shot per round and
     axis) is not below the verified best is pruned without science; a family without a prepared base (new L_D) goes to
     D2's completion API (``Completion.complete`` with the selected L_D, width and absolute kappa); a saved READY record of
     the same design is reused, never re-requested;
  3. signals (A entry, missing rounds only) -> B intake -> route registry (round04 driver);
  4. V2 batch (coarse-first FINE class, HYBRID slots) on all ready schedules; completed Q3 jobs are reused by job key;
  5. the verified-all best (every identity-checked completed job, adopted or not) and the rule-adopted best are both kept.
Stops when the plan proposes nothing, nothing is pending and V2 has no non-small candidate (bases per family in the state).

    python run_rpe_schedule_search.py --problem P --round R --run-dir D [--batch 64] [--plan-only]
``--plan-only`` writes the first decision (proposals, configurations, prices, price floors, completion requests) and stops
before any signal, calibration or Q3 (metadata only).
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from trottertracks import rpe_method_optimizer as O  # noqa: E402
from trottertracks import rpe_kappa_schedule as KS  # noqa: E402
from trottertracks import rpe_schedule_search as W  # noqa: E402
import run_rpe_kappa_neighbourhood as K  # noqa: E402

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)


def read_loc(loc):
    import subprocess
    x = json.loads(subprocess.check_output(['git', 'show', loc['commit'] + ':' + loc['path']], cwd=ROOT))
    for k in loc.get('selector', []):
        x = x[k]
    return x


class WideDriver(K.Driver):
    def __init__(self, a):
        super().__init__(a)
        self.wide = self.spec['search_overrides']['schedule_wide']
        if 'seeded' not in self.state:
            self.seed()

    # ------------------------------------------------------------ state
    def seed(self):
        """Points from the ledger (every configuration with verified saved Q3, incl. saved multiplier configurations) and the
        carried kappa state; family constants of prepared and new splits (exact, from canonical configurations)."""
        led = O.load_json(ROOT / self.spec['ledger_summary'])
        have = {(p['L_D'], Fraction(p['kappa'])) for p in self.state['points']}
        for cid, d in led['by_configuration'].items():
            if d.get('kappa') is None:
                continue
            k = O.frac(d['kappa']);ld = d['L_D']
            if (ld, k) not in have:
                self.state['points'].append(dict(L_D=ld, kappa=str(k), status='EVALUATED', configuration_id=cid, requested_cycle=0,
                                                 request_reason='SAVED_EVALUATED_SCHEDULE'));have.add((ld, k))
        for p in self.state['points']:
            d = led['by_configuration'].get(p.get('alias_of') or p.get('configuration_id') or '')
            if d:
                p['verified_all_best'] = d.get('verified_all');p['rule_adopted_best'] = d.get('rule_adopted')
        fc = self.state.setdefault('family_constants', {})
        for ld, loc in (self.spec.get('new_families') or {}).items():
            cfg = read_loc(loc['configuration']);cfg = cfg.get('configuration', cfg)
            rows = KS.constants(cfg)
            if KS.R_vector(rows, KS._frac(cfg['spec']['kappa']['value'])) != [r['R_outer'] for r in cfg['rounds']]:
                raise ValueError('MAPPER_DOES_NOT_REPRODUCE_NEW_SPLIT_' + str(ld))
            fc.setdefault(str(ld), KS.encode_rows(rows))
            self.state.setdefault('new_split_configurations', {})[str(ld)] = dict(loc['configuration'], configuration_id=cfg['id'])
        for p in self.state['points']:  # a new split is not 'waiting for P0' any more: D2's completion prepares it when selected
            if str(p['L_D']) in (self.spec.get('new_families') or {}) and p['status'] == 'HELD' and str(p.get('reason', '')).startswith('SPLIT_PREPARATION_REQUIRED'):
                p.update(status='COMPLETION', reopened=dict(previous=p.pop('reason'), by='D2_COMPLETION_API'))
        self.state['seeded'] = dict(UTC=K.now(), ledger=self.spec['ledger_summary'], new_families=sorted(self.spec.get('new_families') or {}))
        self.save()

    def families(self):
        return {int(k): W.Family(int(k), rows) for k, rows in self.state.get('family_constants', {}).items()}

    def wide_points(self):
        fams = self.families();out = []
        for p in self.state['points']:
            if p['L_D'] not in fams:
                continue
            best = [x for x in (p.get('best'), p.get('verified_all_best'), p.get('run_completed_best')) if x is not None]
            best = min(O.frac(x) for x in best) if best else None
            st = p['status']
            status = ('FEASIBLE' if best is not None else
                      'NO_FEASIBLE_FOUND' if st in ('EVALUATED', 'ALIAS') and not p.get('promising') else
                      'UNAVAILABLE' if st in ('HELD', 'PRUNED') else 'IN_PROGRESS' if st in ('EVALUATED', 'ALIAS') else 'PENDING')
            out.append(dict(family=p['L_D'], kappa=Fraction(p['kappa']), status=status, best=best, min_lower_over_tau=p.get('min_lower_over_tau')))
        return fams, out

    # ------------------------------------------------------------ plan
    def plan(self):
        fams, pts = self.wide_points()
        floors = sorted((Fraction(p['kappa']), p['L_D'], O.frac(p['price_floor'])) for p in self.state['points'] if p.get('price_floor') is not None)

        def price_floor(ld, R):  # prices grow with R: the floor of a quoted smaller-or-equal schedule bounds this one from below
            f = fams[ld];lb = [fl for k, l, fl in floors if l == ld and all(a <= b for a, b in zip(f.R(k), R))]
            return max(lb) if lb else None
        proposals, decisions, report = W.plan_wide(fams, pts, self.wide, price_floor)
        for q in proposals:
            if self.point(q['family'], q['kappa']) is None:
                new_split = str(q['family']) in (self.state.get('new_split_configurations') or {}) and not any(
                    p['L_D'] == q['family'] and p.get('configuration_id') for p in self.state['points'])
                self.state['points'].append(dict(L_D=q['family'], kappa=str(q['kappa']), status='COMPLETION' if new_split else 'PENDING',
                                                 requested_cycle=self.state['cycle'], request_reason=q['reason'],
                                                 request_evidence={k: v for k, v in q.items() if k not in ('family', 'kappa')}))
        self.state['history'].append(dict(cycle=self.state['cycle'], UTC=K.now(), rule=W.RULE, decisions=decisions, report=report,
                                          proposals=[dict(L_D=q['family'], kappa=str(q['kappa']), reason=q['reason']) for q in proposals]))
        return proposals, report

    # ------------------------------------------------------------ inputs
    def prune(self, ctx):
        """Proven price floor: an allocation of one shot per round and axis is the cheapest of the schedule."""
        fams, pts = self.wide_points();best = min((p['best'] for p in pts if p['status'] == 'FEASIBLE'), default=None)
        by = {c['configuration_id']: c for c in ctx['candidates']}
        for p in self.state['points']:
            c = by.get(p.get('configuration_id'))
            if c is None or c.get('prices') is None:
                continue
            p['price_floor'] = sum(c['prices'])
            if p['status'] == 'PREPARED' and best is not None and p['price_floor'] >= best:
                p.update(status='PRUNED', reason='PROVEN_COST_FLOOR:%s>=%s' % (O.show(p['price_floor']), O.show(best)))

    def completion_requests(self):
        """New splits: D2 completion (READY -> search, HELD -> recorded).  --plan-only records the request only."""
        todo = [p for p in self.state['points'] if p['status'] == 'COMPLETION']
        if not todo:
            return
        cc = self.spec.get('candidate_completion');imported = {}
        if cc and cc.get('resume_preparation_from'):
            for r in read_loc(cc['resume_preparation_from'])['records'].values():
                if r['status'] == 'READY_FOR_SEARCH' and r.get('design') and r.get('method') == 'PR_PAULI':  # never a PR-DF record
                    d = r['design'];imported[(d['L_D'], O.frac(d['kappa']), O.frac(d['delta_pf']))] = r
        run = []
        for p in todo:
            design = dict(L_D=p['L_D'], formula='2nd', delta_over_h='1', kappa=str(p['kappa']), K=2)
            p['completion_request'] = dict(method='PR_PAULI', selection=dict(rule=W.RULE, reason=p.get('request_reason'), cycle=p['requested_cycle']),
                                           design=design)
            saved = next((r for (ld, k, dl), r in imported.items() if ld == p['L_D'] and k == Fraction(p['kappa'])
                          and dl == Fraction(0.1)), None)  # same split, absolute kappa and width (delta_PF = h = binary64 0.1)
            if saved is not None:
                p.update(status='REGISTERED', configuration_id=saved['configuration_id'], completion=dict(status='REUSED_SAVED_READY_RECORD',
                                                                                                         record=saved['key']));continue
            if self.a.plan_only or not cc or self.spec.get('science_approved') is not True:
                p.update(status='HELD', reason='COMPLETION_REQUIRED_NOT_RUN_IN_THIS_INVOCATION');continue
            run.append(p)
        if not run:
            return
        comp = self.completion_api(dict(cc, max_attempted_per_method=0))
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max(1, min(len(run), int(cc.get('parallel', 4))))) as pool:  # independent splits / schedules
            for p, rec in zip(run, pool.map(comp.complete, [p['completion_request'] for p in run])):
                if rec['status'] == 'READY_FOR_SEARCH':
                    p.update(status='REGISTERED', configuration_id=rec['configuration_id'], completion=dict(status=rec['status'], key=rec['key']))
                else:  # HELD: the concrete reason is kept and the other candidates go on
                    p.update(status='HELD', reason='COMPLETION:%s:%s' % (rec['status'], str(rec.get('reason'))[:300]),
                             completion=dict(status=rec['status'], key=rec['key']))

    def completion_api(self, settings):
        from trottertracks.pr_candidate_completion import Completion
        return Completion(self.run / 'completion', settings)

    def write_points(self):
        """A's preparation entry gets only schedules of splits with a prepared base (new splits go through D2 completion)."""
        rows = [dict(L_D=p['L_D'], kappa=p['kappa']) for p in self.state['points']
                if p['status'] not in ('HELD', 'PRUNED', 'COMPLETION') and not p.get('completion')]
        self.points_file.parent.mkdir(parents=True, exist_ok=True);self.points_file.write_text(json.dumps(rows, indent=1) + '\n')

    def absorb(self, res):
        """Rule-adopted bests (V2) plus the cheapest feasible completed job of this run per configuration, adopted or not
        (provisional for planning; the ledger re-verifies every job with the executor's identity checks before reporting)."""
        super().absorb(res)
        done = {}
        for q in (self.run / 'jobs' / 'PR_PAULI').glob('*/REQUEST.json') if (self.run / 'jobs' / 'PR_PAULI').exists() else []:
            job = q.parent
            if not (job / 'OUTCOME.json').exists():
                continue
            out, _ = O.job_outcome(job);r = json.loads(q.read_text())
            if out.get('status') == O.PASS:
                c = O.frac(r['cost']);cid = r['configuration_id']
                done[cid] = min(done.get(cid, c), c)
        for p in self.state['points']:
            cid = p.get('alias_of') if p['status'] == 'ALIAS' else p.get('configuration_id')
            if cid in done:
                p['run_completed_best'] = done[cid]

    # ------------------------------------------------------------ loop
    def batch(self):
        spec = json.loads(json.dumps(self.spec))
        if spec.get('candidate_completion'):
            spec['candidate_completion']['max_attempted_per_method'] = 0  # install READY records only; B2 chooses what to complete
        self.spec, keep = spec, self.spec
        try:
            return super().batch()
        finally:
            self.spec = keep

    def loop(self):
        while self.state['cycle'] < self.a.max_cycles:
            self.state['cycle'] += 1;self.save()
            proposals, report = self.plan();self.save()
            self.completion_requests();self.save()
            stop_here = self.a.plan_only or self.spec.get('science_approved') is not True
            if any(p['status'] in ('PENDING', 'PREPARED') for p in self.state['points']):
                ctx = self.prepare();self.prune(ctx);self.save()
                if stop_here:
                    self.state['stop'] = dict(reason='PLAN_ONLY_FIRST_DECISION_WRITTEN' if self.a.plan_only else
                                              'SCIENCE_NOT_APPROVED_FIRST_DECISION_WRITTEN', UTC=K.now());self.save();return
                self.signals();self.save()
            elif stop_here:
                self.state['stop'] = dict(reason='PLAN_ONLY_NOTHING_TO_PREPARE', UTC=K.now());self.save();return
            code, res = self.batch()
            if res is None:
                self.state['stop'] = dict(reason='V2_INVOCATION_FAILED', returncode=code, UTC=K.now());self.save();return
            self.absorb(res);self.save()
            pending = [p for p in self.state['points'] if p['status'] in ('PENDING', 'PREPARED', 'REGISTERED', 'COMPLETION')]
            promising = sum(p.get('promising') or 0 for p in self.state['points'] if p['status'] == 'EVALUATED')
            self.log(event='CYCLE_END', v2_stop=res['stop_reason'], proposals=len(proposals), pending=len(pending), promising=promising,
                     plan_stop=report['stop'])
            if res['stop_reason'] in ('UNKNOWN_RESULT', 'STALE_JOB_NEEDS_REVIEW', 'NO_ROUTE_ADAPTER'):
                self.state['stop'] = dict(reason='V2_' + res['stop_reason'], UTC=K.now());self.save();return
            if not proposals and not pending and res['stop_reason'] in ('ONLY_SMALL_OR_DEFERRED_LEFT', 'NO_CANDIDATE_LEFT_IN_RANGE'):
                self.state['stop'] = dict(reason='SCHEDULE_WIDE_RULE_EXHAUSTED', UTC=K.now(), v2_stop=res['stop_reason'],
                                          families=report['families']);self.save();return
        self.state['stop'] = dict(reason='MAX_CYCLES', UTC=K.now());self.save()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--problem', type=Path, required=True);ap.add_argument('--round', type=Path, required=True)
    ap.add_argument('--run-dir', type=Path, required=True);ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--max-cycles', type=int, default=200);ap.add_argument('--plan-only', action='store_true')
    a = ap.parse_args();a.problem = a.problem.resolve()
    d = WideDriver(a);d.log(event='DRIVER_START', plan_only=a.plan_only);d.loop();d.log(event='DRIVER_END', stop=d.state.get('stop'))
    print(json.dumps(O.jsonable(d.state.get('stop')), default=str)[:2000])


if __name__ == '__main__':
    main()
