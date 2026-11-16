"""Checks of SCHEDULE_WIDE_V1, the V2 FINE class / slot modes and the D2 completion step (artificial inputs; no science).

  wide_init        no centre: floor schedule + far ladder; kappa=2 / [1/2, 8] are not special; saved schedules are not re-proposed
  wide_refine      after the ladder: refinement towards the cheaper neighbour, moved to an unsampled schedule; then resolved
  wide_stops       proven price floor limits the range; empirical upper stop; NO_FEASIBLE_FOUND is not a proof (reported only)
  new_split_rows   family constants from lambda (no configuration) reproduce the canonical LD4/6/7/8/9 configurations' R
  v2_fine          with coarse_gain_fraction a second configuration's coarse picks come before the first one's fine picks; no pick is
                   dropped; without it the rule is unchanged (same trace as the committed V2)
  v2_slot_modes    INDEPENDENT / HYBRID / CHAIN prefetch different jobs but consume the same answers in the same order
  completion       driver step with an artificial Completion: READY -> registered, HELD -> recorded and the others continue, a saved
                   READY record of the same design (D2 LD4 kappa=2) is reused without calling complete(), D2's PR-DF LD6 record is not
                   taken for PR-Pauli LD6, a second pass calls nothing
Writes SCHEDULE_SEARCH_CHECKS.json into --output.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from trottertracks import rpe_kappa_schedule as KS  # noqa: E402
from trottertracks import rpe_method_optimizer as O  # noqa: E402
from trottertracks import rpe_priority_search as P  # noqa: E402
from trottertracks import rpe_schedule_search as W  # noqa: E402

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)
ST = ROOT / 'docs/research/rpe_method_optimizer_h4_20261116/kappa_abs_round04/run01/KAPPA_STATE.json'
BASE = Fraction('13112614441153360648422410673503886836581176429267589536059228160/539496659426909122096437389293855970805348475396548904610992201')


def git_json(commit, path):
    x = json.loads(subprocess.check_output(['git', 'show', commit + ':' + path], cwd=ROOT))
    return x.get('configuration', x)


def main():
    ap = argparse.ArgumentParser(description=__doc__);ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args();a.output.mkdir(parents=True, exist_ok=True);res = {}
    fc = json.loads(ST.read_text())['family_constants'];f5 = W.Family(5, fc['5']);f3 = W.Family(3, fc['3'])
    F = lambda k, b=None, st='FEASIBLE', fam=5: dict(family=fam, kappa=Fraction(k), status=st if b is not None or st != 'FEASIBLE' else 'IN_PROGRESS',
                                                    best=None if b is None else Fraction(b))
    # init
    pts = [F(2, 501), F(BASE, 437), F(2 * BASE, 410)]
    props, dec, rep = W.plan_wide({5: f5}, pts)
    ks = [p['kappa'] for p in props]
    res['wide_init'] = dict(ok=[p['reason'] for p in props] == ['INIT_FAR_SCHEDULE'] * 5 and ks == [97, 406, 1643, 6589, 26373]
                            and all(f5.R(k) not in {f5.R(p['kappa']) for p in pts} for k in ks), proposals=[str(k) for k in ks],
                            floor_end=float(f5.floor_end))
    # refine: ladder answered (97 dearer than 48.6 by > 5 %, 406 dearer again) -> refine between 48.6 and 97 / 24.3 ; upper stop
    pts2 = pts + [F(97, 470), F(406, 520), F(1643, None, 'NO_FEASIBLE_FOUND'), F(6589, None, 'NO_FEASIBLE_FOUND'), F(26373, None, 'UNAVAILABLE')]
    p2, d2, r2 = W.plan_wide({5: f5}, pts2)
    ref = [p for p in p2 if p['reason'].startswith('REFINE')]
    ok_ref = bool(ref) and all(f5.R(p['kappa']) not in {f5.R(q['kappa']) for q in pts2} for p in p2) and \
        (BASE < ref[0]['kappa'] < 2 * BASE or 2 * BASE < ref[0]['kappa'] < 97)
    # resolution: neighbours adjacent -> no refinement
    nb_lo = f5.next(2 * BASE, 'down');nb_hi = f5.next(2 * BASE, 'up')
    pts3 = [F(2, 501), F(nb_lo, 420), F(2 * BASE, 410), F(nb_hi, 415)]
    p3, d3, r3 = W.plan_wide({5: f5}, pts3, dict(ladder_ratio='65536', max_kappa='60'))
    res['wide_refine'] = dict(ok=ok_ref and r3['families'][5]['refinement'] == 'RESOLVED' and not any(p['reason'].startswith('REFINE') for p in p3),
                              refine=[(p['reason'], str(p['kappa'])) for p in ref], after_adjacent=[(p['reason'], str(p['kappa'])) for p in p3],
                              upper=dict(limit=str(r2['families'][5]['upper_limit']), basis=r2['families'][5]['upper_basis']))
    # stops: proven price floor (artificial floor = events) and empirical upper stop
    floor = lambda fam, R: Fraction(f5.events(R)) * 1000  # artificial price rule: 1000 RZ per event, one shot per round and axis
    ptsM = [F(2, 501 * 10 ** 6), F(BASE, 437 * 10 ** 6), F(2 * BASE, 410 * 10 ** 6)]
    p4, d4, r4 = W.plan_wide({5: f5}, ptsM, None, floor)
    lim = r4['families'][5]['upper_limit']
    res['wide_stops'] = dict(ok=r4['families'][5]['upper_basis'] == 'PROVEN_COST_FLOOR' and floor(5, f5.R(lim)) >= 410 * 10 ** 6
                             and all(p['kappa'] <= lim for p in p4) and [str(p['kappa']) for p in p4 if p['reason'] == 'INIT_FAR_SCHEDULE'] == ['97', '406']
                             and r2['families'][5]['upper_basis'] == 'EMPIRICAL_UPPER_STOP',
                             proposals_with_floor=[(p['reason'], str(p['kappa'])) for p in p4],
                             proven_limit=str(r4['families'][5]['upper_limit']), empirical=r2['families'][5]['upper_basis'],
                             no_feasible_found_is_not_proof='NO_FEASIBLE_FOUND points are sampled schedules; nothing is marked infeasible for all allocations')
    # new split rows from lambda reproduce canonical configurations
    locs = {4: ('8690e962b', 'docs/research/d_pr_candidates_auto_20261116/run02/completion/candidates/3b127c39973a21be33a0/CONFIGURATION.json'),
            6: ('d7a976a629a6db68197b24f8f86734e109c8bf5d', 'docs/research/rpe_split_preparation_20261116/LD6/PREPARATION.0.CONFIGURATION.json')}
    for ld in (7, 8, 9):
        locs[ld] = ('c034dfa02', 'docs/research/pauli_p0_no_drop_preparation_20261116/LD%d/PREPARATION.0.CONFIGURATION.json' % ld)
    rows_ok = {}
    for ld, (c, p) in locs.items():
        cfg = git_json(c, p);s = cfg['spec']
        mine = W.family_rows(KS._frac(s['split']['lambda_R']), KS._frac(s['h']), KS._frac(s['delta_pf']))
        rows_ok[ld] = [(r['c'], r['q']) for r in mine] == [(r['c'], r['q']) for r in KS.constants(cfg)] and \
            KS.R_vector(mine, KS._frac(s['kappa']['value'])) == [r['R_outer'] for r in cfg['rounds']]
    fams = {ld: W.Family(ld, KS.constants(git_json(*locs[ld]))) for ld in locs}
    res['new_split_rows'] = dict(ok=all(rows_ok.values()), per_split=rows_ok,
                                 floor_end={ld: float(f.floor_end) for ld, f in fams.items()})
    # V2 FINE class and slot modes on artificial configurations (as in check_rpe_priority_search)
    import check_rpe_priority_search as CP
    capsum = sum(w * Fraction(1, n) for w, n in zip(CP.W, CP.CAPS))
    cands = [CP.config('A', 5, '1', 1), CP.config('B', 5, '2', Fraction(11, 10))]
    f = CP.mse_fn({'A': Fraction(1, 2) / capsum, 'B': Fraction(3, 5) / capsum})
    kn0 = CP.seeded(cands, f)

    class Back:
        def __init__(self):
            self.prefetched = []

        def answer(self, p):
            lo, up = f(p['configuration_id'], p['vector'])
            return dict(status=O.PASS if up <= CP.TAU else O.FAIL, MSE_lower=lo, MSE_upper=up, calls=1, source='ARTIFICIAL')

    def run(search, prefetch=None):
        kn = dict(kn0);b = Back()
        r = P.run_priority('PR_PAULI', [dict(c) for c in cands], kn, b, dict(CP.SEARCH, **search), CP.TAU, CP.none_generator, 400,
                           prefetch=prefetch, on_missing='STOP', baseline=dict(formula='2nd', delta_over_h='1', kappa_multiplier='1'))
        return r
    base = run(dict(rule=P.RULE_V2, max_steps_per_method=400))
    fine = run(dict(rule=P.RULE_V2, max_steps_per_method=400, coarse_gain_fraction='1/100'))
    first_fine = next((i for i, t in enumerate(fine['trace']) if t['cls'] == 'FINE'), None)
    coarse_after = [t for t in fine['trace'][first_fine:] if t['cls'] in ('UPDATE', 'EXPLORE')] if first_fine is not None else []
    import importlib
    H = types.ModuleType('ps_head');exec(compile(subprocess.check_output(['git', 'show', 'HEAD:src/trottertracks/rpe_priority_search.py'], cwd=ROOT,
                                                                         text=True), 'head', 'exec'), H.__dict__)
    head = H.run_priority('PR_PAULI', [dict(c) for c in cands], dict(kn0), Back(), dict(CP.SEARCH, rule=P.RULE_V2, max_steps_per_method=400), CP.TAU,
                          CP.none_generator, 400, on_missing='STOP', baseline=dict(formula='2nd', delta_over_h='1', kappa_multiplier='1'))
    same_without = [(t['configuration_id'], t['vector']) for t in base['trace']] == [(t['configuration_id'], t['vector']) for t in head['trace']]
    b_before = any(t['configuration_id'] == 'B' for t in fine['trace'][:first_fine]) if first_fine is not None else False
    res['v2_fine'] = dict(ok=first_fine is not None and b_before and same_without and fine['left'].get('FINE', 0) == 0
                          and fine['stop_reason'] == base['stop_reason']
                          and fine['trace'][-1]['method_min_after'] <= base['trace'][-1]['method_min_after'],
                          second_configuration_before_first_fine=b_before, stop=(fine['stop_reason'], base['stop_reason']),
                          classes={k: sum(1 for t in fine['trace'] if t['cls'] == k) for k in P.CLASS_ORDER},
                          final_min_fine=O.show(fine['trace'][-1]['method_min_after']), final_min_base=O.show(base['trace'][-1]['method_min_after']),
                          unchanged_without_fine=same_without)
    orders = {}
    for mode in ('INDEPENDENT', 'HYBRID', 'CHAIN'):
        got = []
        r = run(dict(rule=P.RULE_V2, max_steps_per_method=400, coarse_gain_fraction='1/100', lookahead=6, lookahead_mode=mode,
                     adaptive_lookahead=mode == 'CHAIN'), prefetch=lambda ps, got=got: got.extend((p['configuration_id'], p['vector']) for p in ps))
        orders[mode] = ([(t['configuration_id'], t['vector']) for t in r['trace']], len(set(got)))
    res['v2_slot_modes'] = dict(ok=orders['INDEPENDENT'][0] == orders['HYBRID'][0] == orders['CHAIN'][0],
                                prefetched_distinct={k: v[1] for k, v in orders.items()}, picks=len(orders['CHAIN'][0]))
    # completion step of the driver with an artificial Completion
    import run_rpe_schedule_search as RS
    spec = json.loads((ROOT / 'configs/rpe_method_optimizer/rounds/h4_pr_pauli_schedule_wide_round05.json').read_text())
    spec['science_approved'] = True
    calls = []

    class FakeCompletion:
        def complete(self, req):
            calls.append(req['design'])
            d = req['design']
            if d['L_D'] == 4:
                raise AssertionError('saved READY record must be reused')
            if d['L_D'] == 7:
                return dict(status='HELD_WITH_REASON', reason='PAULI_PRICE:ARTIFICIAL_HOLD', key='k7')
            return dict(status='READY_FOR_SEARCH', configuration_id='ARTIFICIAL_LD%d' % d['L_D'], key='k%d' % d['L_D'])
    drv = object.__new__(RS.WideDriver);drv.spec = spec;drv.run = a.output / 'completion_check';drv.run.mkdir(parents=True, exist_ok=True);drv.a = types.SimpleNamespace(plan_only=False)
    drv.completion_api = lambda settings: FakeCompletion()
    drv.state = dict(cycle=1, points=[dict(L_D=4, kappa='2', status='COMPLETION', requested_cycle=1, request_reason='CARRIED'),
                                      dict(L_D=6, kappa='2', status='COMPLETION', requested_cycle=1, request_reason='CARRIED_PR_DF_RECORD_EXISTS'),
                                      dict(L_D=7, kappa='1', status='COMPLETION', requested_cycle=1, request_reason='INIT_FLOOR_SCHEDULE'),
                                      dict(L_D=8, kappa='1', status='COMPLETION', requested_cycle=1, request_reason='INIT_FLOOR_SCHEDULE')])
    drv.completion_requests()
    st = {p['L_D']: (p['status'], p.get('configuration_id'), (p.get('completion') or {}).get('status'), p.get('reason')) for p in drv.state['points']}
    first_calls = len(calls);drv.completion_requests()
    res['completion'] = dict(ok=st[4][0] == 'REGISTERED' and st[4][2] == 'REUSED_SAVED_READY_RECORD' and st[4][1].startswith('8bf6eb88')
                             and st[6][0] == 'REGISTERED' and st[6][1] == 'ARTIFICIAL_LD6' and st[7][0] == 'HELD' and 'ARTIFICIAL_HOLD' in st[7][3] and st[8][0] == 'REGISTERED'
                             and first_calls == 3 and len(calls) == 3,
                             statuses={k: v[:3] for k, v in st.items()}, complete_calls_first_pass=first_calls, complete_calls_second_pass=len(calls) - first_calls)
    res['ok'] = all(v['ok'] for v in res.values() if isinstance(v, dict))
    (a.output / 'SCHEDULE_SEARCH_CHECKS.json').write_text(json.dumps(O.jsonable(res), indent=1, default=str) + '\n')
    print(json.dumps({k: v['ok'] for k, v in res.items() if isinstance(v, dict)}), 'ALL PASS' if res['ok'] else 'FAIL')
    return 0 if res['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
