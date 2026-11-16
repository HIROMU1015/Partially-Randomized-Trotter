"""Acceptance of METHOD_MINIMUM_PRIORITY_V2 on artificial configurations (labelled ARTIFICIAL; no catalog, no Q3).

Checks
  reaches the same method minimum as V1 (configuration scope) and needs fewer Q3 to get there
  a deferred allocation (dominated by a far failure) is still evaluated when nothing else is left, and may pass
  every explore_every-th pick goes to another configuration; a fresh configuration is explored before far ones
  a configuration without a feasible point gets the nearest kappa's feasible allocations as transfer seeds
  kappa plan: edge minimum -> geometric extension, interior -> midpoints (cheaper side first), bounds / min ratio respected
  saved-log replay: a proposal without an answer is UNEVALUATED, not counted, and the search continues
Writes PRIORITY_ACCEPTANCE.json into --output.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import rpe_method_optimizer as O  # noqa: E402
from trottertracks import rpe_priority_search as P  # noqa: E402

TAU = Fraction(1)
SEARCH = dict(decrement_steps=[1, 2], large_step_headroom='9/10', seed_rules=[], max_steps_per_method=100000)
CAPS = [12, 11, 10, 9, 8, 7, 6, 5, 4, 4, 3, 3, 2]
W = [Fraction((m + 1) ** 2) for m in range(13)]


def config(cid, ld, kappa, price_scale):
    prices = [Fraction(price_scale) * (m + 1) for m in range(13) for _ in O.AXES]
    key = dict(L_D=ld) if kappa == '1' else dict(L_D=ld, formula='2nd', delta_over_h='1', kappa_multiplier=kappa)
    return dict(method='PR_PAULI', family='PR', key=key, configuration_id=cid, ready=True, prices=prices, caps=CAPS,
                name='PR_PAULI:%s:%s' % (ld, cid), needs=[], labels=[], price_status='ARTIFICIAL', signal_source='ARTIFICIAL')


def mse_fn(scales, kinks=None):
    """Artificial MSE: scale_c * sum_m w_m / N_m (X and Y averaged); ``kinks`` override single allocations."""
    def f(cid, v):
        nx, ny = O.split(v)
        if kinks and (cid, tuple(v)) in kinks:
            x = kinks[(cid, tuple(v))]
            return x, x
        x = scales[cid] * sum(w * (Fraction(1, a) + Fraction(1, b)) / 2 for w, a, b in zip(W, nx, ny))
        return x, x
    return f


def seeded(cands, f):
    """Each configuration starts from its cap vector as a saved feasible point (when feasible)."""
    kn = {}
    for c in cands:
        v = tuple(n for n in CAPS for _ in O.AXES);lo, up = f(c['configuration_id'], v)
        kn[(c['configuration_id'], v)] = dict(status=O.PASS if up <= TAU else O.FAIL, MSE_lower=lo, MSE_upper=up, source='SAVED', calls=0,
                                              cost=O.cost(v, c['prices']))
    return kn


def none_generator(*args):
    return None


def check(name, ok, **detail):
    return dict(check=name, status='PASS' if ok else 'FAIL', **detail)


def q3_to(trace, target):
    used = 0
    for t in trace:
        used += t['calls']
        m = t.get('method_min_after', t.get('incumbent_after'))
        if m is not None and m <= target:
            return used
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__);ap.add_argument('--output', type=Path, required=True)
    out = ap.parse_args().output;out.mkdir(parents=True, exist_ok=True);checks = []
    base = (sum(w * Fraction(1, n) for w, n in zip(W, CAPS)))
    scales = {'A': Fraction(1, 2) / base, 'B': Fraction(3, 5) / base, 'C': Fraction(7, 10) / base}
    cands = [config('A', 5, '2', 1), config('B', 5, '1', Fraction(11, 10)), config('C', 3, '1', 2)]
    f = mse_fn(scales)

    # V1 configuration scope (exhaustive per configuration) vs V2.
    v1 = O.run_by_configuration('PR_PAULI', [dict(c) for c in cands], seeded(cands, f), O.FixtureBackend(f, TAU),
                                dict(SEARCH, incumbent_scope='CONFIGURATION'), TAU, none_generator, None)
    v1_min = v1['incumbent']['cost']
    # V1's real spend: the configurations' traces interleaved (they ran in parallel); count all rule steps to reach the min.
    v1_steps = sum(len([t for t in v1['trace'] if t['configuration_id'] == c['configuration_id']]) for c in cands)
    v2 = P.run_priority('PR_PAULI', [dict(c) for c in cands], seeded(cands, f), O.FixtureBackend(f, TAU), SEARCH, TAU, none_generator, None,
                        baseline=dict(formula='2nd', delta_over_h='1', kappa_multiplier='1'))
    v2_min = v2['incumbent']['cost'];v2_to = q3_to(v2['trace'], v1_min)
    lead = [t for t in v1['trace'] if t['configuration_id'] == v1['incumbent']['candidate']['configuration_id']]
    lead_to = next(i + 1 for i, t in enumerate(lead) if t['incumbent_after'] is not None and t['incumbent_after'] <= v1_min)
    checks.append(check('v2_same_method_minimum_fewer_q3_than_v1', v2_min <= v1_min and v2_to is not None and v2_to < v1_steps,
                        v1_min=O.show(v1_min), v2_min=O.show(v2_min), v2_q3_to_v1_min=v2_to,
                        v1_all_steps=v1_steps, v1_leading_configuration_steps_to_min=lead_to, v2_stop=v2['stop_reason']))

    # Deferred allocation: incumbent u (MSE tau/2), far failure f = u - e_m11 (MSE 2 tau); u - 2 e_m11 is dominated by f but
    # passes (artificial non-monotone kink); every other allocation fails.  It is evaluated once nothing else is left.
    one = [config('A', 5, '2', 1)];u = tuple(n for n in CAPS for _ in O.AXES)
    fv = list(u);fv[22] -= 1;fv[23] -= 1;vv = list(u);vv[22] -= 2;vv[23] -= 2;fv, vv = tuple(fv), tuple(vv)

    def fk(cid, v):
        x = Fraction(1, 2) if tuple(v) in (u, vv) else Fraction(2)
        return x, x
    kn = {('A', u): dict(status=O.PASS, MSE_lower=Fraction(1, 2), MSE_upper=Fraction(1, 2), source='SAVED', calls=0),
          ('A', fv): dict(status=O.FAIL, MSE_lower=Fraction(2), MSE_upper=Fraction(2), source='SAVED', calls=0)}
    d = P.run_priority('PR_PAULI', one, kn, O.FixtureBackend(fk, TAU), SEARCH, TAU, none_generator, None, baseline={'kappa_multiplier': '1'})
    hit = [t for t in d['trace'] if tuple(t['vector']) == vv]
    later = [t for t in d['trace'] if hit and t['step'] < hit[0]['step']]
    checks.append(check('deferred_allocation_evaluated_when_nothing_else_left', hit and hit[0]['cls'] == 'DEFERRED' and hit[0]['status'] == O.PASS
                        and all(t['cls'] != 'DEFERRED' for t in later) and d['incumbent']['vector'] == vv
                        and d['stop_reason'] == 'NO_CANDIDATE_LEFT_IN_RANGE' and d['left'] == {k: 0 for k in P.CLASS_ORDER},
                        cls=hit and hit[0]['cls'], pick_status=hit and hit[0]['status'], position=hit and hit[0]['step'], picks=len(d['trace'])))

    # Exploration share: with explore_every=4 another configuration gets every 4th pick while UPDATE picks exist;
    # a fresh configuration (no feasible point) is explored before far ones.
    fresh = config('D', 5, '4', 2);cands2 = [dict(c) for c in cands] + [fresh]  # dearer prices: its seeds cost more than M
    scales2 = dict(scales, D=Fraction(1, 2) / base);f2 = mse_fn(scales2)
    kn2 = seeded(cands, f2)
    a1 = list(tuple(n for n in CAPS for _ in O.AXES));a1[0] -= 1;a1[1] -= 1;a1 = tuple(a1)  # A also holds a cheaper feasible point
    lo, up_ = f2('A', a1);kn2[('A', a1)] = dict(status=O.PASS if up_ <= TAU else O.FAIL, MSE_lower=lo, MSE_upper=up_, source='SAVED', calls=0)
    e = P.run_priority('PR_PAULI', cands2, kn2, O.FixtureBackend(f2, TAU), dict(SEARCH, max_steps_per_method=24), TAU, none_generator, None,
                       baseline=dict(formula='2nd', delta_over_h='1', kappa_multiplier='1'))
    cls = [t['cls'] for t in e['trace']];first_explore = next((t for t in e['trace'] if t['cls'] == 'EXPLORE'), None)
    windows = [cls[i:i + 4] for i in range(0, 20, 4)]
    checks.append(check('exploration_share_and_fresh_configuration_first', all('EXPLORE' in w for w in windows)
                        and first_explore is not None and first_explore['configuration_id'] == 'D',
                        classes=cls, first_explore=first_explore and first_explore['configuration_id']))
    transfer = [t for t in e['trace'] if t['configuration_id'] == 'D' and t['origin'].get('kind') == 'TRANSFER_FROM_NEAREST_KAPPA']
    nearest = sorted((O.cost(k[1], fresh['prices']), k[1]) for k in kn2 if k[0] == 'A' and kn2[k]['status'] == O.PASS)
    checks.append(check('transfer_seed_from_nearest_kappa', transfer and transfer[0]['origin']['from_configuration'] == 'A'
                        and tuple(transfer[0]['vector']) in {v for _, v in nearest},
                        transfer=transfer and transfer[0]['origin']))

    # Kappa plan.
    class S:
        def __init__(self, cost):
            self.best = None if cost is None else (Fraction(cost), (), {})
    s = P.settings({})
    def plan(costs):
        cs = [config('k%s' % k.replace('/', '_'), 5, k, 1) for k in costs]
        st = {c['configuration_id']: S(costs[k]) for c, k in zip(cs, costs)}
        return [(r['kappa_multiplier'], r['reason'], r['kind']) for r in P.kappa_plan(cs, st, s, dict(formula='2nd', delta_over_h='1', kappa_multiplier='1'), {})]
    up = plan({'2/3': 30, '1': 20, '2': 10});down = plan({'2/3': 10, '1': 20, '2': 30});mid = plan({'2/3': 20, '1': 10, '2': 30})
    fine = plan({'1': 10, '17/16': 20, '9/8': 30, '15/16': 12});bound = plan({'4': 10, '8': 5})
    checks.append(check('kappa_plan_edge_interior_bounds', up == [('4', 'EDGE_EXTEND_UP', 'KAPPA_PROPOSAL')]
                        and down == [('4/9', 'EDGE_EXTEND_DOWN', 'KAPPA_PROPOSAL')]
                        and mid == [('5/6', 'INTERIOR_REFINE', 'KAPPA_PROPOSAL'), ('3/2', 'INTERIOR_REFINE', 'KAPPA_PROPOSAL')]
                        and fine == [] and bound == [('16', 'EDGE_EXTEND_UP', 'KAPPA_PROPOSAL_NOT_REQUESTED')],
                        edge_up=up, edge_down=down, interior=mid, fine_spacing=fine, outside_bounds=bound))

    # Replay: only a subset is answered; unanswered proposals are UNEVALUATED and not counted.
    answered = {}
    full = P.run_priority('PR_PAULI', [dict(c) for c in cands], seeded(cands, f), O.FixtureBackend(f, TAU), SEARCH, TAU, none_generator, None,
                          baseline=dict(formula='2nd', delta_over_h='1', kappa_multiplier='1'))
    for i, t in enumerate(full['trace']):
        if i % 5 != 2:  # drop every fifth answer
            answered[(t['configuration_id'], tuple(t['vector']))] = dict(status=t['status'], MSE_lower=t['MSE_lower'], MSE_upper=t['MSE_upper'],
                                                                        source='LOG', calls=1)

    class Log:
        def answer(self, p, allow_new=True):
            hit = answered.get((p['configuration_id'], tuple(p['vector'])))
            return None if hit is None else dict(hit)
    r = P.run_priority('PR_PAULI', [dict(c) for c in cands], seeded(cands, f), Log(), SEARCH, TAU, none_generator, None,
                       baseline=dict(formula='2nd', delta_over_h='1', kappa_multiplier='1'), on_missing='SKIP',
                       free=lambda p: (p['configuration_id'], tuple(p['vector'])) in answered)
    checks.append(check('replay_unanswered_is_unevaluated_and_not_counted', r['unevaluated'] and all(u['status'] == 'UNEVALUATED_NO_ANSWER_IN_LOG'
                        for u in r['unevaluated']) and sum(t['calls'] for t in r['trace']) == len(r['trace'])
                        and all((t['configuration_id'], tuple(t['vector'])) in answered for t in r['trace']),
                        unevaluated=len(r['unevaluated']), picks=len(r['trace']), stop=r['stop_reason']))

    counts = {}
    for k in checks:
        counts[k['status']] = counts.get(k['status'], 0) + 1
    res = dict(schema='RPE_PRIORITY_SEARCH_ACCEPTANCE_V1', source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               counts=counts, checks=checks, science='ARTIFICIAL configurations and MSE; no Q3')
    (out / 'PRIORITY_ACCEPTANCE.json').write_text(json.dumps(O.jsonable(res), indent=1, ensure_ascii=False) + '\n')
    for k in checks:
        print('%-55s %s' % (k['check'], k['status']))
    print(json.dumps(counts))
    return 0 if set(counts) == {'PASS'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
