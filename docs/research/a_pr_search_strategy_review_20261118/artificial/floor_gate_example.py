"""Artificial example: METHOD_MINIMUM_PRIORITY_V2 at 5c8ced4b2 with and without an own-price floor gate on EXPLORE.

Floor gate (proposed): a configuration whose own one-shot-per-axis price floor sum(prices) is already >= the method
minimum M cannot produce any allocation cheaper than M (every allocation has >= 1 shot per axis and positive unit
prices).  Its candidates are moved to DEFERRED (tried only when nothing else is left), never dropped.  No MSE
monotonicity is assumed.  Everything here is artificial (labelled oracle, no H4 data, no Q3).
"""
import argparse, json, sys
from fractions import Fraction
from pathlib import Path

REVIEW = Path(sys.argv[sys.argv.index('--review-root') + 1]) if '--review-root' in sys.argv else None


def load(review_root, gate):
    sys.path.insert(0, str(Path(review_root) / 'src'))
    from trottertracks import rpe_method_optimizer as O
    import types
    path = Path(review_root) / 'src/trottertracks/rpe_priority_search.py'
    text = path.read_text()
    anchor = "            items = local(st);open_ = []\n"
    assert text.count(anchor) == 1
    if gate:
        text = text.replace(anchor, anchor + (
            "            if M is not None and O.cost(tuple(1 for _ in st.c['prices']), st.c['prices']) >= M:\n"
            "                groups['DEFERRED'].extend(dict(p, cls='DEFERRED', floor_gate=True) for p in items)\n"
            "                st.promising = 0\n"
            "                continue\n"))
    mod = types.ModuleType('rps_gate' if gate else 'rps_base'); mod.__file__ = str(path)
    exec(compile(text, str(path), 'exec'), mod.__dict__)
    return O, mod


def scenario():
    w = [Fraction(4 ** m, 4 ** 12) for m in range(13)]           # artificial round weights of the MSE oracle
    base = [Fraction(300 + 40 * 2 ** m) for m in range(13)]       # artificial unit prices, growing with m
    def prices(scale): return [p * scale for m, p in enumerate(base) for _ in 'XY']
    caps = [60] * 13
    cfg = {
        'A_leader': dict(prices=prices(1), kappa='2', LD=3, w=w, s=Fraction(1, 1)),
        'B_floor_above_M': dict(prices=prices(60), kappa='64', LD=3, w=w, s=Fraction(1, 1)),
        'B2_floor_above_M': dict(prices=prices(45), kappa='32', LD=3, w=w, s=Fraction(1, 1)),
        'C_cheaper_kappa': dict(prices=prices(Fraction(4, 5)), kappa='3/2', LD=3, w=w, s=Fraction(23, 20)),
    }
    tau = Fraction(1, 12)
    return cfg, caps, tau


def run(review_root, gate, budget):
    O, P = load(review_root, gate)
    cfg, caps, tau = scenario()
    cands = []
    for name, c in cfg.items():
        cands.append(dict(configuration_id=name, name='ART:' + name, method='ART', family='PR', ready=True, needs=[],
                          prices=c['prices'], caps=caps, key=dict(L_D=c['LD'], formula='2nd', delta_over_h='1'),
                          kappa_value=c['kappa'], labels=[], saved_points=[]))
    def mse(name, v):
        c = cfg[name]
        return c['s'] * sum(wm * (Fraction(1, v[2 * m]) + Fraction(1, v[2 * m + 1])) / 2 for m, wm in enumerate(c['w']))
    calls = []
    class Oracle:                                                   # ARTIFICIAL answer; one new call per new allocation
        def answer(self, p, allow_new=True):
            x = mse(p['configuration_id'], p['vector']); calls.append((p['configuration_id'], p['cost']))
            return dict(status=O.PASS if x <= tau else O.FAIL, MSE_lower=x, MSE_upper=x, source='ARTIFICIAL', calls=1)
    v0 = tuple(n for n in [40, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18] for _ in 'XY')
    knowledge = {('A_leader', v0): dict(status=O.PASS, MSE_lower=mse('A_leader', v0), MSE_upper=mse('A_leader', v0),
                                         source='SAVED', calls=0, label='seed')}
    assert knowledge[('A_leader', v0)]['MSE_upper'] <= tau
    search = dict(decrement_steps=[1, 2], large_step_headroom='9/10', seed_rules=[], max_steps_per_method=10 ** 6,
                  explore_every=4, small_gain_fraction='1/1000', coarse_gain_fraction='1/100', lookahead=1)
    out = P.run_priority('ART', cands, knowledge, Oracle(), search, tau, lambda *a: None, budget, on_missing='STOP')
    M0 = O.cost(v0, cfg['A_leader']['prices'])
    floors = {n: O.cost((1,) * 26, c['prices']) for n, c in cfg.items()}
    curve, best = [], M0
    for t in out['trace']:
        if t['method_min_after'] is not None: best = min(best, t['method_min_after'])
        curve.append(dict(step=t['step'], configuration=t['configuration_id'], cls=t['cls'], cost=str(t['cost']), status=t['status'][:20],
                          method_min=str(best)))
    by = {}
    for t in out['trace']: by[t['configuration_id']] = by.get(t['configuration_id'], 0) + 1
    return dict(gate=gate, budget=budget, stop=out['stop_reason'], new_calls=out['new_calls'], q3_by_configuration=by,
                initial_M=str(M0), final_M=str(best), floors_over_initial_M={n: round(float(f / M0), 3) for n, f in floors.items()},
                curve=curve)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--review-root', required=True); ap.add_argument('--budget', type=int, default=24)
    ap.add_argument('--out'); a = ap.parse_args()
    res = [run(a.review_root, g, a.budget) for g in (False, True)]
    summary = [dict({k: r[k] for k in ('gate', 'budget', 'stop', 'new_calls', 'q3_by_configuration', 'initial_M', 'final_M', 'floors_over_initial_M')},
                    M_after_each_8=[r['curve'][i]['method_min'] for i in range(7, len(r['curve']), 8)]) for r in res]
    print(json.dumps(summary, indent=1))
    if a.out: Path(a.out).write_text(json.dumps(res, indent=1) + '\n')
