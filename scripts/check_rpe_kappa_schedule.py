"""Checks of the schedule-aware kappa search and the C/A common-entry price step (no signal, no Q3).

  mapper      R_m = max(J, ceil(ceil(kappa c_m) / q_m)) reproduces every saved generated configuration (Delta = h and widths)
  intervals   next_distinct: the schedule is constant up to the breakpoint and changes right after it (exact, both
              directions); no third schedule lies between the centre and the representative (dense rational sample)
  plan        with schedules: aliases are one point, first neighbours = first DIFFERENT schedules (LD5: only up, kappa 7;
              LD3: 43/22 and 33/16), an infeasible family stops after them, a sampled-schedule midpoint is moved on;
              without schedules the rule is the committed one (same proposals/decisions as the HEAD version)
  widths      the string "1/5" and the number 0.2 reach A as given and give different q_m / actual widths
  price       driver price step through the C/A entry: Delta = h points -> REUSE_SUPPLIED_PRICES_NO_CALIBRATION (proceed);
              binary64 0.2 points -> held with their pooled calibration cell recorded
Writes KAPPA_SCHEDULE_CHECKS.json into --output.
"""
import argparse
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from trottertracks import rpe_kappa_schedule as KS  # noqa: E402
from trottertracks import rpe_priority_search as P  # noqa: E402

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)
SIG = Path('/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/b2-pauli-kappa-signals-20261116/docs/research/b2_pauli_kappa_signals_20261116')
D = 'docs/research/rpe_method_optimizer_h4_20261116'
NB = dict(center='2', step_ratio='3/2', bounds=['1/2', '8'], min_ratio='9/8', extend_margin='1/100', settle_after_picks=80,
          family_window='3/2')


def saved_configs():
    out = []
    for r in ('kappa_abs_round02', 'kappa_abs_round03'):
        for f in sorted((SIG / r / 'configurations').glob('*.json')):
            out.append((f.name, json.loads(f.read_text())))
    for f in sorted((ROOT / D / 'kappa_abs_round04/width_requests/configurations').glob('*.json')):
        out.append((f.name, json.loads(f.read_text())))
    return out


def head_plan_module():
    src = subprocess.check_output(['git', 'show', 'HEAD:src/trottertracks/rpe_priority_search.py'], cwd=ROOT, text=True)
    mod = types.ModuleType('rpe_priority_search_head');mod.__dict__['__name__'] = 'rpe_priority_search_head'
    exec(compile(src, 'HEAD:rpe_priority_search.py', 'exec'), mod.__dict__)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__);ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--skip-price', action='store_true');a = ap.parse_args();a.output.mkdir(parents=True, exist_ok=True)
    res = {};cfgs = saved_configs()
    fam = {}
    # mapper
    bad = []
    for name, c in cfgs:
        rows = KS.constants(c);k = KS._frac(c['spec']['kappa']['value'])
        if KS.R_vector(rows, k) != [r['R_outer'] for r in c['rounds']]:
            bad.append(name)
        if KS._frac(c['spec']['delta_pf']) == KS._frac(c['spec']['h']):
            fam.setdefault(c['spec']['L_D'], KS.encode_rows(rows))
    res['mapper'] = dict(ok=not bad and len(cfgs) >= 18, configurations=len(cfgs), mismatches=bad)
    # intervals
    sch = KS.Sched(fam, NB['bounds']);two = Fraction(2);bad = [];found = {};rnd = random.Random(7)
    tiny = Fraction(1, 10 ** 30)
    for ld in sorted(fam):
        rows = sch.f[ld];R0 = KS.R_vector(rows, two)
        for d in ('down', 'up'):
            x = KS.next_distinct(rows, two, d)
            if x is None:
                found[(ld, d)] = None
                if d == 'down' and any(R0[i] > r['J'] for i, r in enumerate(rows)):
                    bad.append((ld, d, 'none but a round is above its floor'))
                continue
            lo, hi = x['interval'];R1 = x['R'];found[(ld, d)] = (str(x['kappa']), x['changed_rounds'])
            if not (lo < x['kappa'] <= hi) or KS.R_vector(rows, x['kappa']) != R1 or R1 == R0:
                bad.append((ld, d, 'representative'))
            if KS.R_vector(rows, hi) != R1 or KS.R_vector(rows, lo + tiny) != R1:
                bad.append((ld, d, 'interval not one schedule'))
            if d == 'up' and (KS.R_vector(rows, lo) != R0 or KS.R_vector(rows, hi + tiny) == R1):
                bad.append((ld, d, 'up breakpoints'))
            if d == 'down' and (KS.R_vector(rows, hi + tiny) != R0 or KS.R_vector(rows, lo) == R1):
                bad.append((ld, d, 'down breakpoints'))
            a_, b_ = (two, x['kappa']) if d == 'up' else (x['kappa'], two)
            for _ in range(400):  # dense sample between centre and representative: only the two schedules occur
                k = a_ + (b_ - a_) * Fraction(rnd.randrange(1, 10 ** 6), 10 ** 6)
                if tuple(KS.R_vector(rows, k)) not in (tuple(R0), tuple(R1)):
                    bad.append((ld, d, 'third schedule at %s' % k));break
    expect = {(5, 'down'): None, (5, 'up'): ('7', [12]), (3, 'down'): ('43/22', [11, 12]), (3, 'up'): ('33/16', [12]),
              (2, 'down'): ('175/88', [12]), (2, 'up'): ('343/171', [11, 12]), (1, 'down'): ('531/266', [12]), (1, 'up'): ('683/341', list(range(5, 13)))}
    # LD5 up is bounded at 8 by the plan (Sched); next_distinct alone has no bounds -> compare via Sched
    found_b = {(ld, d): (lambda k: None if k is None else (str(k), sch.describe(ld, two, d)['changed_rounds']))(sch.next(ld, two, d))
               for ld in sorted(fam) for d in ('down', 'up')}
    res['intervals'] = dict(ok=not bad and all(found_b.get(k) == (tuple(v) if v else None) or found_b.get(k) == v for k, v in expect.items()),
                            problems=bad, first_distinct={'%d:%s' % k: v for k, v in found_b.items()})
    # plan with schedules
    def pt(ld, k, best=None, status='EVALUATED', promising=0):
        return dict(L_D=ld, kappa=Fraction(k), status=status, best=None if best is None else Fraction(best), promising=promising, picks=10)
    pts = [pt(5, '4/3', 500), pt(5, '2', 500), pt(5, '3', 500), pt(3, '4/3'), pt(3, '2'), pt(3, '3')]
    prop, dec, _ = P.kappa_neighbourhood_plan(pts, NB, sch)
    got = sorted((p['L_D'], str(p['kappa']), p['reason']) for p in prop)
    want = sorted([(5, '7', 'FIRST_DISTINCT_SCHEDULE_UP'), (3, '43/22', 'FIRST_DISTINCT_SCHEDULE_DOWN'), (3, '33/16', 'FIRST_DISTINCT_SCHEDULE_UP')])
    step2 = pts + [pt(5, '7', 480), pt(3, '43/22'), pt(3, '33/16')]
    prop2, dec2, _ = P.kappa_neighbourhood_plan(step2, NB, sch)
    d2 = {d['L_D']: d['decision'] for d in dec2}
    # a plateau [2, 7] with 7 cheaper by > 1%: the outward step 21/2 is outside the bounds -> refine towards 2; every kappa in
    # (2, 7) has 2's or 7's schedule, so nothing is proposed (the old rule would have proposed the midpoint 9/2 = 2's schedule)
    res['plan'] = dict(ok=got == want and d2.get(3) == 'STOP_FAMILY_NO_FEASIBLE_POINT' and d2.get(5) == 'STOP_FAMILY_RULE_EXHAUSTED' and not prop2,
                       first=got, after=d2, after_proposals=[(p['L_D'], str(p['kappa'])) for p in prop2])
    # snapping: LD3 points 2 (cheap) and 3 -> interior/edge midpoint moves to a schedule not yet sampled
    pts3 = [pt(3, '43/22', 900), pt(3, '2', 800), pt(3, '33/16', 850), pt(3, '3', 1000)]
    prop3, dec3, _ = P.kappa_neighbourhood_plan(pts3, NB, sch)
    sampled = {tuple(KS.R_vector(sch.f[3], p['kappa'])) for p in pts3}
    new = [p for p in prop3 if p['L_D'] == 3]
    # an outward step whose schedule is already sampled moves on: bounds [1/2, 16] (test only), LD5 points 2 and 7 (cheaper by > 1%)
    # -> the step 21/2 has 7's schedule, so the next different schedule above (12.88.., representative 13) is proposed
    nb16 = dict(NB, bounds=['1/2', '16']);s16 = KS.Sched(fam, nb16['bounds'])
    prop4, dec4, _ = P.kappa_neighbourhood_plan([pt(5, '2', 500), pt(5, '7', 480)], nb16, s16)
    moved = [(str(p['kappa']), p['reason']) for p in prop4]
    res['snap'] = dict(ok=all(tuple(KS.R_vector(sch.f[3], p['kappa'])) not in sampled for p in new) and not new
                       and s16.R(5, Fraction(21, 2)) == s16.R(5, 7) and moved == [('13', 'EDGE_EXTEND_ONE_STEP')],
                       adjacent_schedules_proposals=[(str(p['kappa']), p['reason']) for p in new], decisions=[d['decision'] for d in dec3],
                       moved_step=moved)
    # regression: no schedules -> the committed rule
    H = head_plan_module();cases = [pts, step2, pts3, [pt(2, '2', 700), pt(2, '4/3', 650), pt(2, '3', 720)],
                                    [pt(1, '2', 1000), pt(1, '3', 900)], [pt(4, '2', status='HELD')]]
    same = all(H.kappa_neighbourhood_plan(c, NB) == P.kappa_neighbourhood_plan(c, NB) for c in cases)
    res['regression_without_schedules'] = dict(ok=same, cases=len(cases))
    # widths kept apart
    w = json.loads((ROOT / D / 'kappa_abs_round04/width_requests/WIDTH_REQUESTS.json').read_text())
    req = json.loads(next((ROOT / D / 'kappa_abs_round04/width_requests/preparation').glob('width_requests_*/REQUEST.json')).read_text())
    types_ = sorted({type(i['changes']['delta_pf']).__name__ for i in req['items']})
    qv = {(r['L_D'], r['width']['label']): tuple(x['q'] for x in r['schedule']) for r in w['rows']}
    res['widths'] = dict(ok=types_ == ['float', 'str'] and all(qv[(ld, 'exact_1o5')] != qv[(ld, 'binary64_0p2')] for ld in (1, 2, 3, 5)),
                         request_value_types=types_, q_exact_1o5=qv[(5, 'exact_1o5')], q_binary64_0p2=qv[(5, 'binary64_0p2')])
    # price step through the C/A entry (real views, metadata only)
    if not a.skip_price:
        spec = importlib.util.spec_from_file_location('kdrv', ROOT / 'scripts/run_rpe_kappa_neighbourhood.py')
        K = importlib.util.module_from_spec(spec);spec.loader.exec_module(K)
        round_spec = json.loads((ROOT / 'configs/rpe_method_optimizer/rounds/h4_pr_pauli_kappa_schedule_round04.json').read_text())
        if True:  # the entry's outputs (QUOTES, BATCH_INPUT, BATCH) stay under --output/price_step for inspection
            drv = object.__new__(K.Driver);drv.spec = round_spec;drv.run = (a.output / 'price_step').resolve();drv.state = dict(cycle=1)
            drv.a = types.SimpleNamespace(problem=ROOT / 'configs/rpe_method_optimizer/h4_df10_problem.json');drv.log = lambda **e: None
            cands, points = [], []
            for name, c in cfgs:
                if 'kappa_abs_2.json' in name or 'binary64_0p2' in name:
                    cands.append(dict(configuration_id=c['id'], generated_configuration=c))
                    points.append(dict(L_D=c['spec']['L_D'], kappa=name, configuration_id=c['id'], status='PREPARED',
                                       width='binary64_0p2' if 'binary64' in name else 'h'))
            drv.price_batch(dict(candidates=cands), points)
            h_ok = all(p['status'] == 'PREPARED' and p['price_entry']['status'] == 'REUSE_SUPPLIED_PRICES_NO_CALIBRATION'
                       for p in points if p['width'] == 'h')
            w_ok = all(p['status'] == 'HELD' and p['price_entry']['status'] == 'CALIBRATION_REQUEST_READY_NOT_EXECUTED'
                       and any(r['action'] == 'CALIBRATION_REQUEST_READY_NOT_EXECUTED' for r in p['price_entry']['rounds'])
                       for p in points if p['width'] != 'h')
            cells = drv.state.get('calibration_cells') or []
            res['price'] = dict(ok=h_ok and w_ok and len(cells) == sum(1 for p in points if p['width'] != 'h') and len(points) >= 8,
                                points=[(p['L_D'], p['width'], p['status'], p['price_entry']['status'], p['price_entry']['reused_axes'])
                                        for p in points], cells=[(c['cell']['cell_id'], c['actual_step']) for c in cells])
    res['ok'] = all(v['ok'] for v in res.values() if isinstance(v, dict))
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'KAPPA_SCHEDULE_CHECKS.json').write_text(json.dumps(res, indent=1, default=str) + '\n')
    print(json.dumps({k: v['ok'] for k, v in res.items() if isinstance(v, dict)}), 'ALL PASS' if res['ok'] else 'FAIL')
    return 0 if res['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
