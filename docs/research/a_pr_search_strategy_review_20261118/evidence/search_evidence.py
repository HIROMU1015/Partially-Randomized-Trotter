"""Read-only evidence from saved PR search runs (no Q3, no signal, no price fit).

For each run directory:
  * every SEARCH_ADOPT (adopted Q3 result) with its configuration's own one-shot-per-axis price floor sum(prices) from the
    completion's FINAL_QUOTE, compared with the method minimum recorded in the same event;
  * the move kind (ENQUEUE origin) of each adopted job and its verdict; the method-minimum improvements;
  * for the KAPPA_STATE backlog: proposals whose family already has a quoted smaller-or-equal schedule with floor >= the
    method best (the round05 Driver's price_floor rule, which WidthDriver does not pass to plan_wide).

    python search_evidence.py --review-root <5c8ced4b2 checkout> --run NAME=DIR ... --state DIR --output EVIDENCE.json
"""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import glob
import json
from pathlib import Path
import sys


def fr(x):
    if x is None or x == 'None':
        return None
    if isinstance(x, dict):
        return Fraction(int(x['numerator']), int(x['denominator']))
    return Fraction(str(x).replace(',', ''))


def quotes(runs):
    out = {}
    for run in runs:
        for f in glob.glob(str(Path(run) / 'completion/candidates/*/FINAL_QUOTE.json')):
            q = json.load(open(f))
            if q['complete']:
                out[q['configuration_id']] = dict(floor=sum(fr(r['unit_price_RZ']) for r in q['rows']), method=q['method'],
                                                  L_D=q['L_D'], R=tuple(r['R'] for r in q['rows'][::2]),
                                                  rules=sorted({(r.get('model_alternative') or {}).get('rule', r['classification'])
                                                                for r in q['rows']}))
    return out


def adopted(run, floor):
    ev = [json.loads(line) for line in open(Path(run) / 'PROGRESS.jsonl')]
    enq = {e['job']: e for e in ev if e.get('event') == 'ENQUEUE'}
    pending, rows = defaultdict(list), []
    for e in ev:
        if e.get('event') == 'ADOPT':
            pending[(e['method'], e['configuration'][:8])].append(e)
        elif e.get('event') == 'SEARCH_ADOPT':
            q = pending.get((e['method'], e['configuration'][:8]))
            a = q.pop(0) if q else None
            rows.append((e, a, enq.get(a['job'], {}).get('origin') if a else None))
    floor_tags, moves, improvements, by_cfg = Counter(), Counter(), [], Counter()
    previous = {}
    for e, a, origin in rows:
        m = e['method']; M = fr(e.get('method_min'))
        cid = next((k for k in floor if k.startswith(e['configuration'])), None)
        if cid is None or M is None:
            tag = 'NO_QUOTE_IN_COMPLETION'
        else:
            tag = 'OWN_FLOOR_GE_METHOD_MIN' if floor[cid]['floor'] >= M else 'OWN_FLOOR_LT_METHOD_MIN'
            if tag == 'OWN_FLOOR_GE_METHOD_MIN':
                by_cfg[(m, e['configuration'], floor[cid]['L_D'], str(round(float(floor[cid]['floor'] / M), 2)))] += 1
        floor_tags[(m, e.get('cls'), tag)] += 1
        moves[(m, origin, 'PASS' if e['status'].startswith('MODEL_BOUND_MEETS') else 'FAIL' if e['status'].startswith('MODEL_BOUND_EXCEEDS') else e['status'])] += 1
        if m in previous and M is not None and previous[m] is not None and M < previous[m]:
            improvements.append(dict(method=m, UTC=e['UTC'], before=str(previous[m]), after=str(M), cls=e.get('cls'), origin=origin,
                                     configuration=e['configuration']))
        previous[m] = M
    return dict(search_adopt=len(rows), floor_classes={'|'.join(map(str, k)): v for k, v in sorted(floor_tags.items())},
                q3_on_own_floor_ge_min={'|'.join(map(str, k)): v for k, v in sorted(by_cfg.items(), key=lambda x: -x[1])},
                move_kinds={'|'.join(map(str, k)): v for k, v in sorted(moves.items())}, improvements=improvements,
                final_method_min={m: str(v) for m, v in previous.items()})


def backlog(review_root, state_dir, floor):
    sys.path.insert(0, str(Path(review_root) / 'src'))
    from trottertracks import rpe_schedule_search as W
    ks = json.load(open(Path(state_dir) / 'KAPPA_STATE.json'))
    fams = {k: W.Family(k, f['rows']) for k, f in ks['families'].items() if f['status'] == 'READY_METADATA'}
    best = {}
    for p in ks['points']:
        c = [fr(v) for v in (p.get('best'), p.get('confirmed_all_best')) if v is not None]
        if c:
            best[p['method']] = min(best.get(p['method'], min(c)), min(c))
    quoted = defaultdict(list)
    for p in ks['points']:
        q = floor.get(p.get('configuration_id'))
        if q and p['family'] in fams and fams[p['family']].R(Fraction(p['kappa'])) == q['R']:
            quoted[p['family']].append((q['R'], q['floor'], p['kappa']))
    tags = Counter(); reasons = Counter()
    for p in ks['points']:
        if p['family'] not in fams:
            continue
        R = fams[p['family']].R(Fraction(p['kappa']))
        lbs = [fl for R0, fl, _ in quoted[p['family']] if all(a <= b for a, b in zip(R0, R))]
        tag = ('NO_QUOTED_SMALLER_SCHEDULE' if not lbs else
               'DOMINATED_FLOOR_GE_METHOD_BEST' if max(lbs) >= best[p['method']] else 'FLOOR_BELOW_METHOD_BEST')
        tags[(p['method'], p['status'], tag)] += 1
        if p['status'] == 'COMPLETION':
            reasons[(p['method'], p.get('request_reason'), tag)] += 1
    return dict(state=str(state_dir), method_best={m: str(v) for m, v in best.items()}, points=len(ks['points']),
                classes={'|'.join(map(str, k)): v for k, v in sorted(tags.items())},
                completion_backlog_by_reason={'|'.join(map(str, k)): v for k, v in sorted(reasons.items())},
                quoted_families=len(quoted), quoted_points=sum(len(v) for v in quoted.values()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--review-root', required=True); ap.add_argument('--run', action='append', required=True)
    ap.add_argument('--state', required=True); ap.add_argument('--output', required=True)
    a = ap.parse_args()
    runs = dict(r.split('=', 1) for r in a.run)
    floor = quotes(runs.values())
    out = dict(quoted_configurations=len(floor), runs={n: adopted(d, floor) for n, d in runs.items()},
               backlog=backlog(a.review_root, a.state, floor))
    Path(a.output).write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({n: dict(search_adopt=r['search_adopt'], floor_classes=r['floor_classes'], final=r['final_method_min'],
                              improvements=len(r['improvements'])) for n, r in out['runs'].items()}, indent=1))
    print(json.dumps(out['backlog']['completion_backlog_by_reason'], indent=1))


if __name__ == '__main__':
    main()
