"""SCHEDULE_WIDE_V1: choose kappa by physical 13-round schedules over a wide range, without a fixed centre.

Families are (method, L_D, formula, width); within a family the exact mapper (rpe_kappa_schedule) turns kappa into the
13-round event counts R_m, so a schedule is an interval of kappa and every kappa of it is the same candidate.  kappa=2 and
the saved feasible points are ordinary sampled schedules; nothing is centred on them and no bound of an earlier round
(e.g. [1/2, 8]) is inherited.

Decisions (one call proposes up to ``per_decision`` schedules over all families; independent schedules can run in parallel):
  INIT_FLOOR_SCHEDULE   the schedule where every round has its minimum event count (kappa in (0, floor_end])
  INIT_FAR_SCHEDULE     a geometric ladder floor_end * ladder_ratio^j up to the family's range limit; a rung is skipped when a
                        sampled schedule lies within a factor sqrt(ladder_ratio) (saved feasible points count as sampled)
  REFINE                around the cheapest feasible schedule of each family within ``window`` x the global best: the
                        geometric midpoint towards a neighbour (cheaper neighbour first), moved to the next schedule not yet
                        sampled; resolved when the neighbour is the adjacent schedule or the kappa ratio <= refine_ratio
  EXTEND                a family's best is its highest (lowest) sampled schedule: one step explore_ratio outward
  EXPLORE               the widest gap (log kappa) between sampled schedules, any family, while it exceeds explore_ratio and
                        holds a schedule; interleaved with refinement (``exploit_per_explore`` refinements per exploration)
Range and stop bases, all reported per family:
  PROVEN_COST_FLOOR     with ``price_floor(family, R)`` (one shot per round and axis, the family's price rule) >= the global
                        best, no allocation of that schedule or of any larger kappa can be cheaper (prices grow with R)
  EMPIRICAL_UPPER_STOP  the last ``upper_stop_steps`` sampled schedules above the family best are all dearer by >= margin (or
                        have no feasible point) and their costs do not decrease -- an observation, not a proof
  NO_FEASIBILITY_PROGRESS  a family without a feasible point whose last ``upper_stop_steps`` schedules did not lower the
                        smallest MSE lower bound / tau -- an observation, not a proof
  RANGE_LIMIT           max_kappa (only when no other basis applies)
  RESOLVED              refinement reached adjacent schedules or the ratio limit; gaps are all <= explore_ratio
Point statuses used here: FEASIBLE (best = cheapest verified feasible cost), NO_FEASIBLE_FOUND (its shot search ended
without a feasible point; this is not a proof that every allocation fails), IN_PROGRESS, PENDING, UNAVAILABLE (inputs held or,
in a replay, no saved result).
"""
from fractions import Fraction
import math

from trottertracks import rpe_kappa_schedule as KS

RULE = 'SCHEDULE_WIDE_V1'
DEFAULTS = dict(ladder_ratio='4', explore_ratio='2', refine_ratio='33/32', exploit_per_explore=2, window='3/2',
                upper_stop_steps=2, upper_stop_margin='1/20', max_kappa='65536', per_decision=8)
TINY = Fraction(1, 10 ** 12)


def coarse_axes(scope):
    """Initial alternating coordinates and the final endpoint for each axis."""
    axes = {}
    for method, requested in scope.items():
        axes[method] = {}
        for name, values in requested.items():
            ordered = sorted({KS._frac(value) for value in values})
            if ordered:
                axes[method][name] = [str(value) for value in sorted(set(ordered[::2]) | {ordered[-1]})]
    return axes


def coarse_priority(axes, method, coordinates):
    """Rank a ready candidate without excluding intermediate coordinates."""
    return int(any(coordinates.get(name) is None or str(KS._frac(coordinates[name])) not in values
                   for name, values in axes.get(method, {}).items()))


def requested_widths(problem, spec):
    """Absolute requested PF widths, preserving exact JSON numbers.

    A missing declaration means the historical single-width entry.  In
    particular a JSON float is never parsed through str(float).
    """
    raw = spec.get('search_overrides', {}).get('requested_widths',
        problem.get('search', {}).get('requested_widths'))
    if raw is None:
        return None
    if not isinstance(raw, dict) or not raw:
        raise ValueError('requested_widths must map PR method names to nonempty lists')
    out = {}
    for method, values in raw.items():
        if method not in ('PR_PAULI', 'PR_DF') or method not in spec.get('methods', raw):
            raise ValueError('Unsupported requested-width method: '+method)
        if not isinstance(values, list) or not values:
            raise ValueError('Each method needs at least one requested PF width')
        seen = set(); out[method] = []
        for value in values:
            width = KS._frac(value)
            if width <= 0:
                raise ValueError('Requested PF width must be positive')
            if width not in seen:
                out[method].append(dict(raw=value, exact=str(width)))
                seen.add(width)
    return out


def width_family_key(method, ld, formula, delta, K=2):
    """A search family; kappa is deliberately absent from its identity."""
    return '%s|LD%d|%s|delta=%s|K%d' % (method, ld, formula, KS._frac(delta), K)


def settings(s):
    out = dict(DEFAULTS, **(s or {}))
    for k in ('ladder_ratio', 'explore_ratio', 'refine_ratio', 'window', 'upper_stop_margin', 'max_kappa'):
        out[k] = Fraction(str(out[k]))
    return out


def kappa_bounds(s):
    raw = (s or {}).get('kappa_bounds')
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError('kappa_bounds requires inclusive absolute lower/upper bounds')
    lo, hi = (KS._frac(v) for v in raw)
    if not 0 < lo <= hi:
        raise ValueError('Invalid positive inclusive absolute kappa bounds')
    return lo, hi


def kappa_values(s):
    """An explicit finite absolute-kappa grid, with exact membership."""
    raw = (s or {}).get('kappa_values')
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise ValueError('kappa_values requires a nonempty exact positive list')
    values = sorted({KS._frac(v) for v in raw})
    if values[0] <= 0:
        raise ValueError('kappa_values must be positive')
    bounds = kappa_bounds(s)
    if bounds is not None and any(not bounds[0] <= v <= bounds[1] for v in values):
        raise ValueError('kappa_values escape the requested bounds')
    return values


def kappa_allowed(s, value):
    value = KS._frac(value)
    grid, bounds = kappa_values(s), kappa_bounds(s)
    return ((grid is None or value in grid) and
            (bounds is None or bounds[0] <= value <= bounds[1]))


def plan_grid(families, points, s):
    """Enumerate only requested grid values, without off-grid representatives.

    Distinct requested kappas remain visible even if their physical schedules
    coincide. The ordinary completion/evidence identity path owns that reuse.
    No empirical trend closes unvisited grid points.
    """
    grid = kappa_values(s)
    settings_ = settings(s)
    points = [p for p in points if p['family'] in families and KS._frac(p['kappa']) in grid]
    seen = {(p['family'], KS._frac(p['kappa'])) for p in points}
    feasible = [p['best'] for p in points if p['status'] == 'FEASIBLE' and p.get('best') is not None]
    proposals, decisions, report = [], [], {}
    for key in sorted(families, key=str):
        mine = [p for p in points if p['family'] == key]
        missing = [v for v in grid if (key, v) not in seen]
        report[key] = dict(family=key, requested_kappa_values=list(grid),
                           sampled=len(mine), unsampled_kappa_values=missing,
                           upper_basis='EXPLICIT_DISCRETE_GRID',
                           refinement='OPEN' if missing else 'RESOLVED')
    # Distribute each wave across families before proceeding along the grid.
    for value in grid:
        for key in sorted(families, key=str):
            if (key, value) not in seen and len(proposals) < settings_['per_decision']:
                proposals.append(dict(family=key, kappa=value, reason='EXPLICIT_KAPPA_GRID'))
                decisions.append(dict(family=key, decision='EXPLICIT_KAPPA_GRID', kappa=str(value)))
    pending = any(p['status'] in ('PENDING', 'IN_PROGRESS') for p in points)
    stop = None if proposals else 'WAIT_FOR_PENDING' if pending else 'RULE_EXHAUSTED'
    return proposals, decisions, dict(stop=stop, global_best=min(feasible) if feasible else None,
                                     families=report, requested_kappa_values=list(grid))


def family_rows(lambda_R, h, delta, M=12, J=1):
    """Mapper constants without a configuration: T_m = 2^m h, q_m = ceil(T_m / delta), c_m = lambda_R^2 T_m^2 (exact)."""
    lam, h, delta = Fraction(lambda_R), Fraction(h), Fraction(delta)
    rows = []
    for m in range(M + 1):
        T = h * 2 ** m;q = math.ceil(T / delta)
        rows.append(dict(m=m, c=lam * lam * T * T, q=q, J=J))
    return rows


class Family:
    def __init__(self, key, rows):
        self.key, self.rows = key, [dict(m=r['m'], c=Fraction(r['c']), q=int(r['q']), J=int(r['J'])) for r in rows]
        up = KS.next_distinct(self.rows, TINY, 'up')
        self.floor_R = tuple(KS.R_vector(self.rows, TINY))
        self.floor_end = up['interval'][0] if up else None

    def R(self, k):
        return tuple(KS.R_vector(self.rows, Fraction(k)))

    def events(self, R):
        return sum(r['q'] * x for r, x in zip(self.rows, R))

    def floor_rep(self):
        return KS.simplest_between(0, self.floor_end) if self.floor_end else Fraction(1)

    def rep(self, k):
        """The simplest rational kappa of k's schedule (the same physical candidate as k)."""
        k = Fraction(k);up = KS.next_distinct(self.rows, k, 'up');down = KS.next_distinct(self.rows, k, 'down')
        start = down['interval'][1] if down else Fraction(0);end = up['interval'][0] if up else k * 2
        return KS.simplest_between(start, end) if start < end else k

    def next(self, k, direction):
        d = KS.next_distinct(self.rows, Fraction(k), direction)
        return None if d is None else d['kappa']

    def snap(self, k, toward, have_R, stop_at=None):
        """k, or the next schedule in direction ``toward`` not in have_R; None at / beyond stop_at or its schedule."""
        guard = 0
        while k is not None and self.R(k) in have_R and guard < 4096:
            k = self.next(k, toward);guard += 1
        if k is None or (stop_at is not None and ((toward == 'up' and k >= stop_at) or (toward == 'down' and k <= stop_at)
                                                  or self.R(k) == self.R(stop_at))):
            return None
        return k

    def between(self, a, b):
        """A schedule strictly between those of a < b exists."""
        n = self.next(a, 'up')
        return n is not None and n < b and self.R(n) != self.R(b)


class BoundedFamily(Family):
    """The same physical schedule mapper, restricted to a closed kappa domain."""
    def __init__(self, original, bounds):
        self.key, self.rows = original.key, original.rows
        self.lower, self.upper = bounds
        self.floor_R = self.R(self.lower)
        self.floor_end = min([self.upper] + [Fraction(n * r['q']) / r['c']
            for n, r in zip(self.floor_R, self.rows) if r['c'] > 0])

    def floor_rep(self):
        return self.lower

    def rep(self, k):
        representative = max(self.lower, min(self.upper, super().rep(k)))
        if self.R(representative) != self.R(k):
            raise ValueError('Bounded representative changed the physical schedule')
        return representative

    def next(self, k, direction):
        row = KS.next_distinct(self.rows, k, direction, [self.lower, self.upper])
        if row is None or not self.lower <= row['kappa'] <= self.upper:
            return None
        return row['kappa']

    def snap(self, k, toward, have_R, stop_at=None):
        k = max(self.lower, min(self.upper, k))
        return super().snap(k, toward, have_R, stop_at)


def _log(x):
    return math.log(float(x))


def plan_wide(families, points, s=None, price_floor=None, *, price_floor_best=None):
    """families: {key: Family}; points: dicts(family, kappa (Fraction), status, best (Fraction|None), min_lower_over_tau).
    Returns (proposals, decisions, report)."""
    if kappa_values(s) is not None:
        return plan_grid(families, points, s)
    bounds = kappa_bounds(s)
    if bounds is not None:
        families = {k: BoundedFamily(f, bounds) for k, f in families.items()}
        points = [p for p in points if bounds[0] <= p['kappa'] <= bounds[1]]
    s = settings(s)
    if bounds is not None:
        s['max_kappa'] = bounds[1]
    per_family = {k: [] for k in families}
    for p in points:
        if p['family'] in per_family:
            per_family[p['family']].append(p)
    feas = [p['best'] for p in points if p['status'] == 'FEASIBLE' and p.get('best') is not None]
    gbest = min(feas) if feas else None
    cutoff = gbest if price_floor_best is None else price_floor_best
    exploit, explore, init, decisions, report = [], [], [], [], {}
    for key in sorted(families, key=str):
        f, ps = families[key], sorted(per_family[key], key=lambda p: p['kappa'])
        have_R = {f.R(p['kappa']) for p in ps}
        known = [p for p in ps if p['status'] in ('FEASIBLE', 'NO_FEASIBLE_FOUND')]
        fbest = min((p for p in ps if p['status'] == 'FEASIBLE' and p.get('best') is not None), key=lambda p: (p['best'], p['kappa']),
                    default=None)
        rep = dict(family=key, floor_end=f.floor_end, sampled=len(ps), evaluated=len(known),
                   unavailable=sum(1 for p in ps if p['status'] == 'UNAVAILABLE'),
                   kappa_range=[min((p['kappa'] for p in ps), default=None), max((p['kappa'] for p in ps), default=None)],
                   kappa_range_evaluated=[min((p['kappa'] for p in known), default=None), max((p['kappa'] for p in known), default=None)],
                   best=None if fbest is None else dict(kappa=fbest['kappa'], cost=fbest['best']))
        # range limit
        hi, basis = s['max_kappa'], 'RANGE_LIMIT'
        if price_floor is not None and cutoff is not None:
            k = f.floor_end or Fraction(1)
            while k <= s['max_kappa']:
                lb = price_floor(key, f.R(k))
                if lb is not None and lb >= cutoff:
                    hi, basis = k, 'PROVEN_COST_FLOOR';break
                k *= 2
        # empirical upper stop
        ref = fbest['kappa'] if fbest else None
        above = [p for p in known if ref is None or p['kappa'] > ref]
        tail = above[-s['upper_stop_steps']:] if s.get('stop_by_observed_trend', True) and len(above) >= s['upper_stop_steps'] else []
        if fbest and tail:
            costs = [p['best'] if p['status'] == 'FEASIBLE' else None for p in tail]
            dear = all(c is None or c >= fbest['best'] * (1 + s['upper_stop_margin']) for c in costs)
            mono = all(a is None or b is None or b >= a for a, b in zip(costs, costs[1:])) and (costs[-1] is None or costs[0] is None
                                                                                              or costs[-1] >= costs[0])
            if dear and mono and tail[-1]['kappa'] < hi:
                hi, basis = tail[-1]['kappa'], 'EMPIRICAL_UPPER_STOP'
        elif not fbest and tail:
            lows = [p.get('min_lower_over_tau') for p in known]
            if len(lows) > s['upper_stop_steps'] and all(x is not None for x in lows):
                last = lows[-(s['upper_stop_steps'] + 1):]
                if all(b >= a for a, b in zip(last, last[1:])) and tail[-1]['kappa'] < hi:
                    hi, basis = tail[-1]['kappa'], 'NO_FEASIBILITY_PROGRESS'
        rep.update(upper_limit=hi, upper_basis=basis)
        if bounds is not None:
            rep.update(requested_kappa_bounds=list(bounds), bounds_inclusive=[True, True])
        # INIT
        mine = []
        fr = f.floor_rep()
        if f.floor_R not in have_R:
            mine.append((fr, 'INIT_FLOOR_SCHEDULE'))
        if bounds is not None and f.R(bounds[1]) not in have_R and f.R(bounds[1]) != f.floor_R and bounds[1] <= hi:
            mine.append((bounds[1], 'INIT_UPPER_BOUND_SCHEDULE'))
        k = f.floor_end or Fraction(1);half = math.log(float(s['ladder_ratio'])) / 2
        taken = [p['kappa'] for p in ps] + [v for v, _ in mine] + [fr]
        while True:
            k = k * s['ladder_ratio']
            if k > hi:
                break
            if f.R(k) in have_R or any(abs(_log(k / t)) < half for t in taken):
                continue
            mine.append((k, 'INIT_FAR_SCHEDULE'));taken.append(k)
        if mine:
            init += [dict(family=key, kappa=k, reason=r) for k, r in mine]
            decisions.append(dict(family=key, decision='INIT', kappas=[str(k) for k, _ in mine]));report[key] = rep;continue
        # REFINE / EXTEND around the family best (families within window of the global best)
        resolved = True
        if fbest and gbest is not None and fbest['best'] <= s['window'] * gbest:
            i = ps.index(fbest);sides = []
            for j, toward in ((i - 1, 'down'), (i + 1, 'up')):
                if 0 <= j < len(ps):
                    nb = ps[j];c = nb.get('best') if nb['status'] == 'FEASIBLE' else None
                    sides.append((0 if c is not None and c <= fbest['best'] * (1 + s['upper_stop_margin']) else 1, toward, nb))
            sides.sort(key=lambda x: (x[0], x[1]))
            for _, toward, nb in sides:
                a, b = (nb['kappa'], fbest['kappa']) if toward == 'down' else (fbest['kappa'], nb['kappa'])
                if b / a <= s['refine_ratio'] or not f.between(a, b):
                    continue
                mid = Fraction(math.sqrt(float(a * b))).limit_denominator(10 ** 6)
                mid = max(min(mid, b), a)
                k = f.snap(mid, toward, have_R, stop_at=nb['kappa'])
                if k is None:  # the midpoint's schedule is sampled: the first schedule next to the best on that side
                    k = f.snap(fbest['kappa'], toward, have_R, stop_at=nb['kappa'])
                if k is not None and basis == 'PROVEN_COST_FLOOR' and k > hi:
                    k = None  # no allocation of a schedule above the proven floor limit can be cheaper than the global best
                if k is not None:
                    exploit.append((fbest['best'], dict(family=key, kappa=k, reason='REFINE_' + toward.upper(), around=fbest['kappa'],
                                                        neighbour=nb['kappa'])))
                    resolved = False;break
            if resolved:
                top, low = ps[-1], ps[0]
                if fbest is top and fbest['kappa'] * s['explore_ratio'] <= hi:
                    k = f.snap(fbest['kappa'] * s['explore_ratio'], 'up', have_R)
                    if k is not None and k <= hi:
                        exploit.append((fbest['best'], dict(family=key, kappa=k, reason='EXTEND_UP', around=fbest['kappa'])));resolved = False
                if resolved and fbest is low and f.R(fbest['kappa']) != f.floor_R:
                    k = f.snap(fbest['kappa'] / s['explore_ratio'], 'down', have_R)
                    if k is not None:
                        exploit.append((fbest['best'], dict(family=key, kappa=k, reason='EXTEND_DOWN', around=fbest['kappa'])));resolved = False
        rep['refinement'] = 'RESOLVED' if resolved else 'OPEN'
        # EXPLORE gaps (below the family's upper limit)
        gaps = 0
        for a, b in zip(ps, ps[1:]):
            if a['kappa'] >= hi:
                break
            top = min(b['kappa'], hi)
            if top / a['kappa'] > s['explore_ratio'] and f.between(a['kappa'], top):
                k = f.snap(Fraction(math.sqrt(float(a['kappa'] * top))).limit_denominator(10 ** 6), 'up', have_R, stop_at=top)
                if k is not None:
                    explore.append((-_log(top / a['kappa']), dict(family=key, kappa=k, reason='EXPLORE_GAP', gap=[a['kappa'], top])));gaps += 1
        if ps and ps[-1]['kappa'] * s['explore_ratio'] <= hi and basis != 'EMPIRICAL_UPPER_STOP':
            k = f.snap(ps[-1]['kappa'] * s['explore_ratio'], 'up', have_R)
            if k is not None and k <= hi:
                explore.append((-_log(hi / ps[-1]['kappa']), dict(family=key, kappa=k, reason='EXPLORE_UP', gap=[ps[-1]['kappa'], hi])));gaps += 1
        rep['open_gaps'] = gaps;report[key] = rep
    exploit.sort(key=lambda x: (x[0], str(x[1]['family'])));explore.sort(key=lambda x: (x[0], str(x[1]['family'])))
    out, seen = list(init), {(p['family'], families[p['family']].R(p['kappa'])) for p in init}
    ex, xp = [p for _, p in exploit], [p for _, p in explore];n = 0
    while len(out) < max(s['per_decision'], len(init)) and (ex or xp):
        take = ex if ex and (n % (s['exploit_per_explore'] + 1) != s['exploit_per_explore'] or not xp) else xp
        p = take.pop(0);n += 1
        k = (p['family'], families[p['family']].R(p['kappa']))
        if k not in seen:
            seen.add(k);out.append(p)
    if price_floor is not None and price_floor_best is not None:
        kept=[]
        for p in out:
            lower=price_floor(p['family'],families[p['family']].R(p['kappa']))
            if lower is not None and lower>=cutoff:
                decisions.append(dict(family=p['family'],decision='PROVEN_COST_FLOOR',
                    kappa=str(p['kappa']),lower=str(lower),incumbent=str(cutoff)))
            else:
                kept.append(p)
        out=kept
    for p in out:
        p['kappa'] = families[p['family']].rep(p['kappa'])
        if bounds is not None and not bounds[0] <= p['kappa'] <= bounds[1]:
            raise ValueError('Physical schedule proposal escaped the requested absolute domain')
        if p['reason'] not in ('INIT_FLOOR_SCHEDULE', 'INIT_FAR_SCHEDULE'):
            decisions.append(dict(family=p['family'], decision=p['reason'], kappa=str(p['kappa'])))
    pending = [p for p in points if p['status'] in ('PENDING', 'IN_PROGRESS')]
    stop = None
    if not out:
        stop = 'WAIT_FOR_PENDING' if pending else 'RULE_EXHAUSTED'
    return out, decisions, dict(stop=stop, global_best=gbest, families=report)
