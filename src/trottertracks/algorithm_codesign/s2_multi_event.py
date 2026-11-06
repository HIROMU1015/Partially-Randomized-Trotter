"""Finite ordered-pair adapter; no provider, matrix, RNG, or compiler calls."""
from fractions import Fraction as F
from time import perf_counter, process_time
import resource

from .rte_development.boundary import gate, simplify, cancellation_depth, native_price
from .parent_allocation import reweight_events, event_statistics, enclosed_root
from .return_aggregation import Interval
from .g8_bounds import ceil, log_upper

METRICS = ('T', 'CX', '1Q')
PRECISION = dict(root_bits=256, probability_bits=160, eta=F(1, 10**12), rho=F(1, 10**12))


def pair_table(bindings, cache):
    """Same boundary/suffix formula as D2, retaining individual pair identity."""
    irs, prices, suffixes, atoms = [], [], [], []
    for index, binding in enumerate(bindings):
        if binding['index'] != index:
            raise ValueError('event identity/order mismatch')
        e = binding['event']
        a, p, w = (F(e[k]) for k in ('coefficient', 'proposal', 'weight'))
        if a <= 0 or p <= 0 or p*w != a or e['phase_i_power'] not in (0, 1, 2, 3):
            raise ValueError('coefficient/proposal/weight/phase mismatch')
        ir = [gate(g) for g in binding['native_ir']]
        if simplify(ir) != ir:
            raise ValueError('step IR not internally reduced')
        price = native_price(ir, cache)
        if price != {k: binding['native_cost'][k] for k in METRICS}:
            raise ValueError('saved step price mismatch')
        suffix = [dict.fromkeys(METRICS, 0)]
        for g in reversed(ir):
            c = native_price([g], cache)
            suffix.append({k: suffix[-1][k]+c[k] for k in METRICS})
        irs.append(ir); prices.append(price); suffixes.append(suffix)
        atoms.append((a, p, w))
    if sum(z[1] for z in atoms) != 1:
        raise ValueError('complete Closed P5 law required')
    rows = []
    for i, left in enumerate(irs):
        for j, right in enumerate(irs):
            depth = cancellation_depth(left, right)
            add = {k: prices[i][k]+prices[j][k] for k in METRICS}
            bnd = {k: add[k]-2*suffixes[i][depth][k] for k in METRICS}
            if any(type(v) is not int or v < 0 for v in bnd.values()):
                raise ArithmeticError('negative/invalid boundary price')
            a, p, w = (atoms[i][k]*atoms[j][k] for k in range(3))
            rows.append(dict(i=i, j=j, coefficient=a, proposal=p, weight=w,
                             cancellation_depth=depth, A=add, B=bnd))
    return rows


def positive_t_prices(rows):
    bad = [(r['i'], r['j'], k) for r in rows for k in ('A', 'B') if r[k]['T'] <= 0]
    if bad:
        raise ValueError('ZERO_T_ANALYTIC_IS_INAPPLICABLE: '+repr(bad[:8]))


def normalizer_enclosure(rows, recipe):
    """Certificate for the same scores as Parent; group identical costs only."""
    grouped = {}
    for r in rows:
        c = r[recipe]['T']
        grouped[c] = grouped.get(c, F(0))+r['coefficient']
    lo = hi = F(0)
    for cost, coefficient in sorted(grouped.items()):
        rt = enclosed_root(F(cost), PRECISION['root_bits'], PRECISION['rho'])
        lo += coefficient/rt.hi
        hi += coefficient/rt.lo
    return Interval(lo, hi)


def evaluate_law(name, rows, remaining, alpha):
    wall, cpu = perf_counter(), process_time()
    events = [dict(coefficient=r['coefficient'], proposal=r['proposal'], weight=r['weight']) for r in rows]
    proposal_wall = normalizer_wall = 0.0
    normalizer = None
    if name != 'P0':
        recipe = {'qA': 'A', 'qB': 'B'}[name]
        start = perf_counter()
        normalizer = normalizer_enclosure(rows, recipe)
        normalizer_wall = perf_counter()-start
        start = perf_counter()
        events = reweight_events(events, mode='cost_aware', costs=[r[recipe]['T'] for r in rows], **PRECISION)
        proposal_wall = perf_counter()-start
    if remaining <= 0:
        raise ValueError('nonpositive remaining accuracy')
    start = perf_counter()
    stats = event_statistics(events)
    n = ceil(log_upper(2/alpha)*(2*stats['weight_second_moment']/remaining**2+4*stats['range']/(3*remaining)))
    stat_wall = perf_counter()-start
    cells = {}
    for recipe in ('A', 'B'):
        body = {k: sum((e['proposal']*r[recipe][k] for e, r in zip(events, rows)), F(0)) for k in METRICS}
        # X: H,H. Y: H,Sdg,H. Gate counts exclude system preparation.
        axes = {b: dict(body, **{'1Q': body['1Q']+extra}) for b, extra in (('X', 2), ('Y', 3))}
        cost = {k: n*sum(axes[b][k] for b in ('X', 'Y')) for k in METRICS}
        cells[recipe] = dict(body_expected=body, full_known_expected_by_axis=axes,
                             known_two_axis_cost=cost, unknown_common_h_slope=2*n)
    return dict(name=name, events=events, statistics=stats, N_per_axis=n,
                normalizer=normalizer, cells=cells,
                timing=dict(wall_seconds=perf_counter()-wall, cpu_seconds=process_time()-cpu,
                            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                            normalizer_certificate_wall_seconds=normalizer_wall,
                            proposal_with_internal_normalization_wall_seconds=proposal_wall,
                            statistics_and_N_wall_seconds=stat_wall))


def exact_json(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, Interval):
        return dict(lo=exact_json(value.lo), hi=exact_json(value.hi))
    if isinstance(value, dict):
        return {k: exact_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [exact_json(v) for v in value]
    return value
